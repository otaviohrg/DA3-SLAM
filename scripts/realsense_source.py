"""
Intel RealSense colour-stream frame source for DA3-SLAM.

Yields (RGB image, seq_idx, label) tuples — the FrameItem contract consumed by
DA3SLAM.run_stream().  DA3-SLAM is monocular RGB: only the colour stream is
used; the sensor's own depth is ignored (DA3 predicts depth/pose from colour).

The stream is unbounded — iteration ends when `stop_event` is set (wire this to
a SIGINT handler so Ctrl-C ends the stream cleanly, letting the pipeline run its
final graph optimisation and save outputs) or when `max_frames` is reached.

Real-time behaviour is handled by the driver, not by us: while DA3 inference is
busy the pipeline stops pulling frames, and pyrealsense2 drops the backlog — so
the next frame read is a fresh one, not a stale queue.  Keyframe selection is
optical-flow based, so processing frames farther apart in time is fine.

`.timestamps` records seq_idx → device-timestamp (seconds) for TUM export.

Requires `pyrealsense2` (installed in the demo image; see
requirements-realsense.txt).  Imported lazily so the rest of the package — and
this module's --help — loads without it.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import numpy as np

# Spelled out (not imported from da3_slam.slam) so --help works without torch.
FrameItem = tuple[np.ndarray, int, str]


class RealSenseFrameSource:
    """Iterable RealSense colour source yielding FrameItems.

    Args:
        width, height, fps: colour-stream profile (must be a mode the device
                            supports; 640x480x30 is universally available).
        serial:  specific device serial number, or None for the first camera.
        stop_event: when set, iteration stops after the current frame and the
                    pipeline is closed (end-of-stream → graph finalises + saves).
        max_frames: optional hard cap on emitted frames (for quick tests).
        exposure: manual colour exposure in device units (D4xx colour: tenths
                  of a millisecond).  Lower = less motion blur (at the cost of
                  darker frames); disables auto-exposure.  None keeps
                  auto-exposure but turns auto-exposure *priority* off, so AE
                  cannot stretch the exposure past the frame budget — the
                  cheap anti-blur setting for fast handheld motion.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: str | None = None,
        stop_event: threading.Event | None = None,
        max_frames: int | None = None,
        exposure: float | None = None,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.stop_event = stop_event
        self.max_frames = max_frames
        self.exposure = exposure

        self.timestamps: dict[int, float] = {}
        self.n_yielded = 0

    def _configure_exposure(self, rs, dev) -> None:
        """Apply the anti-motion-blur exposure policy to the colour sensor.

        Best-effort: option availability varies by device/firmware, and a
        failed set must never take down the stream.
        """
        try:
            color_sensor = dev.first_color_sensor() if dev else None
        except Exception:
            color_sensor = None
        if color_sensor is None:
            return
        try:
            if self.exposure is not None:
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                color_sensor.set_option(rs.option.exposure, float(self.exposure))
                print(f"[realsense] manual colour exposure {self.exposure:g} "
                      f"(auto-exposure off)", flush=True)
            elif color_sensor.supports(rs.option.auto_exposure_priority):
                # AE priority off: auto-exposure may not lengthen the exposure
                # past the frame budget — caps motion blur under fast motion
                # and keeps the frame rate constant.
                color_sensor.set_option(rs.option.auto_exposure_priority, 0)
                print("[realsense] auto-exposure priority off "
                      "(exposure capped at frame budget)", flush=True)
        except Exception as exc:
            print(f"[realsense] exposure setup skipped ({exc})", flush=True)

    def __iter__(self) -> Iterator[FrameItem]:
        import pyrealsense2 as rs  # lazy: only needed for the live demo

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        # rgb8 → get_data() is already RGB, matching the FrameItem contract
        # (no BGR→RGB conversion, unlike cv2.imread).
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
        )

        print(f"[realsense] opening {self.width}x{self.height}@{self.fps} "
              f"(format rgb8) ...", flush=True)
        profile = pipeline.start(config)
        dev = profile.get_device()
        name = dev.get_info(rs.camera_info.name) if dev else "RealSense"
        usb = (dev.get_info(rs.camera_info.usb_type_descriptor)
               if dev and dev.supports(rs.camera_info.usb_type_descriptor) else "?")
        print(f"[realsense] streaming from {name} (USB {usb})", flush=True)

        if str(usb).startswith("2"):
            print("[realsense] WARNING: device negotiated a USB 2.x link — the "
                  "D455 needs a USB-3 (SuperSpeed) port + data cable to stream. "
                  "Expect no/dropped frames; try a blue USB-3 port or --fps 15.",
                  flush=True)

        self._configure_exposure(rs, dev)

        seq_idx = 0
        heartbeat = max(self.fps, 1)  # ~one line per second of capture
        # Tolerate slow USB warmup, but surface a truly dead stream instead of
        # hanging or dying on a single miss.
        timeouts = 0
        max_timeouts = 10
        try:
            while self.stop_event is None or not self.stop_event.is_set():
                if self.max_frames is not None and seq_idx >= self.max_frames:
                    break
                # Bounded wait so a stalled camera surfaces instead of hanging
                # forever (5 s ≫ one frame interval at any sane FPS).
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                except RuntimeError as exc:  # librealsense timeout
                    timeouts += 1
                    print(f"[realsense] no frame ({timeouts}/{max_timeouts}): "
                          f"{exc}", flush=True)
                    if timeouts >= max_timeouts:
                        raise RuntimeError(
                            "RealSense delivered no frames. The device is on a "
                            f"USB {usb} link — connect it to a USB-3 (SuperSpeed) "
                            "port with a USB-3 data cable, or lower the profile "
                            "(--fps 15 / --width 424 --height 240)."
                        ) from exc
                    continue
                timeouts = 0
                color = frames.get_color_frame()
                if not color:
                    continue
                # Copy out of the RealSense-owned buffer (it is recycled on the
                # next wait_for_frames call while the pipeline still holds it).
                image = np.asanyarray(color.get_data()).copy()
                timestamp_s = color.get_timestamp() / 1000.0  # device ms → seconds
                self.timestamps[seq_idx] = timestamp_s
                self.n_yielded += 1
                if seq_idx % heartbeat == 0:
                    print(f"[realsense] captured {seq_idx + 1} frames — "
                          f"move the camera to add keyframes", flush=True)
                yield image, seq_idx, f"rs:{timestamp_s:.3f}"
                seq_idx += 1
        finally:
            pipeline.stop()
            print(f"[realsense] stopped after {seq_idx} frames", flush=True)

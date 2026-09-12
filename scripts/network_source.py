"""
Network frame source — the GPU host's end of the robot link.

Mirrors `RealSenseFrameSource` exactly: an iterable of `FrameItem`
`(RGB HxWx3 uint8, seq_idx, label)` with a `stop_event` for clean shutdown and
a `.timestamps` dict for TUM export.  Everything downstream of
`DA3SLAM.run_stream()` is therefore identical to a local-camera run, which is
the point — the robot test must exercise the same system that was benchmarked,
not a variant of it.

Pairs with `scripts/spot_frame_sender.py`; see that file for the wire format.

TIMESTAMPS
----------
`.timestamps` maps seq_idx -> the CAPTURE timestamp taken on the robot, never
the arrival time here.  Network jitter would otherwise be baked into the
trajectory's time base and make it unscoreable against an external reference.

SEQ_IDX GAPS ARE NORMAL
-----------------------
The sender drops the oldest frames under backpressure, so `seq_idx` is
strictly increasing but not contiguous.  That satisfies the pipeline's
contract (unique and increasing, keying the pose-graph node), and keyframe
selection handles the gaps — it measures optical flow between the frames it
actually receives.
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import cv2
import numpy as np

HEADER = struct.Struct("<4sIdI")
MAGIC = b"DA3F"


class NetworkFrameSource:
    """Iterable RGB source reading `spot_frame_sender.py` over TCP.

    Args:
        host, port: address of the sender running on the robot.
        stop_event: when set, iteration ends after the current frame so the
                    graph finalises and the trajectory is still saved.
        max_frames: optional cap (quick tests).
        connect_timeout: seconds to keep retrying the initial connection —
                    the robot-side sender is often started second.
        recv_timeout: seconds without data before the stream is considered
                    dead.  A walking robot on WiFi drops out; ending the
                    stream cleanly beats blocking forever, because the
                    pipeline still saves what it has.
    """

    def __init__(
        self,
        host: str,
        port: int = 5555,
        stop_event: threading.Event | None = None,
        max_frames: int | None = None,
        connect_timeout: float = 60.0,
        recv_timeout: float = 10.0,
    ):
        self.host = host
        self.port = port
        self.stop_event = stop_event or threading.Event()
        self.max_frames = max_frames
        self.connect_timeout = connect_timeout
        self.recv_timeout = recv_timeout

        # seq_idx -> capture timestamp (seconds), for TUM export
        self.timestamps: dict[int, float] = {}
        # `n_yielded` matches RealSenseFrameSource's attribute name, so the
        # entry points can treat the two sources interchangeably.
        self.n_yielded = 0
        self._sock: socket.socket | None = None

    # ── connection ────────────────────────────────────────────────────────────

    def _connect(self) -> socket.socket:
        deadline = time.time() + self.connect_timeout
        last: Exception | None = None
        attempt = 0
        while time.time() < deadline and not self.stop_event.is_set():
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5.0)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(self.recv_timeout)
                print(f"[network] connected to {self.host}:{self.port}")
                return sock
            except OSError as exc:
                last = exc
                attempt += 1
                if attempt == 1:
                    print(f"[network] waiting for sender at "
                          f"{self.host}:{self.port} ...")
                time.sleep(1.0)
        raise ConnectionError(
            f"could not connect to {self.host}:{self.port} within "
            f"{self.connect_timeout:.0f}s ({last})")

    def _recv_exactly(self, sock: socket.socket, n: int) -> bytes | None:
        """Read exactly n bytes, or None if the stream ended or stalled."""
        chunks = []
        remaining = n
        while remaining:
            try:
                chunk = sock.recv(remaining)
            except socket.timeout:
                print(f"[network] no data for {self.recv_timeout:.0f}s — "
                      f"treating the stream as ended")
                return None
            except OSError as exc:
                print(f"[network] socket error: {exc}")
                return None
            if not chunk:
                return None                      # sender closed
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    # ── iteration ─────────────────────────────────────────────────────────────

    def __iter__(self):
        self._sock = self._connect()
        emitted = 0
        t0 = time.time()
        try:
            while not self.stop_event.is_set():
                header = self._recv_exactly(self._sock, HEADER.size)
                if header is None:
                    break
                magic, seq, stamp, nbytes = HEADER.unpack(header)
                if magic != MAGIC:
                    print(f"[network] bad magic {magic!r} — stream desynced")
                    break
                payload = self._recv_exactly(self._sock, nbytes)
                if payload is None:
                    break

                bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8),
                                   cv2.IMREAD_COLOR)
                if bgr is None:
                    print(f"[network] frame {seq}: JPEG decode failed, skipping")
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                self.timestamps[seq] = stamp
                self.n_yielded += 1
                emitted += 1
                if emitted % 100 == 0:
                    print(f"[network] {emitted} frames  "
                          f"{emitted / max(time.time() - t0, 1e-6):.1f} fps in  "
                          f"latency {time.time() - stamp:.2f}s")

                yield rgb, int(seq), f"net:{seq}:{stamp:.6f}"

                if self.max_frames and emitted >= self.max_frames:
                    break
        finally:
            self.close()

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

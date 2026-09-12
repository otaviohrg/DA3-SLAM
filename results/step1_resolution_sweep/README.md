# Branch B · Step 1 — Resolution sweep (T1 + F1)

Fixed backbone `nested-giant`, resolutions {336, 392, 448, 504}, 3 repeats on
**frozen keyframes**. Headline metric: Sim3-aligned ATE (monocular). Repeat
noise band ≈ 1×10⁻⁵ m (bf16), far below every gap reported here. Single server
GPU (RTX 3090-class) — no Jetson, so latency/memory are server-only.

Interactive report (all three tables + figures + synthesis):
https://claude.ai/code/artifact/382a9590-8972-4d16-9a67-343a235f1cc3

## Contents
- `step1_report.html` — self-contained report (open in a browser).
- `tables/T1_<dataset>_final.json` — `{res: [ate_mean, repeat_band, backbone_s, peak_mb, map_points, chamfer_m]}`.
- `figures/F1_<dataset>.png` — ATE-vs-backbone-latency curve (+ map-detail overlay for TUM/Replica).
- `raw_rows/*.jsonl` — one row per run (config + ATE + per-stage latency + backbone-only latency + peak memory). Regenerate tables/figures with `scripts/plot_sweep.py` / `scripts/plot_f1.py`.

## T1 — TUM RGB-D (all 9 fr1 sequences)
| res | ATE Sim3 (m) | backbone (s) | peak mem (GB) | map points | Chamfer (m) |
|----:|:---:|---:|---:|---:|---:|
| 336 | 0.0538 | 10.3 | 10.4 | 3.74 M | 0.237 |
| 392 | 0.0504 | 14.2 | 11.1 | 4.84 M | 0.187 |
| **448** | **0.0482** | 19.8 | 11.9 | 6.32 M | 0.123 |
| 504 | 0.0509 | 23.7 | 12.9 | 7.99 M | native |

**Nearly free** — ATE flat across resolution; knee 448; clean trajectory-vs-map dissociation.

## T1 — Replica (8 indoor scenes)
| res | ATE Sim3 (m) | backbone (s) | peak mem (GB) | map points | Chamfer (m) |
|----:|:---:|---:|---:|---:|---:|
| 336 | 0.1527 | 11.1 | 10.0 | 4.30 M | 0.249 |
| 392 | 0.1110 | 15.5 | 10.4 | 5.73 M | 0.166 |
| 448 | 0.0792 | 20.0 | 11.0 | 7.36 M | 0.210 |
| **504** | **0.0650** | 25.2 | 11.6 | 9.18 M | native |

**Strong lever** — ATE halves with resolution; no free knee, 504 wins.

## T1 — UAS (4 GT sequences, fisheye, km-scale; no map-detail)
ATE as **% of trajectory length** (the honest metric at km-scale).

| res | ATE (% path) | backbone (s) | peak mem (GB) |
|----:|:---:|---:|---:|
| 336 | 9.1% | 94.3 | 10.6 |
| 392 | 7.9% | 128.8 | 11.4 |
| 448 | 9.8% | 169.1 | 12.3 |
| **504** | **6.7%** | 226.9 | 13.3 |

**Bistable — no knee.** Per-sequence ATE (% of path):

| sequence | 336 | 392 | 448 | 504 |
|---|---:|---:|---:|---:|
| fyllingsdalen_tunnel (1275 m) | 16.1% | 6.3% | 15.6% | 6.2% |
| hornbill (1427 m) | 5.1% | 4.8% | 5.8% | 5.5% |
| frozen_lake (826 m) | 9.4% | 10.4% | 9.1% | 8.6% |
| campus_fog (670 m, held-out) | 5.7% | 10.0% | 8.7% | 6.6% |

## Finding
The "free resolution knee" **does not generalize**: nearly free on TUM, a strong
monotonic lever on Replica, bistable on UAS (tunnels flip between tracking and
~200 m drift at adjacent resolutions). Static resolution reduction is a
domain-specific lever, not a safe default cut.

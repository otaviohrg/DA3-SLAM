# TUM RGB-D — uncalibrated monocular SLAM

ATE RMSE in metres, 9 Freiburg-1 sequences. Mean over completed repeats
(a repeat counts only when all 9 sequences scored). Lower is better.

Sim(3) = scale-free alignment, the standard monocular metric.
SE(3)  = rigid alignment, so it also penalises wrong metric scale.
`scale` is the Umeyama scale factor; 1.000 means metric-accurate.

Protocol notes:
- DROID-SLAM* intrinsics are GeoCalib-estimated per sequence from the first
  frame (the published uncalibrated protocol). It has no native uncalibrated
  mode. The heuristic fallback focal = max(W,H) is NOT used.
- Published column: VGGT-SLAM 2.0, arXiv:2601.19887, Table I (uncalibrated).
- EC3R-SLAM is excluded: its public repo ships recover_trajectory only as
  __pycache__/save_file.cpython-310.pyc and never uploaded save_file.py, so it
  crashes on 6 of 9 sequences and cannot be run faithfully.
- Repeats: AMB3R and DROID-SLAM* are stochastic; the rest reproduced
  bit-exact or to 3e-05, so their spread is 0 by nature, not by tolerance.

# TUM RGB-D — Sim(3) ATE RMSE (m)

Scale-free (Umeyama with scale) alignment — the standard monocular
metric, and the one the published comparison tables use.
Lower is better. Mean over completed repeats.

| system | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | **avg** | published avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DROID-SLAM* | 0.2052 | 0.0291 | 0.0914 | 0.0449 | 0.0574 | 0.5990 | 0.0586 | 0.0710 | 0.0128 | **0.1299** | 0.158 |
| MASt3R-SLAM* | 0.0699 | 0.0350 | 0.0548 | 0.0558 | 0.0347 | 0.1185 | 0.0406 | 0.1157 | 0.0196 | **0.0605** | 0.060 |
| ViSTA-SLAM | 0.1044 | 0.0299 | 0.0296 | 0.0698 | 0.0523 | 0.0676 | 0.0230 | 0.0799 | 0.0149 | **0.0524** | 0.052 |
| VGGT-SLAM 2.0 | 0.0602 | 0.0256 | 0.0265 | 0.0430 | 0.0329 | 0.0698 | 0.0290 | 0.0380 | 0.0198 | **0.0383** | 0.041 |
| AMB3R | 0.0516 | 0.0209 | 0.0297 | 0.0335 | 0.0319 | 0.0721 | 0.0245 | 0.0467 | 0.0116 | **0.0358** | — |
| DA3-Streaming | 0.0944 | 0.0797 | 0.0602 | 0.0862 | 0.1150 | 0.1828 | 0.0540 | 0.2135 | 0.0430 | **0.1032** | — |
| DASH-SLAM | 0.0565 | 0.0157 | 0.0214 | 0.0218 | 0.0258 | 0.0455 | 0.0201 | 0.0263 | 0.0071 | **0.0267** | — |

## Recovered scale and repeats

| system | mean scale | repeats |
|---|---|---|
| DROID-SLAM* | 0.989 | 3 |
| MASt3R-SLAM* | 0.919 | 3 |
| ViSTA-SLAM | 0.584 | 3 |
| VGGT-SLAM 2.0 | 1.605 | 3 |
| AMB3R | 1.563 | 3 |
| DA3-Streaming | 0.871 | 3 |
| DASH-SLAM | 1.017 | 3 |

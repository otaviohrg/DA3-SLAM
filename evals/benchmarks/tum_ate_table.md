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

## Sim(3) ATE RMSE (m) — headline

| system | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | **avg** | published avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DROID-SLAM* | 0.2052 | 0.0291 | 0.0914 | 0.0449 | 0.0574 | 0.5990 | 0.0586 | 0.0710 | 0.0128 | **0.1299** | 0.158 |
| MASt3R-SLAM* | 0.0699 | 0.0350 | 0.0548 | 0.0558 | 0.0347 | 0.1185 | 0.0406 | 0.1157 | 0.0196 | **0.0605** | 0.060 |
| ViSTA-SLAM | 0.1044 | 0.0299 | 0.0296 | 0.0698 | 0.0523 | 0.0676 | 0.0230 | 0.0799 | 0.0149 | **0.0524** | 0.052 |
| VGGT-SLAM 2.0 | 0.0602 | 0.0256 | 0.0265 | 0.0430 | 0.0329 | 0.0698 | 0.0290 | 0.0380 | 0.0198 | **0.0383** | 0.041 |
| AMB3R | 0.0516 | 0.0209 | 0.0297 | 0.0335 | 0.0319 | 0.0721 | 0.0245 | 0.0467 | 0.0116 | **0.0358** | — |
| DA3-Streaming | 0.0944 | 0.0797 | 0.0602 | 0.0862 | 0.1150 | 0.1828 | 0.0540 | 0.2135 | 0.0430 | **0.1032** | — |
| DASH-SLAM | 0.0565 | 0.0157 | 0.0214 | 0.0218 | 0.0258 | 0.0455 | 0.0201 | 0.0263 | 0.0071 | **0.0267** | — |

## SE(3) ATE RMSE (m) — scale-sensitive

| system | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | **avg** | published avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DROID-SLAM* | 0.2624 | 0.0605 | 0.1019 | 0.0474 | 0.3391 | 0.6833 | 0.1786 | 0.4809 | 0.0154 | **0.2411** | n/a |
| MASt3R-SLAM* | 0.1238 | 0.0438 | 0.0620 | 0.3821 | 0.0976 | 0.1446 | 0.0490 | 0.1742 | 0.0200 | **0.1219** | n/a |
| ViSTA-SLAM | 0.1940 | 0.9130 | 0.7411 | 1.4343 | 0.2632 | 0.8621 | 0.0669 | 0.0959 | 0.1268 | **0.5219** | n/a |
| VGGT-SLAM 2.0 | 0.0952 | 0.2113 | 0.2745 | 0.2922 | 0.3877 | 0.3328 | 0.0297 | 0.5709 | 0.0489 | **0.2492** | n/a |
| AMB3R | 0.0791 | 0.1652 | 0.2734 | 0.1450 | 0.4225 | 0.2666 | 0.0249 | 0.6594 | 0.0457 | **0.2313** | n/a |
| DA3-Streaming | 0.0990 | 0.0888 | 0.1249 | 0.4897 | 0.1381 | 0.1953 | 0.0831 | 0.2189 | 0.0471 | **0.1650** | n/a |
| DASH-SLAM | 0.0582 | 0.0705 | 0.0764 | 0.1783 | 0.0609 | 0.0527 | 0.0201 | 0.0903 | 0.0239 | **0.0701** | n/a |

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

## Published per-sequence reference (uncalibrated)

arXiv:2601.19887 Table I. For checking that baselines were reproduced,
not for restating as our measurements.

| system | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | avg |
|---|---|---|---|---|---|---|---|---|---|---|
| DROID-SLAM* | 0.202 | 0.032 | 0.091 | 0.064 | 0.045 | 0.918 | 0.056 | 0.045 | 0.012 | 0.163 |
| MASt3R-SLAM* | 0.070 | 0.035 | 0.055 | 0.056 | 0.035 | 0.118 | 0.041 | 0.114 | 0.020 | 0.060 |
| ViSTA-SLAM | 0.104 | 0.030 | 0.030 | 0.070 | 0.052 | 0.067 | 0.023 | 0.080 | 0.015 | 0.052 |
| VGGT-SLAM 2.0 | 0.050 | 0.025 | 0.029 | 0.102 | 0.026 | 0.063 | 0.026 | 0.038 | 0.014 | 0.041 |

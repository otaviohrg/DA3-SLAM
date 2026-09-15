# Replica — uncalibrated monocular SLAM

ATE RMSE in metres, 8 sequences. Mean over completed repeats
(a repeat counts only when every sequence scored). Lower is better.

Sim(3) = scale-free alignment, the standard monocular metric.
SE(3)  = rigid alignment, so it also penalises wrong metric scale.
`scale` is the Umeyama factor; 1.000 means metric-accurate.

Repeats: 1 for systems measured deterministic on TUM, 2 for the
stochastic ones (AMB3R, DROID-SLAM). Variance is not reported, so a
second identical run would add nothing.

## Sim(3) ATE RMSE (m)

Scale-free alignment — the metric the published tables use.

| system | room0 | room1 | room2 | office0 | office1 | office2 | office3 | office4 | **avg** | published |
|---|---|---|---|---|---|---|---|---|---|---|
| DROID-SLAM* | 1.0321 | 0.5542 | 0.6417 | 0.8034 | 0.4566 | 0.9039 | 1.1552 | 0.8769 | **0.8030** | 0.193 |
| ViSTA-SLAM | 0.0688 | 0.0934 | 0.1361 | 0.0744 | 0.1934 | 0.1177 | 0.0485 | 0.1302 | **0.1078** | — |
| MASt3R-SLAM* | 0.0298 | 0.0375 | 0.0934 | 0.0320 | 0.0415 | 0.0445 | 0.0365 | 0.0451 | **0.0450** | 0.045 |
| AMB3R | 0.0159 | 0.0420 | 0.0407 | 0.0468 | 0.0358 | 0.0599 | 0.0243 | 0.0367 | **0.0378** | — |
| VGGT-SLAM 2.0 | 0.0326 | 0.0476 | 0.0418 | 0.0251 | 0.0175 | 0.0243 | 0.0293 | 0.0251 | **0.0304** | 0.043 |
| DA3-Streaming | 0.0136 | 0.0204 | 0.0168 | 0.0250 | 0.0178 | 0.0187 | 0.0146 | 0.0228 | **0.0187** | — |
| DASH-SLAM | 0.0092 | 0.0094 | 0.0068 | 0.0093 | 0.0058 | 0.0076 | 0.0096 | 0.0138 | **0.0089** | — |

## Recovered scale and repeats

| system | mean scale | repeats |
|---|---|---|
| DROID-SLAM* | 0.743 | 2 |
| ViSTA-SLAM | 1.287 | 1 |
| MASt3R-SLAM* | 1.085 | 1 |
| AMB3R | 2.977 | 2 |
| VGGT-SLAM 2.0 | 3.148 | 1 |
| DA3-Streaming | 1.041 | 1 |
| DASH-SLAM | 1.067 | 1 |

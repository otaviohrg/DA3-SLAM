# TUM RGB-D — calibrated monocular SLAM

ATE RMSE in metres, 9 sequences. Mean over completed repeats
(a repeat counts only when every sequence scored). Lower is better.

Sim(3) = scale-free alignment, the standard monocular metric.
SE(3)  = rigid alignment, so it also penalises wrong metric scale.
`scale` is the Umeyama factor; 1.000 means metric-accurate.

Repeats: 1 for systems measured deterministic on TUM, 2 for the
stochastic ones (AMB3R, DROID-SLAM). Variance is not reported, so a
second identical run would add nothing.

## Sim(3) ATE RMSE (m)

Scale-free alignment — the metric the published tables use.

| system | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | **avg** | published |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DROID-SLAM | 0.1553 | 0.0167 | 0.5036 | 0.0250 | 0.0167 | 0.0708 | 0.0574 | 0.0453 | 0.0401 | **0.1034** | 0.038 |
| DPV-SLAM | 0.1533 | 0.0206 | 0.0384 | 0.1298 | 0.0240 | 0.3298 | 0.0296 | 0.0839 | 0.0128 | **0.0913** | 0.076 |
| DPV-SLAM++ | 0.1751 | 0.0215 | 0.0368 | 0.1372 | 0.0197 | 0.2364 | 0.0263 | 0.0904 | 0.0118 | **0.0839** | 0.054 |
| MASt3R-SLAM | 0.0482 | 0.0161 | 0.0235 | 0.0250 | 0.0196 | 0.0612 | 0.0231 | 0.0403 | 0.0089 | **0.0295** | 0.030 |

## Recovered scale and repeats

| system | mean scale | repeats |
|---|---|---|
| DROID-SLAM | 1.062 | 2 |
| DPV-SLAM | 1.062 | 2 |
| DPV-SLAM++ | 1.054 | 2 |
| MASt3R-SLAM | 1.021 | 1 |

# TUM RGB-D — calibrated monocular SLAM

ATE RMSE in metres, 9 sequences. Mean over completed repeats
(a repeat counts only when every sequence scored). Lower is better.

Sim(3) = scale-free alignment, the standard monocular metric.
SE(3)  = rigid alignment, so it also penalises wrong metric scale.
`scale` is the Umeyama factor; 1.000 means metric-accurate.

Repeats: 1 for systems measured deterministic on TUM, 2 for the
stochastic ones (AMB3R, DROID-SLAM). Variance is not reported, so a
second identical run would add nothing.

## SE(3) ATE RMSE (m)

Rigid alignment, so this also penalises wrong metric scale — what
Sim(3) hides. No published SE(3) column exists: the literature reports
Sim(3) only, because scale is free for methods that do not recover it.
Read it together with the scale table below.

| system | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | **avg** | published |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DROID-SLAM | 0.1655 | 0.0699 | 0.5470 | 0.0512 | 0.2371 | 0.2509 | 0.4711 | 0.4863 | 0.0424 | **0.2579** | n/a |
| DPV-SLAM | 1.9290 | 0.0219 | 0.0951 | 0.3343 | 0.2931 | 0.3647 | 0.0338 | 0.4874 | 0.0201 | **0.3977** | n/a |
| DPV-SLAM++ | 574.0893 | 0.0222 | 0.0850 | 0.4038 | 0.2798 | 0.3163 | 0.0307 | 0.4843 | 0.0252 | **63.9707** | n/a |
| MASt3R-SLAM | 0.0747 | 0.0924 | 0.1931 | 0.3727 | 0.0499 | 0.0635 | 0.0231 | 0.1505 | 0.0229 | **0.1159** | n/a |

## Recovered scale and repeats

| system | mean scale | repeats |
|---|---|---|
| DROID-SLAM | 1.062 | 2 |
| DPV-SLAM | 1.062 | 2 |
| DPV-SLAM++ | 1.054 | 2 |
| MASt3R-SLAM | 1.021 | 1 |

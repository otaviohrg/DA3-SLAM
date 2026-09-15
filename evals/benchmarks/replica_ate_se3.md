# Replica — uncalibrated monocular SLAM

ATE RMSE in metres, 8 sequences. Mean over completed repeats
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

| system | room0 | room1 | room2 | office0 | office1 | office2 | office3 | office4 | **avg** | published |
|---|---|---|---|---|---|---|---|---|---|---|
| DROID-SLAM* | 1.1308 | 0.9009 | 0.6829 | 0.9216 | 0.8031 | 0.9346 | 1.1986 | 6.9779 | **1.6938** | n/a |
| ViSTA-SLAM | 0.2387 | 0.1140 | 0.2131 | 0.0813 | 0.2012 | 0.4728 | 0.4116 | 0.4945 | **0.2784** | n/a |
| MASt3R-SLAM* | 0.1454 | 0.0404 | 0.1255 | 0.0620 | 0.0471 | 0.0802 | 0.1564 | 0.1358 | **0.0991** | n/a |
| AMB3R | 0.7808 | 0.5886 | 0.6589 | 0.5197 | 0.2333 | 0.9244 | 0.9927 | 0.9191 | **0.7022** | n/a |
| VGGT-SLAM 2.0 | 0.8442 | 0.5036 | 0.5776 | 0.5434 | 0.2469 | 0.9049 | 1.0285 | 0.9329 | **0.6978** | n/a |
| DA3-Streaming | 0.0589 | 0.0533 | 0.1108 | 0.0279 | 0.0446 | 0.0244 | 0.0722 | 0.0650 | **0.0571** | n/a |
| DASH-SLAM | 0.0451 | 0.0822 | 0.1032 | 0.1208 | 0.0262 | 0.0117 | 0.0141 | 0.0445 | **0.0560** | n/a |

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

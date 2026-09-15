# 7-Scenes — calibrated monocular SLAM

ATE RMSE in metres, 7 sequences. Mean over completed repeats
(a repeat counts only when every sequence scored). Lower is better.

Sim(3) = scale-free alignment, the standard monocular metric.
SE(3)  = rigid alignment, so it also penalises wrong metric scale.
`scale` is the Umeyama factor; 1.000 means metric-accurate.

Repeats: 1 for systems measured deterministic on TUM, 2 for the
stochastic ones (AMB3R, DROID-SLAM). Variance is not reported, so a
second identical run would add nothing.

Protocol: seq-01 of each scene — stated in EC3R (arXiv:2510.02080)
and hard-coded in MASt3R-SLAM's dataloader.

## Sim(3) ATE RMSE (m)

Scale-free alignment — the metric the published tables use.

| system | chess | fire | heads | office | pumpkin | redkitchen | stairs | **avg** | published |
|---|---|---|---|---|---|---|---|---|---|
| DPV-SLAM++ | 0.7789 | 0.7514 | 0.4341 | 0.6173 | 0.7574 | 0.5544 | 0.5187 | **0.6303** | — |
| DPV-SLAM | 0.7994 | 0.7970 | 0.4322 | 0.6105 | 0.6674 | 0.5746 | 0.4700 | **0.6216** | — |
| DROID-SLAM | 0.2079 | 0.1668 | 0.2015 | 0.3277 | 0.5917 | 0.2535 | 0.6412 | **0.3415** | — |
| GO-SLAM | 0.0399 | 0.0279 | 0.0162 | 0.1161 | 0.1494 | 0.0571 | 0.0242 | **0.0615** | — |
| MASt3R-SLAM | 0.0399 | 0.0300 | 0.0111 | 0.1041 | 0.0879 | 0.0486 | 0.0184 | **0.0486** | — |

## Recovered scale and repeats

| system | mean scale | repeats |
|---|---|---|
| DPV-SLAM++ | 0.205 | 2 |
| DPV-SLAM | 0.114 | 2 |
| DROID-SLAM | 1.201 | 2 |
| GO-SLAM | 1.974 | 1 |
| MASt3R-SLAM | 0.957 | 1 |

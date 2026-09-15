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

## SE(3) ATE RMSE (m)

Rigid alignment, so this also penalises wrong metric scale — what
Sim(3) hides. No published SE(3) column exists: the literature reports
Sim(3) only, because scale is free for methods that do not recover it.
Read it together with the scale table below.

| system | chess | fire | heads | office | pumpkin | redkitchen | stairs | **avg** | published |
|---|---|---|---|---|---|---|---|---|---|
| DPV-SLAM++ | 0.9726 | 2.3720 | 10.6958 | 1.8797 | 10.5470 | 1.3857 | 1.9100 | **4.2519** | n/a |
| DPV-SLAM | 0.9618 | 3.5423 | 8.3724 | 2.6261 | 7.8971 | 21.7323 | 14.2394 | **8.4816** | n/a |
| DROID-SLAM | 0.3585 | 0.3423 | 0.3537 | 0.3794 | 0.5960 | 0.3454 | 0.6570 | **0.4332** | n/a |
| GO-SLAM | 0.4746 | 0.5493 | 0.3793 | 0.4190 | 0.4993 | 0.3119 | 0.1881 | **0.4031** | n/a |
| MASt3R-SLAM | 0.2170 | 0.1620 | 0.1491 | 0.1264 | 0.0880 | 0.0538 | 0.1419 | **0.1340** | n/a |

## Recovered scale and repeats

| system | mean scale | repeats |
|---|---|---|
| DPV-SLAM++ | 0.205 | 2 |
| DPV-SLAM | 0.114 | 2 |
| DROID-SLAM | 1.201 | 2 |
| GO-SLAM | 1.974 | 1 |
| MASt3R-SLAM | 0.957 | 1 |

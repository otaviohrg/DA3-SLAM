# 7-Scenes — uncalibrated monocular SLAM

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
| DROID-SLAM* | 0.5395 | 0.6431 | 0.5401 | 0.3785 | 0.4756 | 0.1015 | 0.4521 | **0.4472** | n/a |
| VGGT-SLAM 2.0 | 0.5699 | 0.3809 | 0.0991 | 0.3255 | 0.6140 | 0.3945 | 0.7172 | **0.4430** | n/a |
| MASt3R-SLAM* | 0.2310 | 0.1283 | 0.1709 | 0.1265 | 0.1307 | 0.0978 | 0.1652 | **0.1501** | n/a |
| ViSTA-SLAM | 0.1103 | 0.4831 | 1.2863 | 0.2816 | 0.1882 | 0.2021 | 0.4544 | **0.4294** | n/a |
| AMB3R | 0.5959 | 0.3287 | 0.1446 | 0.4021 | 0.4959 | 0.3154 | 0.4327 | **0.3879** | n/a |
| DA3-Streaming | 0.0573 | 0.0839 | 0.0719 | 0.1219 | 0.1577 | 0.0627 | 0.0965 | **0.0931** | n/a |
| DASH-SLAM | 0.0496 | 0.0298 | 0.0590 | 0.1141 | 0.1436 | 0.0420 | 0.1636 | **0.0859** | n/a |

## Recovered scale and repeats

| system | mean scale | repeats |
|---|---|---|
| DROID-SLAM* | 1.322 | 1 |
| VGGT-SLAM 2.0 | 2.288 | 1 |
| MASt3R-SLAM* | 0.940 | 1 |
| ViSTA-SLAM | 0.691 | 1 |
| AMB3R | 2.054 | 2 |
| DA3-Streaming | 0.971 | 1 |
| DASH-SLAM | 1.016 | 1 |

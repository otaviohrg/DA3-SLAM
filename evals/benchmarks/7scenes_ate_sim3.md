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

## Sim(3) ATE RMSE (m)

Scale-free alignment — the metric the published tables use.

| system | chess | fire | heads | office | pumpkin | redkitchen | stairs | **avg** | published |
|---|---|---|---|---|---|---|---|---|---|
| DROID-SLAM* | 0.4768 | 0.6430 | 0.2042 | 0.2859 | 0.2527 | 0.0921 | 0.3985 | **0.3362** | 0.078 |
| VGGT-SLAM 2.0 | 0.0387 | 0.0273 | 0.0208 | 0.1073 | 0.1358 | 0.0527 | 0.0915 | **0.0677** | 0.072 |
| MASt3R-SLAM* | 0.0626 | 0.0461 | 0.0294 | 0.1028 | 0.1140 | 0.0741 | 0.0320 | **0.0659** | 0.065 |
| ViSTA-SLAM | 0.0727 | 0.0353 | 0.0276 | 0.0552 | 0.1286 | 0.0345 | 0.0332 | **0.0553** | — |
| AMB3R | 0.0345 | 0.0262 | 0.0216 | 0.0681 | 0.1428 | 0.0491 | 0.0303 | **0.0532** | — |
| DA3-Streaming | 0.0554 | 0.0537 | 0.0702 | 0.0942 | 0.1532 | 0.0625 | 0.0851 | **0.0820** | — |
| DASH-SLAM | 0.0357 | 0.0240 | 0.0141 | 0.0961 | 0.1309 | 0.0408 | 0.1149 | **0.0652** | — |

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

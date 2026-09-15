# Replica — calibrated monocular SLAM

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
| DPV-SLAM++ | 1.0222 | 0.7713 | 0.9068 | 0.8302 | 0.4591 | 1.2194 | 1.2082 | 1.0628 | **0.9350** | — |
| DPV-SLAM | 0.9326 | 0.7508 | 0.8620 | 0.8521 | 0.4258 | 1.1108 | 1.1588 | 1.1011 | **0.8993** | — |
| DROID-SLAM | 0.9618 | 0.5042 | 0.8231 | 0.4280 | 0.4608 | 0.8639 | 0.9373 | 0.4723 | **0.6814** | — |
| MASt3R-SLAM | 0.0057 | 0.0157 | 0.0112 | 0.0134 | 0.0067 | 0.0214 | 0.0266 | 0.0210 | **0.0152** | — |
| GO-SLAM | 0.0045 | 0.0040 | 0.0027 | 0.0032 | 0.0039 | 0.0035 | 0.0054 | 0.0065 | **0.0042** | — |

## Recovered scale and repeats

| system | mean scale | repeats |
|---|---|---|
| DPV-SLAM++ | 0.176 | 2 |
| DPV-SLAM | 0.254 | 2 |
| DROID-SLAM | 1.029 | 2 |
| MASt3R-SLAM | 1.096 | 1 |
| GO-SLAM | 3.409 | 1 |

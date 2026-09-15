# Replica — calibrated monocular SLAM

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
| DPV-SLAM++ | 3.9095 | 10.4031 | 1.7307 | 11.1601 | 2.1459 | 1.9536 | 1.8617 | 13.7592 | **5.8655** | n/a |
| DPV-SLAM | 1.6107 | 5.8904 | 2.1071 | 40.6352 | 2.1253 | 4.6738 | 1.3635 | 9.1892 | **8.4494** | n/a |
| DROID-SLAM | 0.9823 | 0.5083 | 0.8543 | 0.5311 | 0.7821 | 0.9579 | 1.2063 | 0.6238 | **0.8058** | n/a |
| MASt3R-SLAM | 0.1365 | 0.0362 | 0.0935 | 0.0768 | 0.0380 | 0.0215 | 0.1993 | 0.1417 | **0.0929** | n/a |
| GO-SLAM | 0.8769 | 0.4256 | 0.7275 | 0.6226 | 0.2169 | 0.9678 | 1.0835 | 0.9203 | **0.7301** | n/a |

## Recovered scale and repeats

| system | mean scale | repeats |
|---|---|---|
| DPV-SLAM++ | 0.176 | 2 |
| DPV-SLAM | 0.254 | 2 |
| DROID-SLAM | 1.029 | 2 |
| MASt3R-SLAM | 1.096 | 1 |
| GO-SLAM | 3.409 | 1 |

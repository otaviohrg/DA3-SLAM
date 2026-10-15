# Branch B · Step 2 — Model-size ladder (T2 + F2)

Fixed resolution **504**, five sizes `small → base → large → giant →
nested-giant`, 3 repeats on the **same frozen keyframes as Step 1**. Headline:
Sim3-aligned ATE (monocular). Single server GPU (RTX 3090-class) — no Jetson,
so latency/memory are server-only.

Interactive report: https://claude.ai/code/artifact/a6e7b11e-72f2-4556-941e-76f3f59286ee

## Contents
- `step2_report.html` — self-contained report.
- `tables/T2_<dataset>_final.json` — `{size: [ate_mean, ate_median, backbone_s, peak_gb, map_points, chamfer_m]}`.
- `figures/F2_<dataset>.png` — ATE (+ map-detail) vs model size.
- `raw_rows/*.jsonl` — per-run logs (lean triple + `*_md_*` map-detail rep-0 passes).

## T2 — TUM (9 fr1 seqs) · Replica (8 scenes) · ATE Sim3 (m)
| size | TUM ATE | TUM backbone | TUM mem | Replica ATE | Replica backbone | Replica mem |
|---|---:|---:|---:|---:|---:|---:|
| small | 0.1848 | 2.3s | 1.7 GB | 0.1897 | 2.7s | 1.4 GB |
| base | 0.1483 | 5.1s | 3.4 GB | 0.1794 | 5.1s | 2.8 GB |
| large | 0.1384 | 10.5s | 6.5 GB | 0.1888 | 11.2s | 5.4 GB |
| giant | 0.1774 | 19.1s | 11.5 GB | 0.1593 | 20.0s | 10.3 GB |
| **nested-giant** | **0.0509** | 23.8s | 12.8 GB | **0.0650** | 24.9s | 11.6 GB |

Map point count is **flat across size** (TUM ~8.0 M, Replica ~9.2 M) — depth is
dense at every size, so density is not a size lever. Chamfer-to-nested is
pose-confounded on this axis (clouds are pose-projected), so it tracks ATE
rather than measuring map detail independently.

## T2 — UAS (4 GT seqs, km-scale, no map-detail) · ATE as % of trajectory length
| size | fyllingsdalen | hornbill | frozen_lake | campus_fog | mean | backbone | mem |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 7.1% | 8.6% | 13.4% | 11.3% | 10.1% | 25.8s | 1.8 GB |
| base | 16.2% | 6.6% | 12.8% | 6.9% | 10.6% | 48.7s | 3.6 GB |
| large | 16.0% | 7.2% | 11.7% | 8.2% | 10.8% | 96.8s | 6.8 GB |
| **giant** | 2.9% | 5.5% | 9.1% | 9.0% | **6.6%** | 174.4s | 12.0 GB |
| nested-giant | 6.2% | 5.5% | 8.6% | 6.6% | 6.7% | 214.7s | 13.3 GB |

## Finding
**H2 is false.** Trajectory ATE does not saturate below the largest model. On
both indoor datasets only the *nested-giant* architecture reaches good ATE
(0.051 / 0.065 m); the entire plain ladder (small→giant) is flat at ~3× worse —
so it is "nested vs not", not "bigger vs smaller". On UAS, giant and nested are
a shared top tier that trades the lead by sequence. Combined with Step 1
(resolution not a lever either), **neither static knob gives a generalizable
Pareto win** — the speedup must come from temporal reuse, not model shrinking.

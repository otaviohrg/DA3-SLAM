# Branch B · Step 3 — Stacking & the Pareto frontier (T3 + F3)

Size × resolution grid `{giant, nested-giant} × {392, 448, 504}`, 3 repeats on
the same frozen keyframes as Steps 1–2. Tests whether the two levers compound,
and where the Pareto frontier reaches relative to the `nested-giant @504`
default. Sim3-aligned ATE; **UAS as % of trajectory length**. Single server
GPU — latency/memory server-only.

Interactive report: https://claude.ai/code/artifact/1c33ee7e-d517-432f-80e3-e89232bd1ba5

## Contents
- `step3_report.html` — self-contained report.
- `tables/T3_<dataset>.json` — `{"<size>_<res>": ate}` grid cells.
- `figures/F3_<dataset>.png` — Pareto scatter (ATE vs latency + memory panel).
- `raw_rows/f3_rows_<dataset>.jsonl` — all config rows feeding F3 (UAS ate already % of path).

## T3 — size × resolution grid (ATE)
**TUM (m)** · **Replica (m)** · **UAS (% path)**

| size | TUM 392 | 448 | 504 | Replica 392 | 448 | 504 | UAS 392 | 448 | 504 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| giant | 0.1739 | 0.1466 | 0.1774 | 0.2101 | 0.1842 | 0.1593 | 6.7% | **5.8%** | 6.6% |
| nested-giant | 0.0504 | **0.0482** | 0.0509 | 0.1110 | 0.0792 | **0.0650** | 7.9% | 9.8% | 6.7% |

## F3 — Pareto frontier vs default (nested-giant @504)
- **TUM** — default not on frontier; best accuracy = nested @448 (0.0482 m, 20s). Frontier = nested resolution line + tiny/inaccurate small/base@504.
- **Replica** — default **is** on the frontier and is the best accuracy point (0.0650 m); no cheaper config at comparable ATE.
- **UAS** — **giant @448 dominates the default**: 5.8% vs 6.7%, 139s vs 215s (−35% compute), 11.0 vs 13.3 GB.

## Finding — Branch B closes
The levers **do not compound** on the indoor datasets: model size dominates
(only the nested-giant architecture is accurate) and the both-cut corner
(giant + low res) collapses, so the best config is just the nested-giant
resolution knee — @448 (TUM), @504 (Replica). Only on UAS does a cheaper point
(giant@448) actually beat the default, and UAS is the bistable domain where
held-out validation previously overturned a giant win — so it is a candidate,
not a safe default.

**Decision:** keep `nested-giant @504` as the baseline (or @448 for
latency-critical indoor use). Static reduction is not a generalizable free win;
the real speedup must come from temporal reuse (Branch C). Every later
experiment is reported on top of this baseline.

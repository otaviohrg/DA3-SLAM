# Token merging × submap size on UAS (bf16 backbones)

Full grid: **7 submap sizes {8, 16, 32, 48, 64, 96, 128} × 2 arms {unmerged,
FastVGGT merging from global block 0} × 4 UAS GT sequences**, nested-giant @504
with **bf16 ViT backbones**, 1 repeat, loop closure on. 56/56 cells completed,
zero OOM — bf16 is what makes submaps above ~48 reachable at all (in fp32 the
wall is between 48 and 64 frames).

Keyframes were frozen once per sequence at submap 128 and replayed by every
cell, so submap size is purely a batching parameter and is not confounded with
keyframe *selection* through `keyframe.max_submap_size`.

## Contents
- `tables/merging_agg.csv` — one row per (submap size, arm) cell.
- `tables/summary.json` — the same aggregates plus the per-cell comparisons.
- `raw_rows/merging_rows.jsonl` — all 56 runs, one JSON object each.
- `figures/submap_scaling.png` — ATE / latency / memory vs submap size, per arm.
- `figures/per_sequence.png` — ATE vs submap size, one panel per sequence.
- `figures/frontier.png` — ATE vs latency + ATE vs memory.

Regenerate: `python scripts/report_merging.py --rows raw_rows/merging_rows.jsonl --out_dir .`

## T1 — ATE Sim3 as % of GT path length (mean over 4 sequences)

| submap | unmerged | merged | Δ backbone | Δ peak mem |
|---:|---:|---:|---:|---:|
| **8** | **5.66%** | 6.40% | −0.5% | −5.0% |
| 16 | 6.34% | 9.09% | −1.8% | −4.5% |
| 32 | 9.65% | 11.64% | −7.3% | −2.2% |
| 48 | 13.39% | 13.52% | −14.8% | −2.2% |
| 64 | 12.88% | 13.26% | −20.1% | +0.9% |
| 96 | 12.69% | 14.10% | −26.9% | +13.3% |
| 128 | 12.73% | 11.87% | **−32.6%** | **+16.0%** |

Absolute cost of the largest submaps: backbone 205.9 s → 323.7 s and peak
memory 7.99 GB → 14.19 GB going from submap 16 to 128 (unmerged).

## Findings

**1. Smaller submaps are better, decisively.** ATE more than doubles from
submap 8 (5.66%) to submap 48 (13.39%), then plateaus. The plan's H2 — "fewer
submap boundaries means less drift" — is not merely unsupported, it is
**backwards**: every extra boundary was supposed to be a place drift enters,
but larger DA3 batches evidently lose more than the boundaries cost. The best
cell in the entire grid is the smallest size tested (and the single best run is
fyllingsdalen at submap 8, 3.51%), which raises the obvious follow-up: the knee
may be below 8, and nobody has looked there.

**2. Large submaps are strictly dominated.** They are worse on all three axes at
once — worse ATE, more backbone time (205.9 s → 323.7 s from 16 to 128, because
attention grows quadratically within a batch while the frame count is fixed),
and nearly double the memory. There is no trade to make here; submap 128 is not
an operating point anyone would choose.

**3. Merging costs accuracy at 6 of 7 sizes** (+0.13 to +2.75 percentage
points), and only starts paying real speed past submap 48 — by which point the
baseline ATE has already degraded past the point of interest. Its best speed
result (−32.6% at submap 128) buys +16% memory at an ATE that is twice the
submap-8 baseline. The one size where merging wins on ATE (128, −0.86 pp) is
the worst operating point in the grid.

**4. Per-sequence, the trend is consistent** (`figures/per_sequence.png`):
frozen_lake, fyllingsdalen and hornbill all rise sharply from submap 8 to ~48
then flatten. campus_fog is the exception — it dips at 32 and again at 96/128 —
so the mean is not hiding disagreement about the main effect, only about the
tail.

## Caveats

- **1 repeat.** The report prints `n/a (1 rep)` in its `beyond noise?` column
  rather than comparing differences against a zero-width band. The submap-size
  effect (a 2.4x change in ATE) is far too large to be run-to-run noise; the
  merged-vs-unmerged deltas at submaps 8, 48 and 64 (+0.13 to +0.74 pp) are
  **not** established and would need ≥3 repeats to call either way. Append them
  with `sweep_merging.py --resume --repeat_offset 1`.
- **All-bf16.** This grid does not compare fp32 against bf16 — every cell uses
  bf16 backbones. Against the fp32 Branch B result at submap 16
  (`results/step2_size_ladder`, UAS mean 6.7%, 13.3 GB) this grid's 6.34% /
  7.99 GB is suggestive that bf16 is roughly accuracy-neutral, but the two used
  different frozen keyframe lists so it is not a controlled comparison. That
  experiment is still unrun and is the one worth doing next.
- The first attempt at this grid was killed by the **host** OOM killer at 24 GB
  RSS (not GPU) partway through campus_fog; the sweep now runs one sequence per
  container and prints host RSS per cell.

## Decision

Confirms the negative result in `plan_fastvggt-token-merging.txt`. Token merging
has no operating point on this system: where it is fast the accuracy is already
ruined by the submap size it needs, and where the accuracy is good it is worth
~0.5%. The actionable finding is unrelated to merging — **submap size should be
swept downward from 16, not upward**, and bf16 backbones cut memory 40% for
free-so-far.

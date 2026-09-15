"""Emit the final TUM table (Sim3 + SE3 ATE, per sequence + average).

Values are the mean over completed repeats; a repeat counts only when all 9
sequences scored. Published column is VGGT-SLAM 2.0 arXiv:2601.19887 Table I
(uncalibrated). DROID-SLAM* uses GeoCalib-estimated intrinsics, per sequence,
from the first frame — the heuristic fallback (focal = max(W,H)) is NOT the
published protocol and is not used here.
"""
import json, glob, statistics as st
from pathlib import Path

SEQ = ["360","desk","desk2","floor","plant","room","rpy","teddy","xyz"]
OUT = Path("/home/pierre-yves/otavio/SLAM/DA3-SLAM/evals/benchmarks")
DASH = "/home/pierre-yves/otavio/SLAM/DA3-SLAM/outputs/tum_full"
SYS = [("DASH-SLAM", None), ("AMB3R", None), ("VGGT-SLAM", "VGGT-SLAM 2.0"),
       ("ViSTA-SLAM", "ViSTA-SLAM"), ("MASt3R-SLAM-uncal", "MASt3R-SLAM*"),
       ("DA3-Streaming", None), ("DROID-SLAM-uncal", "DROID-SLAM*")]
LABEL = {"DASH-SLAM":"DASH-SLAM", "AMB3R":"AMB3R", "VGGT-SLAM":"VGGT-SLAM 2.0",
         "ViSTA-SLAM":"ViSTA-SLAM", "MASt3R-SLAM-uncal":"MASt3R-SLAM*",
         "DA3-Streaming":"DA3-Streaming", "DROID-SLAM-uncal":"DROID-SLAM*"}
pub = json.load(open("/tmp/published_tum.json"))
pubseq = pub.get("per_sequence_uncalibrated", {})

def collect(sysname):
    base = DASH if sysname == "DASH-SLAM" else f"/tmp/tum_full/{sysname}"
    sim3, se3, scale, nrep = {s: [] for s in SEQ}, {s: [] for s in SEQ}, [], 0
    for r in (1, 2, 3):
        fs = glob.glob(f"{base}/r{r}/*/results.json")
        if len(fs) != 9:
            continue
        nrep += 1
        for p in fs:
            d = json.load(open(p))
            n = d["sequence"].replace("rgbd_dataset_freiburg1_", "")
            sim3[n].append(d["ate_sim3"]["rmse"])
            se3[n].append(d["ate_se3"]["rmse"])
            scale.append(d["ate_sim3"].get("scale", float("nan")))
    if not nrep:
        return None
    return ({s: st.mean(v) for s, v in sim3.items()},
            {s: st.mean(v) for s, v in se3.items()},
            st.mean(scale), nrep)

rows = []
for sysname, pubkey in SYS:
    got = collect(sysname)
    if got:
        rows.append((LABEL[sysname], pubkey, *got))
# Row order: worst-first by Sim(3) mean, EXCEPT that the two DA3-backbone
# systems are kept adjacent at the bottom — DA3-Streaming immediately before
# DASH-SLAM — so the comparison that matters most (same backbone, different
# SLAM layer) reads off consecutive rows instead of being split by baselines.
# Both metric tables share this order so rows line up between them.
TAIL = ["DA3-Streaming", "DASH-SLAM"]
rows.sort(key=lambda r: st.mean(r[2].values()), reverse=True)
rows = ([r for r in rows if r[0] not in TAIL]
        + [r for name in TAIL for r in rows if r[0] == name])

md = ["# TUM RGB-D — uncalibrated monocular SLAM",
      "",
      "ATE RMSE in metres, 9 Freiburg-1 sequences. Mean over completed repeats",
      "(a repeat counts only when all 9 sequences scored). Lower is better.",
      "",
      "Sim(3) = scale-free alignment, the standard monocular metric.",
      "SE(3)  = rigid alignment, so it also penalises wrong metric scale.",
      "`scale` is the Umeyama scale factor; 1.000 means metric-accurate.",
      "",
      "Protocol notes:",
      "- DROID-SLAM* intrinsics are GeoCalib-estimated per sequence from the first",
      "  frame (the published uncalibrated protocol). It has no native uncalibrated",
      "  mode. The heuristic fallback focal = max(W,H) is NOT used.",
      "- Published column: VGGT-SLAM 2.0, arXiv:2601.19887, Table I (uncalibrated).",
      "- EC3R-SLAM is excluded: its public repo ships recover_trajectory only as",
      "  __pycache__/save_file.cpython-310.pyc and never uploaded save_file.py, so it",
      "  crashes on 6 of 9 sequences and cannot be run faithfully.",
      "- Repeats: AMB3R and DROID-SLAM* are stochastic; the rest reproduced",
      "  bit-exact or to 3e-05, so their spread is 0 by nature, not by tolerance.",
      ""]

for title, idx in (("## Sim(3) ATE RMSE (m) — headline", 2),
                   ("## SE(3) ATE RMSE (m) — scale-sensitive", 3)):
    md += [title, "",
           "| system | " + " | ".join(SEQ) + " | **avg** | published avg |",
           "|" + "---|" * (len(SEQ) + 3)]
    for label, pubkey, s3, se, sc, nrep in rows:
        d = s3 if idx == 2 else se
        avg = st.mean(d.values())
        pv = pub["uncalibrated"].get(pubkey) if pubkey else None
        pcell = f"{pv:.3f}" if (pv and idx == 2) else ("—" if idx == 2 else "n/a")
        md.append(f"| {label} | " + " | ".join(f"{d[s]:.4f}" for s in SEQ)
                  + f" | **{avg:.4f}** | {pcell} |")
    md.append("")

md += ["## Recovered scale and repeats", "",
       "| system | mean scale | repeats |", "|---|---|---|"]
for label, pubkey, s3, se, sc, nrep in rows:
    md.append(f"| {label} | {sc:.3f} | {nrep} |")
md.append("")

md += ["## Published per-sequence reference (uncalibrated)", "",
       "arXiv:2601.19887 Table I. For checking that baselines were reproduced,",
       "not for restating as our measurements.", "",
       "| system | " + " | ".join(SEQ) + " | avg |", "|" + "---|" * (len(SEQ) + 2)]
for k, v in pubseq.items():
    if k == "EC3R-SLAM":
        continue
    md.append(f"| {k} | " + " | ".join(f"{v[s]:.3f}" for s in SEQ)
              + f" | {st.mean(v.values()):.3f} |")
md.append("")

(OUT / "tum_ate_table.md").write_text("\n".join(md))

# ── per-metric files ──────────────────────────────────────────────────────────
# Same numbers as the combined table, split so each can be dropped straight into
# a paper without editing. Row order is shared (Sim(3) mean, worst first, the two
# DA3-backbone systems last) so the two files line up row for row.
HEAD = md[:md.index("## Sim(3) ATE RMSE (m) — headline")]

def one_metric(idx, title, blurb, fname):
    out = [title, ""] + blurb + [""]
    out += ["| system | " + " | ".join(SEQ) + " | **avg** | published avg |",
            "|" + "---|" * (len(SEQ) + 3)]
    for label, pubkey, s3, se, sc, nrep in rows:
        d = s3 if idx == 2 else se
        pv = pub["uncalibrated"].get(pubkey) if pubkey else None
        pcell = f"{pv:.3f}" if (pv and idx == 2) else ("—" if idx == 2 else "n/a")
        out.append(f"| {label} | " + " | ".join(f"{d[s]:.4f}" for s in SEQ)
                   + f" | **{st.mean(d.values()):.4f}** | {pcell} |")
    out += ["", "## Recovered scale and repeats", "",
            "| system | mean scale | repeats |", "|---|---|---|"]
    for label, pubkey, s3, se, sc, nrep in rows:
        out.append(f"| {label} | {sc:.3f} | {nrep} |")
    (OUT / fname).write_text("\n".join(HEAD + out) + "\n")
    print(f"  wrote {OUT/fname}")

one_metric(2, "# TUM RGB-D — Sim(3) ATE RMSE (m)",
           ["Scale-free (Umeyama with scale) alignment — the standard monocular",
            "metric, and the one the published comparison tables use.",
            "Lower is better. Mean over completed repeats."],
           "tum_ate_sim3.md")

one_metric(3, "# TUM RGB-D — SE(3) ATE RMSE (m)",
           ["Rigid alignment (no scale freedom), so this metric ALSO penalises a",
            "wrong metric scale — what Sim(3) hides. Lower is better.",
            "",
            "There is no published SE(3) column: the uncalibrated tables in the",
            "literature report Sim(3) only, because monocular scale is free for",
            "methods that do not recover it. Read this table together with the",
            "`mean scale` column below — a system whose scale is far from 1.000",
            "is being flattered by the Sim(3) table.",
            "",
            "Measured scale: DASH-SLAM 1.017 (metric to 1.7%), vs 1.563 (AMB3R),",
            "1.605 (VGGT-SLAM 2.0) and 0.584 (ViSTA-SLAM)."],
           "tum_ate_se3.md")

# per-metric CSVs too
for mname, idx in (("sim3", 2), ("se3", 3)):
    lines = ["system," + ",".join(SEQ) + ",avg,scale,repeats"]
    for label, pubkey, s3, se, sc, nrep in rows:
        d = s3 if idx == 2 else se
        lines.append(f"{label}," + ",".join(f"{d[s]:.6f}" for s in SEQ)
                     + f",{st.mean(d.values()):.6f},{sc:.4f},{nrep}")
    (OUT / f"tum_ate_{mname}.csv").write_text("\n".join(lines) + "\n")
    print(f"  wrote {OUT/f'tum_ate_{mname}.csv'}")

csv = ["system,metric," + ",".join(SEQ) + ",avg,scale,repeats"]
for label, pubkey, s3, se, sc, nrep in rows:
    for mname, d in (("sim3", s3), ("se3", se)):
        csv.append(f"{label},{mname}," + ",".join(f"{d[s]:.6f}" for s in SEQ)
                   + f",{st.mean(d.values()):.6f},{sc:.4f},{nrep}")
(OUT / "tum_ate_table.csv").write_text("\n".join(csv) + "\n")

print(f"  wrote {OUT/'tum_ate_table.md'}")
print(f"  wrote {OUT/'tum_ate_table.csv'}")
print(f"  {len(rows)} systems x {len(SEQ)} sequences, both metrics")

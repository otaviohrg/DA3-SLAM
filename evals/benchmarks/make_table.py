"""Emit Sim(3) and SE(3) ATE tables for one dataset x mode.

    python make_table.py --dataset tum      --mode uncalibrated
    python make_table.py --dataset replica  --mode calibrated

Values are the mean over completed repeats; a repeat counts only when EVERY
sequence scored, so a half-finished run is never silently averaged.  Row order
is worst-first by Sim(3) mean, with the two DA3-backbone systems pinned last
(DA3-Streaming then DASH-SLAM) so the same-backbone comparison reads off
consecutive rows.
"""
from __future__ import annotations
import argparse, glob, json, statistics as st
from pathlib import Path

OUT = Path(__file__).resolve().parent
DASH_LOCAL = "/home/pierre-yves/otavio/SLAM/DA3-SLAM/outputs"

DATASETS = {
    "tum":     dict(root="/tmp/tum_full",     dash=f"{DASH_LOCAL}/tum_full",
                    n=9, pub="/tmp/published_tum.json",
                    seqs=["360","desk","desk2","floor","plant","room","rpy","teddy","xyz"],
                    strip="rgbd_dataset_freiburg1_", title="TUM RGB-D"),
    "replica": dict(root="/tmp/replica_full", dash="/tmp/replica_full/DASH-SLAM",
                    n=8, pub="/tmp/published_replica.json",
                    seqs=["room0","room1","room2","office0","office1","office2","office3","office4"],
                    strip="", title="Replica"),
    "7scenes": dict(root="/tmp/scenes7_full", dash="/tmp/scenes7_full/DASH-SLAM",
                    n=7, pub="/tmp/published_7scenes.json",
                    seqs=["chess","fire","heads","office","pumpkin","redkitchen","stairs"],
                    strip="", title="7-Scenes"),
}
# repo dir -> label; label -> published-table key
UNCAL = [("DASH-SLAM","DASH-SLAM",None), ("AMB3R","AMB3R",None),
         ("VGGT-SLAM","VGGT-SLAM 2.0","VGGT-SLAM"),
         ("ViSTA-SLAM","ViSTA-SLAM","ViSTA-SLAM"),
         ("MASt3R-SLAM-uncal","MASt3R-SLAM*","MASt3R-SLAM"),
         ("DA3-Streaming","DA3-Streaming",None),
         ("DROID-SLAM-uncal","DROID-SLAM*","DROID-SLAM")]
CAL   = [("ORB-SLAM3","ORB-SLAM3","ORB-SLAM3"), ("DeepV2D","DeepV2D","DeepV2D"),
         ("DPV-SLAM","DPV-SLAM","DPV-SLAM"), ("DPV-SLAM++","DPV-SLAM++","DPV-SLAM++"),
         ("GO-SLAM","GO-SLAM","GO-SLAM"),
         ("DROID-SLAM-cal","DROID-SLAM","DROID-SLAM"),
         ("MASt3R-SLAM-cal","MASt3R-SLAM","MASt3R-SLAM")]
TAIL = ["DA3-Streaming", "DASH-SLAM"]


def published(pubfile, mode):
    try:
        d = json.load(open(pubfile))
    except Exception:
        return {}
    return d.get("calibrated" if mode == "calibrated" else "uncalibrated", {})


def collect(cfg, sysdir, label):
    base = cfg["dash"] if label == "DASH-SLAM" else f"{cfg['root']}/{sysdir}"
    per3, perE, scale, nrep = {}, {}, [], 0
    for r in (1, 2, 3):
        fs = glob.glob(f"{base}/r{r}/*/results.json")
        if len(fs) != cfg["n"]:
            continue
        nrep += 1
        for p in fs:
            d = json.load(open(p))
            # 7-Scenes sequences are recorded as "<scene>_seq-01"; the column
            # keys are bare scene names, so the suffix has to come off or every
            # per-scene cell renders as "—" while the average still computes.
            name = d["sequence"].replace(cfg["strip"], "").replace("_seq-01", "")
            per3.setdefault(name, []).append(d["ate_sim3"]["rmse"])
            perE.setdefault(name, []).append(d["ate_se3"]["rmse"])
            scale.append(d["ate_sim3"].get("scale", float("nan")))
    if not nrep:
        return None
    return ({k: st.mean(v) for k, v in per3.items()},
            {k: st.mean(v) for k, v in perE.items()}, st.mean(scale), nrep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--mode", choices=["uncalibrated", "calibrated"], default="uncalibrated")
    a = ap.parse_args()
    cfg = dict(DATASETS[a.dataset])
    # CALIBRATED RUNS LIVE IN A DIFFERENT TREE.
    # The calibrated sweep writes /tmp/{tum,replica,scenes7}_cal; the
    # uncalibrated one writes /tmp/{tum_full,replica_full,scenes7_full}.  Using
    # the uncalibrated root for both silently produced "no complete repeats"
    # instead of a calibrated table.
    if a.mode == "calibrated":
        cfg["root"] = {"tum": "/tmp/tum_cal", "replica": "/tmp/replica_cal",
                       "7scenes": "/tmp/scenes7_cal"}[a.dataset]
        cfg["dash"] = cfg["root"] + "/DASH-SLAM"   # unused: DASH is uncal-only
    SEQ = cfg["seqs"]
    pub = published(cfg["pub"], a.mode)
    systems = CAL if a.mode == "calibrated" else UNCAL

    rows = []
    for sysdir, label, pubkey in systems:
        got = collect(cfg, sysdir, label)
        if got:
            rows.append((label, pubkey, *got))
    if not rows:
        print(f"  no complete repeats for {a.dataset}/{a.mode} yet"); return
    rows.sort(key=lambda r: st.mean(r[2].values()), reverse=True)
    rows = ([r for r in rows if r[0] not in TAIL]
            + [r for name in TAIL for r in rows if r[0] == name])

    tag = "" if a.mode == "uncalibrated" else "_cal"
    head = [f"# {cfg['title']} — {a.mode} monocular SLAM", "",
            f"ATE RMSE in metres, {cfg['n']} sequences. Mean over completed repeats",
            "(a repeat counts only when every sequence scored). Lower is better.", "",
            "Sim(3) = scale-free alignment, the standard monocular metric.",
            "SE(3)  = rigid alignment, so it also penalises wrong metric scale.",
            "`scale` is the Umeyama factor; 1.000 means metric-accurate.", "",
            "Repeats: 1 for systems measured deterministic on TUM, 2 for the",
            "stochastic ones (AMB3R, DROID-SLAM). Variance is not reported, so a",
            "second identical run would add nothing.", ""]
    if a.dataset == "7scenes":
        head += ["Protocol: seq-01 of each scene — stated in EC3R (arXiv:2510.02080)",
                 "and hard-coded in MASt3R-SLAM's dataloader.", ""]

    def emit(idx, title, blurb, fname):
        out = head + [title, ""] + blurb + ["",
              "| system | " + " | ".join(SEQ) + " | **avg** | published |",
              "|" + "---|" * (len(SEQ) + 3)]
        for label, pubkey, s3, se, sc, nrep in rows:
            d = s3 if idx == 2 else se
            pv = pub.get(pubkey) if pubkey else None
            pv = pv.get("avg") if isinstance(pv, dict) else pv
            cell = f"{pv:.3f}" if (pv and idx == 2) else ("—" if idx == 2 else "n/a")
            vals = " | ".join(f"{d[s]:.4f}" if s in d else "—" for s in SEQ)
            out.append(f"| {label} | {vals} | **{st.mean(d.values()):.4f}** | {cell} |")
        out += ["", "## Recovered scale and repeats", "",
                "| system | mean scale | repeats |", "|---|---|---|"]
        for label, pubkey, s3, se, sc, nrep in rows:
            out.append(f"| {label} | {sc:.3f} | {nrep} |")
        (OUT / fname).write_text("\n".join(out) + "\n")
        print(f"  wrote {OUT/fname}")

    emit(2, f"## Sim(3) ATE RMSE (m)",
         ["Scale-free alignment — the metric the published tables use."],
         f"{a.dataset}{tag}_ate_sim3.md")
    emit(3, f"## SE(3) ATE RMSE (m)",
         ["Rigid alignment, so this also penalises wrong metric scale — what",
          "Sim(3) hides. No published SE(3) column exists: the literature reports",
          "Sim(3) only, because scale is free for methods that do not recover it.",
          "Read it together with the scale table below."],
         f"{a.dataset}{tag}_ate_se3.md")

    for mname, idx in (("sim3", 2), ("se3", 3)):
        lines = ["system," + ",".join(SEQ) + ",avg,scale,repeats"]
        for label, pubkey, s3, se, sc, nrep in rows:
            d = s3 if idx == 2 else se
            lines.append(f"{label}," + ",".join(f"{d[s]:.6f}" if s in d else ""
                                                for s in SEQ)
                         + f",{st.mean(d.values()):.6f},{sc:.4f},{nrep}")
        f = OUT / f"{a.dataset}{tag}_ate_{mname}.csv"
        f.write_text("\n".join(lines) + "\n")
        print(f"  wrote {f}")
    print(f"  {len(rows)} systems x {len(SEQ)} sequences")


if __name__ == "__main__":
    main()

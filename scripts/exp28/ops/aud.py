"""EXP-28 AUD (ledger EXP-28 C, arrow reference per run record 7) over the line-B labels of the training scenes.

Pool = what build_route_stream draws from: training-scene label rows of kind correction/anchor, call_index < 48,
distance_to_goal > 3 m (R1).  Lines:
  pixel out of window (u not in [100,540] or v not in [32,455]) among the labeller's pixel targets <= 1%
  pixel ahead distance (straight line) p95 <= 3.1 m
  turn-arrow count 1-4: TV between the pool's turn targets and the base's own turn answers on the same states <= 0.15
  DAgger STOP targets = 0
Reported only: stairs (labeller pixel target vs agent height > 0.3 m), the anchors' pixels (the model's own).
The stream-side lines (no STOP target, no reversal left after the drop) are read from the stream build's own
report by the caller.
"""
import argparse, collections, glob, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--label-root", required=True)
ap.add_argument("--train-scenes-file", required=True)
ap.add_argument("--first-calls", type=int, default=48)
ap.add_argument("--min-distance", type=float, default=3.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

train = {l.strip().split("-", 1)[-1] for l in open(a.train_scenes_file) if l.strip() and not l.startswith("#")}
ARW = set("←→")
def arrows(s):
    return len(s) if s and set(s) <= ARW else 0
def pix(s):
    try:
        u, v = (int(x) for x in s.split())
        return u, v
    except Exception:
        return None

base_turns, tgt_turns = collections.Counter(), collections.Counter()
kinds = collections.Counter()
corr_pix, anch_pix, ahead, stairs = [], [], [], 0
stop_targets = 0
files = 0
for f in glob.glob(os.path.join(a.label_root, "*", "*.s2labels.jsonl")):
    scene = os.path.basename(os.path.dirname(f))
    if scene not in train:
        continue
    files += 1
    for line in open(f):
        r = json.loads(line)
        if r.get("call_index", 0) >= a.first_calls or (r.get("distance_to_goal") or 0) <= a.min_distance:
            continue
        k = arrows(r.get("model_turn1"))
        if r.get("model_kind") == "turn" and k:
            base_turns[k] += 1
        kind = r.get("kind")
        if kind not in ("correction", "anchor"):
            continue
        kinds[kind] += 1
        t1 = r.get("target_turn1") or ""
        if "STOP" in t1.upper() or t1.strip() == "■":
            stop_targets += 1
        k = arrows(t1)
        if k:
            tgt_turns[k] += 1
        p = pix(r.get("target_turn2") or "")
        if p is not None:
            (corr_pix if kind == "correction" else anch_pix).append(p)
            d = r.get("debug") or {}
            if kind == "correction" and d.get("pixel_ahead_euclid_m") is not None:
                ahead.append(float(d["pixel_ahead_euclid_m"]))
                pw, pos = d.get("pixel_point_world"), d.get("position")
                if pw and pos and abs(pw[1] - pos[1]) > 0.3:
                    stairs += 1

def dist(c):
    n = sum(c.values())
    return {k: (c[k] / n if n else 0.0) for k in (1, 2, 3, 4)}, n
def outside(ps):
    return sum(1 for u, v in ps if not (100 <= u <= 540 and 32 <= v <= 455))
def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None

bd, bn = dist(base_turns)
td, tn = dist(tgt_turns)
tv = 0.5 * sum(abs(bd[k] - td[k]) for k in (1, 2, 3, 4))
out_c = outside(corr_pix)
p95 = q(ahead, 0.95)
res = {
    "label_files": files, "pool": dict(kinds),
    "pixel_out_of_window": {"k": out_c, "n": len(corr_pix), "rate": out_c / len(corr_pix) if corr_pix else None,
                            "line": 0.01, "pass": bool(corr_pix) and out_c / len(corr_pix) <= 0.01},
    "pixel_ahead_p95_m": {"value": p95, "n": len(ahead), "line": 3.1, "pass": p95 is not None and p95 <= 3.1},
    "arrow_tv_vs_base": {"value": round(tv, 4), "targets": {str(k): round(v, 4) for k, v in td.items()}, "n_targets": tn,
                         "base": {str(k): round(v, 4) for k, v in bd.items()}, "n_base": bn, "line": 0.15, "pass": tn > 0 and tv <= 0.15},
    "dagger_stop_targets": {"value": stop_targets, "pass": stop_targets == 0},
    "report_only": {"stairs_share": stairs / len(ahead) if ahead else None,
                    "anchor_pixels_out_of_window": {"k": outside(anch_pix), "n": len(anch_pix)}},
}
res["pass"] = all(res[k]["pass"] for k in ("pixel_out_of_window", "pixel_ahead_p95_m", "arrow_tv_vs_base", "dagger_stop_targets"))
json.dump(res, open(a.out, "w"), indent=1)
print(("AUD PASS " if res["pass"] else "AUD FAIL ") + json.dumps({k: res[k].get("value", res[k].get("rate")) for k in
      ("pixel_out_of_window", "pixel_ahead_p95_m", "arrow_tv_vs_base", "dagger_stop_targets")}))
sys.exit(0 if res["pass"] else 1)

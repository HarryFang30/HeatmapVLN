"""EXP-28 S0 read of the first committed records: platform split, SR of the first 20, timings (CPU only)."""
import glob, json, os, re, sys, tarfile

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/exp28/collect_base/records"
rows = []
for sha in glob.glob(os.path.join(root, "*", "*.s2rec.tar.sha256")):
    tar = sha[: -len(".sha256")]
    try:
        with tarfile.open(tar) as t:
            ep = json.load(t.extractfile("episode.json"))
    except Exception as exc:  # an unreadable record fails S0
        rows.append({"bad": f"{tar}: {exc}"})
        continue
    shard = int(re.search(r"shard_(\d+)\.json", ep["episode_list"]).group(1))
    o = ep.get("outcome", {})
    rows.append({"t": ep["committed_unix"], "shard": shard, "part": "heldout" if shard < 20 else "train",
                 "platform": "910b" if not ep["server"]["address"].startswith("127.0.0.1:5247") else "4090",
                 "success": o.get("success"), "steps": o.get("steps"), "calls": ep.get("num_calls"),
                 "navmesh_sha": bool(ep.get("navmesh_sha256")), "geo": ep.get("geodesic_distance")})
bad = [r["bad"] for r in rows if "bad" in r]
ok = sorted((r for r in rows if "bad" not in r), key=lambda r: r["t"])
first = ok[:20]
out = {"records": len(ok), "unreadable": bad,
       "by_platform": {p: sum(r["platform"] == p for r in ok) for p in ("4090", "910b")},
       "by_part": {p: sum(r["part"] == p for r in ok) for p in ("heldout", "train")},
       "heldout_not_on_910b": sum(r["part"] == "heldout" and r["platform"] != "910b" for r in ok),
       "missing_navmesh_sha": sum(not r["navmesh_sha"] for r in ok),
       "first20_n": len(first), "first20_sr": round(sum(bool(r["success"]) for r in first) / max(len(first), 1), 3),
       "sr_all": round(sum(bool(r["success"]) for r in ok) / max(len(ok), 1), 3),
       "span_min": round((ok[-1]["t"] - ok[0]["t"]) / 60, 1) if ok else None}
print(json.dumps(out))

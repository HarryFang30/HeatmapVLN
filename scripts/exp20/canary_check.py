"""EXP-20 canary verdict (docs/experiments/exp20-ablation-runbook.md section 2), run inside fjl-habitat:
  python canary_check.py <src> <canary output root> <arm a1|a2|a3> <verdict.json>
1. preflight evidence in every model log (A1: also the bridge-off line);
2. 2 episodes per shard 00/01 finished, ppa_applied_calls > 0 per shard;
3. against the 09-28 A0 canary, every call before an episode's first ready trajectory call is identical
   (the first divergence after it is recorded, not judged)."""
import json
import sys
from pathlib import Path

src, root, arm, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
sys.path.insert(0, src)
from scripts.deploy.nav_agent_habitat_check import compare_logs  # noqa: E402
from scripts.exp19.select_cases import parse_client_log  # noqa: E402

REF = Path("/workspace/eval_runs/canary_cuda_seed42/runtime/20260928_212453_1550/logs")
runtime = sorted((root / "runtime").iterdir())[-1]
checks, ok = {}, True
models = sorted((runtime / "logs").glob("model_*.log"))
need = ["Formal PPA online AMB3R runtime enabled"] + (["PPA bridge off (EXP-20 A1)"] if arm == "a1" else [])
checks["preflight"] = {m.name: all(n in m.read_text(errors="replace") for n in need) for m in models}
ok &= bool(models) and all(checks["preflight"].values())
checks["shards"] = {}
for shard in ("00", "01"):
    rows = [json.loads(x) for x in (root / "workers" / f"shard_{shard}" / "progress.json").read_text().splitlines()
            if x.strip()]
    log = runtime / "logs" / f"client_shard_{shard}.log"
    cmp = compare_logs(log, REF / f"client_shard_{shard}.log")
    mine, _, _ = parse_client_log(log)
    eps = []
    for e in cmp["episodes"]:
        block = next(b for b in mine.values() if f"{b['scene_id']}_{b['episode_id']:04d}" == e["episode"])
        ready = next((c["call_index"] for c in block["calls"] if c["vo_ready"] and c["kind"] == "trajectory"), None)
        first = e["first_divergent_call"]
        same_before_ready = first is None or (ready is not None and first >= ready)
        eps.append({**{k: e[k] for k in ("episode", "calls", "reference_calls", "identical_calls",
                                         "first_divergent_call")},
                    "first_ready_trajectory_call": ready, "identical_before_first_ready": same_before_ready})
    applied = sum(int(r.get("ppa_applied_calls", 0)) for r in rows)
    s_ok = len(rows) == 2 and len(eps) == 2 and applied > 0 and all(x["identical_before_first_ready"] for x in eps)
    checks["shards"][shard] = {"episodes": eps, "rows": len(rows), "ppa_applied_calls": applied, "pass": s_ok,
                               "sr_info_only": [r.get("success") for r in rows]}
    ok &= s_ok
verdict = {"arm": arm, "output": str(root), "runtime": runtime.name, "pass": bool(ok), "checks": checks}
out.write_text(json.dumps(verdict, indent=1) + "\n")
print(json.dumps({"arm": arm, "pass": bool(ok)}))

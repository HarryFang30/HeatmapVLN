import json

from scripts.tools import compare_client_logs, paired_progress_monitor


def _rows(path, pairs):
    path.write_text("".join(json.dumps({"scene_id": s, "episode_id": e, "success": ok, "spl": 0.5 * ok}) + "\n"
                            for s, e, ok in pairs))
    return path


def test_paired_monitor_waits_then_fires_the_stop_rule(tmp_path):
    ref = _rows(tmp_path / "ref.jsonl", [("A", i, 1.0) for i in range(1000)])
    (tmp_path / "w").mkdir()
    # 10 points worse on the 300 episodes done so far: too few to stop
    _rows(tmp_path / "w" / "p0.json", [("A", i, 0.0 if i % 10 == 0 else 1.0) for i in range(300)])
    argv = ["--candidate", str(tmp_path / "w" / "*.json"), "--reference", str(ref), "--min-episodes", "600"]
    assert paired_progress_monitor.main(argv) == 0
    _rows(tmp_path / "w" / "p1.json", [("A", i, 0.0 if i % 10 == 0 else 1.0) for i in range(300, 700)])
    out = tmp_path / "m.json"
    assert paired_progress_monitor.main(argv + ["--out", str(out)]) == 3
    report = json.loads(out.read_text())
    assert report["sr"]["n"] == 700 and report["sr"]["delta_pt"] == -10.0 and report["abort"]
    # the same deficit within the rule's tolerance does not stop the run
    assert paired_progress_monitor.main(argv + ["--abort-delta", "-20"]) == 0
    assert paired_progress_monitor.main(["--candidate", str(tmp_path / "none*"), "--reference", str(ref)]) == 2


def _log(path, episodes):
    lines = []
    for n, (scene, ep, calls, success) in enumerate(episodes, 1):
        lines.append(f"[{n}/{len(episodes)}] Episode {scene}_{ep}: walk")
        for step, kind, text in calls:
            lines.append(f"  step_id: {step}, RPC kind={kind}, VLM output: {text}")
        lines.append(f"  => success: {success}, spl: 1.0, os: 1.0, ne: 0.5, vlm_calls: {len(calls)}, trajectory_calls: 1, steps: 10")
    path.write_text("\n".join(lines) + "\n")


def test_client_logs_compare_call_for_call(tmp_path):
    calls = [(0, "native_actions", "→→"), (3, "trajectory", "450 57"), (9, "stop", "STOP")]
    _log(tmp_path / "a_client_shard_00.log", [("S", 1, calls, 1.0), ("S", 2, calls, 1.0)])
    changed = [calls[0], (3, "trajectory", "451 57"), calls[2]]
    _log(tmp_path / "b_client_shard_00.log", [("S", 1, calls, 1.0), ("S", 2, changed, 1.0), ("S", 3, calls, 1.0)])
    out = tmp_path / "cmp.json"
    rc = compare_client_logs.main(["--a", str(tmp_path / "a_*.log"), "--b", str(tmp_path / "b_*.log"),
                                   "--out", str(out), "--require-identical"])
    report = json.loads(out.read_text())
    assert rc == 1 and report["shared_episodes"] == 2 and report["identical"] == 1
    diverged = next(e for e in report["episodes"] if not e["identical"])
    assert diverged["episode_id"] == 2 and diverged["first_divergence"]["call_index"] == 1

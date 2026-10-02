"""EXP-21 sensitivity switches: System1's S / M overrides on the model server, K and the subset lists in the CUDA
launcher, and the fixed stratified subset (scripts/exp21/make_subset.py)."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.exp21 import make_subset as ms

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "run_ppa_r2r_val_unseen_cuda.sh"
CONFIG = REPO / "configs" / "ppa_action_refine_v2_8gpu.yaml"


def _cohorts(root: Path, per_shard: int = 20) -> Path:
    """8 synthetic cohort shards; episode i of shard s has geodesic distance (8 * i + s) / 10 m."""
    root.mkdir(parents=True, exist_ok=True)
    for s in range(8):
        eps = [{"episode_id": 8 * i + s, "scene_id": f"scene{(8 * i + s) % 5}"} for i in range(per_shard)]
        (root / f"shard_0{s}.json").write_text(json.dumps(
            {"cohort_name": f"rr_{s:02d}_of_08", "dataset_sha256": "x", "episodes": eps, "num_shards": 8,
             "shard_index": s}))
        data = {"episodes": [{"episode_id": e["episode_id"], "scene_id": f"mp3d/{e['scene_id']}/{e['scene_id']}.glb",
                              "info": {"geodesic_distance": e["episode_id"] / 10.0}} for e in eps]}
        with gzip.open(root / f"dataset_shard_0{s}.json.gz", "wt") as fh:
            json.dump(data, fh)
    return root


def test_subset_is_stratified_deterministic_and_keeps_cohort_order(tmp_path):
    cohorts = _cohorts(tmp_path / "cohorts")
    for out in ("a", "b"):
        assert ms.main(["--cohorts", str(cohorts), "--out", str(tmp_path / out), "--n", "40"]) == 0
    for s in range(8):  # deterministic
        assert (tmp_path / "a" / f"shard_0{s}.json").read_bytes() == (tmp_path / "b" / f"shard_0{s}.json").read_bytes()
    man = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert man["n"] == 40 and man["total_episodes"] == 160 and [k["chosen"] for k in man["strata"]] == [10] * 4
    kept = []
    for s in range(8):
        sub = json.loads((tmp_path / "a" / f"shard_0{s}.json").read_text())
        full = json.loads((cohorts / f"shard_0{s}.json").read_text())
        ids = [e["episode_id"] for e in sub["episodes"]]
        order = [e["episode_id"] for e in full["episodes"]]
        assert ids == [i for i in order if i in set(ids)]  # cohort order kept
        assert sub["shard_index"] == s and sub["parent_cohort"] == full["cohort_name"]
        kept += sub["episodes"]
    assert len(kept) == 40
    # distances are 0.0 .. 15.9 m evenly: each quartile is 40 episodes, the 10 lowest sha1 keys of each are kept
    for k in range(4):
        members = [e for e in range(160) if k * 40 <= e < (k + 1) * 40]
        want = sorted(members, key=lambda e: hashlib.sha1(f"scene{e % 5}:{e}".encode()).hexdigest())[:10]
        assert sorted(e["episode_id"] for e in kept if k * 40 <= e["episode_id"] < (k + 1) * 40) == sorted(want)


def test_launcher_passes_k_s_m_and_subset_lists_and_skips_the_full_merge():
    sh = LAUNCHER.read_text(encoding="utf-8")
    assert 'NUM_HISTORY="${PPA_EVAL_NUM_HISTORY:-8}"' in sh and '--num_history "$NUM_HISTORY"' in sh
    assert 'MODEL_EXTRA+=(--nextdit_num_sample_trajs "$NUM_SAMPLE_TRAJS")' in sh
    assert 'MODEL_EXTRA+=(--nextdit_num_inference_steps "$NUM_INFERENCE_STEPS")' in sh
    assert '--episode_list "$EPISODE_LISTS_DIR/shard_0${shard}.json"' in sh
    assert '"$EPISODE_LISTS_DIR" == "$COHORTS_DIR" ]]; then' in sh  # a subset run is never merged as a full one
    assert 'grep -F "Sensitivity override (EXP-21): nextdit.$key"' in sh  # the run dies without the evidence


@pytest.fixture()
def runtime_cls(monkeypatch, tmp_path):
    pytest.importorskip("vla_rpc")
    from scripts.evaluation import rpc_model_server as server

    for name in ("PPA_DATA_ROOT", "PPA_AMB3R_CACHE_ROOT", "PPA_STAGE2_OUTPUT_ROOT", "PPA_TENSORBOARD_ROOT",
                 "PPA_ACTION_REFINE_OUTPUT_ROOT"):
        monkeypatch.setenv(name, str(tmp_path))
    return server.HeatmapVLNRuntime


def _cfg(runtime_cls, **overrides):
    args = SimpleNamespace(config=str(CONFIG), internnav_model_path="/tmp/internnav",
                           pano_latent_adapter_checkpoint=None, **overrides)
    return object.__new__(runtime_cls)._load_runtime_config(args)


def test_server_overrides_s_and_m_only_when_asked(runtime_cls):
    base = _cfg(runtime_cls)
    nd = base["model"]["action_head"]["nextdit"]
    assert (nd["num_sample_trajs"], nd["num_inference_steps"]) == (32, 10)
    assert _cfg(runtime_cls, nextdit_num_sample_trajs=None, nextdit_num_inference_steps=None) == base
    s = _cfg(runtime_cls, nextdit_num_sample_trajs=8, nextdit_num_inference_steps=None)
    m = _cfg(runtime_cls, nextdit_num_sample_trajs=None, nextdit_num_inference_steps=2)
    assert s["model"]["action_head"]["nextdit"]["num_sample_trajs"] == 8
    assert m["model"]["action_head"]["nextdit"]["num_inference_steps"] == 2
    for cfg, key in ((s, "num_sample_trajs"), (m, "num_inference_steps")):  # nothing else moves
        cfg["model"]["action_head"]["nextdit"][key] = base["model"]["action_head"]["nextdit"][key]
        assert cfg == base
    with pytest.raises(ValueError, match="positive integer"):
        _cfg(runtime_cls, nextdit_num_sample_trajs=0, nextdit_num_inference_steps=None)


def test_full_run_summary_reads_the_merged_progress_jsonl():
    """The merge tool writes merged/progress.jsonl; the summary used to look for progress.json and fail a finished
    full run with "no episodes were recorded" (A0 seed 42, 2026-10-02)."""
    sh = LAUNCHER.read_text(encoding="utf-8")
    assert '(root / "progress.jsonl", root / "progress.json")' in sh

"""EXP-20 A1 (no history injection): ``rpc_model_server --ppa_bridge_off`` lets System1 sample from Z instead of
the injected Z~ on the same call, and changes nothing else in the response; the launcher passes the switch and
checks the server took it.  Uses the model-server fakes of ``test_latency_timing`` (needs ``vla_rpc``)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from tests import test_latency_timing as tl
from tests.test_latency_timing import model_server  # noqa: F401  (fixture)

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "run_ppa_r2r_val_unseen_cuda.sh"


def _ready_call(server, vla_pb2, monkeypatch, bridge_off):
    """One ready two-turn planning call; returns (the plan System1 sampled from, the response payload)."""
    monkeypatch.delenv(tl.latency.TIMING_ENV, raising=False)
    kwargs, answers, _ = tl.MODEL_CASES["ready_two_turns"]
    runtime = tl._model_runtime(server, answers)
    if bridge_off is not None:
        runtime.ppa_bridge_off = bridge_off
    seen = []

    def trajectory_from_projected(plan, **kw):
        seen.append(plan.clone())
        return tl._trajectory(plan, **kw)
    runtime.model.nextdit_action_head.get_trajectory_from_projected = trajectory_from_projected
    servicer = server.HeatmapVLNRPCServicer(runtime)
    errors = []
    context = tl.SimpleNamespace(set_details=errors.append, set_code=errors.append)
    response = servicer.InferJSON(tl._model_request(vla_pb2, 0, **kwargs), context)
    assert errors == [] and len(seen) == 1
    return seen[0], json.loads(response.json_payload)


def test_bridge_off_samples_from_z_and_only_adds_a_flag(model_server, monkeypatch):
    server, vla_pb2 = model_server
    z0 = torch.ones(1, 4, tl.PLAN_DIM)  # the fakes' form_plan: Z = ones, Z~ = ones + 0.01
    plan_on, deployed = _ready_call(server, vla_pb2, monkeypatch, None)  # a runtime without the attribute
    plan_off, ablated = _ready_call(server, vla_pb2, monkeypatch, True)
    plan_false, explicit = _ready_call(server, vla_pb2, monkeypatch, False)
    assert torch.equal(plan_on, z0 + 0.01) and torch.equal(plan_false, z0 + 0.01)
    assert torch.equal(plan_off, z0)
    assert "ppa_bridge_off" not in deployed and explicit == deployed
    assert ablated.pop("ppa_bridge_off") is True
    assert ablated["ppa_applied"] is True  # the client's ready-call check still holds
    assert ablated == deployed  # the fakes' trajectory ignores the plan: nothing else differs


def test_the_server_has_the_switch_and_the_launcher_passes_and_checks_it():
    text = (LAUNCHER.parents[0] / "evaluation" / "rpc_model_server.py").read_text(encoding="utf-8")
    assert '"--ppa_bridge_off"' in text and "PPA bridge off (EXP-20 A1)" in text
    sh = LAUNCHER.read_text(encoding="utf-8")
    assert 'BRIDGE_OFF="${PPA_EVAL_BRIDGE_OFF:-0}"' in sh
    assert "MODEL_EXTRA=(--ppa_bridge_off)" in sh and '"${MODEL_EXTRA[@]}"' in sh
    assert re.search(r'grep -F "PPA bridge off \(EXP-20 A1\)"', sh)  # the run dies without the server's evidence

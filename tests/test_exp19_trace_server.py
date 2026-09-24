"""EXP-19 [B]: the trace entry point reads the deployed call and changes nothing.

The real ``HeatmapVLNRuntime._plan_panoramic_native`` runs end to end on CPU with
fake Qwen / History Head / NextDiT objects, a real ``PastPlanActionChain``
(bridge with trained-like, non-zero weights) and real JPEG blobs, so the
class-level wrappers are exercised on exactly the call sites the server uses.
Needs ``vla_rpc`` (skipped otherwise, like tests/test_rpc_pano_two_phase.py):
``PYTHONPATH=$R/rpc/src:. python -m pytest tests/test_exp19_trace_server.py``.
"""

import copy
import io
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

pytest.importorskip("vla_rpc")

from vla_rpc.proto import vla_pb2  # noqa: E402

from scripts.evaluation import rpc_model_server as S  # noqa: E402
from scripts.evaluation.rpc_protocol import build_rpc_sampling_metadata  # noqa: E402
from scripts.exp19 import rpc_model_server_trace as trace  # noqa: E402
from src.models.action.nextdit_action_head import NextDiTActionHead  # noqa: E402
from src.models.past_plan_action import PastPlanActionChain  # noqa: E402

PLAN_DIM, MEMORY_DIM, HIDDEN_DIM, HEADS = 16, 8, 12, 2
SCENE, EPISODE = "zsNo4HB9uLZ", 7
EP_KEY = "zsNo4HB9uLZ_0007"
VIEWS = ("front", "right", "back", "left")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Tokenizer:
    eos_token_id = 99

    def __init__(self):
        self.text = "180 176"  # native System2 pixel goal "row col"

    def encode(self, _text, add_special_tokens=False):
        return [41, 42]

    def decode(self, _ids, skip_special_tokens=True):
        return self.text


class _ImageProcessor:
    def __call__(self, images, return_tensors):
        return {
            "pixel_values": torch.zeros(len(images), 3),
            "image_grid_thw": torch.ones(len(images), 3, dtype=torch.long),
        }


class _Processor:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.image_processor = _ImageProcessor()

    def apply_chat_template(self, _messages, **_kwargs):
        ids = torch.tensor([[1, 2, 3]])
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.zeros(4, 3),
            "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        }


class _Qwen:
    def __init__(self):
        self.model = self
        self.fail = None
        generator = torch.Generator().manual_seed(3)
        self.traj_hs = torch.randn(1, 4, HIDDEN_DIM, generator=generator).to(torch.bfloat16)

    def generate(self, **kwargs):
        if self.fail is not None:
            raise self.fail
        suffix = torch.tensor([[7, 8]], dtype=kwargs["input_ids"].dtype)
        return SimpleNamespace(sequences=torch.cat([kwargs["input_ids"], suffix], dim=1))

    def generate_latents(self, **_kwargs):
        return self.traj_hs


class _FakeNextDiT(NextDiTActionHead):
    """Skips the heavy __init__ but inherits the (wrapped) public API."""

    def __init__(self):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(
            latent_emb_size=PLAN_DIM,
            predict_steps=32,
            guidance_scale=1.0,
            num_inference_steps=10,
            num_sample_trajs=32,
        )
        generator = torch.Generator().manual_seed(5)
        self.cond_projector = nn.Linear(HIDDEN_DIM, PLAN_DIM)
        with torch.no_grad():  # seeded, so every _runtime() is the same model
            for parameter in self.cond_projector.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3)
        self.cond_projector = self.cond_projector.to(torch.bfloat16).requires_grad_(False)
        self.direction = torch.randn(PLAN_DIM, generator=generator)
        self.calls = []

    def sample(self, cond, generator, num_sample_trajs=32, predict_step_nums=32):
        """Deterministic in (cond, generator): drift set by the plan + seeded noise."""
        noise = torch.randn((num_sample_trajs, predict_step_nums, 3), generator=generator)
        turn = torch.tanh((cond.float()[0] @ self.direction).mean() * 4.0)
        delta = torch.zeros_like(noise)
        delta[..., 0] = 0.8  # 0.2 m forward per step after action_scale 4
        delta[..., 1] = 0.8 * turn
        return (delta + 0.3 * noise).to(cond.dtype)

    def _record(self, kind, cond, traj_images, generator, num_sample_trajs, predict_step_nums):
        out = self.sample(cond, generator, num_sample_trajs, predict_step_nums)
        self.calls.append(
            SimpleNamespace(
                kind=kind,
                cond=cond,
                traj_images=traj_images,
                generator=generator,
                state_after=None if generator is None else generator.get_state().clone(),
            )
        )
        return out

    def generate_traj_from_projected(self, traj_cond, traj_images=None, predict_step_nums=32,
                                     guidance_scale=1.0, num_inference_steps=10, num_sample_trajs=32,
                                     heatmap_tokens=None, heatmap_mask=None, heatmap_valid=None,
                                     generator=None, initial_noise=None):
        return self._record("projected", traj_cond, traj_images, generator, num_sample_trajs, predict_step_nums)

    def generate_traj(self, traj_hidden_states, traj_images=None, predict_step_nums=32,
                      guidance_scale=1.0, num_inference_steps=10, num_sample_trajs=32,
                      heatmap_tokens=None, heatmap_mask=None, heatmap_valid=None,
                      generator=None, initial_noise=None):
        cond = self.cond_projector(traj_hidden_states)
        return self._record("native", cond, traj_images, generator, num_sample_trajs, predict_step_nums)


def _history_head(*, inputs, num_histories, history_rel_poses, explicit_history_mask, return_memory_tokens):
    generator = torch.Generator().manual_seed(11)
    mask = explicit_history_mask
    valid = mask.float()
    logits = torch.randn(1, 8, 4, 64, 64, generator=generator) * 3.0
    visibility = torch.randn(1, 8, 4, generator=generator) * valid[..., None]
    spatial = torch.softmax(logits.flatten(-2), dim=-1).view_as(logits)
    view_none = torch.softmax(torch.cat([torch.zeros(1, 8, 1), visibility], dim=-1), dim=-1)
    slot = valid[:, :, None, None, None]
    return {
        "heatmaps": torch.sigmoid(logits) * slot,
        "heatmap_logits": logits * slot,
        "visibility": visibility,
        "heatmaps_gated": spatial * view_none[..., 1:, None, None] * slot,
        "none_probability": view_none[..., 0] * valid + (1.0 - valid),
        "history_mask": mask,
        "history_memory": torch.randn(1, 8, MEMORY_DIM, generator=generator) * valid[..., None],
        "history_memory_mask": mask,
        "panoramic_vit_features": torch.zeros(1),  # only read by the (faked) Future Head
    }


def _fake_decode_future(self, plan_z, *, past_output, past_head, time_mask=None):
    probability = torch.sigmoid(plan_z.float().mean(dim=-1))[..., None].expand(1, 4, 4).clone()
    maps = torch.ones(1, 4, 4, 64, 64) * probability[..., None, None]
    return {
        "future_visibility_probability": probability,
        "future_heatmaps_gated": maps,
        "future_heatmaps": maps.clone(),
    }


def _runtime():
    runtime = object.__new__(S.HeatmapVLNRuntime)
    runtime.require_deterministic_sampling = True
    runtime.device = torch.device("cpu")
    runtime.processor = _Processor()
    runtime.train_cfg = {
        "data": {
            "image_size": [384, 384],
            "trajectory": {
                "system2_sft_protocol": "internnav",
                "structured_pano_output": False,
                "traj_image_size": [224, 224],
            },
        }
    }
    chain = PastPlanActionChain(
        plan_dim=PLAN_DIM,
        memory_dim=MEMORY_DIM,
        bridge_heads=HEADS,
        max_delta_ratio=0.05,
    ).eval()
    with torch.no_grad():  # a trained bridge: W_o is exactly zero only at init
        generator = torch.Generator().manual_seed(13)
        for parameter in chain.bridge.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)
    runtime.model = SimpleNamespace(
        qwen2_5_vl=_Qwen(),
        latent_queries=torch.zeros(1, 4, HIDDEN_DIM),
        config=SimpleNamespace(dtype=torch.bfloat16),
        nextdit_action_head=_FakeNextDiT(),
        past_plan_action=chain,
        heatmap_vln=object(),
        _forward_frozen_single_view_heatmap=_history_head,
    )
    runtime.pano_latent_adapter = None
    runtime.has_nextdit = True
    runtime.num_sample_trajs = 32
    runtime.action_scale = 4.0
    runtime.ppa_stage0_action_arm = "disabled"
    runtime.ppa_online_amb3r = True
    return runtime


def _jpeg(size, color):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _request(call_index, *, pose_ready=True, num_history=5):
    """A request as r2r_val_unseen._rpc_plan_panoramic builds it (ppa arm)."""
    blobs = [
        vla_pb2.BinaryBlob(name=f"current/{view}", data=_jpeg((32, 32), (40 * i, 20, 30)))
        for i, view in enumerate(VIEWS)
    ]
    for slot in range(num_history):
        for i, view in enumerate(VIEWS):
            blobs.append(
                vla_pb2.BinaryBlob(name=f"history/{slot}/{view}", data=_jpeg((32, 32), (30 * slot, 40 * i, 5)))
            )
    blobs.append(vla_pb2.BinaryBlob(name="lookdown", data=_jpeg((640, 480), (90, 90, 90))))
    history_steps = [2 * slot for slot in range(num_history)]
    current = 2 * num_history + 3
    payload = {
        "instruction": "walk past the sofa and stop at the door",
        "num_history": num_history,
        "vlm_image_size": [384, 384],
        "traj_image_size": [224, 224],
        "system1_coord_order": "generated",
        "trajectory_selection": "mean",
        "trajectory_x_sign": 1.0,
        "trajectory_heading_alignment": "none",
        "require_deterministic_sampling": True,
        "phase": "joint",
        "deterministic_sampling": build_rpc_sampling_metadata(
            protocol_seed=42,
            scene_id=SCENE,
            episode_id=EPISODE,
            system2_call_index=call_index,
        ),
        "pose_provider": "amb3r_vo_da3",
        "pose_ready": pose_ready,
        "vo_current_frame_id": current,
        "vo_history_frame_ids": history_steps,
        "vo_provider_phase": "stateful_backend" if pose_ready else "map_warmup",
        "vo_trajectory_revision": 2,
        "current_capture_step": current,
        "history_capture_steps": history_steps,
        "history_age_steps": [current - step for step in history_steps],
    }
    if pose_ready:
        payload["history_rel_poses"] = [
            [-0.25 * (num_history - slot), 0.05 * slot, 1.0, 0.0] for slot in range(num_history)
        ]
    return payload, blobs


def _untraced(call_index, **kwargs):
    """The same call through the original method (no context -> wrappers inert)."""
    original = S.HeatmapVLNRuntime._plan_panoramic_native._exp19_original
    return original(_runtime(), *_request(call_index, **kwargs))


def _read(root, call_index):
    stem = root / EP_KEY / f"call_{call_index:03d}"
    record = json.loads(stem.with_suffix(".json").read_text())
    with np.load(stem.with_suffix(".npz")) as data:
        arrays = {name: data[name] for name in data.files}
    return record, arrays


def _dumps(response):
    return json.dumps(response, sort_keys=True)


@pytest.fixture()
def traced(monkeypatch, tmp_path):
    # The Future Head needs the real History Head decoders; fake it before
    # install() so the class-level wrapper sits on top of the fake.
    monkeypatch.setattr(PastPlanActionChain, "decode_future", _fake_decode_future)
    returned = []
    real = S.HeatmapVLNRuntime._plan_panoramic_native

    def spy(self, payload, blobs):
        response = real(self, payload, blobs)
        returned.append(response)
        return response

    monkeypatch.setattr(S.HeatmapVLNRuntime, "_plan_panoramic_native", spy)
    root = tmp_path / "trace"
    tracer = trace.install(root, diagnostics=True)
    try:
        yield SimpleNamespace(tracer=tracer, root=root, returned=returned)
    finally:
        trace.uninstall()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bridge_call_is_traced_and_the_response_is_untouched(traced):
    runtime = _runtime()
    payload, blobs = _request(0)
    payload_before = copy.deepcopy(payload)

    response = runtime._plan_panoramic_native(payload, blobs)

    assert response is traced.returned[-1]  # the original's object, not a copy
    assert payload == payload_before
    assert response["kind"] == "trajectory" and response["ppa_applied"] is True
    assert _dumps(response) == _dumps(_untraced(0))

    record, arrays = _read(traced.root, 0)
    assert record["schema"] == "exp19-call-trace-v1"
    assert (record["scene_id"], record["episode_id"], record["ep_key"]) == (SCENE, EPISODE, EP_KEY)
    assert record["system2_call_index"] == 0
    assert record["per_call_seed"] == payload["deterministic_sampling"]["per_call_seed"]
    assert record["protocol_seed"] == 42
    for key in ("current_capture_step", "history_capture_steps", "history_age_steps",
                "vo_current_frame_id", "vo_history_frame_ids", "pose_ready",
                "vo_provider_phase", "instruction"):
        assert record[key] == payload[key]
    assert record["request"] == payload_before
    assert record["response"] == json.loads(json.dumps(response))
    assert record["blob_names"] == [blob.name for blob in blobs]
    assert record["trajectory_path"] == "ppa"
    assert record["has_past_output"] is True and record["has_future_output"] is True
    assert record["actions_match"] is True
    assert record["recomputed_actions"] == response["actions"]
    assert record["trace_warnings"] == []
    assert record["checks"]["calls"] == {"decode_future": 1, "form_plan": 1, "past_output": 1, "projected": 1}
    assert all(value is True for name, value in record["checks"].items() if name != "calls")

    expected_jpegs = {f"jpeg__current__{view}" for view in VIEWS}
    expected_jpegs |= {f"jpeg__history__{slot}__front" for slot in range(5)}
    expected_jpegs.add("jpeg__lookdown")
    assert {name for name in arrays if name.startswith("jpeg__")} == expected_jpegs
    assert arrays["jpeg__current__front"].tobytes() == blobs[0].data
    assert Image.open(io.BytesIO(arrays["jpeg__lookdown"].tobytes())).size == (640, 480)

    assert arrays["hist_heatmaps_gated"].shape == (8, 4, 64, 64)
    assert arrays["hist_heatmaps_gated"].dtype == np.float16
    assert arrays["hist_heatmaps"].dtype == np.float16
    assert arrays["hist_visibility_logits"].shape == (8, 4)
    assert arrays["hist_none_probability"].shape == (8,)
    assert arrays["hist_mask"].dtype == bool and arrays["hist_mask"].sum() == 5
    exact = _history_head(
        inputs=None, num_histories=[8], history_rel_poses=None,
        explicit_history_mask=torch.tensor([[True] * 5 + [False] * 3]), return_memory_tokens=True,
    )["heatmaps"][0].numpy()
    flat = exact.reshape(8, 4, -1).argmax(-1)
    np.testing.assert_array_equal(arrays["hist_view_peak_yx"][..., 0], flat // 64)
    np.testing.assert_array_equal(arrays["hist_view_peak_yx"][..., 1], flat % 64)
    for name in ("plan_z0", "plan_z", "delta_z"):
        assert arrays[name].shape == (4, PLAN_DIM) and arrays[name].dtype == np.float32
    assert arrays["delta_token_ratio"].shape == (4,)
    assert np.all(arrays["delta_token_ratio"] <= 0.05 + 1e-2)  # trust-region cap (bf16)
    assert arrays["fut_heatmaps_gated"].shape == (4, 4, 64, 64)
    assert arrays["fut_visibility_probability"].shape == (4, 4)
    assert arrays["trajectory_raw"].shape == (32, 32, 3) and arrays["trajectory_raw"].dtype == np.float32
    assert arrays["selected_path_xy"].shape == (33, 2) and arrays["selected_path_xy"].dtype == np.float64
    np.testing.assert_array_equal(arrays["selected_path_xy"], np.asarray(record["selected_path_xy"]))
    # The response's own summary rounds the same mean-path endpoint to cm.
    goal = re.search(r"traj_goal=\(([-\d.]+),([-\d.]+)\)", response["trajectory_summary"])
    np.testing.assert_allclose(arrays["selected_path_xy"][-1], [float(goal[1]), float(goal[2])], atol=0.0051)
    assert record["shapes"] == {name: list(array.shape) for name, array in arrays.items()}
    assert "trace_failed" not in record
    # The json is written after the npz and carries that write's time.
    assert record["timing_s"]["npz_compress"] >= 0.0 and record["timing_s"]["npz_write"] >= 0.0


def test_bridge_attention_recompute_reproduces_delta_z(traced):
    runtime = _runtime()
    runtime._plan_panoramic_native(*_request(0))
    record, arrays = _read(traced.root, 0)

    diagnostic = record["diagnostic"]
    assert diagnostic["diagnostic_only"] is True and diagnostic["errors"] == []
    check = diagnostic["bridge_attention_check"]
    assert check["capped"] is True and check["reproduces_delta_z"] is True
    assert check["max_abs_delta_z"] > 0.0
    assert diagnostic["bridge_attention_available"] is True
    weights = arrays["bridge_attention"]
    assert weights.shape == (HEADS, 4, 8)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-5)
    assert np.all(weights[..., 5:] == 0.0)  # padded history slots are masked keys


def test_counterfactual_and_replay_use_fresh_generators(traced):
    runtime = _runtime()
    head = runtime.model.nextdit_action_head
    payload, blobs = _request(3)
    response = runtime._plan_panoramic_native(payload, blobs)
    seed = payload["deterministic_sampling"]["per_call_seed"]

    deployed, counterfactual, replay = head.calls
    assert counterfactual.generator is not deployed.generator
    assert replay.generator is not deployed.generator
    assert counterfactual.generator is not replay.generator
    assert all(call.generator.initial_seed() == seed for call in head.calls)
    # The deployed generator was not advanced after the deployed draw.
    assert torch.equal(deployed.generator.get_state(), deployed.state_after)
    assert counterfactual.traj_images is deployed.traj_images

    record, arrays = _read(traced.root, 3)
    assert torch.equal(counterfactual.cond[0].float(), torch.from_numpy(arrays["plan_z0"]))
    assert torch.equal(deployed.cond[0].float(), torch.from_numpy(arrays["plan_z"]))
    assert replay.cond is deployed.cond
    expected = head.sample(counterfactual.cond, torch.Generator().manual_seed(seed))
    np.testing.assert_array_equal(arrays["cf_trajectory_raw"], expected.float().numpy())

    diagnostic = record["diagnostic"]
    cf = diagnostic["counterfactual_no_memory"]
    assert cf["plan"] == "plan_z0" and cf["generator_seed"] == seed
    assert cf["actions_changed"] == (cf["actions"] != response["actions"])
    np.testing.assert_allclose(
        cf["endpoint_shift_m"],
        np.linalg.norm(arrays["cf_selected_path_xy"][-1] - arrays["selected_path_xy"][-1]),
    )
    assert diagnostic["replay_same_plan"]["bitwise_equal"] is True
    assert diagnostic["replay_same_plan"]["actions_equal"] is True


def test_diagnostics_off_runs_nextdit_once(traced):
    traced.tracer.diagnostics = False
    runtime = _runtime()
    runtime._plan_panoramic_native(*_request(0))
    assert len(runtime.model.nextdit_action_head.calls) == 1
    record, arrays = _read(traced.root, 0)
    assert record["diagnostic"]["skipped_reason"] == "EXP19_TRACE_DIAGNOSTICS=0"
    assert record["diagnostic"]["counterfactual_no_memory"] is None
    assert "cf_trajectory_raw" not in arrays and "bridge_attention" not in arrays
    assert record["actions_match"] is True


def test_original_exceptions_propagate(traced):
    runtime = _runtime()
    runtime.model.qwen2_5_vl.fail = RuntimeError("qwen exploded")
    with pytest.raises(RuntimeError, match="qwen exploded"):
        runtime._plan_panoramic_native(*_request(0))
    assert trace._active_context() is None
    assert not (traced.root / EP_KEY).exists()

    runtime.model.qwen2_5_vl.fail = None
    runtime._plan_panoramic_native(*_request(0))
    assert (traced.root / EP_KEY / "call_000.json").exists()


def test_trace_errors_do_not_alter_the_response(traced, monkeypatch):
    def broken_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(trace.CallTracer, "_write", broken_write)
    runtime = _runtime()
    response = runtime._plan_panoramic_native(*_request(0))

    assert response is traced.returned[-1]
    assert _dumps(response) == _dumps(_untraced(0))
    assert not (traced.root / EP_KEY / "call_000.npz").exists()
    fallback = json.loads((traced.root / EP_KEY / "call_000.json").read_text())
    assert fallback["trace_failed"] is True and "disk full" in fallback["traceback"]
    lines = (traced.root / EP_KEY / "trace_errors.jsonl").read_text().splitlines()
    error = json.loads(lines[-1])
    assert error["where"] == "trace_call" and error["system2_call_index"] == 0
    assert "disk full" in error["traceback"]


def test_a_failed_npz_write_still_leaves_a_joinable_call_json(traced, monkeypatch):
    real_write = trace._atomic_write_bytes

    def npz_write_fails(path, data):
        if path.suffix == ".npz":
            raise OSError("npz write failed")
        real_write(path, data)

    monkeypatch.setattr(trace, "_atomic_write_bytes", npz_write_fails)
    runtime = _runtime()
    payload, blobs = _request(0)
    payload_before = copy.deepcopy(payload)
    response = runtime._plan_panoramic_native(payload, blobs)

    assert response is traced.returned[-1]
    assert _dumps(response) == _dumps(_untraced(0))
    assert not (traced.root / EP_KEY / "call_000.npz").exists()
    record = json.loads((traced.root / EP_KEY / "call_000.json").read_text())
    assert record["schema"] == "exp19-call-trace-v1"
    assert record["trace_failed"] is True and record["npz"] is None
    assert "npz write failed" in record["traceback"]
    assert (record["scene_id"], record["episode_id"], record["ep_key"]) == (SCENE, EPISODE, EP_KEY)
    assert record["system2_call_index"] == 0
    assert record["per_call_seed"] == payload["deterministic_sampling"]["per_call_seed"]
    for key in ("current_capture_step", "history_capture_steps", "history_age_steps",
                "vo_current_frame_id", "vo_history_frame_ids", "pose_ready", "vo_provider_phase"):
        assert record[key] == payload[key]
    assert record["request"] == payload_before
    assert record["response"] == json.loads(json.dumps(response))
    error = json.loads((traced.root / EP_KEY / "trace_errors.jsonl").read_text().splitlines()[-1])
    assert error["where"] == "trace_call" and "npz write failed" in error["traceback"]


def test_a_failing_fallback_write_never_reaches_the_server(traced, monkeypatch):
    def every_write_fails(_path, _data):
        raise OSError("read-only file system")

    monkeypatch.setattr(trace, "_atomic_write_bytes", every_write_fails)
    runtime = _runtime()
    response = runtime._plan_panoramic_native(*_request(0))

    assert response is traced.returned[-1]
    assert _dumps(response) == _dumps(_untraced(0))
    assert not (traced.root / EP_KEY / "call_000.json").exists()
    error = json.loads((traced.root / EP_KEY / "trace_errors.jsonl").read_text().splitlines()[-1])
    assert error["where"] == "trace_call" and "read-only file system" in error["traceback"]


def test_a_failing_diagnostic_is_recorded_and_the_rest_still_runs(traced, monkeypatch):
    def broken_attention(*_args, **_kwargs):
        raise ValueError("attention recompute failed")

    monkeypatch.setattr(trace, "recompute_bridge_attention", broken_attention)
    runtime = _runtime()
    response = runtime._plan_panoramic_native(*_request(0))
    assert _dumps(response) == _dumps(_untraced(0))

    record, arrays = _read(traced.root, 0)
    diagnostic = record["diagnostic"]
    assert diagnostic["bridge_attention_available"] is False
    assert [error["diagnostic"] for error in diagnostic["errors"]] == ["bridge_attention"]
    assert diagnostic["counterfactual_no_memory"] is not None
    assert "bridge_attention" not in arrays and "cf_trajectory_raw" in arrays
    error = json.loads((traced.root / EP_KEY / "trace_errors.jsonl").read_text().splitlines()[-1])
    assert error["where"] == "diagnostic.bridge_attention"


def test_warmup_call_traces_the_native_trajectory(traced):
    runtime = _runtime()
    response = runtime._plan_panoramic_native(*_request(1, pose_ready=False))
    assert response["ppa_skip_reason"] == "amb3r_map_warmup"
    assert _dumps(response) == _dumps(_untraced(1, pose_ready=False))

    record, arrays = _read(traced.root, 1)
    assert record["trajectory_path"] == "warmup"
    assert record["has_past_output"] is False and record["has_future_output"] is False
    assert record["actions_match"] is True
    # The History Head is consulted and declines (returns None) before the map.
    assert record["checks"]["calls"] == {"past_output": 1, "warmup": 1}
    assert record["diagnostic"]["counterfactual_no_memory"] is None
    assert record["diagnostic"]["skipped_reason"] == "the bridge did not run on this call"
    assert "hist_heatmaps" not in arrays and "plan_z" not in arrays
    assert arrays["trajectory_raw"].shape == (32, 32, 3)
    assert record["pose_ready"] is False and record["vo_provider_phase"] == "map_warmup"


def test_arrow_call_is_traced_without_a_trajectory(traced):
    runtime = _runtime()
    runtime.processor.tokenizer.text = "←←"
    response = runtime._plan_panoramic_native(*_request(2))
    assert response["kind"] == "native_actions"

    record, arrays = _read(traced.root, 2)
    assert record["trajectory_path"] is None
    assert record["actions_match"] is None and record["recomputed_actions"] is None
    assert record["selected_path_xy"] is None
    assert record["checks"]["calls"] == {}
    assert "jpeg__lookdown" in arrays and "trajectory_raw" not in arrays


def test_restarted_episode_moves_the_earlier_trace_aside(traced):
    runtime = _runtime()
    runtime._plan_panoramic_native(*_request(0))
    runtime._plan_panoramic_native(*_request(1))
    runtime._plan_panoramic_native(*_request(0))
    names = sorted(path.name for path in (traced.root / EP_KEY).iterdir())
    assert names == ["call_000.json", "call_000.npz"]
    assert (traced.root / "_superseded" / f"{EP_KEY}__1" / "call_001.json").exists()


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "selection,x_sign,heading",
    [("mean", 1.0, None), ("mean", -1.0, None), ("mean", 1.0, 35.0), ("endpoint_medoid", 1.0, None)],
)
def test_postprocess_equals_the_deployed_action_path(seed, selection, x_sign, heading):
    generator = torch.Generator().manual_seed(seed)
    drift = torch.tensor([0.8, 0.3 * (seed - 1.5), 0.0])
    trajectory = (torch.randn(32, 32, 3, generator=generator) + drift).to(torch.bfloat16)

    post = trace.Postprocess(
        num_sample_trajs=32, action_scale=4.0, selection=selection, x_sign=x_sign, target_heading_deg=heading
    )
    path, actions, anti_deadlock = post(trajectory)

    deployed = S._finalize_local_actions(
        S.traj_to_actions(
            trajectory,
            num_sample_trajs=32,
            action_scale=4.0,
            trajectory_selection=selection,
            trajectory_x_sign=x_sign,
            target_heading_deg=heading,
        )
    )
    deployed_anti_deadlock = bool(deployed) and deployed[0] == S.ActionCode.STOP
    if deployed_anti_deadlock:
        deployed = [S.ActionCode.LEFT]
    assert actions == deployed and anti_deadlock == deployed_anti_deadlock
    assert path.shape == (33, 2) and np.all(path[0] == 0.0)


def test_postprocess_applies_the_anti_deadlock_rule():
    post = trace.Postprocess(num_sample_trajs=32, action_scale=4.0)
    path, actions, anti_deadlock = post(torch.zeros(32, 32, 3))
    assert actions == [S.ActionCode.LEFT] and anti_deadlock is True
    assert np.all(path == 0.0)


def test_trace_dir_is_required(tmp_path):
    with pytest.raises(ValueError, match="EXP19_TRACE_DIR"):
        trace.trace_config_from_env({})
    with pytest.raises(ValueError, match="EXP19_TRACE_DIAGNOSTICS"):
        trace.trace_config_from_env({"EXP19_TRACE_DIR": str(tmp_path), "EXP19_TRACE_DIAGNOSTICS": "yes"})
    assert trace.trace_config_from_env({"EXP19_TRACE_DIR": str(tmp_path)}) == (tmp_path.resolve(), True)
    assert trace.trace_config_from_env(
        {"EXP19_TRACE_DIR": str(tmp_path), "EXP19_TRACE_DIAGNOSTICS": "0"}
    ) == (tmp_path.resolve(), False)


def test_main_refuses_to_start_without_a_trace_dir(monkeypatch):
    monkeypatch.delenv("EXP19_TRACE_DIR", raising=False)
    monkeypatch.setattr(S, "main", lambda: pytest.fail("the server must not start"))
    assert trace.main() == 2
    assert not trace._PATCHES


def test_install_reaches_the_model_classes_and_uninstall_restores(tmp_path):
    def current(owner, name):
        return owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)

    originals = {(owner, name): current(owner, name) for owner, name, _make in trace._targets()}
    trace.install(tmp_path)
    try:
        for (owner, name), original in originals.items():
            assert current(owner, name)._exp19_original is original
        with pytest.raises(RuntimeError, match="already installed"):
            trace.install(tmp_path)

        runtime = _runtime()
        trace.verify_patch_reach(runtime)

        class _OverridingHead(_FakeNextDiT):
            def get_trajectory_from_projected(self, *args, **kwargs):
                raise AssertionError("never called")

        runtime.model.nextdit_action_head = _OverridingHead()
        with pytest.raises(RuntimeError, match="_OverridingHead.get_trajectory_from_projected"):
            trace.verify_patch_reach(runtime)

        # Only the deployed ppa-online-amb3r arm can be traced.
        for name, value, message in (
            ("ppa_online_amb3r", False, "not the ppa-online-amb3r arm"),
            ("system2_cognition_arm", True, "system2_cognition_arm"),
            ("ppa_stage0_action_arm", "treatment", "ppa_stage0_action_arm=treatment"),
        ):
            runtime = _runtime()
            setattr(runtime, name, value)
            with pytest.raises(RuntimeError, match=message):
                trace.verify_patch_reach(runtime)
    finally:
        trace.uninstall()
    for (owner, name), original in originals.items():
        assert current(owner, name) is original


def test_the_server_refuses_to_serve_a_model_it_cannot_trace(monkeypatch, tmp_path):
    def fake_init(self, args):  # stands in for the model load
        self.__dict__.update(_runtime().__dict__)
        self.system2_cognition_arm = args.system2_cognition_arm

    monkeypatch.setattr(S.HeatmapVLNRuntime, "__init__", fake_init)
    trace.install(tmp_path)
    try:
        S.HeatmapVLNRuntime(SimpleNamespace(system2_cognition_arm=False))
        with pytest.raises(RuntimeError, match="system2_cognition_arm"):
            S.HeatmapVLNRuntime(SimpleNamespace(system2_cognition_arm=True))
    finally:
        trace.uninstall()

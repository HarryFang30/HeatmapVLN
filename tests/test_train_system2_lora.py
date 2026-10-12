import dataclasses
import math
import re

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.train_system2_lora import (
    LORA_TARGET_PATTERN,
    Runtime,
    batch_plan,
    check_lora_targets,
    checkpoint_dirs,
    gather_rank_rows,
    lora_config,
    lora_wrapped_modules,
    per_sample_token_mean_loss,
    restore_checkpoint,
    save_checkpoint,
    shared_index,
    summarize_parity,
    warmup_cosine,
)
from scripts.tools.eval_system2_lora_offline import (
    distance_to_polyline,
    gate_report,
    lookdown_pixel_to_world,
    score_answer,
    summarize_mode,
)
from src.data.s2_sft_dataset import IGNORE_INDEX, CallLabel, CallRef


def _rt(tmp_path=None):
    return Runtime(device=torch.device("cpu"), backend="gloo", rank=0, local_rank=0, world_size=1)


# ---------------------------------------------------------------------------
# Loss, schedule, batch plan
# ---------------------------------------------------------------------------


def test_loss_is_a_per_sample_mean_over_supervised_next_tokens():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    labels = torch.full((2, 5), IGNORE_INDEX)
    labels[0, 2], labels[0, 3] = 4, 7
    labels[1, 4] = 1
    per_sample, counts = per_sample_token_mean_loss(logits, labels)
    want0 = (F.cross_entropy(logits[0, 1:2], torch.tensor([4])) + F.cross_entropy(logits[0, 2:3], torch.tensor([7]))) / 2
    want1 = F.cross_entropy(logits[1, 3:4], torch.tensor([1]))
    torch.testing.assert_close(per_sample, torch.stack([want0, want1]))
    assert counts.tolist() == [2, 1]
    labels[1, 4] = IGNORE_INDEX
    with pytest.raises(RuntimeError, match="no supervised token"):
        per_sample_token_mean_loss(logits, labels)


def test_loss_upcasts_bf16_logits_and_keeps_the_graph():
    logits = torch.randn(1, 4, 9, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([[IGNORE_INDEX, 3, 5, IGNORE_INDEX]])
    per_sample, _ = per_sample_token_mean_loss(logits, labels)
    assert per_sample.dtype == torch.float32
    per_sample.sum().backward()
    assert logits.grad is not None and logits.grad[0, 2:].abs().sum() == 0


def test_warmup_then_cosine_to_zero():
    factor = warmup_cosine(100, 0.03)
    assert factor(0) == pytest.approx(1 / 3) and factor(2) == pytest.approx(1.0)
    assert factor(3) == pytest.approx(1.0)
    values = [factor(s) for s in range(3, 100)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert factor(100) == pytest.approx(0.0, abs=1e-12)
    assert warmup_cosine(10, 0.0)(0) == pytest.approx(1.0)


def test_global_batch_is_split_into_micro_batches_and_accumulation():
    assert batch_plan(64, 1, 8) == 8
    assert batch_plan(64, 2, 4) == 8
    assert batch_plan(64, 1, 1) == 64
    with pytest.raises(ValueError):
        batch_plan(64, 3, 8)


# ---------------------------------------------------------------------------
# LoRA placement
# ---------------------------------------------------------------------------


class _Mlp(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(d, d) for _ in range(4))


class _Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn, self.mlp = _Attn(d), _Mlp(d)


class _VisionBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = _Mlp(d)  # Qwen2.5-VL's vision blocks also have gate/up/down_proj
        self.attn = nn.Module()
        self.attn.qkv, self.attn.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)


class _TinyQwen(nn.Module):
    """The module tree of Qwen2_5_VLForConditionalGeneration in transformers 4.51."""

    def __init__(self, layers=2, d=8):
        super().__init__()
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList(_VisionBlock(d) for _ in range(2))
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_Layer(d) for _ in range(layers))
        self.lm_head = nn.Linear(d, 5)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer.mlp.down_proj(layer.self_attn.q_proj(x))
        return self.lm_head(x)


def _peft_tiny(layers=2):
    peft = pytest.importorskip("peft")
    model = _TinyQwen(layers).to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    config = dataclasses.replace(lora_config(4, 8, 0.0), task_type=None)
    return peft.get_peft_model(model, config)


def test_lora_reaches_every_language_layer_and_nothing_else():
    names = ["model.layers.0.self_attn.q_proj", "model.layers.27.mlp.down_proj", "visual.blocks.0.mlp.gate_proj",
             "visual.blocks.0.attn.qkv", "lm_head", "model.layers.3.self_attn.rotary_emb", "model.embed_tokens"]
    assert [n for n in names if re.fullmatch(LORA_TARGET_PATTERN, n)] == names[:2]
    peft_model = _peft_tiny()
    wrapped = lora_wrapped_modules(peft_model)
    assert len(wrapped) == 2 * 7 and not any(n.startswith("visual") for n in wrapped)
    info = check_lora_targets(peft_model, 2)
    assert info["wrapped_modules"] == 14
    # fp32 adapters on a bf16 base
    assert {p.dtype for n, p in peft_model.named_parameters() if p.requires_grad} == {torch.float32}
    with pytest.raises(RuntimeError, match="wraps 14 modules"):
        check_lora_targets(peft_model, 3)
    with pytest.raises(RuntimeError, match="must be frozen"):
        check_lora_targets(peft_model, 2, trainable=False)


def test_a_fresh_adapter_is_a_no_op():
    peft_model = _peft_tiny()
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    with peft_model.disable_adapter():
        base = peft_model(x)
    torch.testing.assert_close(peft_model(x), base, rtol=0, atol=0)


def test_checkpoint_round_trip_and_run_key_guard(tmp_path):
    peft_model = _peft_tiny()
    params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(10, 0.1))
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    peft_model(x).float().pow(2).mean().backward()
    optimizer.step()
    scheduler.step()
    trained = {n: p.detach().clone() for n, p in peft_model.named_parameters() if p.requires_grad}
    save_checkpoint(tmp_path, peft_model, optimizer, scheduler, step=3, run_key="k", keep=2)
    (tmp_path / "checkpoints" / "step_000009").mkdir()  # incomplete: no COMPLETE marker
    assert [p.name for p in checkpoint_dirs(tmp_path)] == ["step_000003"]

    fresh = _peft_tiny()
    fresh_params = [p for p in fresh.parameters() if p.requires_grad]
    fresh_opt = torch.optim.AdamW(fresh_params, lr=1e-3)
    fresh_sched = torch.optim.lr_scheduler.LambdaLR(fresh_opt, warmup_cosine(10, 0.1))
    with pytest.raises(RuntimeError, match="different run"):
        restore_checkpoint(checkpoint_dirs(tmp_path)[-1], fresh, fresh_opt, fresh_sched, run_key="other")
    step = restore_checkpoint(checkpoint_dirs(tmp_path)[-1], fresh, fresh_opt, fresh_sched, run_key="k")
    assert step == 3 and fresh_sched.last_epoch == scheduler.last_epoch
    for name, param in fresh.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param, trained[name])
    for s in (4, 5):
        save_checkpoint(tmp_path, peft_model, optimizer, scheduler, step=s, run_key="k", keep=2)
    assert [p.name for p in checkpoint_dirs(tmp_path)] == ["step_000004", "step_000005"]


# ---------------------------------------------------------------------------
# Single-process plumbing and the G0 summary
# ---------------------------------------------------------------------------


def test_single_process_plumbing(tmp_path):
    rt = _rt()
    assert shared_index(rt, tmp_path / "x.pkl", lambda: ("built", 1)) == ("built", 1)
    assert not (tmp_path / "x.pkl").exists()
    rows = [{"a": 1}, {"a": 2}]
    assert gather_rank_rows(rows, rt, tmp_path / "shards", "t") == rows


def test_parity_summary():
    rows = [
        {"kind": "trajectory", "recorded": ["↓", "1 2"], "exact": True, "turn1_match": True, "class_match": True},
        {"kind": "trajectory", "recorded": ["↓", "1 2"], "exact": False, "turn1_match": True, "class_match": True},
        {"kind": "stop", "recorded": ["STOP"], "exact": True, "turn1_match": True, "class_match": True},
        {"kind": "native_actions", "recorded": ["←"], "exact": True, "turn1_match": True, "class_match": True},
    ]
    report = summarize_parity(rows)
    assert report["exact_rate"] == 0.75 and report["two_turn"] == {"n": 2, "exact_rate": 0.5}
    assert report["g0_pass"] is False and len(report["mismatches"]) == 1
    assert summarize_parity(rows[2:])["g0_pass"] is True


# ---------------------------------------------------------------------------
# Offline gates
# ---------------------------------------------------------------------------


def test_lookdown_back_projection_agrees_with_the_exp19_projection():
    gt = pytest.importorskip("scripts.exp19.gt")
    path_xy = np.array([[2.0, 0.5], [1.4, -0.3]])  # robot frame: x forward, y left, on the floor
    cam = gt.path_camera_points(path_xy, pitch_deg=30.0)
    uv = gt.project_to_decision_image(cam, (640, 480))  # pixel index coordinates
    yaw = math.radians(30.0)
    rotation = [math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0]  # about +y (up)
    position = np.array([1.0, 0.1, -2.0])
    body = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    for (forward, left), point_cam, (u, v) in zip(path_xy, cam, uv):
        col, row = int(round(u)), int(round(v))
        depth = np.zeros((480, 640), dtype=np.float32)
        depth[row, col] = -point_cam[2]
        world = lookdown_pixel_to_world(row, col, depth, position, rotation)
        # Habitat body frame: forward = -z, left = -x, floor 0 m below the body origin.
        want = position + body @ np.array([-left, 0.0, -forward])
        assert np.linalg.norm(world - want) < 0.02  # one pixel of rounding at ~2 m
    assert lookdown_pixel_to_world(0, 0, np.zeros((480, 640)), position, rotation) is None
    assert lookdown_pixel_to_world(480, 0, np.ones((480, 640)), position, rotation) is None


def test_distance_to_polyline():
    path = [[0, 0, 0], [2, 0, 0], [2, 0, 2]]
    assert distance_to_polyline(np.array([1.0, 0.0, 0.5]), path) == pytest.approx(0.5)
    assert distance_to_polyline(np.array([3.0, 0.0, 3.0]), path) == pytest.approx(math.sqrt(2))
    assert distance_to_polyline(np.array([0.0, 1.0, 0.0]), [[0, 0, 0]]) == pytest.approx(1.0)


def _ref(agree, oracle_turns, recorded=("↓", "100 200"), distance=5.0, path=None):
    label = CallLabel(0, False, agree, "x", tuple(oracle_turns), None, distance, path)
    return CallRef("r.tar", "scene", 1, 0, tuple(recorded), distance, label)


def test_answers_are_scored_against_record_and_oracle():
    nowhere = lambda text: None  # noqa: E731
    keep = score_answer(("↓", "100 200"), _ref(True, ["STOP"]), pixel_world=nowhere, path_tolerance_m=1.0)
    assert keep["exact_recorded"] and keep["class"] == "pixel" and "corrected" not in keep
    turn = score_answer(("→",), _ref(False, ["→→"]), pixel_world=nowhere, path_tolerance_m=1.0)
    assert turn["oracle_class_match"] and turn["corrected"] is True
    wrong_way = score_answer(("←",), _ref(False, ["→→"]), pixel_world=nowhere, path_tolerance_m=1.0)
    assert wrong_way["corrected"] is False
    path = ((0.0, 0.0, 0.0), (0.0, 0.0, -5.0))
    on_path = score_answer(("↓", "9 9"), _ref(False, ["↓", "1 1"], path=path),
                           pixel_world=lambda t: np.array([0.5, 0.0, -2.0]), path_tolerance_m=1.0)
    off_path = score_answer(("↓", "9 9"), _ref(False, ["↓", "1 1"], path=path),
                            pixel_world=lambda t: np.array([1.5, 0.0, -2.0]), path_tolerance_m=1.0)
    unplaced = score_answer(("↓", "9 9"), _ref(False, ["↓", "1 1"]), pixel_world=nowhere, path_tolerance_m=1.0)
    assert on_path["corrected"] is True and off_path["corrected"] is False and unplaced["corrected"] is None


def test_gate_summary_and_thresholds():
    def answer(stop=False, exact=True, corrected=None, cls_match=None):
        out = {"stop": stop, "exact_recorded": exact}
        if corrected is not None or cls_match is not None:
            out.update({"corrected": corrected, "oracle_class_match": cls_match})
        return out

    rows = []
    for i in range(10):  # agreeing, far: the adapter keeps 9 of the off answers, no stop
        rows.append({"agree": True, "distance_to_goal": 6.0, "record_device": "npu", "on_off_identical": i > 0,
                     "on": answer(exact=i > 0), "off": answer()})
    rows.append({"agree": False, "distance_to_goal": 1.0, "record_device": "npu", "on_off_identical": False,
                 "on": answer(stop=True, exact=False, corrected=True, cls_match=True),
                 "off": answer(exact=True, corrected=False, cls_match=False)})
    rows.append({"agree": None, "distance_to_goal": None, "record_device": "npu", "on_off_identical": True,
                 "on": answer(), "off": answer()})
    on = summarize_mode(rows, "on", r_stop_m=2.0, false_stop_m=3.0, device="npu")
    off = summarize_mode(rows, "off", r_stop_m=2.0, false_stop_m=3.0, device="npu")
    assert {k: on["G1_keep"][k] for k in ("n", "matched", "rate")} == {"n": 10, "matched": 9, "rate": 0.9}
    assert on["G2_false_stop"]["n"] == 10 and on["G2_false_stop"]["rate"] == 0.0
    assert on["G3_stop_recall"] == {"n": 1, "stops": 1, "rate": 1.0, "distance_le_m": 2.0}
    assert on["G4_correction"]["corrected_rate"] == 1.0 and off["G4_correction"]["corrected_rate"] == 0.0
    assert on["calls_without_distance"] == 1
    gates = gate_report(on, off)
    assert gates["G0_adapter_off_replay"]["value"] == 1.0
    assert gates["G1_keep"]["pass"] and gates["G2_false_stop"]["pass"] and gates["all_pass"]
    rows[0]["on"]["stop"] = True  # one false stop in ten far calls
    assert not gate_report(summarize_mode(rows, "on", r_stop_m=2.0, false_stop_m=3.0, device="npu"), off)["G2_false_stop"]["pass"]


def test_g0_counts_only_calls_recorded_on_this_platform():
    def row(device, exact):
        answer = {"stop": False, "exact_recorded": exact}
        return {"agree": True, "distance_to_goal": 6.0, "record_device": device, "on_off_identical": True,
                "on": dict(answer), "off": dict(answer)}

    # NPU records replay exactly; a fifth of the CUDA records differ by a few pixels on the NPU.
    rows = [row("npu", True) for _ in range(50)] + [row("cuda", i % 5 != 0) for i in range(50)]
    off = summarize_mode(rows, "off", r_stop_m=2.0, false_stop_m=3.0, device="npu")
    on = summarize_mode(rows, "on", r_stop_m=2.0, false_stop_m=3.0, device="npu")
    assert off["replay_exact"]["rate"] == 0.9  # pooled: would fail the 0.98 line for a platform reason
    assert off["replay_exact_same_device"] == {"device": "npu", "n": 50, "matched": 50, "rate": 1.0}
    assert gate_report(on, off)["G0_adapter_off_replay"]["pass"]
    assert on["G1_keep"]["rate"] == 1.0 and on["G1_keep_vs_recorded"]["rate"] == 0.9


def test_stream_restart_shard_is_the_start_of_the_workers_round():
    from scripts.train_system2_lora import stream_start_shard

    # 256-sample shards, global batch 64 -> 4 steps per shard; 4 workers read 4 shards at a time.
    assert stream_start_shard(0, 64, 256, 4) == 0
    assert stream_start_shard(15, 64, 256, 4) == 0   # step 15 is still inside the first round of 4 shards
    assert stream_start_shard(16, 64, 256, 4) == 4
    assert stream_start_shard(100, 64, 256, 4) == 24  # 25 shards consumed; round 24-27 restarts whole
    assert stream_start_shard(100, 64, 256, 0) == 25  # no workers: sequential


def test_stream_resume_empties_the_buffer_and_names_the_restart_shard(tmp_path):
    from types import SimpleNamespace

    from scripts.train_system2_lora import prepare_stream_resume

    for name in ("stream-000024.tar", "stream-000150.tar", "stream-000024.tar.done.r0", "stream_manifest.json"):
        (tmp_path / name).write_text("x")
    rt = SimpleNamespace(is_main=True, barrier=lambda: None, rank=0)
    prepare_stream_resume(tmp_path, 24, rt, tmp_path / "log.jsonl")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["NEED_PUSH_FROM", "log.jsonl", "stream_manifest.json"]
    assert (tmp_path / "NEED_PUSH_FROM").read_text().strip() == "24"

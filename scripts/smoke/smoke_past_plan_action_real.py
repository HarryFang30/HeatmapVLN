#!/usr/bin/env python3
"""One-GPU, one-real-batch Past -> Plan -> Action contract smoke.

This is a development validation, not a training launcher.  It consumes one
fixed R2R expert endpoint (clip_004345, t=19), proves the fresh zero bridge is
bitwise native under one explicit noise tensor, then performs exactly one
Stage-2 forward/backward/AdamW update and audits gradients.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT_DEFAULT = Path("/mnt/afs/lixiaoou/intern/fjl/HeatmapVLN")
CONFIG_DEFAULT = (
    Path(__file__).resolve().parents[2]
    / "configs/ppa_stage2_real_one_batch.yaml"
)
CHECKPOINT_DEFAULT = Path(
    "/mnt/afs/lixiaoou/intern/fjl/"
    "model/output_heatmap_internnav_single_view_v1_4gpu/"
    "runs/run_20260803_143402/checkpoints/best.pth"
)
DATASET_ROOT = Path("/mnt/afs/lixiaoou/intern/fjl/r2r_paronamic_data")
EXACT_CLIP = DATASET_ROOT / "train/17DRP5sb8fy/clip_004345"
EXACT_CURRENT_T = 19
EXACT_SAMPLE_ID = "train/17DRP5sb8fy/clip_004345@000019"
EXACT_PIXEL_GOAL = [189, 303]
EXACT_GOAL_LEN = 13
CACHE_ROOT = Path(
    "/mnt/afs/lixiaoou/intern/fjl/.codex_tmp/"
    "ppa_r2r_cache_20260813/cache"
)
REPORT_DEFAULT = Path(
    "/mnt/afs/lixiaoou/intern/fjl/.codex_tmp/"
    "ppa_real_smoke_20260813/report.json"
)
VISUALIZATION_DEFAULT = Path(
    "/mnt/afs/lixiaoou/intern/fjl/.codex_tmp/"
    "ppa_real_smoke_20260813/future_heatmap_strip.png"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    parser.add_argument(
        "--visualization", type=Path, default=VISUALIZATION_DEFAULT
    )
    parser.add_argument("--gpu", default="0", help="one physical GPU id")
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _finite_scalar(value: Any, *, name: str) -> float:
    import torch

    _require(torch.is_tensor(value) and value.ndim == 0, f"{name} must be scalar")
    _require(bool(torch.isfinite(value)), f"{name} is non-finite")
    return float(value.detach().float().cpu())


def _snapshot_scheduler_training_state(scheduler: Any) -> dict[str, Any]:
    """Capture the exact flow-matching training schedule before inference."""
    return {
        "timesteps": scheduler.timesteps.detach().clone(),
        "sigmas": scheduler.sigmas.detach().clone(),
        "step_index": getattr(scheduler, "_step_index", None),
        "begin_index": getattr(scheduler, "_begin_index", None),
        "had_num_inference_steps": hasattr(scheduler, "num_inference_steps"),
        "num_inference_steps": getattr(scheduler, "num_inference_steps", None),
    }


def _restore_scheduler_training_state(
    scheduler: Any,
    state: dict[str, Any],
) -> None:
    """Undo inference-time scheduler mutation before the Stage-2 loss."""
    scheduler.timesteps = state["timesteps"]
    scheduler.sigmas = state["sigmas"]
    scheduler._step_index = state["step_index"]
    scheduler._begin_index = state["begin_index"]
    if state["had_num_inference_steps"]:
        scheduler.num_inference_steps = state["num_inference_steps"]
    elif hasattr(scheduler, "num_inference_steps"):
        delattr(scheduler, "num_inference_steps")

    expected = int(scheduler.config.num_train_timesteps)
    _require(
        int(scheduler.timesteps.numel()) == expected,
        "failed to restore the exact flow-matching training timesteps",
    )
    _require(
        int(scheduler.sigmas.numel()) in {expected, expected + 1},
        "failed to restore the exact flow-matching training sigmas",
    )


def _gradient_report(
    named_parameters: Iterable[tuple[str, Any]],
    *,
    family: str,
) -> tuple[dict[str, Any], tuple[str, Any]]:
    import torch

    selected = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    _require(bool(selected), f"{family}: no trainable parameters")
    with_grad = []
    nonzero = []
    squared_norm = torch.zeros((), dtype=torch.float64)
    max_abs = 0.0
    for name, parameter in selected:
        gradient = parameter.grad
        if gradient is None:
            continue
        _require(bool(torch.isfinite(gradient).all()), f"{family}: non-finite gradient at {name}")
        with_grad.append((name, parameter))
        abs_max = float(gradient.detach().float().abs().max().cpu())
        max_abs = max(max_abs, abs_max)
        squared_norm += gradient.detach().double().square().sum().cpu()
        if abs_max > 0.0:
            nonzero.append((name, parameter))
    _require(bool(with_grad), f"{family}: every gradient is None")
    _require(bool(nonzero), f"{family}: every gradient is exactly zero")
    report = {
        "trainable_tensors": len(selected),
        "tensors_with_grad": len(with_grad),
        "nonzero_grad_tensors": len(nonzero),
        "grad_l2": float(squared_norm.sqrt()),
        "grad_abs_max": max_abs,
        "first_nonzero_tensor": nonzero[0][0],
    }
    return report, nonzero[0]


def _render_future_strip(
    *,
    dataset: Any,
    gt_heatmaps: Any,
    pred_heatmaps_gated: Any,
    output_path: Path,
) -> None:
    """Render the same F|R|B|L magma-strip convention as Past maps.

    The four horizontal groups are future horizons rather than observation
    frames.  RGB is intentionally repeated because every prediction is made
    from the same current observation.  Both GT and prediction use one fixed
    [0,1] color scale; per-row normalization would destroy confidence
    semantics.
    """

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import colormaps

    views = dataset._load_all_views(EXACT_CLIP, EXACT_CURRENT_T)
    _require(tuple(views.shape[:2]) == (4, 3), "display panorama shape drift")
    gt = gt_heatmaps.detach().float().cpu().numpy()
    pred = pred_heatmaps_gated.detach().float().cpu().numpy()
    _require(gt.shape == (4, 4, 64, 64), f"GT display shape drift: {gt.shape}")
    _require(pred.shape == gt.shape, f"prediction display shape drift: {pred.shape}")
    _require(np.isfinite(gt).all() and np.isfinite(pred).all(), "display maps non-finite")

    tile = 64
    gap = 2
    row_gap = 4
    left = 142
    top = 30
    group_width = 4 * tile + 3 * gap
    content_width = 4 * group_width + 3 * 8
    canvas = Image.new(
        "RGB",
        (left + content_width, top + 3 * tile + 2 * row_gap),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    direction_labels = ("F", "R", "B", "L")
    horizon_labels = ("steps 1-8", "steps 9-16", "steps 17-24", "steps 25-32")
    row_labels = (
        "Current RGB (same input)",
        "GT future trajectory",
        "Pred future (untrained)",
    )
    magma = colormaps["magma"]

    rgb_tiles = []
    for view in views:
        array = view.detach().float().cpu().permute(1, 2, 0).numpy()
        array = np.clip(array, 0.0, 1.0)
        image = Image.fromarray((array * 255.0 + 0.5).astype(np.uint8))
        rgb_tiles.append(image.resize((tile, tile), Image.Resampling.BILINEAR))

    def heatmap_tile(value: np.ndarray) -> Image.Image:
        rgba = magma(np.clip(value, 0.0, 1.0), bytes=True)
        return Image.fromarray(rgba[..., :3], mode="RGB")

    for row, label in enumerate(row_labels):
        y = top + row * (tile + row_gap)
        draw.text((5, y + tile // 2 - 5), label, fill=(240, 240, 240), font=font)
        for horizon in range(4):
            group_x = left + horizon * (group_width + 8)
            if row == 0:
                tiles = rgb_tiles
            else:
                maps = gt[horizon] if row == 1 else pred[horizon]
                tiles = [heatmap_tile(maps[view]) for view in range(4)]
            for view, image in enumerate(tiles):
                x = group_x + view * (tile + gap)
                canvas.paste(image, (x, y))
                draw.text((x + 3, y + 2), direction_labels[view], fill=(255, 230, 90), font=font)
            if row == 0:
                draw.text((group_x + 4, 5), horizon_labels[horizon], fill=(255, 230, 90), font=font)
            if horizon < 3:
                separator_x = group_x + group_width + 3
                draw.line((separator_x, top, separator_x, canvas.height), fill=(215, 180, 20), width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _assert_exact_inputs(cfg: dict[str, Any], checkpoint: Path) -> None:
    trajectory = cfg["data"]["trajectory"]
    nextdit = cfg["model"]["action_head"]["nextdit"]
    stage = cfg["training"]["stages"][0]
    _require(Path(cfg["data"]["root"]).resolve() == DATASET_ROOT, "dataset root drift")
    _require(
        Path(trajectory["amb3r_pose_cache_root"]).resolve() == CACHE_ROOT,
        "AMB3R cache root drift",
    )
    _require(trajectory["require_amb3r_pose_cache"] is True, "AMB3R must be required")
    _require(trajectory["random_subsequence"] is False, "cache identity forbids subsequences")
    _require(trajectory["future_heatmap"]["enabled"] is True, "Future target disabled")
    _require(trajectory["enable_trajectory_augmentation"] is False, "target/action drift")
    _require(trajectory["load_depth"] is True, "expert labels need source geometry")
    _require(trajectory["panoramic_vlm_input"] is False, "native System2 path drift")
    _require(
        cfg["model"]["llm"]["model_path"] == nextdit["internnav_model_path"],
        "System2/System1 model roots differ",
    )
    _require(not nextdit.get("internnav_system1_path"), "external System1 override")
    _require(not nextdit.get("pretrained_system1_path"), "external System1 checkpoint")
    _require(not nextdit.get("dav2_ckpt_path"), "external DAV2 checkpoint")
    _require(stage["past_plan_action_stage"] == "stage2_joint", "not Stage2")
    _require(stage["required_history_pose_provider"] == "amb3r_vo_cache", "provider drift")
    _require(stage["trajectory_sequence_mode"] == "first_only", "action layout drift")
    _require(stage["trainable_modules"] == ["past_plan_action", "heatmap_vln"], "scope drift")
    _require(checkpoint.resolve() == CHECKPOINT_DEFAULT, "single-view best checkpoint drift")


def _build_exact_sample(cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Use the real factory while overriding only clip enumeration in-process."""

    import src.data.trajectory_dataset as trajectory_module
    from src.data.factory import build_trajectory_dataset

    base_class = trajectory_module.VLNTrajectoryDataset
    exact_clip = EXACT_CLIP.resolve()

    class ExactClipTrajectoryDataset(base_class):
        def _enumerate_clips(self) -> list[Path]:
            _require(self.root.resolve() == DATASET_ROOT, "dataset root changed before enumeration")
            _require(self.split == "train", f"expected train split, got {self.split!r}")
            _require(exact_clip.is_dir(), f"missing exact R2R clip: {exact_clip}")
            return [exact_clip]

    trajectory_module.VLNTrajectoryDataset = ExactClipTrajectoryDataset
    try:
        dataset = build_trajectory_dataset(cfg, split="train", max_clips=1)
    finally:
        trajectory_module.VLNTrajectoryDataset = base_class

    matches = [
        index
        for index, (clip_index, current_t) in enumerate(dataset.sample_index)
        if dataset.clips[clip_index].resolve() == exact_clip
        and int(current_t) == EXACT_CURRENT_T
    ]
    _require(len(matches) == 1, f"expected one exact endpoint, got indices={matches}")
    sample = dataset[matches[0]]
    _require(sample.get("sample_identity") == EXACT_SAMPLE_ID, "dataset retried another endpoint")
    _require(sample.get("history_pose_provider") == "amb3r_vo_cache", "GT pose fallback")
    _require(sample.get("pixel_goal") == EXACT_PIXEL_GOAL, "native pixel goal drift")
    _require(int(sample.get("pixel_goal_relative_len", -1)) == EXACT_GOAL_LEN, "goal_len drift")
    _require(tuple(sample["history_rel_poses"].shape) == (8, 4), "AMB3R pose shape drift")
    _require(tuple(sample["heatmap"].shape) == (8, 4, 64, 64), "History target shape drift")
    _require(tuple(sample["gt_visibility"].shape) == (8, 4), "History visibility shape drift")
    _require(tuple(sample["future_trajectory_heatmap"].shape) == (4, 4, 64, 64), "Future target shape drift")
    _require(tuple(sample["trajectory"].shape) == (12, 32, 3), "expert action sequence drift")
    _require(tuple(sample["traj_images"].shape) == (12, 224, 224, 3), "System1 image sequence drift")
    return dataset, sample


def _collate_exact_sample(cfg: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoProcessor

    from src.data.amb3r_pose_cache import AMB3R_POSE_PROVIDER
    from src.data.future_trajectory_batch import assert_no_future_teacher_inputs
    from src.data.internnav_heatmap_control_collator import InternNavHeatmapControlCollator

    model_path = cfg["model"]["llm"]["model_path"]
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    collator = InternNavHeatmapControlCollator(
        processor,
        n_traj_query=4,
        max_seq_length=8192,
        teacher_force_system2_answer=True,
        include_future_trajectory_targets=True,
        required_history_pose_provider=AMB3R_POSE_PROVIDER,
    )
    batch = collator([sample])
    required = {
        "pano_inputs",
        "pano_num_histories",
        "heatmap_single_view_inputs",
        "heatmap_single_view_num_histories",
        "history_valid_mask",
        "history_rel_poses",
        "history_pose_provider",
        "heatmap",
        "gt_visibility",
        "trajectory",
        "trajectory_valid",
        "traj_images",
        "future_trajectory_heatmap",
        "future_trajectory_visibility",
        "future_trajectory_time_mask",
    }
    missing = sorted(required - set(batch))
    _require(not missing, f"joint collator dropped required PPA fields: {missing}")
    _require(batch["history_pose_provider"] == [AMB3R_POSE_PROVIDER], "provider drift in batch")
    _require(batch["native_system2_num_histories"] == [8], "native history count drift")
    _require(tuple(batch["trajectory"].shape) == (1, 32, 3), "first-only action drift")
    _require(tuple(batch["traj_images"].shape) == (1, 2, 224, 224, 3), "eval-matched image pair drift")
    _require(tuple(batch["heatmap"].shape) == (1, 8, 4, 64, 64), "History batch shape drift")
    _require(tuple(batch["future_trajectory_heatmap"].shape) == (1, 4, 4, 64, 64), "Future batch shape drift")
    assert_no_future_teacher_inputs(batch)
    forbidden = {
        "current_pose",
        "current_camera_pose",
        "current_agent_pose",
        "history_poses",
        "current_depth",
    }
    leaked = sorted(forbidden & set(batch))
    _require(not leaked, f"GT pose/depth reached model batch: {leaked}")
    return batch


def _model_forward(model: Any, batch: dict[str, Any], device: Any) -> dict[str, Any]:
    history_rel_poses = batch["history_rel_poses"].to(device, non_blocking=True)
    return model(
        video_frames=batch["current_frame"].unsqueeze(1),
        instruction_text=list(batch["text"]),
        current_observation=batch["current_frame"],
        panoramic_inputs=batch["pano_inputs"],
        panoramic_num_histories=batch["pano_num_histories"],
        panoramic_text_anchor_positions=batch.get("pano_text_anchor_positions"),
        heatmap_single_view_inputs=batch["heatmap_single_view_inputs"],
        heatmap_single_view_num_histories=batch["heatmap_single_view_num_histories"],
        heatmap_control_history_mask=batch["heatmap_control_history_mask"],
        history_valid_mask=batch["history_valid_mask"],
        history_age_steps=batch["history_age_steps"],
        history_rel_poses=history_rel_poses,
        sample_trajectory=False,
        return_heatmaps=True,
        return_heatmap_logits=True,
        return_actions=True,
        return_future_heatmaps=True,
        return_lm_loss=False,
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    _require((repo_root / "src").is_dir(), f"invalid repository root: {repo_root}")
    _require(config_path.is_file(), f"missing smoke config: {config_path}")
    _require(checkpoint.is_file(), f"missing single-view best: {checkpoint}")
    _require(EXACT_CLIP.is_dir(), f"missing exact clip: {EXACT_CLIP}")
    cache_npz = CACHE_ROOT / "17DRP5sb8fy/clip_004345/amb3r_pose_cache.npz"
    _require(cache_npz.is_file(), f"missing strict endpoint-v2 cache: {cache_npz}")

    sys.path.insert(0, str(repo_root))

    import torch

    _require(torch.cuda.is_available(), "CUDA/MACA is not available")
    _require(torch.cuda.device_count() == 1, "smoke must see exactly one GPU")
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda:0")

    from scripts.training.model_builder import (
        assert_complete_internnav_system1_load,
        build_model,
        set_trainable_modules,
    )
    from scripts.training.optimizer import build_optimizer
    from scripts.training.pose_adaptation import (
        assert_required_history_pose_provider,
        load_past_plan_action_initialization,
    )
    from scripts.training.train_loop import (
        _apply_bridge_only_train_mode,
        _prepare_trajectory_sequence_inputs,
    )
    from scripts.training.utils import (
        build_future_heatmap_loss_fn,
        build_heatmap_loss_fn,
    )
    from src.config_schema import load_and_validate_config
    from src.models.past_plan_action import (
        compute_shared_plan_action_losses,
        verify_stage0_treatment_equivalence,
    )
    from src.models.action.treatment_spec import TrajectoryPostprocessConfig
    from src.models.past_plan_action_loss import (
        PastPlanActionLossWeights,
        compose_past_plan_action_loss,
    )
    from src.models.past_plan_action_training import (
        assert_native_frozen_and_gradient_free,
    )

    cfg = load_and_validate_config(config_path)
    _assert_exact_inputs(cfg, checkpoint)
    stage_cfg = cfg["training"]["stages"][0]

    started = time.perf_counter()
    dataset_started = time.perf_counter()
    _dataset, sample = _build_exact_sample(cfg)
    batch = _collate_exact_sample(cfg, sample)
    assert_required_history_pose_provider(batch, stage_cfg)
    dataset_seconds = time.perf_counter() - dataset_started

    model_started = time.perf_counter()
    model = build_model(cfg, verbose=True, device="cuda:0", enable_action_head=True)
    system1_tensor_count = assert_complete_internnav_system1_load(model)
    backbone = model.vlm_backbone
    if backbone.model is None:
        backbone._load_model()
    model._ensure_heatmap_vln()
    init_report = load_past_plan_action_initialization(
        model,
        checkpoint,
        # Development smoke starts from the existing complete Past Head.  A
        # formal Stage-2 run must instead load its trained Stage-1 Future Head.
        stage="stage1_map_pretrain",
    )
    set_trainable_modules(model, stage_cfg, logging.getLogger("ppa-smoke"))
    optimizer = build_optimizer(model, cfg, stage_cfg)
    model_seconds = time.perf_counter() - model_started

    _require(init_report["loaded_heatmap_head_tensors"] == 79, "incomplete Past Head")
    _require(init_report["loaded_future_head_tensors"] == 0, "smoke must use fresh Future Head")
    _require(init_report["bridge_zero_initialized"] is True, "bridge is not zero initialized")
    _require(model.latent_queries.requires_grad is False, "native TRAJ queries trainable")

    model.eval()
    model.nextdit_action_head.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    stage0_started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        stage0_output = _model_forward(model, batch, device)
    _require(tuple(stage0_output["plan_z0"].shape) == (1, 4, 768), "native Plan shape")
    _require(tuple(stage0_output["plan_z"].shape) == (1, 4, 768), "bridged Plan shape")
    _require(torch.equal(stage0_output["plan_z"], stage0_output["plan_z0"]), "zero bridge changed Plan")
    _require(tuple(stage0_output["future_heatmaps"].shape) == (1, 4, 4, 64, 64), "Future output shape")
    _require(tuple(stage0_output["future_visibility_logits"].shape) == (1, 4, 4), "Future visibility shape")
    _require(bool(torch.isfinite(stage0_output["future_heatmaps"]).all()), "Future output non-finite")
    _render_future_strip(
        dataset=_dataset,
        gt_heatmaps=batch["future_trajectory_heatmap"][0],
        pred_heatmaps_gated=stage0_output["future_heatmaps_gated"][0],
        output_path=args.visualization.expanduser().resolve(),
    )

    traj_images = batch["traj_images"].to(device, non_blocking=True)
    action_head = model.nextdit_action_head
    scheduler_training_state = _snapshot_scheduler_training_state(
        action_head.noise_scheduler
    )
    noise_shape = (
        int(stage0_output["plan_z"].shape[0]) * int(action_head.config.num_sample_trajs),
        int(action_head.config.predict_steps),
        int(action_head.config.action_dim),
    )
    noise_generator = torch.Generator(device=device).manual_seed(20260813)
    explicit_noise = torch.randn(
        noise_shape,
        generator=noise_generator,
        device=device,
        dtype=torch.float32,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        equivalence = verify_stage0_treatment_equivalence(
            action_head=action_head,
            plan_z0=stage0_output["plan_z0"],
            plan_z=stage0_output["plan_z"],
            traj_images=traj_images,
            initial_noise=explicit_noise,
            postprocess_config=TrajectoryPostprocessConfig(
                num_sample_trajs=int(action_head.config.num_sample_trajs),
                action_scale=float(cfg["data"]["trajectory"]["action_scale"]),
                trajectory_selection="mean",
                trajectory_x_sign=1.0,
                target_heading_deg=None,
            ),
            old_heatmap_control_enabled=False,
            pano_latent_adapter_enabled=False,
        )
    _restore_scheduler_training_state(
        action_head.noise_scheduler,
        scheduler_training_state,
    )
    torch.cuda.synchronize(device)
    stage0_seconds = time.perf_counter() - stage0_started

    model.train()
    _apply_bridge_only_train_mode(model, stage_cfg, logging.getLogger("ppa-smoke"))
    optimizer.zero_grad(set_to_none=True)
    backward_started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = _model_forward(model, batch, device)
        gt_trajectory, trajectory_valid, train_traj_images = (
            _prepare_trajectory_sequence_inputs(
                batch["trajectory"].to(device, non_blocking=True),
                batch["trajectory_valid"].to(device, non_blocking=True),
                batch["traj_images"].to(device, non_blocking=True),
                mode="first_only",
            )
        )
        action_plan = compute_shared_plan_action_losses(
            action_head=model.nextdit_action_head,
            plan_z0=output["plan_z0"],
            plan_z=output["plan_z"],
            gt_trajectory=gt_trajectory,
            trajectory_valid=trajectory_valid,
            traj_images=train_traj_images,
            preserve_weight=0.0,
            delta_weight=0.0,
        )

    history_loss_fn = build_heatmap_loss_fn(cfg, device)
    future_loss_fn = build_future_heatmap_loss_fn(cfg, device)
    with torch.autocast(device_type="cuda", enabled=False):
        history_loss = history_loss_fn(
            output["visibility"].float(),
            output["heatmaps"].float(),
            gt_vis=batch["gt_visibility"].to(device, dtype=torch.float32),
            gt_heatmaps=batch["heatmap"].to(device, dtype=torch.float32),
            history_mask=batch["history_mask"].to(device),
            pred_heatmap_logits=output["heatmap_logits"].float(),
        )
        future_loss = future_loss_fn(
            pred_visibility_logits=output["future_visibility_logits"].float(),
            pred_heatmaps=output["future_heatmaps"].float(),
            pred_heatmap_logits=output["future_heatmap_logits"].float(),
            gt_visibility=batch["future_trajectory_visibility"].to(
                device, dtype=torch.float32
            ),
            gt_heatmaps=batch["future_trajectory_heatmap"].to(
                device, dtype=torch.float32
            ),
            future_time_mask=batch["future_trajectory_time_mask"].to(
                device, dtype=torch.bool
            ),
        )
        weights = PastPlanActionLossWeights(
            action=float(cfg["loss"]["trajectory_weight"]),
            history=float(cfg["loss"]["history_weight"]),
            future=float(cfg["loss"]["future_weight"]),
            preserve=float(cfg["loss"]["preserve_weight"]),
            delta_z=float(cfg["loss"]["delta_z_weight"]),
        )
        losses = compose_past_plan_action_loss(
            stage="stage2_joint",
            history_loss=history_loss,
            future_loss=future_loss,
            action_plan_losses=action_plan,
            weights=weights,
        )
    losses["total"].backward()

    future_grad, future_probe = _gradient_report(
        model.past_plan_action.future_head.named_parameters(), family="Future Head"
    )
    bridge_grad, bridge_probe = _gradient_report(
        model.past_plan_action.bridge.named_parameters(), family="M->Z bridge"
    )
    shared_past_grad, shared_past_probe = _gradient_report(
        model.heatmap_vln.named_parameters(), family="shared Past Head"
    )

    frozen_with_grad = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    _require(not frozen_with_grad, f"frozen tensors received gradients: {frozen_with_grad[:8]}")
    assert_native_frozen_and_gradient_free(
        model.qwen2_5_vl,
        model.llm_projector,
        model.nextdit_action_head,
        model.nextdit_action_head.cond_projector,
    )
    _require(model.latent_queries.grad is None, "frozen latent_queries received a gradient")

    bridge_out = model.past_plan_action.bridge.cross_attention.out_proj
    _require(bridge_out.weight.grad is not None, "bridge out_proj.weight grad is None")
    _require(bool((bridge_out.weight.grad != 0).any()), "bridge out_proj.weight grad is zero")

    probes = {
        "future": (future_probe[0], future_probe[1], future_probe[1].detach().clone()),
        "bridge": (bridge_probe[0], bridge_probe[1], bridge_probe[1].detach().clone()),
        "shared_past": (
            shared_past_probe[0],
            shared_past_probe[1],
            shared_past_probe[1].detach().clone(),
        ),
    }
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    _require(bool(torch.isfinite(grad_norm)), "global gradient norm is non-finite")
    optimizer.step()
    changed = {
        family: not torch.equal(parameter.detach(), before)
        for family, (_name, parameter, before) in probes.items()
    }
    _require(all(changed.values()), f"optimizer did not update every family: {changed}")
    _require(
        not torch.equal(bridge_out.weight.detach(), torch.zeros_like(bridge_out.weight)),
        "bridge remained exact zero after its Stage2 update",
    )
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - backward_started

    scalar_losses = {
        key: _finite_scalar(value, name=f"loss.{key}")
        for key, value in losses.items()
    }
    _require(scalar_losses["total"] > 0.0, "total loss must be positive")
    _require(scalar_losses["history"] > 0.0, "History loss must be positive")
    _require(scalar_losses["future"] > 0.0, "Future loss must be positive")
    _require(scalar_losses["action"] > 0.0, "Action loss must be positive")

    return {
        "status": "passed",
        "contract": "real-single-gpu-stage0-treatment-plus-stage2-one-batch-v2",
        "repo_root": str(repo_root),
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_initializer_mode": "stage1_map_pretrain (complete Past only; fresh Future for smoke)",
        "sample_identity": EXACT_SAMPLE_ID,
        "current_t": EXACT_CURRENT_T,
        "pixel_goal": EXACT_PIXEL_GOAL,
        "goal_len": EXACT_GOAL_LEN,
        "history_pose_provider": "amb3r_vo_cache",
        "model_batch_has_gt_pose_or_depth": False,
        "system1_required_tensor_count": int(system1_tensor_count),
        "stage0": dataclasses.asdict(equivalence),
        "future_shapes": {
            "heatmaps": list(output["future_heatmaps"].shape),
            "visibility": list(output["future_visibility_logits"].shape),
        },
        "losses": scalar_losses,
        "gradient_families": {
            "future": future_grad,
            "bridge": bridge_grad,
            "shared_past": shared_past_grad,
        },
        "optimizer_family_changed": changed,
        "global_grad_norm_before_clip": float(grad_norm.detach().float().cpu()),
        "peak_cuda_memory_gib": float(torch.cuda.max_memory_allocated(device) / 2**30),
        "timing_seconds": {
            "dataset_and_collator": dataset_seconds,
            "model_load_and_init": model_seconds,
            "stage0_bitwise": stage0_seconds,
            "stage2_forward_backward_step": backward_seconds,
            "total": time.perf_counter() - started,
        },
        "formal_stage2_note": (
            "This smoke intentionally does not claim navigation quality. A formal Stage2 run "
            "must initialize Future from a trained Stage1 deployment checkpoint."
        ),
        "visualization": str(args.visualization.expanduser().resolve()),
        "visualization_semantics": (
            "F|R|B|L; fixed vmin=0/vmax=1; four 8-step horizons; "
            "prediction is fresh/untrained and is not a quality claim"
        ),
    }


def main() -> int:
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report_path = args.report.expanduser().resolve()
    try:
        report = _run(args)
    except Exception as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

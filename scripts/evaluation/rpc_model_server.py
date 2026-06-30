#!/usr/bin/env python3
"""HeatmapVLN model-side RPC server.

Run this in the model environment.  It intentionally does not import Habitat;
the Habitat process sends RGB observations through the vla_rpc bridge and
receives discrete Habitat actions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
from concurrent import futures
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import grpc
import numpy as np
import torch
from PIL import Image

from scripts.training.utils import _normalize_state_key, load_config
from src.models.heatmap.input_constructor import (
    construct_input,
    parse_structured_pano_output,
    structured_condition_text,
    vlm_output_requests_stop,
    vlm_output_requests_turn,
)
from src.models.runtime_compat import install_flash_attn_stub, install_numpy_legacy_aliases
from vla_rpc.core.image import decode_jpeg_to_rgb
from vla_rpc.proto import vla_pb2, vla_pb2_grpc

LOGGER = logging.getLogger("heatmapvln-rpc-server")

MAX_STEPS = 8
MAX_LOCAL_STEPS = 4
PROTO_VERSION = "heatmapvln-r2r-json-v1"
LOCAL_FJL_ROOT = Path(os.environ.get("HEATMAPVLN_FJL_ROOT", "/mnt/afs/lixiaoou/intern/fjl"))
LOCAL_INTERNNAV_MODEL_PATH = Path(
    os.environ.get("HEATMAPVLN_INTERNNAV_MODEL_PATH", str(LOCAL_FJL_ROOT / "InternNav-Model"))
)


def _default_internnav_model_path() -> str:
    for raw in (
        os.environ.get("INTERNNAV_MODEL_PATH"),
        os.environ.get("INTERNNAV_BACKBONE"),
        str(LOCAL_INTERNNAV_MODEL_PATH),
    ):
        if not raw:
            continue
        candidate = Path(os.path.expandvars(os.path.expanduser(str(raw))))
        if candidate.exists():
            return str(candidate.resolve())
    return os.environ.get("INTERNNAV_MODEL_PATH", "")


class ActionCode:
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


def _extract_checkpoint_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")
    for key in ("model_state_dict", "trainable_state_dict", "state_dict"):
        state_dict = ckpt.get(key)
        if isinstance(state_dict, dict):
            return state_dict
    if all(torch.is_tensor(value) for value in ckpt.values()):
        return ckpt
    raise KeyError(
        "Checkpoint does not contain model_state_dict/trainable_state_dict/state_dict: "
        f"{checkpoint_path}"
    )


def _extract_checkpoint_config(checkpoint_path: str | None) -> dict:
    if not checkpoint_path:
        return {}
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        return {}
    cfg = ckpt.get("config", {})
    return cfg if isinstance(cfg, dict) else {}


def _state_has_prefix(state_dict: dict[str, torch.Tensor] | None, prefix: str) -> bool:
    if not state_dict:
        return False
    return any(_normalize_state_key(key).startswith(prefix) for key in state_dict)


def _looks_action_only(state_dict: dict[str, torch.Tensor]) -> bool:
    if not state_dict:
        return False
    prefixes = {_normalize_state_key(key).split(".", 1)[0] for key in state_dict}
    return prefixes.issubset({"latent_queries", "nextdit_action_head"})


def _checkpoint_has_base_weights(state_dict: dict[str, torch.Tensor] | None) -> bool:
    return (
        _state_has_prefix(state_dict, "qwen2_5_vl.")
        or _state_has_prefix(state_dict, "qwen3_5.")
        or _state_has_prefix(state_dict, "qwen3_5_vl.")
        or _state_has_prefix(state_dict, "heatmap_vln.")
    )


def _requires_base_checkpoint(cfg: dict, checkpoint_cfg: dict | None = None) -> bool:
    for source in (checkpoint_cfg, cfg):
        if not isinstance(source, dict):
            continue
        stages = source.get("training", {}).get("stages", [])
        if not stages:
            continue
        stage_cfg = stages[0]
        if stage_cfg.get("requires_base_checkpoint") or stage_cfg.get("bridge_only"):
            return True
    return False


def _load_compatible_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    checkpoint_path: str,
    label: str,
) -> int:
    current_state = model.state_dict()
    normalized_to_actual = {_normalize_state_key(name): name for name in current_state}
    remapped: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    skipped_missing: list[str] = []
    for name, value in state_dict.items():
        normalized_name = _normalize_state_key(name)
        actual_name = normalized_to_actual.get(normalized_name)
        if actual_name is None:
            skipped_missing.append(name)
            continue
        if current_state[actual_name].shape != value.shape:
            skipped_shape.append(
                f"{actual_name}: ckpt {tuple(value.shape)} vs "
                f"model {tuple(current_state[actual_name].shape)}"
            )
            continue
        remapped[actual_name] = value
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    LOGGER.info(
        "%s loaded: %s (loaded=%d/%d, missing=%d, unexpected=%d)",
        label,
        checkpoint_path,
        len(remapped),
        len(state_dict),
        len(missing),
        len(unexpected),
    )
    if skipped_missing:
        LOGGER.info("Skipped unmatched keys: %d; examples: %s", len(skipped_missing), skipped_missing[:5])
    if skipped_shape:
        LOGGER.info("Skipped shape-mismatched keys: %d; examples: %s", len(skipped_shape), skipped_shape[:3])
    return len(remapped)


def _load_pano_latent_adapter(checkpoint_path: str, hidden_dim: int, device: torch.device):
    from scripts.evaluation.eval_pano_latent_adapter import _load_adapter_from_checkpoint

    fallback = argparse.Namespace(
        adapter_hidden_dim=2048,
        adapter_dropout=0.0,
        residual=False,
        pre_norm=False,
    )
    adapter, _saved_args = _load_adapter_from_checkpoint(
        Path(checkpoint_path).expanduser(),
        dim=hidden_dim,
        fallback_args=fallback,
        device=device,
    )
    return adapter


def _maybe_apply_pano_latent_adapter(
    traj_hs: torch.Tensor,
    adapter,
    *,
    view_id: str | None = None,
    pixel_goal: list[int] | None = None,
    image_size: tuple[int, int] | None = None,
    cond_projector: torch.nn.Module | None = None,
) -> torch.Tensor:
    if adapter is None:
        return traj_hs
    orig_dtype = traj_hs.dtype
    adapter_param = next(adapter.parameters(), None)
    adapter_dtype = adapter_param.dtype if adapter_param is not None else orig_dtype

    if hasattr(adapter, "geometry_token"):
        from src.models.adapters import view_ids_to_indices

        if pixel_goal is None:
            raise RuntimeError("Geometry-aware pano adapter requires pixel_goal")
        goal_view = (view_id or "front").lower()
        if goal_view not in {"front", "right", "back", "left"}:
            goal_view = "front"
        if image_size is None:
            image_size = (384, 384)
        width, height = int(image_size[0]), int(image_size[1])
        view_indices = view_ids_to_indices([goal_view], device=traj_hs.device)
        pixel_xy = torch.tensor(
            [[int(pixel_goal[0]), int(pixel_goal[1])]],
            device=traj_hs.device,
            dtype=adapter_dtype,
        )
        image_hw = torch.tensor([[height, width]], device=traj_hs.device, dtype=adapter_dtype)
        out = adapter(traj_hs.to(dtype=adapter_dtype), view_indices, pixel_xy, image_hw)
        return out.to(dtype=orig_dtype)

    if hasattr(adapter, "mlp"):
        adapted = adapter(traj_hs.to(dtype=adapter_dtype))
        if cond_projector is not None:
            proj_dtype = next(cond_projector.parameters()).dtype
            adapted = cond_projector(adapted.to(dtype=proj_dtype))
        return adapted.to(dtype=orig_dtype)

    out = adapter(traj_hs.to(dtype=adapter_dtype))
    return out.to(dtype=orig_dtype)


def _normalize_multimodal_inputs(inputs: dict[str, torch.Tensor]) -> None:
    if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
        vgt = inputs["video_grid_thw"]
        if vgt.shape[0] > 0 and vgt[:, 0].max() > 1:
            inputs["video_grid_thw"] = torch.repeat_interleave(vgt, vgt[:, 0], dim=0)
            inputs["video_grid_thw"][:, 0] = 1


def _parse_pano_view_id(llm_output: str) -> str | None:
    parsed = parse_structured_pano_output(llm_output, image_size=None)
    return parsed.view_id


def _parse_pixel_goal(
    llm_output: str,
    image_size: tuple[int, int],
    *,
    allow_legacy_coord: bool = True,
) -> list[int] | None:
    parsed = parse_structured_pano_output(
        llm_output,
        image_size=image_size,
        allow_legacy_coord=allow_legacy_coord,
    )
    if parsed.kind in {"pixel", "legacy_coord"} and parsed.pixel_goal is not None:
        return list(parsed.pixel_goal)
    if not re.search(r"\d", llm_output or ""):
        return None
    return None


def _condition_output_ids_for_pixel_goal(
    output_ids: torch.Tensor,
    prompt_len: int,
    tokenizer,
    pixel_goal: list[int],
    llm_output: str,
    coord_order: str = "generated",
    view_id: str | None = None,
    structured_output: bool = False,
) -> torch.Tensor:
    parsed = parse_structured_pano_output(llm_output, image_size=None)
    use_structured = structured_output or parsed.kind == "pixel"
    if use_structured:
        resolved_view = (view_id or parsed.view_id or "front").lower()
        desired_text = structured_condition_text(resolved_view, pixel_goal)
        generated_text = (llm_output or "").strip()
        if generated_text == desired_text:
            return output_ids
        replacement = tokenizer.encode(desired_text, add_special_tokens=False)
    else:
        coord = [int(c) for c in re.findall(r"\d+", llm_output or "")]
        if coord_order == "generated":
            desired = [int(pixel_goal[0]), int(pixel_goal[1])]
        elif coord_order == "internnav_yx":
            desired = [int(pixel_goal[1]), int(pixel_goal[0])]
        else:
            raise ValueError(f"Unsupported coord_order: {coord_order}")
        if len(coord) >= 2 and [coord[0], coord[1]] == desired:
            return output_ids
        replacement = tokenizer.encode(f"{desired[0]} {desired[1]}", add_special_tokens=False)

    if not replacement:
        return output_ids
    replacement_ids = torch.tensor([replacement], device=output_ids.device, dtype=output_ids.dtype)
    generated_suffix = output_ids[:, prompt_len:]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if (
        eos_token_id is not None
        and generated_suffix.numel() > 0
        and int(generated_suffix[0, -1].item()) == int(eos_token_id)
    ):
        eos = torch.tensor([[eos_token_id]], device=output_ids.device, dtype=output_ids.dtype)
        replacement_ids = torch.cat([replacement_ids, eos], dim=1)
    return torch.cat([output_ids[:, :prompt_len], replacement_ids], dim=1)


def _trajectory_from_condition(action_head, traj_condition: torch.Tensor, *, traj_images: torch.Tensor | None):
    if traj_condition.shape[-1] == int(action_head.config.latent_emb_size):
        return action_head.get_trajectory_from_projected(traj_condition, traj_images=traj_images)
    return action_head.get_trajectory(traj_condition, traj_images=traj_images)


def _lookdown_to_traj_tensor(lookdown_img: Image.Image, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.array(lookdown_img)).to(device=device, dtype=torch.bfloat16) / 255.0


def _finalize_local_actions(action_list: list[int]) -> list[int]:
    if len(action_list) < MAX_STEPS:
        action_list = list(action_list) + [ActionCode.STOP] * (MAX_STEPS - len(action_list))
    if len(action_list) >= MAX_LOCAL_STEPS:
        action_list = action_list[:MAX_LOCAL_STEPS]
    return [int(action) for action in action_list]


def reconstruct_xy_from_delta(delta_xyt: np.ndarray) -> np.ndarray:
    start_xy = np.zeros((len(delta_xyt), 2))
    delta_xy = delta_xyt[:, :, :2]
    cumsum_xy = np.cumsum(delta_xy, axis=1)
    batch_size = delta_xyt.shape[0]
    steps = delta_xyt.shape[1]
    xy = np.zeros((batch_size, steps + 1, 2))
    xy[:, 0] = start_xy
    xy[:, 1:] = start_xy[:, None, :] + cumsum_xy
    return xy


def trajectory_xy_path_len(trajectory: np.ndarray) -> float:
    if trajectory.ndim != 2 or trajectory.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1).sum())


def _trajectory_to_discrete_actions_close_to_goal(
    trajectory: np.ndarray,
    step_size: float = 0.25,
    turn_angle_deg: float = 15,
    lookahead: int = 4,
) -> list[int]:
    actions: list[int] = []
    yaw = 0.0
    pos = trajectory[0]
    turn_angle_rad = np.deg2rad(turn_angle_deg)
    goal = trajectory[-1]

    def normalize_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    while np.linalg.norm(pos - goal) > 0.2:
        dists = np.linalg.norm(trajectory - pos, axis=1)
        nearest_idx = np.argmin(dists)
        target_idx = min(nearest_idx + lookahead, len(trajectory) - 1)
        target = trajectory[target_idx]
        target_dir = target - pos
        if np.linalg.norm(target_dir) < 1e-6:
            break
        target_yaw = np.arctan2(target_dir[1], target_dir[0])
        delta_yaw = normalize_angle(target_yaw - yaw)
        n_turns = round(delta_yaw / turn_angle_rad)
        if n_turns > 0:
            actions += [ActionCode.LEFT] * n_turns
        elif n_turns < 0:
            actions += [ActionCode.RIGHT] * (-n_turns)
        yaw = normalize_angle(yaw + n_turns * turn_angle_rad)
        next_pos = pos + step_size * np.array([np.cos(yaw), np.sin(yaw)])
        if np.linalg.norm(next_pos - goal) > np.linalg.norm(pos - goal):
            break
        actions.append(ActionCode.FORWARD)
        pos = next_pos
    return actions


def _endpoint_medoid_index(all_trajectory: np.ndarray) -> int:
    endpoints = all_trajectory[:, -1, :2]
    dists = np.linalg.norm(endpoints[:, None, :] - endpoints[None, :, :], axis=-1)
    return int(np.argmin(dists.sum(axis=1)))


def _path_medoid_index(all_trajectory: np.ndarray) -> int:
    flat = all_trajectory[:, :, :2].reshape(all_trajectory.shape[0], -1)
    dists = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    return int(np.argmin(dists.sum(axis=1)))


def _median_endpoint_nearest_index(all_trajectory: np.ndarray) -> int:
    endpoints = all_trajectory[:, -1, :2]
    median_endpoint = np.median(endpoints, axis=0)
    return int(np.argmin(np.linalg.norm(endpoints - median_endpoint[None, :], axis=-1)))


def _forward_candidate_stats(all_trajectory: np.ndarray) -> list[tuple[int, int, float, list[int]]]:
    candidates: list[tuple[int, int, float, list[int]]] = []
    for idx, trajectory in enumerate(all_trajectory):
        actions = _trajectory_to_discrete_actions_close_to_goal(trajectory)
        forward_count = sum(1 for action in actions if action == ActionCode.FORWARD)
        if forward_count <= 0:
            continue
        candidates.append((idx, forward_count, trajectory_xy_path_len(trajectory), actions))
    return candidates


def select_trajectory_xy(all_trajectory: np.ndarray, selection: str = "mean") -> tuple[np.ndarray, int | None]:
    if all_trajectory.ndim != 3 or all_trajectory.shape[0] == 0:
        raise ValueError(f"Expected all_trajectory shape (B,T,2), got {all_trajectory.shape}")
    if selection == "mean":
        return np.mean(all_trajectory, axis=0), None
    if selection == "endpoint_medoid":
        idx = _endpoint_medoid_index(all_trajectory)
        return all_trajectory[idx], idx
    if selection == "path_medoid":
        idx = _path_medoid_index(all_trajectory)
        return all_trajectory[idx], idx
    if selection == "median_endpoint_nearest":
        idx = _median_endpoint_nearest_index(all_trajectory)
        return all_trajectory[idx], idx

    forward_candidates = _forward_candidate_stats(all_trajectory)
    if selection == "forward_or_medoid":
        if forward_candidates:
            medoid_idx = _endpoint_medoid_index(all_trajectory)
            medoid_endpoint = all_trajectory[medoid_idx, -1, :2]
            median_path_len = float(np.median([trajectory_xy_path_len(traj) for traj in all_trajectory]))

            def score(item: tuple[int, int, float, list[int]]) -> tuple[float, int, float]:
                idx, forward_count, path_len, _actions = item
                endpoint_dist = float(np.linalg.norm(all_trajectory[idx, -1, :2] - medoid_endpoint))
                return (endpoint_dist, -forward_count, abs(path_len - median_path_len))

            idx = min(forward_candidates, key=score)[0]
            return all_trajectory[idx], idx
        idx = _endpoint_medoid_index(all_trajectory)
        return all_trajectory[idx], idx
    if selection == "longest_forward":
        if forward_candidates:
            idx = max(forward_candidates, key=lambda item: (item[2], item[1]))[0]
            return all_trajectory[idx], idx
        idx = _endpoint_medoid_index(all_trajectory)
        return all_trajectory[idx], idx
    raise ValueError(f"Unsupported trajectory selection: {selection}")


def traj_to_actions(
    dp_actions: torch.Tensor,
    num_sample_trajs: int = 32,
    action_scale: float = 4.0,
    trajectory_selection: str = "mean",
) -> list[int]:
    trajs = dp_actions[:num_sample_trajs].float().detach().cpu().numpy()
    trajs[:, :, :2] /= action_scale
    all_trajectory = reconstruct_xy_from_delta(trajs)
    trajectory, _selected_idx = select_trajectory_xy(all_trajectory, trajectory_selection)
    actions = _trajectory_to_discrete_actions_close_to_goal(trajectory)
    return actions if actions else [ActionCode.STOP]


def _trajectory_debug_summary(trajectory: torch.Tensor, num_sample_trajs: int, action_scale: float) -> str:
    if trajectory is None or trajectory.numel() == 0:
        return "trajectory=empty"
    trajs = trajectory[:num_sample_trajs].float().detach().cpu().numpy().copy()
    if trajs.ndim != 3 or trajs.shape[-1] < 2:
        return f"trajectory_shape={tuple(trajectory.shape)}"
    trajs[:, :, :2] /= float(action_scale)
    cumsum_xy = np.cumsum(trajs[:, :, :2], axis=1)
    xy = np.concatenate([np.zeros((trajs.shape[0], 1, 2), dtype=cumsum_xy.dtype), cumsum_xy], axis=1)
    mean_xy = xy.mean(axis=0)
    goal_xy = mean_xy[-1]
    direct = float(np.linalg.norm(goal_xy))
    path_len = float(np.linalg.norm(np.diff(mean_xy, axis=0), axis=1).sum())
    return f"traj_goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f}), direct={direct:.2f}, path_len={path_len:.2f}"


def _pil_from_blob(blob: vla_pb2.BinaryBlob, image_size: tuple[int, int] | None = None) -> Image.Image:
    arr = decode_jpeg_to_rgb(blob.data)
    image = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    if image_size is not None and image.size != image_size:
        image = image.resize(image_size)
    return image


def _blobs_by_name(blobs) -> dict[str, vla_pb2.BinaryBlob]:
    return {blob.name: blob for blob in blobs}


class HeatmapVLNRuntime:
    def __init__(self, args: argparse.Namespace):
        install_numpy_legacy_aliases()
        if os.environ.get("HEATMAPVLN_FORCE_FLASH_ATTN_STUB", "0") == "1":
            install_flash_attn_stub(LOGGER)
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.cfg = self._load_runtime_config(args)
        self.model, self.train_cfg = self._load_model(args, self.device)
        self.processor = self.model.qwen2_5_vl.processor
        self.processor.tokenizer.padding_side = "left"
        self.action_scale = self.train_cfg.get("data", {}).get("trajectory", {}).get("action_scale", 4.0)
        self.num_sample_trajs = (
            self.train_cfg.get("model", {})
            .get("action_head", {})
            .get("nextdit", {})
            .get("num_sample_trajs", 32)
        )
        self.has_nextdit = self.model.nextdit_action_head is not None and self.model.latent_queries is not None
        self.pano_latent_adapter = self._load_adapter(args)
        if self.pano_latent_adapter is None and getattr(self.model, "pano_latent_adapter", None) is not None:
            self.pano_latent_adapter = self.model.pano_latent_adapter
            self.pano_latent_adapter.eval()
            LOGGER.info("Using model-attached pano latent adapter")
        LOGGER.info("NextDiT action head available: %s", self.has_nextdit)
        LOGGER.info("action_scale=%s num_sample_trajs=%s", self.action_scale, self.num_sample_trajs)

    def _load_runtime_config(self, args: argparse.Namespace) -> dict:
        cfg = load_config(args.config)
        internnav_path = (
            args.internnav_model_path
            or os.environ.get("INTERNNAV_MODEL_PATH")
            or cfg.get("paths", {}).get("internnav_model_path", "")
        )
        if internnav_path:
            internnav_path = os.path.expandvars(os.path.expanduser(str(internnav_path)))
            cfg.setdefault("paths", {})["internnav_model_path"] = internnav_path
            cfg.setdefault("model", {}).setdefault("llm", {})["model_path"] = internnav_path
            nextdit = cfg["model"].setdefault("action_head", {}).setdefault("nextdit", {})
            nextdit["internnav_model_path"] = internnav_path
            nextdit["internnav_system1_path"] = ""
            LOGGER.info("InternNav model path: %s", internnav_path)
        nextdit = cfg.get("model", {}).get("action_head", {}).get("nextdit", {})
        adapter_cfg = nextdit.get("pano_latent_adapter", {})
        if args.pano_latent_adapter_checkpoint and isinstance(adapter_cfg, dict):
            adapter_cfg["pretrained_path"] = ""
        return cfg

    def _load_model(self, args: argparse.Namespace, device: torch.device):
        from scripts.training.model_builder import build_model

        model = build_model(self.cfg, device=str(device), verbose=True)
        model = model.to(device)
        checkpoint_cfg = _extract_checkpoint_config(args.checkpoint)
        if not args.base_checkpoint and checkpoint_cfg:
            recorded_base = checkpoint_cfg.get("runtime", {}).get("base_checkpoint")
            if recorded_base and Path(recorded_base).exists():
                args.base_checkpoint = str(Path(recorded_base).resolve())
                LOGGER.info("Auto-loading base checkpoint from Stage 2 metadata: %s", args.base_checkpoint)

        base_state_dict = _extract_checkpoint_state_dict(args.base_checkpoint) if args.base_checkpoint else None
        checkpoint_state_dict = _extract_checkpoint_state_dict(args.checkpoint) if args.checkpoint else None
        if (
            _requires_base_checkpoint(self.cfg, checkpoint_cfg)
            and not args.base_checkpoint
            and not _checkpoint_has_base_weights(checkpoint_state_dict)
        ):
            raise ValueError("This config/checkpoint requires --base_checkpoint")
        if checkpoint_state_dict and _looks_action_only(checkpoint_state_dict) and not args.base_checkpoint:
            LOGGER.warning("Main checkpoint contains only action-head weights and no base checkpoint was loaded")

        model.qwen2_5_vl._load_model()
        if (
            _state_has_prefix(base_state_dict, "heatmap_vln.")
            or _state_has_prefix(checkpoint_state_dict, "heatmap_vln.")
        ):
            model._ensure_heatmap_vln()

        if base_state_dict:
            _load_compatible_state_dict(model, base_state_dict, args.base_checkpoint, label="Base checkpoint")
        if checkpoint_state_dict:
            _load_compatible_state_dict(model, checkpoint_state_dict, args.checkpoint, label="Main checkpoint")
        del checkpoint_state_dict
        del base_state_dict
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.eval()
        return model, self.cfg

    def _load_adapter(self, args: argparse.Namespace):
        if not args.pano_latent_adapter_checkpoint:
            return None
        hidden_dim = int(self.train_cfg.get("model", {}).get("llm", {}).get("hidden_dim", 3584))
        LOGGER.info("Loading pano latent adapter from %s", args.pano_latent_adapter_checkpoint)
        return _load_pano_latent_adapter(args.pano_latent_adapter_checkpoint, hidden_dim, self.device)

    def plan_panoramic(self, payload: dict[str, Any], blobs) -> dict[str, Any]:
        blob_map = _blobs_by_name(blobs)
        vlm_image_size = tuple(payload.get("vlm_image_size") or self.train_cfg["data"]["image_size"])
        traj_image_size = tuple(
            payload.get("traj_image_size")
            or self.train_cfg.get("data", {}).get("trajectory", {}).get("traj_image_size", [224, 224])
        )
        current_views = {
            view: _pil_from_blob(blob_map[f"current/{view}"], vlm_image_size)
            for view in ("front", "right", "back", "left")
        }
        history_panoramas: list[dict[str, Image.Image]] = []
        for hist_idx in range(int(payload.get("num_history", 0))):
            history_panoramas.append({
                view: _pil_from_blob(blob_map[f"history/{hist_idx}/{view}"], vlm_image_size)
                for view in ("front", "right", "back", "left")
            })
        lookdown_img = _pil_from_blob(blob_map["lookdown"], traj_image_size)
        instruction = str(payload.get("instruction", ""))
        trajectory_cfg = self.train_cfg.get("data", {}).get("trajectory", {})
        internnav_protocol = trajectory_cfg.get("system2_sft_protocol", "direct").lower() == "internnav"
        structured_pano_output = bool(trajectory_cfg.get("structured_pano_output", True))
        system1_coord_order = str(payload.get("system1_coord_order", "generated"))
        if system1_coord_order == "auto":
            system1_coord_order = "generated"
        trajectory_selection = str(payload.get("trajectory_selection", "mean"))
        oracle_system2 = payload.get("oracle_system2")
        if not isinstance(oracle_system2, dict):
            oracle_system2 = None
        oracle_system2_text = ""
        if oracle_system2 is not None:
            oracle_system2_text = str(oracle_system2.get("text") or "").strip()

        messages = construct_input(
            current_views=current_views,
            history_panoramas=history_panoramas,
            instruction=instruction,
            pixel_goal=[0, 0],
            internnav_protocol=internnav_protocol,
            structured_pano_output=structured_pano_output,
        )
        messages = [m for m in messages if m["role"] != "assistant"]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        _normalize_multimodal_inputs(inputs)

        prompt_len = inputs["input_ids"].shape[1]
        if oracle_system2_text:
            oracle_ids = self.processor.tokenizer.encode(
                oracle_system2_text,
                add_special_tokens=False,
            )
            if not oracle_ids:
                raise ValueError("oracle_system2.text produced no tokens")
            oracle_suffix = torch.tensor(
                [oracle_ids],
                device=inputs["input_ids"].device,
                dtype=inputs["input_ids"].dtype,
            )
            output_ids = torch.cat([inputs["input_ids"], oracle_suffix], dim=1)
            llm_output = oracle_system2_text
        else:
            with torch.no_grad():
                output_ids = self.model.qwen2_5_vl.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                ).sequences
            llm_output = self.processor.tokenizer.decode(
                output_ids[0][prompt_len:],
                skip_special_tokens=True,
            )
        response: dict[str, Any] = {
            "ok": True,
            "proto_v": PROTO_VERSION,
            "llm_output": llm_output,
            "system2_source": "oracle" if oracle_system2_text else "model",
            "oracle_system2": oracle_system2,
            "actions": [],
            "terminal": False,
            "kind": "unknown",
        }
        if vlm_output_requests_stop(llm_output):
            response.update({"kind": "stop", "terminal": True, "actions": [ActionCode.STOP]})
            return response

        turn_dir = vlm_output_requests_turn(llm_output)
        if turn_dir is not None:
            action = ActionCode.LEFT if turn_dir == "left" else ActionCode.RIGHT
            response.update({"kind": "turn", "actions": [int(action)], "turn_direction": turn_dir})
            return response

        pixel_goal = _parse_pixel_goal(
            llm_output,
            vlm_image_size,
            allow_legacy_coord=not structured_pano_output,
        )
        pano_goal_view = _parse_pano_view_id(llm_output) or "front"
        response["pixel_goal"] = pixel_goal
        response["pano_goal_view"] = pano_goal_view

        if self.has_nextdit and pixel_goal is not None:
            lookdown_t = _lookdown_to_traj_tensor(lookdown_img, self.device)
            pix_goal_image = lookdown_t.clone()
            traj_images = torch.stack([pix_goal_image, lookdown_t]).unsqueeze(0).to(self.device)
            lq = self.model.latent_queries.expand(1, -1, -1).to(
                device=self.device,
                dtype=self.model.config.dtype,
            )
            condition_output_ids = _condition_output_ids_for_pixel_goal(
                output_ids=output_ids,
                prompt_len=prompt_len,
                tokenizer=self.processor.tokenizer,
                pixel_goal=pixel_goal,
                llm_output=llm_output,
                coord_order=system1_coord_order,
                view_id=pano_goal_view,
                structured_output=structured_pano_output,
            )
            with torch.no_grad():
                traj_hs = self.model.qwen2_5_vl.generate_latents(
                    output_ids=condition_output_ids,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    latent_queries=lq,
                    attention_mask=inputs.get("attention_mask"),
                    mm_token_type_ids=inputs.get("mm_token_type_ids"),
                )
                if self.pano_latent_adapter is not None:
                    traj_hs = _maybe_apply_pano_latent_adapter(
                        traj_hs,
                        self.pano_latent_adapter,
                        view_id=pano_goal_view,
                        pixel_goal=pixel_goal,
                        image_size=vlm_image_size,
                        cond_projector=self.model.nextdit_action_head.cond_projector
                        if self.model.nextdit_action_head is not None
                        else None,
                    )
                trajectory = _trajectory_from_condition(
                    self.model.nextdit_action_head,
                    traj_hs,
                    traj_images=traj_images,
                )
            local_actions = _finalize_local_actions(
                traj_to_actions(
                    trajectory,
                    num_sample_trajs=self.num_sample_trajs,
                    action_scale=self.action_scale,
                    trajectory_selection=trajectory_selection,
                )
            )
            if local_actions and local_actions[0] == ActionCode.STOP:
                local_actions = [ActionCode.LEFT]
                response["anti_deadlock"] = True
            response.update({
                "kind": "trajectory",
                "actions": [int(action) for action in local_actions],
                "trajectory_summary": _trajectory_debug_summary(
                    trajectory,
                    self.num_sample_trajs,
                    self.action_scale,
                ),
            })
            return response

        response.update({"kind": "fallback_stop", "terminal": True, "actions": [ActionCode.STOP]})
        return response


class HeatmapVLNRPCServicer(vla_pb2_grpc.VLAServicer):
    def __init__(self, runtime: HeatmapVLNRuntime):
        self.runtime = runtime
        self.started = int(torch.cuda.Event(enable_timing=False) is not None)
        self.requests_processed = 0
        self.model_version = "heatmapvln-r2r"

    def InferJSON(self, request: vla_pb2.JSONRequest, context) -> vla_pb2.JSONResponse:
        try:
            payload = json.loads(request.json_payload) if request.json_payload else {}
            if request.method != "plan_panoramic":
                raise ValueError(f"Unsupported method: {request.method}")
            output = self.runtime.plan_panoramic(payload, request.blobs)
            self.requests_processed += 1
            return vla_pb2.JSONResponse(
                ts=request.ts,
                json_payload=json.dumps(output, ensure_ascii=False),
                model_v=self.model_version,
            )
        except Exception as exc:
            LOGGER.exception("InferJSON failed")
            context.set_details(str(exc))
            context.set_code(grpc.StatusCode.INTERNAL)
            return vla_pb2.JSONResponse(ts=request.ts, json_payload=json.dumps({"ok": False, "error": str(exc)}))

    def HealthCheck(self, request: vla_pb2.HealthCheckRequest, context) -> vla_pb2.HealthCheckResponse:
        return vla_pb2.HealthCheckResponse(
            status=vla_pb2.HealthCheckResponse.SERVING,
            message="HeatmapVLN model server is running",
            version=PROTO_VERSION,
            requests_processed=self.requests_processed,
        )

    def GetServerInfo(self, request: vla_pb2.Empty, context) -> vla_pb2.ServerInfo:
        return vla_pb2.ServerInfo(
            version=PROTO_VERSION,
            model_version=self.model_version,
            max_batch_size=1,
            supported_formats=["json+jpeg"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HeatmapVLN model-side RPC server")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base_checkpoint", default=None)
    parser.add_argument("--pano_latent_adapter_checkpoint", default=None)
    parser.add_argument("--internnav_model_path", default=_default_internnav_model_path())
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    runtime = HeatmapVLNRuntime(args)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers),
        options=[
            ("grpc.max_send_message_length", 128 * 1024 * 1024),
            ("grpc.max_receive_message_length", 128 * 1024 * 1024),
        ],
    )
    vla_pb2_grpc.add_VLAServicer_to_server(HeatmapVLNRPCServicer(runtime), server)
    address = f"{args.host}:{args.port}"
    server.add_insecure_port(address)
    server.start()
    LOGGER.info("HeatmapVLN RPC server listening on %s", address)

    def _shutdown(_signum, _frame):
        LOGGER.info("Stopping HeatmapVLN RPC server")
        server.stop(grace=5)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    server.wait_for_termination()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Collect compact InternNav teacher labels on an existing HeatmapVLN dataset.

This script does not run Habitat and does not save images.  It reads the
already-collected trajectory dataset, feeds front-view history/current frames
plus the current front_down frame to InternNav, and writes JSONL sidecar labels
that can be used for distillation or offline audits.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import itertools
import json
import os
import random
import re
import sys
import types
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image

from scripts.training.utils import load_config
from src.data.factory import build_trajectory_dataset

DEFAULT_IMAGE_TOKEN = "<image>"
MAX_STEPS = 8
MAX_LOCAL_STEPS = 4
PROMPT_TEMPLATE = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. "
    "Where should you go next to stay on track? Please output the next waypoint's "
    "coordinates in the image. Please output STOP when you have successfully completed the task."
)
INTERNNAV_CONJUNCTIONS = [
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
]


def _install_flash_attn_stub() -> None:
    """Match the standalone InternNav eval script when using SDPA attention."""

    def _noop(*_args, **_kwargs):
        raise RuntimeError("flash_attn stub called; use --no-flash-attn-stub with flash_attention_2")

    def _make_stub(name: str, attrs: dict[str, Any] | None = None):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        module.__version__ = "2.8.3"
        for key, value in (attrs or {}).items():
            setattr(module, key, value)
        sys.modules[name] = module
        return module

    if "flash_attn" in sys.modules:
        return

    fa = _make_stub("flash_attn", {
        "flash_attn_func": _noop,
        "flash_attn_varlen_func": _noop,
    })
    _make_stub("flash_attn_2_cuda")
    fa_iface = _make_stub("flash_attn.flash_attn_interface", {
        "flash_attn_func": _noop,
        "flash_attn_varlen_func": _noop,
    })
    fa_bert = _make_stub("flash_attn.bert_padding", {
        "index_first_axis": _noop,
        "pad_input": _noop,
        "unpad_input": _noop,
    })
    fa_layers = _make_stub("flash_attn.layers", {})
    fa_rotary = _make_stub("flash_attn.layers.rotary", {
        "apply_rotary_emb": _noop,
    })
    fa.flash_attn_interface = fa_iface
    fa.bert_padding = fa_bert
    fa.layers = fa_layers
    fa_layers.rotary = fa_rotary


def _patch_numpy_aliases() -> None:
    if not hasattr(np, "float"):
        np.float = np.float64  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = np.int64  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = np.bool_  # type: ignore[attr-defined]


def _patch_internnav_depthanything() -> None:
    """Avoid loading a separate DepthAnything checkpoint during from_pretrained."""
    import internnav.model.basemodel.internvla_n1.internvla_n1_arch as arch_mod

    def _patched_build_dav2(_config):
        from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {"vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}}
        dav2_model = DepthAnythingV2(**model_configs["vits"])
        return dav2_model.pretrained

    arch_mod.build_depthanythingv2 = _patched_build_dav2


def _torch_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def _cast_tensor_for_save(tensor: torch.Tensor, dtype_name: str) -> torch.Tensor:
    dtype_name = dtype_name.lower()
    out = tensor.detach().cpu()
    if dtype_name == "source":
        return out
    return out.to(_torch_dtype(dtype_name))


def _set_sample_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_and_clean(text: str) -> list[str]:
    parts = re.split(r"(<image>)", text)
    out: list[str] = []
    for part in parts:
        if part == DEFAULT_IMAGE_TOKEN:
            out.append(part)
        else:
            clean = part.replace("\n", "").strip()
            if clean:
                out.append(clean)
    return out


def _tensor_chw_to_pil(tensor: torch.Tensor, resize: tuple[int, int] | None = None) -> Image.Image:
    arr = (
        tensor.detach()
        .cpu()
        .float()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )
    arr = np.rint(arr * 255.0).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    if resize is not None and img.size != resize:
        img = img.resize(resize, Image.BILINEAR)
    return img


def _traj_image_to_tensor_hwc(sample: dict[str, Any], resize: tuple[int, int]) -> torch.Tensor:
    ti = sample.get("traj_images")
    # Generic all/stop_turn mode can include no-pixel-goal states.  The dataset
    # pads those states with zero traj_images, so prefer the real current
    # lookdown frame when there is no pixel-goal label.
    if ti is not None and (sample.get("pixel_goal") is not None or sample.get("lookdown_frame") is None):
        if not torch.is_tensor(ti):
            ti = torch.as_tensor(ti)
        img = ti[0].detach().cpu().float().clamp(0.0, 1.0)
        if img.ndim != 3 or img.shape[-1] != 3:
            raise RuntimeError(f"Expected traj_images[0] as HWC, got {tuple(img.shape)}")
        if tuple(img.shape[:2][::-1]) != resize:
            arr = np.rint(img.numpy() * 255.0).astype(np.uint8)
            pil = Image.fromarray(arr, mode="RGB").resize(resize, Image.BILINEAR)
            img = torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0)
        return img

    lookdown = sample.get("lookdown_frame")
    if lookdown is None:
        raise RuntimeError("Sample has neither traj_images nor lookdown_frame")
    pil = _tensor_chw_to_pil(lookdown, resize=resize)
    return torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0)


def _build_images_dp(
    sample: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    image_size: tuple[int, int],
) -> torch.Tensor:
    img = _traj_image_to_tensor_hwc(sample, resize=image_size)
    pair = torch.stack([img, img], dim=0).unsqueeze(0)
    return pair.to(device=device, dtype=dtype)


def _build_front_images(
    sample: dict[str, Any],
    front_resize: tuple[int, int] | None,
) -> tuple[list[Image.Image], Image.Image]:
    history = sample.get("history_frames")
    if history is None:
        history_pil: list[Image.Image] = []
    else:
        if not torch.is_tensor(history):
            history = torch.as_tensor(history)
        history_pil = [_tensor_chw_to_pil(frame, resize=front_resize) for frame in history]

    current = sample["current_frame"]
    if not torch.is_tensor(current):
        current = torch.as_tensor(current)
    current_pil = _tensor_chw_to_pil(current, resize=front_resize)
    return history_pil, current_pil


def _build_lookdown_image(sample: dict[str, Any], front_resize: tuple[int, int] | None) -> Image.Image:
    lookdown = sample.get("lookdown_frame")
    if lookdown is not None:
        if not torch.is_tensor(lookdown):
            lookdown = torch.as_tensor(lookdown)
        return _tensor_chw_to_pil(lookdown, resize=front_resize)

    img_hwc = _traj_image_to_tensor_hwc(sample, resize=front_resize or (224, 224))
    arr = np.rint(img_hwc.numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _content_from_text_with_images(text: str, images: list[Image.Image], start_image_id: int = 0) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    image_id = start_image_id
    for part in _split_and_clean(text):
        if part == DEFAULT_IMAGE_TOKEN:
            content.append({"type": "image", "image": images[image_id]})
            image_id += 1
        else:
            content.append({"type": "text", "text": part})
    if image_id != len(images):
        raise RuntimeError(f"Prompt consumed {image_id} images but {len(images)} were provided")
    return content


def _strip_instruction_final_period(instruction: str) -> str:
    instruction = (instruction or "").strip()
    if instruction.endswith("."):
        return instruction[:-1]
    return instruction


def _choose_conjunction(args: argparse.Namespace, rng: random.Random) -> str:
    if args.conjunction_mode == "random":
        return rng.choice(INTERNNAV_CONJUNCTIONS)
    return args.fixed_conjunction


def _build_first_turn(
    sample: dict[str, Any],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    front_resize = None
    if args.front_width > 0 and args.front_height > 0:
        front_resize = (args.front_width, args.front_height)

    history_pil, current_pil = _build_front_images(sample, front_resize)
    input_images = history_pil + [current_pil]

    instruction = _strip_instruction_final_period(str(sample.get("text", "")))
    prompt_text = PROMPT_TEMPLATE.replace("<instruction>.", instruction)
    if history_pil:
        prompt_text += f" These are your historical observations: {(DEFAULT_IMAGE_TOKEN + chr(10)) * len(history_pil)}."
    prompt_text += f" {_choose_conjunction(args, rng)}{DEFAULT_IMAGE_TOKEN}."

    messages = [{"role": "user", "content": _content_from_text_with_images(prompt_text, input_images)}]
    return messages, input_images


def _build_second_turn(
    first_messages: list[dict[str, Any]],
    first_images: list[Image.Image],
    first_output: str,
    sample: dict[str, Any],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    front_resize = None
    if args.front_width > 0 and args.front_height > 0:
        front_resize = (args.front_width, args.front_height)

    lookdown = _build_lookdown_image(sample, front_resize)
    second_images = first_images + [lookdown]
    second_user_text = f"{_choose_conjunction(args, rng)}{DEFAULT_IMAGE_TOKEN}."
    messages = list(first_messages)
    messages.append({"role": "assistant", "content": [{"type": "text", "text": first_output}]})
    messages.append({
        "role": "user",
        "content": _content_from_text_with_images(second_user_text, [lookdown]),
    })
    return messages, second_images


def _parse_text_actions(output: str) -> list[int]:
    actions2idx = OrderedDict({
        "STOP": [0],
        "↑": [1],
        "←": [2],
        "→": [3],
        "↓": [5],
    })
    regex = re.compile("|".join(re.escape(action) for action in actions2idx))
    matches = regex.findall(output or "")
    return list(itertools.chain.from_iterable(actions2idx[m] for m in matches))


def _parse_coord(output: str) -> tuple[list[int] | None, list[int] | None]:
    nums = [int(x) for x in re.findall(r"\d+", output or "")]
    if len(nums) < 2:
        return None, None
    coord_uv = [nums[0], nums[1]]
    internnav_yx = [nums[1], nums[0]]
    return coord_uv, internnav_yx


def _pad_actions(actions: list[int], max_steps: int = MAX_STEPS) -> list[int]:
    out = [int(a) for a in actions[:max_steps]]
    if len(out) < max_steps:
        out += [0] * (max_steps - len(out))
    return out


def _action_summary(actions: list[int], max_steps: int = MAX_STEPS) -> dict[str, Any]:
    padded = _pad_actions(actions, max_steps=max_steps)
    return {
        "actions8": padded,
        "local4": padded[:MAX_LOCAL_STEPS],
        "forward_count8": int(sum(1 for a in padded if a == 1)),
        "first_action": int(padded[0]) if padded else 0,
    }


def _path_len_from_scaled_delta(delta: np.ndarray, action_scale: float) -> float:
    if delta.ndim != 2 or delta.shape[0] == 0:
        return 0.0
    delta_xy = delta[:, :2].astype(np.float64) / float(action_scale)
    xy = np.concatenate([np.zeros((1, 2), dtype=np.float64), np.cumsum(delta_xy, axis=0)], axis=0)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _trajectory_xy_from_scaled_delta(delta: np.ndarray, action_scale: float) -> np.ndarray:
    delta_xy = delta[:, :2].astype(np.float64) / float(action_scale)
    return np.concatenate([np.zeros((1, 2), dtype=np.float64), np.cumsum(delta_xy, axis=0)], axis=0)


def _summarize_sample_trajectories(
    dp_actions: torch.Tensor,
    traj_to_actions_fn: Callable[[torch.Tensor], list[int]],
    action_scale: float,
) -> dict[str, Any]:
    """Summarize all sampled trajectories without storing large tensors in JSON."""
    dp_cpu = dp_actions.float().detach().cpu()
    if dp_cpu.ndim != 3 or dp_cpu.shape[0] == 0:
        return {
            "path_len_m": [],
            "path_len_mean_m": 0.0,
            "path_len_median_m": 0.0,
            "path_len_std_m": 0.0,
            "endpoint_std_xy_m": 0.0,
            "forward_candidate_count": 0,
            "per_sample_actions4": [],
        }

    path_lens: list[float] = []
    endpoints: list[np.ndarray] = []
    actions4: list[list[int]] = []
    forward_counts: list[int] = []

    for i in range(int(dp_cpu.shape[0])):
        arr = dp_cpu[i].numpy()
        xy = _trajectory_xy_from_scaled_delta(arr, action_scale)
        path_lens.append(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()))
        endpoints.append(xy[-1])
        actions = _pad_actions(traj_to_actions_fn(dp_cpu[i:i + 1].clone()))
        actions4.append(actions[:MAX_LOCAL_STEPS])
        forward_counts.append(int(sum(1 for a in actions if a == 1)))

    path_arr = np.asarray(path_lens, dtype=np.float64)
    endpoints_arr = np.stack(endpoints, axis=0)
    forward_indices = [i for i, count in enumerate(forward_counts) if count > 0]
    longest_forward_idx = (
        max(forward_indices, key=lambda i: path_lens[i]) if forward_indices else None
    )
    median_path = float(np.median(path_arr))
    median_idx = int(np.argmin(np.abs(path_arr - median_path)))

    return {
        "path_len_m": [round(float(v), 5) for v in path_lens],
        "path_len_mean_m": round(float(path_arr.mean()), 5),
        "path_len_median_m": round(median_path, 5),
        "path_len_std_m": round(float(path_arr.std()), 5),
        "path_len_min_m": round(float(path_arr.min()), 5),
        "path_len_max_m": round(float(path_arr.max()), 5),
        "endpoint_std_xy_m": round(float(np.linalg.norm(endpoints_arr.std(axis=0))), 5),
        "forward_candidate_count": int(len(forward_indices)),
        "forward_candidate_pct": round(float(100.0 * len(forward_indices) / len(path_lens)), 3),
        "forward_counts8": forward_counts,
        "per_sample_actions4": actions4,
        "median_path_index": median_idx,
        "longest_forward_index": (
            int(longest_forward_idx) if longest_forward_idx is not None else None
        ),
    }


def _tensor_sidecar_path(args: argparse.Namespace, dataset_index: int) -> Path | None:
    if not args.tensor_output_dir:
        return None
    root = Path(args.tensor_output_dir).expanduser()
    shard = int(dataset_index) // int(args.tensor_shard_size)
    return root / f"shard_{shard:05d}" / f"{int(dataset_index):08d}.pt"


def _save_tensor_sidecar(
    path: Path,
    *,
    dataset_index: int,
    mode: str,
    traj_latents: torch.Tensor | None,
    dp_actions: torch.Tensor | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dataset_index": int(dataset_index),
        "mode": mode,
        "tensor_save_dtype": args.tensor_save_dtype,
    }
    saved: dict[str, Any] = {}
    if traj_latents is not None and args.save_traj_latents:
        payload["traj_latents"] = _cast_tensor_for_save(traj_latents, args.tensor_save_dtype)
        saved["traj_latents_shape"] = list(traj_latents.shape)
    if dp_actions is not None and args.save_dp_actions:
        payload["dp_actions"] = _cast_tensor_for_save(dp_actions, args.tensor_save_dtype)
        saved["dp_actions_shape"] = list(dp_actions.shape)
    if len(payload) > 3:
        torch.save(payload, path)
        saved["path"] = str(path)
    return saved


def _round_nested(value: Any, digits: int = 5) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_round_nested(v, digits=digits) for v in value]
    return value


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _normalize_image_grid_thw(inputs: Any) -> torch.Tensor:
    thw = inputs.image_grid_thw
    if torch.is_tensor(thw):
        return thw
    return torch.cat([item.unsqueeze(0) for item in thw], dim=0)


def _generate_text(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    images: list[Image.Image],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[str, torch.Tensor, Any, int]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            past_key_values=None,
            return_dict_in_generate=True,
        ).sequences
    prompt_len = int(inputs.input_ids.shape[1])
    output = processor.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    return output.strip(), output_ids, inputs, prompt_len


def _run_system1(
    model: Any,
    traj_to_actions_fn: Callable[[torch.Tensor], list[int]],
    output_ids: torch.Tensor,
    inputs: Any,
    sample: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    action_scale: float,
    args: argparse.Namespace,
    dataset_index: int,
    mode: str,
) -> dict[str, Any]:
    pixel_values = inputs.pixel_values
    image_grid_thw = _normalize_image_grid_thw(inputs)
    traj_images = _build_images_dp(sample, device=device, dtype=dtype, image_size=(args.traj_image_size, args.traj_image_size))

    with torch.inference_mode():
        traj_latents = model.generate_latents(output_ids, pixel_values, image_grid_thw)
        dp_actions = model.generate_traj(
            traj_latents,
            traj_images,
            None,
            predict_step_nums=args.predict_steps,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_sample_trajs=args.num_sample_trajs,
        )

    action_list = traj_to_actions_fn(dp_actions.clone())
    action_stats = _action_summary(action_list)
    mean_scaled = dp_actions.float().mean(dim=0).detach().cpu().numpy()
    result = {
        **action_stats,
        "mean_path_len_m": round(_path_len_from_scaled_delta(mean_scaled, action_scale), 5),
        "num_sample_trajs": int(dp_actions.shape[0]),
        "sample_traj_summary": _summarize_sample_trajectories(
            dp_actions, traj_to_actions_fn, action_scale,
        ),
    }
    tensor_path = _tensor_sidecar_path(args, dataset_index)
    if tensor_path is not None:
        saved = _save_tensor_sidecar(
            tensor_path,
            dataset_index=dataset_index,
            mode=mode,
            traj_latents=traj_latents,
            dp_actions=dp_actions,
            args=args,
        )
        if saved:
            result["tensor_sidecar"] = saved
    if args.save_mean_trajectory:
        result["mean_traj_scaled"] = _round_nested(mean_scaled.tolist(), digits=args.round_digits)
    return result


def _load_teacher(args: argparse.Namespace, device: torch.device):
    internnav_repo = Path(args.internnav_repo).expanduser().resolve()
    if not internnav_repo.exists():
        raise FileNotFoundError(f"InternNav repo not found: {internnav_repo}")
    if str(internnav_repo) not in sys.path:
        sys.path.insert(0, str(internnav_repo))

    if args.flash_attn_stub:
        _install_flash_attn_stub()
    _patch_numpy_aliases()
    _patch_internnav_depthanything()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    from internnav.model.basemodel.internvla_n1.internvla_n1 import (
        InternVLAN1ForCausalLM,
        InternVLAN1ModelConfig,
    )
    from internnav.model.utils.vln_utils import traj_to_actions

    try:
        AutoConfig.register("internvla_n1", InternVLAN1ModelConfig)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(InternVLAN1ModelConfig, InternVLAN1ForCausalLM)
    except ValueError:
        pass

    model_path = Path(args.model_path).expanduser().resolve()
    print(f"[teacher] loading processor/model from {model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(str(model_path))
    processor.tokenizer.padding_side = "left"
    model = InternVLAN1ForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=_torch_dtype(args.torch_dtype),
        attn_implementation=args.attn_implementation,
        device_map={"": device},
    )
    model.eval()

    system1_type = str(model.get_system1_type()) if hasattr(model, "get_system1_type") else ""
    print(f"[teacher] loaded system1_type={system1_type}", flush=True)
    if args.require_nextdit and "nextdit" not in system1_type:
        raise RuntimeError(
            f"Expected an InternNav NextDiT teacher, got system1_type={system1_type!r}. "
            "Pass --no-require-nextdit if this is intentional."
        )
    return model, processor, traj_to_actions


def _load_done_indices(output: Path) -> set[int]:
    done: set[int] = set()
    if not output.exists():
        return done
    with output.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = rec.get("dataset_index")
            if idx is not None:
                done.add(int(idx))
    return done


def _sample_metadata(dataset: Any, idx: int) -> dict[str, Any]:
    meta: dict[str, Any] = {"dataset_index": int(idx)}
    if hasattr(dataset, "sample_index"):
        clip_idx, current_t = dataset.sample_index[idx]
        meta.update({"clip_idx": int(clip_idx), "current_t": int(current_t)})
        if hasattr(dataset, "clips"):
            clip_dir = Path(dataset.clips[clip_idx])
            meta["clip_dir"] = str(clip_dir)
        try:
            clip_meta = dataset._load_meta(clip_idx)
            for key in ("scene_id", "episode_id", "trajectory_id", "num_frames"):
                if key in clip_meta:
                    meta[key] = _jsonable(clip_meta[key])
        except Exception:
            pass
    return meta


def _sample_kind(sample: dict[str, Any]) -> str:
    if float(sample.get("is_stop", 0.0)) > 0.5 or int(sample.get("discrete_action", 1)) == 0:
        return "stop"
    if sample.get("pixel_goal") is not None:
        return "pixel"
    if sample.get("turn_actions") or int(sample.get("discrete_action", 1)) in {2, 3, 4, 5}:
        return "turn"
    return "forward_or_unknown"


def _sample_passes_mode(sample: dict[str, Any], args: argparse.Namespace) -> bool:
    kind = _sample_kind(sample)
    if args.sample_mode == "pixel":
        if args.require_pixel_goal and sample.get("pixel_goal") is None:
            return False
        return args.include_stop or kind != "stop"
    if args.sample_mode == "stop_turn":
        return kind in {"stop", "turn"}
    if args.sample_mode == "all":
        if not args.include_stop and kind == "stop":
            return False
        if args.require_pixel_goal and sample.get("pixel_goal") is None:
            return False
        return True
    raise ValueError(f"Unknown sample mode: {args.sample_mode}")


def _collect_one(
    idx: int,
    sample: dict[str, Any],
    dataset: Any,
    model: Any,
    processor: Any,
    traj_to_actions_fn: Callable[[torch.Tensor], list[int]],
    device: torch.device,
    dtype: torch.dtype,
    action_scale: float,
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, Any]:
    _set_sample_seed(args.seed + int(idx) * 1009 + args.shard_index)
    first_messages, first_images = _build_first_turn(sample, args, rng)
    first_output, first_output_ids, first_inputs, first_prompt_len = _generate_text(
        model, processor, first_messages, first_images, device, args,
    )
    first_coord_uv, first_goal_yx = _parse_coord(first_output)
    first_actions = _parse_text_actions(first_output)

    final_output = first_output
    final_output_ids = first_output_ids
    final_inputs = first_inputs
    prompt_len = first_prompt_len
    second_output = None
    coord_uv = first_coord_uv
    goal_yx = first_goal_yx
    mode = "coord" if coord_uv is not None else "text_actions"

    if coord_uv is None and args.two_turn_lookdown and 5 in first_actions:
        second_messages, second_images = _build_second_turn(
            first_messages, first_images, first_output, sample, args, rng,
        )
        second_output, second_output_ids, second_inputs, second_prompt_len = _generate_text(
            model, processor, second_messages, second_images, device, args,
        )
        second_coord_uv, second_goal_yx = _parse_coord(second_output)
        if second_coord_uv is not None:
            final_output = second_output
            final_output_ids = second_output_ids
            final_inputs = second_inputs
            prompt_len = second_prompt_len
            coord_uv = second_coord_uv
            goal_yx = second_goal_yx
            mode = "coord_after_lookdown"

    teacher: dict[str, Any] = {
        "mode": mode,
        "first_output": first_output,
        "first_actions": _pad_actions(first_actions),
        "prompt_len": prompt_len,
    }
    if second_output is not None:
        teacher["second_output"] = second_output
    if coord_uv is not None:
        teacher["coord_uv"] = coord_uv
        teacher["internnav_pixel_goal_yx"] = goal_yx
        try:
            teacher["system1"] = _run_system1(
                model,
                traj_to_actions_fn,
                final_output_ids,
                final_inputs,
                sample,
                device,
                dtype,
                action_scale,
                args,
                dataset_index=idx,
                mode=mode,
            )
            for key in ("actions8", "local4", "forward_count8", "first_action"):
                teacher[key] = teacher["system1"][key]
        except Exception as exc:
            teacher["system1_error"] = repr(exc)
            if not args.skip_system1_errors:
                raise
    else:
        teacher.update(_action_summary(first_actions))

    pixel_goal = sample.get("pixel_goal")
    trajectory_valid = sample.get("trajectory_valid")
    rec = {
        "status": "ok",
        **_sample_metadata(dataset, idx),
        "teacher": teacher,
        "dataset_label": {
            "pixel_goal_uv": _jsonable(pixel_goal) if pixel_goal is not None else None,
            "sample_kind": _sample_kind(sample),
            "pixel_goal_relative_len": _jsonable(sample.get("pixel_goal_relative_len")),
            "discrete_action": int(sample.get("discrete_action", -1)),
            "is_stop": float(sample.get("is_stop", 0.0)),
            "turn_actions": _jsonable(sample.get("turn_actions", [])),
            "trajectory_valid": _jsonable(trajectory_valid),
        },
    }
    if args.include_text:
        rec["instruction"] = str(sample.get("text", ""))
    return rec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect InternNav teacher sidecar labels from an existing dataset")
    p.add_argument("--config", default="configs/train_config_internnav_8gpu_stage2_wider.yaml")
    p.add_argument("--root", default=os.environ.get("PANORAMIC_DATA_ROOT", "/workspace/r2r_panoramic_data"))
    p.add_argument("--split", default="train")
    p.add_argument("--output", required=True, help="Output JSONL sidecar path")
    p.add_argument("--internnav-repo", default=os.environ.get("INTERNNAV_REPO", "~/InternNav"))
    p.add_argument("--model-path", default=os.environ.get("INTERNNAV_MODEL_PATH", "/workspace/InternNav_Model"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--flash-attn-stub", dest="flash_attn_stub", action="store_true", default=True)
    p.add_argument("--no-flash-attn-stub", dest="flash_attn_stub", action="store_false")
    p.add_argument("--require-nextdit", dest="require_nextdit", action="store_true", default=True)
    p.add_argument("--no-require-nextdit", dest="require_nextdit", action="store_false")
    p.add_argument("--num-samples", type=int, default=0, help="0 means all filtered samples")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-stop", action="store_true", help="Keep dataset stop samples")
    p.add_argument(
        "--sample-mode",
        choices=["pixel", "all", "stop_turn"],
        default="pixel",
        help=(
            "pixel: InternNav pixel-goal samples for System1 distillation; "
            "all: generic dataset index including STOP/turn/no-pixel states; "
            "stop_turn: only STOP/turn/no-pixel states for System2 policy labels."
        ),
    )
    p.add_argument("--allow-no-pixel-goal", dest="require_pixel_goal", action="store_false")
    p.set_defaults(require_pixel_goal=True)
    p.add_argument("--all-samples", action="store_true", help="Ignore config require_sft_target and iterate the generic dataset index")
    p.add_argument("--include-text", action="store_true")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--skip-errors", dest="skip_errors", action="store_true", default=True)
    p.add_argument("--no-skip-errors", dest="skip_errors", action="store_false")
    p.add_argument("--skip-system1-errors", dest="skip_system1_errors", action="store_true", default=True)
    p.add_argument("--no-skip-system1-errors", dest="skip_system1_errors", action="store_false")
    p.add_argument("--front-width", type=int, default=0, help="0 keeps dataset resolution")
    p.add_argument("--front-height", type=int, default=0, help="0 keeps dataset resolution")
    p.add_argument("--traj-image-size", type=int, default=224)
    p.add_argument("--conjunction-mode", choices=["fixed", "random"], default="fixed")
    p.add_argument("--fixed-conjunction", default="you can see ")
    p.add_argument("--two-turn-lookdown", dest="two_turn_lookdown", action="store_true", default=True)
    p.add_argument("--no-two-turn-lookdown", dest="two_turn_lookdown", action="store_false")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--predict-steps", type=int, default=32)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--num-sample-trajs", type=int, default=32)
    p.add_argument("--save-mean-trajectory", dest="save_mean_trajectory", action="store_true", default=True)
    p.add_argument("--no-save-mean-trajectory", dest="save_mean_trajectory", action="store_false")
    p.add_argument(
        "--tensor-output-dir",
        default=None,
        help="Optional directory for .pt tensor sidecars containing traj_latents and/or dp_actions.",
    )
    p.add_argument("--tensor-shard-size", type=int, default=1000)
    p.add_argument(
        "--tensor-save-dtype",
        default="bfloat16",
        choices=["source", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
    )
    p.add_argument("--save-traj-latents", dest="save_traj_latents", action="store_true", default=True)
    p.add_argument("--no-save-traj-latents", dest="save_traj_latents", action="store_false")
    p.add_argument("--save-dp-actions", dest="save_dp_actions", action="store_true", default=True)
    p.add_argument("--no-save-dp-actions", dest="save_dp_actions", action="store_false")
    p.add_argument("--round-digits", type=int, default=5)
    p.add_argument("--progress-interval", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_samples:
        args.sample_mode = "all"
    if args.sample_mode in {"all", "stop_turn"}:
        args.require_pixel_goal = False
        args.include_stop = True
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")

    rng = random.Random(args.seed + args.shard_index)
    torch.manual_seed(args.seed + args.shard_index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + args.shard_index)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = _torch_dtype(args.torch_dtype)

    cfg = load_config(args.config)
    cfg["data"]["root"] = args.root
    if "paths" in cfg:
        cfg["paths"]["dataset_root"] = args.root
    traj_cfg = cfg.setdefault("data", {}).setdefault("trajectory", {})
    traj_cfg["panoramic_vlm_input"] = False
    traj_cfg["load_lookdown_for_system2"] = True
    traj_cfg["load_traj_images"] = True
    traj_cfg["enable_trajectory_augmentation"] = False
    if args.sample_mode in {"all", "stop_turn"}:
        traj_cfg["require_sft_target"] = False
    elif args.sample_mode == "pixel":
        traj_cfg["require_sft_target"] = True

    action_scale = float(traj_cfg.get("action_scale", 4.0))
    print(
        f"[dataset] root={args.root} split={args.split} sample_mode={args.sample_mode} "
        f"require_sft_target={traj_cfg.get('require_sft_target')} "
        f"require_pixel_goal={args.require_pixel_goal} include_stop={args.include_stop} "
        f"shard={args.shard_index}/{args.num_shards}",
        flush=True,
    )
    dataset = build_trajectory_dataset(
        cfg,
        split=args.split,
        enable_augmentation=False,
        enable_trajectory_augmentation=False,
        load_history_heatmap=False,
        panoramic_vlm_input=False,
        load_lookdown_for_system2=True,
        load_traj_images=True,
    )
    print(f"[dataset] samples={len(dataset)}", flush=True)

    model, processor, traj_to_actions_fn = _load_teacher(args, device)

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_indices(out_path) if args.resume else set()
    if done:
        print(f"[resume] loaded {len(done)} completed dataset indices from {out_path}", flush=True)
    if args.tensor_output_dir:
        print(
            f"[tensor] output_dir={Path(args.tensor_output_dir).expanduser()} "
            f"save_traj_latents={args.save_traj_latents} save_dp_actions={args.save_dp_actions} "
            f"dtype={args.tensor_save_dtype}",
            flush=True,
        )

    attempted = 0
    written = 0
    skipped = 0
    errors = 0
    kind_counts: dict[str, int] = {}
    with out_path.open("a", encoding="utf-8", buffering=1) as f:
        for idx in range(len(dataset)):
            if args.num_shards > 1 and idx % args.num_shards != args.shard_index:
                continue
            if idx in done:
                skipped += 1
                continue
            if args.num_samples > 0 and written >= args.num_samples:
                break

            try:
                sample = dataset[idx]
                if not _sample_passes_mode(sample, args):
                    skipped += 1
                    continue

                kind = _sample_kind(sample)
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
                rec = _collect_one(
                    idx,
                    sample,
                    dataset,
                    model,
                    processor,
                    traj_to_actions_fn,
                    device,
                    dtype,
                    action_scale,
                    args,
                    rng,
                )
                attempted += 1
                written += 1
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            except Exception as exc:
                errors += 1
                if not args.skip_errors:
                    raise
                err_rec = {
                    "status": "error",
                    **_sample_metadata(dataset, idx),
                    "error": repr(exc),
                }
                f.write(json.dumps(err_rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1

            if args.progress_interval > 0 and (attempted + errors) % args.progress_interval == 0:
                print(
                    f"[progress] attempted={attempted} written={written} skipped={skipped} "
                    f"errors={errors} last_idx={idx}",
                    flush=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(
        f"[done] output={out_path} attempted={attempted} written={written} skipped={skipped} "
        f"errors={errors} kind_counts={kind_counts}",
        flush=True,
    )


if __name__ == "__main__":
    main()

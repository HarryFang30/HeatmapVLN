"""Shared helpers for pano-aligned InternNav teacher latent conditioning."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import types
from typing import Any

import torch

from src.models.heatmap.input_constructor import (
    STRUCTURED_PANO_OUTPUT_SUFFIX,
    VIEW_NAMES,
    format_structured_pano_assistant_text,
    structured_condition_text,
)

NATIVE_TEACHER_SIDECAR_SCHEMA = "heatmapvln.native_teacher.v2"
NATIVE_TEACHER_ALIGNMENT_VERSION = "same_goal_frame_front_down_yx_v1"


def has_structured_pano_pixel_goal(sample: dict[str, Any]) -> bool:
    kind = str(sample.get("pano_sample_kind") or "").lower()
    if kind and kind != "pixel":
        return False
    if sample.get("pano_pixel_goal") is None:
        return False
    view_id = str(sample.get("pano_view_id") or "").lower()
    return view_id in VIEW_NAMES


def structured_assistant_from_sample(sample: dict[str, Any]) -> str:
    text = format_structured_pano_assistant_text(
        sample.get("pano_view_id"),
        sample.get("pano_pixel_goal"),
        sample_kind=sample.get("pano_sample_kind"),
        is_stop=float(sample.get("is_stop", 0.0)) > 0.5,
    )
    if text is None or "pixel:" not in text:
        raise RuntimeError(
            "Sample has no structured pano pixel goal "
            f"(view={sample.get('pano_view_id')} kind={sample.get('pano_sample_kind')})"
        )
    return text


def append_structured_pano_suffix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated = copy.deepcopy(messages)
    user_content = updated[0]["content"]
    for item in reversed(user_content):
        if item.get("type") == "text":
            item["text"] = str(item["text"]) + STRUCTURED_PANO_OUTPUT_SUFFIX
            break
    return updated


def condition_on_pano_coord(
    processor: Any,
    first_messages: list[dict[str, Any]],
    first_images: list[Any],
    sample: dict[str, Any],
    device: torch.device,
) -> tuple[str, torch.Tensor, Any, int, list[int], list[int], str]:
    """Build single-turn InternNav context aligned with dataset pano goal."""
    assistant_text = structured_assistant_from_sample(sample)
    pano_pixel_goal = sample.get("pano_pixel_goal")
    coord_uv = [int(pano_pixel_goal[0]), int(pano_pixel_goal[1])]
    goal_yx = [coord_uv[1], coord_uv[0]]
    pano_view_id = str(sample.get("pano_view_id")).lower()

    messages = append_structured_pano_suffix(first_messages)
    full_messages = [
        *messages,
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    inputs = processor(text=[full_text], images=first_images, return_tensors="pt").to(device)
    prompt_len = int(inputs.input_ids.shape[1])
    return (
        assistant_text,
        inputs.input_ids,
        inputs,
        prompt_len,
        coord_uv,
        goal_yx,
        pano_view_id,
    )


def find_cond_projector(model: Any) -> torch.nn.Module | None:
    for attr in ("cond_projector", "traj_cond_projector"):
        module = getattr(model, attr, None)
        if isinstance(module, torch.nn.Module):
            return module
    nested = getattr(model, "model", None)
    if nested is not None and nested is not model:
        for attr in ("cond_projector", "traj_cond_projector"):
            module = getattr(nested, attr, None)
            if isinstance(module, torch.nn.Module):
                return module
    return None


def project_traj_latents_to_768(model: Any, traj_latents: torch.Tensor) -> torch.Tensor:
    cond_projector = find_cond_projector(model)
    if cond_projector is None:
        raise RuntimeError("InternNav teacher model has no cond_projector")
    projector_dtype = next(cond_projector.parameters()).dtype
    return cond_projector(traj_latents.to(dtype=projector_dtype)).detach()


@torch.no_grad()
def generate_teacher_latents_768(
    teacher_model: Any,
    processor: Any,
    output_ids: torch.Tensor,
    inputs: Any,
    device: torch.device,
) -> torch.Tensor:
    from scripts.evaluation.collect_internnav_teacher_sidecar import _normalize_image_grid_thw

    pixel_values = inputs.pixel_values
    image_grid_thw = _normalize_image_grid_thw(inputs)
    traj_latents = teacher_model.generate_latents(output_ids, pixel_values, image_grid_thw)
    return project_traj_latents_to_768(teacher_model, traj_latents)


def make_teacher_turn_args(seed: int = 42) -> Any:
    return types.SimpleNamespace(
        front_width=0,
        front_height=0,
        conjunction_mode="fixed",
        fixed_conjunction="you can see ",
        seed=seed,
    )


@torch.no_grad()
def compute_aligned_teacher_latents_3584_batch(
    teacher_model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    device: torch.device,
    *,
    turn_args: Any,
) -> torch.Tensor:
    """Extract aligned teacher **raw** latents (3584-dim, before cond_projector).

    Same as ``compute_aligned_teacher_latents_768_batch`` but returns the
    VLM hidden-state latents instead of cond_projector projections.
    Used as training targets for ``PanoLatentSpaceAdapter``.
    """
    from scripts.evaluation.collect_internnav_teacher_sidecar import (
        _build_first_turn,
        _normalize_image_grid_thw,
    )

    rng = random.Random(int(getattr(turn_args, "seed", 42)))
    if not samples:
        raise RuntimeError("Empty teacher batch")

    latents: list[torch.Tensor] = []
    for sample in samples:
        if not has_structured_pano_pixel_goal(sample):
            raise RuntimeError(
                "Aligned teacher batch contains a sample without a structured pano pixel goal"
            )
        first_messages, first_images = _build_first_turn(sample, turn_args, rng)
        _, output_ids, inputs, _, _, _, _ = condition_on_pano_coord(
            processor, first_messages, first_images, sample, device,
        )
        pixel_values = inputs.pixel_values
        image_grid_thw = _normalize_image_grid_thw(inputs)
        traj_latents = teacher_model.generate_latents(
            output_ids, pixel_values, image_grid_thw,
        )
        if traj_latents.dim() == 3 and traj_latents.shape[0] == 1:
            traj_latents = traj_latents.squeeze(0)
        latents.append(traj_latents)
    return torch.stack(latents, dim=0).to(device)


@torch.no_grad()
def compute_aligned_teacher_latents_768_batch(
    teacher_model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    device: torch.device,
    *,
    turn_args: Any,
) -> torch.Tensor:
    """Extract aligned teacher latents for a batch of pano samples.

    InternNav's released ``generate_latents`` only supports one sample per
    call, so keep the teacher forward sequential even when adapter training is
    batched. Each call receives the complete tokenized conversation.
    """
    from scripts.evaluation.collect_internnav_teacher_sidecar import _build_first_turn

    rng = random.Random(int(getattr(turn_args, "seed", 42)))
    if not samples:
        raise RuntimeError("Empty teacher batch")

    latents: list[torch.Tensor] = []
    for sample in samples:
        if not has_structured_pano_pixel_goal(sample):
            raise RuntimeError(
                "Aligned teacher batch contains a sample without a structured pano pixel goal"
            )
        first_messages, first_images = _build_first_turn(sample, turn_args, rng)
        _, output_ids, inputs, _, _, _, _ = condition_on_pano_coord(
            processor,
            first_messages,
            first_images,
            sample,
            device,
        )
        latent_768 = generate_teacher_latents_768(
            teacher_model,
            processor,
            output_ids,
            inputs,
            device,
        )
        if latent_768.dim() == 3 and latent_768.shape[0] == 1:
            latent_768 = latent_768.squeeze(0)
        latents.append(latent_768)
    return torch.stack(latents, dim=0).to(device)


def sidecar_alignment_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "pano_view_id": sample.get("pano_view_id"),
        "pano_pixel_goal": sample.get("pano_pixel_goal"),
        "pano_sample_kind": sample.get("pano_sample_kind"),
        "structured_assistant_text": structured_assistant_from_sample(sample)
        if has_structured_pano_pixel_goal(sample)
        else None,
        "condition_text": structured_condition_text(
            str(sample.get("pano_view_id")).lower(),
            [int(v) for v in sample["pano_pixel_goal"]],
        )
        if has_structured_pano_pixel_goal(sample)
        else None,
        "pano_pixel_goal_relative_len": sample.get("pano_pixel_goal_relative_len"),
        "pano_goal_frame_idx": sample.get("pano_goal_frame_idx"),
        "aligned_native_pixel_goal_uv": sample.get("aligned_native_pixel_goal_uv"),
        "aligned_native_goal_frame_idx": sample.get("aligned_native_goal_frame_idx"),
        "aligned_native_visible": bool(sample.get("aligned_native_visible", False)),
    }


def aligned_native_sidecar_contract(
    sample: dict[str, Any],
    *,
    stable_sample_key: str,
    current_t: int,
) -> dict[str, Any]:
    """Build a fail-closed identity contract for one aligned native teacher.

    The fingerprint binds JSONL and tensor sidecars to the exact student pano
    waypoint and its projection into the native ``front_down`` camera.
    """
    if not stable_sample_key:
        raise ValueError("aligned native sidecar requires a stable_sample_key")
    if not has_structured_pano_pixel_goal(sample):
        raise ValueError("aligned native sidecar requires a structured pano pixel goal")
    if str(sample.get("pano_view_id") or "").lower() != "front":
        raise ValueError("aligned native sidecar only supports front pano goals")
    native_uv = sample.get("aligned_native_pixel_goal_uv")
    if native_uv is None or len(native_uv) < 2:
        raise ValueError("aligned native sidecar requires a visible front_down projection")

    relative_len = int(sample.get("pano_pixel_goal_relative_len") or 0)
    pano_frame_idx = int(sample.get("pano_goal_frame_idx") or -1)
    native_frame_idx = int(sample.get("aligned_native_goal_frame_idx") or -1)
    expected_frame_idx = int(current_t) + relative_len
    if relative_len <= 0 or pano_frame_idx != expected_frame_idx or native_frame_idx != pano_frame_idx:
        raise ValueError(
            "aligned native sidecar goal-frame mismatch: "
            f"current_t={current_t} relative_len={relative_len} "
            f"pano={pano_frame_idx} native={native_frame_idx}"
        )

    pano_uv = [int(v) for v in sample["pano_pixel_goal"][:2]]
    native_uv = [int(v) for v in native_uv[:2]]
    identity = {
        "alignment_version": NATIVE_TEACHER_ALIGNMENT_VERSION,
        "stable_sample_key": str(stable_sample_key),
        "current_t": int(current_t),
        "goal_frame_idx": pano_frame_idx,
        "student_goal": {
            "view_id": "front",
            "pixel_uv": pano_uv,
            "relative_len": relative_len,
        },
        "native_goal": {
            "view_id": "front_down",
            "pixel_uv": native_uv,
            "text_yx": [native_uv[1], native_uv[0]],
            "relative_len": relative_len,
        },
        "source_coord_order": "uv",
        "teacher_text_coord_order": "yx",
    }
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "sidecar_schema": NATIVE_TEACHER_SIDECAR_SCHEMA,
        "alignment_version": NATIVE_TEACHER_ALIGNMENT_VERSION,
        "alignment_fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "alignment_contract": identity,
    }


def validate_aligned_native_sidecar_contract_fields(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Validate a native-v2 JSONL record without loading RGB or tensor files.

    This is intentionally fail-closed: it binds the fingerprinted contract to
    the duplicated teacher and dataset-label fields that the trainer consumes.
    Dataset path/index identity is checked separately by the trainer because
    it needs the live dataset instance.
    """

    def pair(value: Any, name: str) -> list[int]:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            raise ValueError(f"{name} must be a 2-element coordinate")
        return [int(value[0]), int(value[1])]

    if record.get("sidecar_schema") != NATIVE_TEACHER_SIDECAR_SCHEMA:
        raise ValueError("native sidecar schema mismatch")
    if record.get("alignment_version") != NATIVE_TEACHER_ALIGNMENT_VERSION:
        raise ValueError("native sidecar alignment version mismatch")

    contract = record.get("alignment_contract")
    if not isinstance(contract, dict):
        raise ValueError("native sidecar has no alignment_contract")
    if contract.get("alignment_version") != NATIVE_TEACHER_ALIGNMENT_VERSION:
        raise ValueError("alignment_contract version mismatch")
    if contract.get("source_coord_order") != "uv":
        raise ValueError("student coordinate order must be uv")
    if contract.get("teacher_text_coord_order") != "yx":
        raise ValueError("teacher text coordinate order must be yx")

    stable_key = str(record.get("stable_sample_key") or "")
    if not stable_key or contract.get("stable_sample_key") != stable_key:
        raise ValueError("stable_sample_key is not bound to alignment_contract")
    current_t = int(record.get("current_t", -1))
    if current_t < 0 or int(contract.get("current_t", -1)) != current_t:
        raise ValueError("current_t is not bound to alignment_contract")

    student = contract.get("student_goal")
    native = contract.get("native_goal")
    if not isinstance(student, dict) or not isinstance(native, dict):
        raise ValueError("alignment_contract goal fields are missing")
    if student.get("view_id") != "front" or native.get("view_id") != "front_down":
        raise ValueError("native sidecar must align front to front_down")
    student_uv = pair(student.get("pixel_uv"), "student_goal.pixel_uv")
    native_uv = pair(native.get("pixel_uv"), "native_goal.pixel_uv")
    native_yx = pair(native.get("text_yx"), "native_goal.text_yx")
    if native_yx != [native_uv[1], native_uv[0]]:
        raise ValueError("native teacher yx text is not the reverse of image uv")
    student_len = int(student.get("relative_len", 0))
    native_len = int(native.get("relative_len", 0))
    goal_frame = int(contract.get("goal_frame_idx", -1))
    if student_len <= 0 or native_len != student_len:
        raise ValueError("student/native relative waypoint lengths differ")
    if goal_frame != current_t + student_len:
        raise ValueError("goal_frame_idx does not match current_t + relative_len")

    canonical = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if record.get("alignment_fingerprint") != expected_fingerprint:
        raise ValueError("alignment_fingerprint does not match alignment_contract")

    teacher = record.get("teacher") or {}
    if pair(teacher.get("coord_uv"), "teacher.coord_uv") != native_uv:
        raise ValueError("teacher.coord_uv does not match native goal")
    if pair(
        teacher.get("internnav_pixel_goal_yx"),
        "teacher.internnav_pixel_goal_yx",
    ) != native_yx:
        raise ValueError("teacher yx text coordinate does not match native goal")
    if str(teacher.get("conditioned_coord_text") or "").strip() != (
        f"{native_yx[0]} {native_yx[1]}"
    ):
        raise ValueError("teacher conditioned_coord_text does not match native yx")
    if str(teacher.get("pano_view_id") or "").lower() != "front":
        raise ValueError("teacher pano_view_id must be front")
    if int(teacher.get("goal_frame_idx", -1)) != goal_frame:
        raise ValueError("teacher goal_frame_idx does not match alignment_contract")

    label = record.get("dataset_label") or {}
    if str(label.get("pano_view_id") or "").lower() != "front":
        raise ValueError("dataset_label pano_view_id must be front")
    if pair(label.get("pano_pixel_goal"), "dataset_label.pano_pixel_goal") != student_uv:
        raise ValueError("dataset_label student pixel does not match alignment_contract")
    if pair(
        label.get("aligned_native_pixel_goal_uv"),
        "dataset_label.aligned_native_pixel_goal_uv",
    ) != native_uv:
        raise ValueError("dataset_label native pixel does not match alignment_contract")
    if int(label.get("pano_pixel_goal_relative_len", 0)) != student_len:
        raise ValueError("dataset_label relative_len does not match alignment_contract")
    if int(label.get("pano_goal_frame_idx", -1)) != goal_frame:
        raise ValueError("dataset_label pano goal frame does not match alignment_contract")
    if int(label.get("aligned_native_goal_frame_idx", -1)) != goal_frame:
        raise ValueError("dataset_label native goal frame does not match alignment_contract")
    if not bool(label.get("aligned_native_visible", False)):
        raise ValueError("dataset_label native projection is not visible")
    expected_structured = structured_condition_text("front", student_uv)
    if label.get("structured_assistant_text") != expected_structured:
        raise ValueError("dataset_label structured assistant text is inconsistent")
    if label.get("condition_text") != expected_structured:
        raise ValueError("dataset_label condition text is inconsistent")
    return contract

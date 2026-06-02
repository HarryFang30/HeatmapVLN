"""Shared helpers for pano-aligned InternNav teacher latent conditioning."""

from __future__ import annotations

import copy
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
def compute_aligned_teacher_latents_768_batch(
    teacher_model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    device: torch.device,
    *,
    turn_args: Any,
) -> torch.Tensor:
    """Batched aligned teacher latent extraction.

    Builds per-sample InternNav contexts, then runs the processor and
    ``generate_latents`` once across the whole batch.  Falls back to
    sequential per-sample processing for single-sample batches.
    """
    from scripts.evaluation.collect_internnav_teacher_sidecar import (
        _build_first_turn,
        _normalize_image_grid_thw,
    )

    rng = random.Random(int(getattr(turn_args, "seed", 42)))
    batch_size = len(samples)
    if batch_size == 0:
        raise RuntimeError("Empty teacher batch")
    if batch_size == 1:
        # Single sample — no batching benefit.
        sample = samples[0]
        if not has_structured_pano_pixel_goal(sample):
            raise RuntimeError(
                "Aligned teacher batch contains a sample without a structured pano pixel goal"
            )
        first_messages, first_images = _build_first_turn(sample, turn_args, rng)
        _, output_ids, inputs, _, _, _, _ = condition_on_pano_coord(
            processor, first_messages, first_images, sample, device,
        )
        latent_768 = generate_teacher_latents_768(
            teacher_model, processor, output_ids, inputs, device,
        )
        if latent_768.dim() == 3 and latent_768.shape[0] == 1:
            latent_768 = latent_768.squeeze(0)
        return latent_768.unsqueeze(0).to(device)

    # Build per-sample contexts.
    full_texts: list[str] = []
    all_images: list[Any] = []
    output_ids_list: list[torch.Tensor] = []
    for sample in samples:
        if not has_structured_pano_pixel_goal(sample):
            raise RuntimeError(
                "Aligned teacher batch contains a sample without a structured pano pixel goal"
            )
        first_messages, first_images = _build_first_turn(sample, turn_args, rng)
        assistant_text = structured_assistant_from_sample(sample)
        messages = append_structured_pano_suffix(first_messages)
        full_messages = [
            *messages,
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]
        full_text = processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False,
        )
        full_texts.append(full_text)
        all_images.extend(first_images)
        # Encode assistant output standalone for generate_latents conditioning.
        output_ids_list.append(
            processor.tokenizer.encode(assistant_text, add_special_tokens=False, return_tensors="pt")[0]
        )

    # Batch through the processor: all texts + all images at once.
    inputs = processor(
        text=full_texts, images=all_images, return_tensors="pt", padding=True,
    ).to(device)
    pixel_values = inputs.pixel_values
    image_grid_thw = _normalize_image_grid_thw(inputs)

    # Pad output_ids to max length for batched generate_latents.
    max_len = max(len(ids) for ids in output_ids_list)
    padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, ids in enumerate(output_ids_list):
        padded_ids[i, :len(ids)] = ids.to(device=device)

    # Single batched teacher forward.
    traj_latents = teacher_model.generate_latents(padded_ids, pixel_values, image_grid_thw)
    traj_latents_768 = project_traj_latents_to_768(teacher_model, traj_latents)

    # Unbatch: each sample's latents have shape [n_query, 768].
    if traj_latents_768.dim() == 3 and traj_latents_768.shape[0] == batch_size:
        return traj_latents_768.to(device)
    # Safety fallback: generate_latents may have different batching semantics.
    latents: list[torch.Tensor] = []
    for sample in samples:
        _, output_ids, inputs_i, _, _, _, _ = condition_on_pano_coord(
            processor,
            _build_first_turn(sample, turn_args, rng)[0],
            _build_first_turn(sample, turn_args, rng)[1],
            sample, device,
        )
        latent_768 = generate_teacher_latents_768(
            teacher_model, processor, output_ids, inputs_i, device,
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
    }

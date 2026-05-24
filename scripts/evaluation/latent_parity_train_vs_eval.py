#!/usr/bin/env python3
"""
Latent parity: eval (A), training collator forward (B), and isolated generate_latents (C).

Path C (highest priority): same ``pano_inputs`` as B, strip trailing TRAJ tokens,
call ``generate_latents`` on identical input_ids / pixel_values / image_grid_thw,
compare to B's ``traj_hidden_states``.

Decision:
  - C vs B cosine > 0.99 → generate_latents OK; A/B gap is prompt / token sequence.
  - C vs B cosine << 1   → generate_latents ≠ train forward; fix eval latent extraction.
  - C ≈ B but B short   → fix eval wiring, but Stage2 bridge still weak vs GT.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from scripts.evaluation.oracle_pixel_goal_bridge_test import (
    build_eval_aligned_messages,
    load_eval_model,
)
from scripts.evaluation.r2r_val_unseen import (
    _condition_output_ids_for_pixel_goal,
    _finalize_local_actions,
    _lookdown_to_traj_tensor,
    _normalize_multimodal_inputs,
    _system1_coord_order,
    _trajectory_debug_summary,
    traj_to_actions,
)
from scripts.training.utils import load_config

from src.data.factory import build_trajectory_dataset
from src.data.panoramic_tokenized_collator import PanoramicTokenizedCollator
from src.models.qwen2_5_vl.integration import TRAJ_TOKEN_INDEX

LOGGER = logging.getLogger("latent_parity")
COSINE_MATCH = 0.99

PATH_LEN_RE = re.compile(r"path_len=([0-9.]+)")


def _parse_path_len(summary: str) -> float | None:
    m = PATH_LEN_RE.search(summary)
    return float(m.group(1)) if m else None


def _gt_trajectory_for_summary(sample: dict) -> torch.Tensor | None:
    traj = sample.get("trajectory")
    if traj is None:
        return None
    if torch.is_tensor(traj):
        t = traj
    else:
        t = torch.as_tensor(traj)
    if t.dim() == 3:
        t = t[0]
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t


def _build_eval_traj_images(sample: dict, device: torch.device, traj_image_size: tuple[int, int]):
    from scripts.evaluation.oracle_pixel_goal_bridge_test import _ensure_pil_from_sample_tensor

    lookdown_pil = _ensure_pil_from_sample_tensor(sample["lookdown_frame"])
    if lookdown_pil.size != traj_image_size:
        lookdown_pil = lookdown_pil.resize(traj_image_size)
    traj_t = _lookdown_to_traj_tensor(lookdown_pil, device)
    return torch.stack([traj_t, traj_t]).unsqueeze(0).to(device)


def _build_train_batch(
    sample: dict,
    processor,
    n_traj_query: int,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Exact Stage2 training tokenization (PanoramicTokenizedCollator + internnav)."""
    collator = PanoramicTokenizedCollator(
        processor,
        n_traj_query=n_traj_query,
        sft_mode=True,
        sft_protocol="internnav",
    )
    batch_item = {k: v for k, v in sample.items()}
    batch = collator([batch_item])
    pano_inputs = batch["pano_inputs"]
    _normalize_multimodal_inputs(pano_inputs)
    return pano_inputs, batch["pano_num_histories"]


def _strip_trailing_traj_tokens(
    input_ids: torch.Tensor,
    n_traj_query: int,
) -> torch.Tensor:
    """Return collator input_ids without the trailing TRAJ placeholders (path C)."""
    if n_traj_query <= 0:
        return input_ids
    tail = input_ids[:, -n_traj_query:]
    if not bool((tail == TRAJ_TOKEN_INDEX).all().item()):
        got = tail[0].tolist()
        raise RuntimeError(
            f"Expected last {n_traj_query} tokens to be TRAJ_TOKEN_INDEX={TRAJ_TOKEN_INDEX}, "
            f"got {got}"
        )
    return input_ids[:, :-n_traj_query].contiguous()


def _latent_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.detach().float().reshape(1, -1)
    b = b.detach().float().reshape(1, -1)
    if a.shape != b.shape:
        return {
            "cosine_mean": float("nan"),
            "cosine_min": float("nan"),
            "max_abs_diff": float("nan"),
            "norm_ratio": float("nan"),
            "l2_diff": float("nan"),
        }
    cos = F.cosine_similarity(a, b, dim=1).item()
    diff = a - b
    return {
        "cosine_mean": cos,
        "cosine_min": cos,
        "max_abs_diff": float(diff.abs().max().item()),
        "norm_ratio": float(a.norm().item() / (b.norm().item() + 1e-8)),
        "l2_diff": float(diff.norm().item()),
    }


def _per_query_cosine(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    a = a.detach().float().squeeze(0)
    b = b.detach().float().squeeze(0)
    if a.shape != b.shape:
        return []
    return [
        float(F.cosine_similarity(a[i : i + 1], b[i : i + 1]).item())
        for i in range(a.shape[0])
    ]


def _run_get_trajectory(
    model,
    traj_hs: torch.Tensor,
    traj_images: torch.Tensor,
    *,
    num_sample_trajs: int,
    action_scale: float,
) -> dict:
    with torch.no_grad():
        trajectory = model.nextdit_action_head.get_trajectory(
            traj_hs,
            traj_images=traj_images,
        )
    summary = _trajectory_debug_summary(trajectory, num_sample_trajs, action_scale)
    actions = _finalize_local_actions(
        traj_to_actions(
            trajectory,
            num_sample_trajs=num_sample_trajs,
            action_scale=action_scale,
        )
    )
    path_len = _parse_path_len(summary)
    return {
        "trajectory_summary": summary,
        "path_len": path_len,
        "actions": actions,
        "forward_count": sum(1 for a in actions if a == 1),
        "zero_pad_pattern": actions[:4] == [0, 0, 0, 0],
    }


def parity_forward_one(
    model,
    sample: dict,
    device: torch.device,
    *,
    num_sample_trajs: int,
    action_scale: float,
    traj_image_size: tuple[int, int],
    coord_order: str,
    n_traj_query: int,
) -> dict:
    pixel_goal = [int(sample["pixel_goal"][0]), int(sample["pixel_goal"][1])]
    coord_text = f"{pixel_goal[0]} {pixel_goal[1]}"
    processor = model.qwen2_5_vl.processor
    if processor is None:
        raise RuntimeError("processor is None")

    # ── Path A: eval / generate_latents ──────────────────────────────────
    messages = build_eval_aligned_messages(sample)
    prefill = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_len = int(prefill["input_ids"].shape[1])

    messages_with_gold = [
        *messages,
        {
            "role": "assistant",
            "content": [{"type": "text", "text": coord_text}],
        },
    ]
    full = processor.apply_chat_template(
        messages_with_gold,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    output_ids = full["input_ids"].to(device)
    inputs_a = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in full.items()}
    _normalize_multimodal_inputs(inputs_a)

    condition_output_ids = _condition_output_ids_for_pixel_goal(
        output_ids=output_ids,
        prompt_len=prompt_len,
        tokenizer=processor.tokenizer,
        pixel_goal=pixel_goal,
        llm_output=coord_text,
        coord_order=coord_order,
    )
    lq = model.latent_queries.expand(1, -1, -1).to(device=device, dtype=model.config.dtype)
    with torch.no_grad():
        traj_hs_a = model.qwen2_5_vl.generate_latents(
            output_ids=condition_output_ids,
            pixel_values=inputs_a.get("pixel_values"),
            image_grid_thw=inputs_a.get("image_grid_thw"),
            latent_queries=lq,
        )

    # ── Path B: training collator + _forward_model_inputs ────────────────
    pano_inputs, pano_num_histories = _build_train_batch(sample, processor, n_traj_query)
    pano_inputs = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in pano_inputs.items()
    }
    with torch.no_grad():
        qwen_out = model.qwen2_5_vl(
            history_frames=sample["history_frames"].unsqueeze(0).to(device),
            current_frame=sample["current_frame"].unsqueeze(0).to(device),
            panoramic_inputs=pano_inputs,
            panoramic_num_histories=pano_num_histories,
            latent_queries=lq,
            return_hidden_states=False,
        )
    traj_hs_b = qwen_out["traj_hidden_states"]
    if traj_hs_b is None:
        raise RuntimeError("Training path did not return traj_hidden_states")

    # ── Path C: B's pano_inputs minus TRAJ suffix → generate_latents ─────
    output_ids_c = _strip_trailing_traj_tokens(pano_inputs["input_ids"], n_traj_query)
    with torch.no_grad():
        traj_hs_c = model.qwen2_5_vl.generate_latents(
            output_ids=output_ids_c,
            pixel_values=pano_inputs.get("pixel_values"),
            image_grid_thw=pano_inputs.get("image_grid_thw"),
            latent_queries=lq,
        )

    traj_images = _build_eval_traj_images(sample, device, traj_image_size)
    traj_a = _run_get_trajectory(
        model, traj_hs_a, traj_images,
        num_sample_trajs=num_sample_trajs, action_scale=action_scale,
    )
    traj_b = _run_get_trajectory(
        model, traj_hs_b, traj_images,
        num_sample_trajs=num_sample_trajs, action_scale=action_scale,
    )
    traj_c = _run_get_trajectory(
        model, traj_hs_c, traj_images,
        num_sample_trajs=num_sample_trajs, action_scale=action_scale,
    )

    gt_t = _gt_trajectory_for_summary(sample)
    gt_summary = None
    gt_path_len = None
    if gt_t is not None:
        gt_summary = _trajectory_debug_summary(gt_t, 1, action_scale)
        gt_path_len = _parse_path_len(gt_summary)

    latent_ab = _latent_metrics(traj_hs_a, traj_hs_b)
    latent_cb = _latent_metrics(traj_hs_c, traj_hs_b)
    return {
        "pixel_goal": pixel_goal,
        "coord_text": coord_text,
        "latent_ab": latent_ab,
        "latent_cb": latent_cb,
        "latent_ab_cosine_per_query": _per_query_cosine(traj_hs_a, traj_hs_b),
        "latent_cb_cosine_per_query": _per_query_cosine(traj_hs_c, traj_hs_b),
        "path_a": traj_a,
        "path_b": traj_b,
        "path_c": traj_c,
        "gt_trajectory_summary": gt_summary,
        "gt_path_len": gt_path_len,
        "seq_len_eval": int(condition_output_ids.shape[1]),
        "seq_len_train": int(pano_inputs["input_ids"].shape[1]),
        "seq_len_c_input": int(output_ids_c.shape[1]),
        "pano_num_histories": int(pano_num_histories[0]),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Latent parity: generate_latents vs train forward")
    p.add_argument("--config", default="configs/train_config_internnav_8gpu.yaml")
    p.add_argument("--base-checkpoint", default="checkpoints/stage1-s2_latest.pth")
    p.add_argument("--stage2-checkpoint", default="checkpoints/stage2_latest.pth")
    p.add_argument("--root", default="/workspace/r2r_panoramic_audit_v2")
    p.add_argument("--split", default="train")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="logs/latent_parity_train_vs_eval.jsonl")
    return p.parse_args()


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{100 * n / total:.1f}%"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    cfg = load_config(args.config)
    cfg["data"]["root"] = args.root
    if "paths" in cfg:
        cfg["paths"]["dataset_root"] = args.root

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    traj_cfg = cfg.get("model", {}).get("action_head", {}).get("nextdit", {})
    num_sample_trajs = int(traj_cfg.get("num_sample_trajs", 32))
    action_scale = float(cfg.get("data", {}).get("action_scale", 4.0))
    traj_size = tuple(cfg.get("data", {}).get("traj_image_size", [224, 224]))
    n_traj_query = int(traj_cfg.get("n_query", 4))
    coord_order = _system1_coord_order(
        argparse.Namespace(system1_coord_order="auto"),
        panoramic_internnav_protocol=True,
    )

    model = load_eval_model(
        cfg,
        Path(args.base_checkpoint),
        Path(args.stage2_checkpoint),
        device,
    )

    dataset = build_trajectory_dataset(cfg, split=args.split)
    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    records = []
    skipped = 0
    SHORT = 0.5

    for idx in indices:
        if len(records) >= args.num_samples:
            break
        sample = dataset[idx]
        if sample.get("pixel_goal") is None:
            skipped += 1
            continue
        if float(sample.get("is_stop", 0.0)) > 0.5:
            skipped += 1
            continue

        try:
            result = parity_forward_one(
                model,
                sample,
                device,
                num_sample_trajs=num_sample_trajs,
                action_scale=action_scale,
                traj_image_size=traj_size,
                coord_order=coord_order,
                n_traj_query=n_traj_query,
            )
        except Exception as exc:
            LOGGER.exception("Failed idx=%s: %s", idx, exc)
            continue

        rec = {"dataset_index": idx, **result}
        records.append(rec)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    n = len(records)
    if n == 0:
        print("No samples processed.", flush=True)
        return 1

    cos_ab = [r["latent_ab"]["cosine_mean"] for r in records if not np.isnan(r["latent_ab"]["cosine_mean"])]
    cos_cb = [r["latent_cb"]["cosine_mean"] for r in records if not np.isnan(r["latent_cb"]["cosine_mean"])]
    path_a = [r["path_a"]["path_len"] for r in records if r["path_a"]["path_len"] is not None]
    path_b = [r["path_b"]["path_len"] for r in records if r["path_b"]["path_len"] is not None]
    path_c = [r["path_c"]["path_len"] for r in records if r["path_c"]["path_len"] is not None]
    gt_lens = [r["gt_path_len"] for r in records if r["gt_path_len"] is not None]

    a_short = sum(1 for p in path_a if p < SHORT)
    b_short = sum(1 for p in path_b if p < SHORT)
    c_short = sum(1 for p in path_c if p < SHORT)
    cb_match = sum(1 for c in cos_cb if c >= COSINE_MATCH)
    c_match_b_short = sum(
        1 for r in records
        if r["latent_cb"]["cosine_mean"] >= COSINE_MATCH
        and r["path_b"]["path_len"] is not None
        and r["path_b"]["path_len"] < SHORT
    )
    a_short_b_ok = sum(
        1 for r in records
        if r["path_a"]["path_len"] is not None and r["path_b"]["path_len"] is not None
        and r["path_a"]["path_len"] < SHORT and r["path_b"]["path_len"] >= SHORT
    )
    both_short = sum(
        1 for r in records
        if r["path_b"]["path_len"] is not None and r["path_c"]["path_len"] is not None
        and r["path_b"]["path_len"] < SHORT and r["path_c"]["path_len"] < SHORT
    )

    print("\n===== Latent Parity A / B / C =====", flush=True)
    print(f"samples:              {n} (skipped: {skipped})", flush=True)
    if cos_cb:
        print(
            f"C vs B cosine:        avg={np.mean(cos_cb):.4f} "
            f"min={np.min(cos_cb):.4f} max={np.max(cos_cb):.4f}",
            flush=True,
        )
        print(f"C vs B >= {COSINE_MATCH}:     {cb_match}/{n} ({_pct(cb_match, n)})", flush=True)
    if cos_ab:
        print(
            f"A vs B cosine:        avg={np.mean(cos_ab):.4f} "
            f"(prompt+eval path; not isolated)",
            flush=True,
        )
    if path_a:
        print(f"path_len A (eval):    mean={np.mean(path_a):.3f} median={np.median(path_a):.3f}", flush=True)
    if path_b:
        print(f"path_len B (train):   mean={np.mean(path_b):.3f} median={np.median(path_b):.3f}", flush=True)
    if path_c:
        print(f"path_len C (gen_lat): mean={np.mean(path_c):.3f} median={np.median(path_c):.3f}", flush=True)
    if gt_lens:
        print(f"path_len GT label:    mean={np.mean(gt_lens):.3f} median={np.median(gt_lens):.3f}", flush=True)
    print(f"path_len < {SHORT}m — A: {_pct(a_short, n)}  B: {_pct(b_short, n)}  C: {_pct(c_short, n)}", flush=True)
    print(f"A short & B ok:       {a_short_b_ok}/{n} ({_pct(a_short_b_ok, n)})", flush=True)
    print(f"B & C both short:     {both_short}/{n} ({_pct(both_short, n)})", flush=True)
    print(f"C≈B but B short:      {c_match_b_short}/{n} ({_pct(c_match_b_short, n)})", flush=True)
    print(f"jsonl:                {out_path}", flush=True)

    median_cb = float(np.median(cos_cb)) if cos_cb else 0.0
    median_gt = float(np.median(gt_lens)) if gt_lens else 0.0
    median_b = float(np.median(path_b)) if path_b else 0.0

    if cos_cb and median_cb >= COSINE_MATCH and cb_match >= n * 0.9:
        print(
            f"\nVERDICT [C≈B]: generate_latents matches train forward (median cosine={median_cb:.4f}). "
            "Fix eval prompt / token sequence (path A), not generate_latents internals.",
            flush=True,
        )
        if median_b < SHORT and median_gt >= SHORT:
            print(
                "  Bridge still weak: train latents (B) do not yield GT-scale trajectories.",
                flush=True,
            )
    elif cos_cb and median_cb < 0.95:
        print(
            f"\nVERDICT [C≠B]: generate_latents diverges from train forward (median cosine={median_cb:.4f}). "
            "Replace eval latent extraction with _forward_model_inputs(..., latent_queries=...) "
            "on collator pano_inputs; do not use hand-rolled inputs_embeds/rope path.",
            flush=True,
        )
    elif c_match_b_short >= n * 0.5:
        print(
            "\nVERDICT [C≈B, B short]: latent extraction OK; Stage2 bridge / training is the bottleneck.",
            flush=True,
        )
    else:
        print("\nVERDICT: Mixed — inspect jsonl latent_cb_cosine_per_query.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""EXP-17: read a trained System2 arm by *generation* on held-out scenes.

The EXP-13/14 evaluator scores the first supervised token under teacher forcing.
That is exact for a bare answer, but an EXP-17 arm first writes a cognition
prefix and only then decides, so the decision is no longer the first token and
must be decoded the way it is deployed: greedily, from the prompt alone.

Passes per state (``--passes``):

``natural``
    greedy decode of the first assistant turn from the prompt.  Gives the
    prefix fields, the decision, the generated preservation numbers and the
    natural prefix/decision association (test 2b).
``placeholder``
    the content-free prefix (every slot and the progress read 未知) is forced
    after the prompt, then the decision is decoded.  The same weights, the same
    prompt, only the prefix content differs: the drop from ``natural`` is the
    load-bearing test (2a).  Only meaningful for an arm trained with
    ``prefix_placeholder_fraction > 0``.
``no_pose``
    the pose tokens are blanked (``geometry`` mode only), natural decode.  Reads
    how much of the cognition the arm computes without odometry.

Decision classes read from the decoded first turn after the prefix is split
off: ``stop`` (contains STOP), ``turn_left``/``turn_right`` (leading arrow
run), ``lookdown`` (the released "↓" that precedes a pixel goal, i.e. "keep
walking"), ``other``.  Targets come from the relabel plan: ``correct_turn``
states want the oracle's arrow, ``correct_stop``/``keep_stop`` want STOP,
``keep_pixel`` wants exactly "↓", ``keep_turn`` wants native's own arrow.

Reported (per pass): recovery_turn_accuracy, stop_recall, stop_false_alarm,
stop_false_alarm_normal, preservation_generated (keep_pixel ∧ normal: the
decoded decision is exactly "↓"), nonpixel_on_normal (turn or STOP where native
kept walking), decision distribution.  For prefix arms: prefix well-formedness,
slot view / distance accuracy (micro and macro over the four views), progress
accuracy; placeholder drops; and the natural association: risk differences of
decision correctness given the relevant prefix component is right vs wrong,
with the irrelevant component as the specificity control, episode-clustered
bootstrap CIs.

Sharded runs merge with ``--merge``.  Nothing is trained here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA = "heatmapvln-exp17-system2-cognition-prefix-v1"
STOP_TEXT = "STOP"
ARROW_LEFT, ARROW_RIGHT, ARROW_DOWN = "←", "→", "↓"
STOP_TARGET_KINDS = ("correct_stop", "keep_stop")
IGNORE_INDEX = -100
VISION_START_TOKEN_ID = 151652  # Qwen2.5-VL <|vision_start|>; one per image in the prompt
_DIGITS = re.compile(r"\d")


def perturb_rel_poses_np(
    poses,
    *,
    translation_m: float,
    rotation_deg: float,
    ages,
    drift: bool,
    seed: int,
):
    """EXP-15's pose-noise model (numpy): additive metres, a yaw rotation of (cos, sin).

    Reads an arm trained on simulator-true poses at the deployment noise level
    before any closed loop is spent.  Deterministic per ``seed``.
    """
    import math

    import numpy as np

    out = np.array(poses, dtype=np.float32, copy=True)
    if out.ndim != 3 or out.shape[-1] != 4:
        raise ValueError(f"poses must be [B,K,4], got {out.shape}")
    batch, slots, _ = out.shape
    rng = np.random.default_rng(int(seed))
    if drift:
        if ages is None:
            raise ValueError("drift noise needs history_age_steps")
        scale = np.sqrt(1.0 + np.asarray(ages, dtype=np.float32).reshape(batch, slots))
    else:
        scale = np.ones((batch, slots), dtype=np.float32)
    if translation_m > 0.0:
        out[:, :, 0] += rng.normal(0.0, 1.0, size=(batch, slots)).astype(np.float32) * translation_m * scale
        out[:, :, 1] += rng.normal(0.0, 1.0, size=(batch, slots)).astype(np.float32) * translation_m * scale
    if rotation_deg > 0.0:
        delta = math.radians(rotation_deg) * scale * rng.normal(0.0, 1.0, size=(batch, slots)).astype(np.float32)
        cos_d, sin_d = np.cos(delta), np.sin(delta)
        cos_y, sin_y = out[:, :, 2].copy(), out[:, :, 3].copy()
        out[:, :, 2] = cos_y * cos_d - sin_y * sin_d
        out[:, :, 3] = sin_y * cos_d + cos_y * sin_d
    return out


def state_noise_seed(base_seed: int, sample_key: str) -> int:
    import hashlib

    digest = hashlib.md5(f"{int(base_seed)}:{sample_key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def prompt_inputs_from_batch(pano_inputs: dict[str, Any]) -> dict[str, Any]:
    """Cut a teacher-forced, batch-of-one row down to the prompt of its first assistant turn.

    The supervised span starts at the first label that is not IGNORE_INDEX; the
    prompt is everything before it.  Images that only appear later in the
    conversation (the look-down view of the second turn) must be dropped from
    ``pixel_values`` / ``image_grid_thw`` too, or Qwen refuses the mismatch
    between image tokens and image features.
    """
    import torch

    labels = pano_inputs["labels"][0]
    supervised = (labels != IGNORE_INDEX).nonzero(as_tuple=False)
    if supervised.numel() == 0:
        raise RuntimeError("no supervised position: the target text was not found in the tokenized row")
    prompt_len = int(supervised[0].item())
    input_ids = pano_inputs["input_ids"][:, :prompt_len]
    out: dict[str, Any] = {"input_ids": input_ids}
    for key in ("attention_mask", "mm_token_type_ids"):
        value = pano_inputs.get(key)
        if torch.is_tensor(value):
            out[key] = value[:, :prompt_len]
    grid = pano_inputs.get("image_grid_thw")
    pixels = pano_inputs.get("pixel_values")
    if torch.is_tensor(grid) and torch.is_tensor(pixels):
        n_images = int((input_ids[0] == VISION_START_TOKEN_ID).sum().item())
        if n_images > int(grid.shape[0]):
            raise RuntimeError(f"prompt references {n_images} images but the row carries {int(grid.shape[0])}")
        kept = grid[:n_images]
        n_patches = int(kept.prod(dim=1).sum().item()) if n_images else 0
        out["image_grid_thw"] = kept
        out["pixel_values"] = pixels[:n_patches]
    return out


# ----------------------------------------------------------------- decisions
def decision_of(text: str) -> tuple[str, str | None]:
    """Classify a decoded first turn (prefix already removed)."""
    stripped = text.strip()
    if STOP_TEXT in stripped:
        return "stop", None
    if stripped.startswith(ARROW_LEFT):
        return "turn_left", "left"
    if stripped.startswith(ARROW_RIGHT):
        return "turn_right", "right"
    if stripped.startswith(ARROW_DOWN):
        return "lookdown", None
    if _DIGITS.search(stripped):
        return "pixel", None
    return "other", None


def target_of(plan: dict[str, Any]) -> tuple[str, str | None]:
    kind = str(plan["kind"])
    first = str(plan["target_texts"][0])
    if kind in STOP_TARGET_KINDS:
        return "stop", None
    if kind in ("correct_turn", "keep_turn"):
        return ("turn_left", "left") if first.startswith(ARROW_LEFT) else ("turn_right", "right")
    return "lookdown", None


def decision_correct(predicted: tuple[str, str | None], target: tuple[str, str | None]) -> bool:
    if target[0] == "stop":
        return predicted[0] == "stop"
    if target[0].startswith("turn"):
        return predicted[1] == target[1]
    return predicted[0] == "lookdown"


# ------------------------------------------------------------------- scoring
def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return sum(1 for r in rows if r[key]) / len(rows) if rows else None


def score_prefix(fields: dict[str, Any] | None, truth: dict[str, Any] | None) -> dict[str, Any]:
    """Per-state prefix correctness against the rendered ground truth."""
    if truth is None:
        return {"prefix_wellformed": fields is not None}
    if fields is None:
        return {
            "prefix_wellformed": False,
            "progress_correct": False,
            "slot_view_hits": [],
            "slot_view_truth": [],
            "slot_dist_hits": [],
            "slot_view_acc": 0.0,
        }
    view_hits: list[bool] = []
    view_truth: list[str] = []
    dist_hits: list[bool] = []
    for predicted, expected in zip(fields["slots"], truth["slots"]):
        if not isinstance(expected, tuple):
            continue  # padded slot: nothing to grade
        ok_view = isinstance(predicted, tuple) and predicted[0] == expected[0]
        ok_dist = isinstance(predicted, tuple) and predicted[1] == expected[1]
        view_hits.append(bool(ok_view))
        view_truth.append(expected[0])
        dist_hits.append(bool(ok_dist))
    return {
        "prefix_wellformed": True,
        "progress_correct": fields["progress"] == truth["progress"],
        "progress_predicted": fields["progress"],
        "progress_truth": truth["progress"],
        "slot_view_hits": view_hits,
        "slot_view_truth": view_truth,
        "slot_dist_hits": dist_hits,
        "slot_view_acc": (sum(view_hits) / len(view_hits)) if view_hits else None,
    }


def _clustered_risk_difference(
    rows: list[dict[str, Any]],
    *,
    outcome: str,
    condition: str,
    cluster: str = "episode_key",
    draws: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """RD = P(outcome | condition true) - P(outcome | condition false), episode-clustered bootstrap."""
    import random

    usable = [r for r in rows if r.get(condition) is not None and r.get(outcome) is not None]

    def rd(subset: list[dict[str, Any]]) -> float | None:
        pos = [r for r in subset if r[condition]]
        neg = [r for r in subset if not r[condition]]
        if not pos or not neg:
            return None
        return sum(1 for r in pos if r[outcome]) / len(pos) - sum(1 for r in neg if r[outcome]) / len(neg)

    point = rd(usable)
    clusters: dict[str, list[dict[str, Any]]] = {}
    for r in usable:
        clusters.setdefault(str(r.get(cluster)), []).append(r)
    keys = list(clusters)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(draws if keys else 0):
        picked = [clusters[rng.choice(keys)] for _ in keys]
        value = rd([r for group in picked for r in group])
        if value is not None:
            samples.append(value)
    samples.sort()
    ci = (
        [samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]]
        if samples
        else None
    )
    return {
        "risk_difference": point,
        "ci95": ci,
        "n_condition_true": sum(1 for r in usable if r[condition]),
        "n_condition_false": sum(1 for r in usable if not r[condition]),
    }


def summarise(states: list[dict[str, Any]], passes: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"states": len(states), "passes": {}, "by_kind": {}}
    kinds = Counter(s["relabel_kind"] for s in states)
    report["by_kind"] = dict(kinds)
    for pass_name in passes:
        rows = [dict(s, **s["passes"][pass_name]) for s in states if pass_name in s["passes"]]
        turn = [r for r in rows if r["relabel_kind"] == "correct_turn"]
        stop = [r for r in rows if r["relabel_kind"] in STOP_TARGET_KINDS]
        non_stop = [r for r in rows if r["relabel_kind"] not in STOP_TARGET_KINDS]
        normal_keep = [r for r in rows if r["relabel_kind"] == "keep_pixel" and r["source_type"] == "dagger_normal"]
        keep = [r for r in rows if r["relabel_kind"] == "keep_pixel"]
        summary: dict[str, Any] = {
            "states": len(rows),
            "recovery_turn_accuracy": _rate(turn, "decision_correct"),
            "recovery_turn_states": len(turn),
            "stop_recall": _rate(stop, "predicted_is_stop"),
            "stop_recall_states": len(stop),
            "stop_false_alarm": _rate(non_stop, "predicted_is_stop"),
            "stop_false_alarm_states": len(non_stop),
            "stop_false_alarm_normal": _rate(normal_keep, "predicted_is_stop"),
            "preservation_generated": _rate(normal_keep, "decision_correct"),
            "preservation_generated_states": len(normal_keep),
            "preservation_generated_all_keep": _rate(keep, "decision_correct"),
            "nonpixel_on_normal": _rate(normal_keep, "predicted_nonpixel"),
            "decision_distribution": dict(Counter(r["decision"] for r in rows)),
            "decision_distribution_by_kind": {
                kind: dict(Counter(r["decision"] for r in rows if r["relabel_kind"] == kind)) for kind in kinds
            },
        }
        graded = [r for r in rows if r.get("prefix_truth_present")]
        if graded:
            view_hits = [h for r in graded for h in r.get("slot_view_hits", [])]
            view_truth = [t for r in graded for t in r.get("slot_view_truth", [])]
            per_view = {}
            for view in ("前", "右", "后", "左"):
                pairs = [h for h, t in zip(view_hits, view_truth) if t == view]
                per_view[view] = (sum(pairs) / len(pairs)) if pairs else None
            present = [v for v in per_view.values() if v is not None]
            progress_truth = Counter(r.get("progress_truth") for r in graded if r.get("progress_truth"))
            progress_macro = []
            for char in progress_truth:
                subset = [r for r in graded if r.get("progress_truth") == char]
                progress_macro.append(sum(1 for r in subset if r.get("progress_correct")) / len(subset))
            summary["prefix"] = {
                "wellformed": _rate(graded, "prefix_wellformed"),
                "slot_view_micro_acc": (sum(view_hits) / len(view_hits)) if view_hits else None,
                "slot_view_macro_acc": (sum(present) / len(present)) if present else None,
                "slot_view_per_view_acc": per_view,
                "slot_distance_micro_acc": (
                    lambda hits: (sum(hits) / len(hits)) if hits else None
                )([h for r in graded for h in r.get("slot_dist_hits", [])]),
                "progress_acc": _rate(graded, "progress_correct"),
                "progress_macro_acc": (sum(progress_macro) / len(progress_macro)) if progress_macro else None,
                "progress_truth_dist": dict(progress_truth),
            }
        report["passes"][pass_name] = summary

    natural = report["passes"].get("natural")
    if natural and "placeholder" in report["passes"]:
        placeholder = report["passes"]["placeholder"]
        report["placeholder_drop_pt"] = {
            key: (
                round(100.0 * (natural[key] - placeholder[key]), 2)
                if natural.get(key) is not None and placeholder.get(key) is not None
                else None
            )
            for key in ("recovery_turn_accuracy", "stop_recall", "stop_false_alarm", "preservation_generated")
        }
    if natural and "no_pose" in report["passes"]:
        no_pose = report["passes"]["no_pose"]
        report["no_pose_drop_pt"] = {
            key: (
                round(100.0 * (natural[key] - no_pose[key]), 2)
                if natural.get(key) is not None and no_pose.get(key) is not None
                else None
            )
            for key in ("recovery_turn_accuracy", "stop_recall", "stop_false_alarm")
        }

    # 2b: natural association between the relevant prefix component and the decision.
    nat_rows = [dict(s, **s["passes"]["natural"]) for s in states if "natural" in s["passes"] and s["passes"]["natural"].get("prefix_truth_present")]
    if nat_rows:
        stop_rows = [
            dict(r, stop_decision_correct=(r["predicted_is_stop"] == (r["relabel_kind"] in STOP_TARGET_KINDS)),
                 view_mostly_right=(r.get("slot_view_acc") is not None and r["slot_view_acc"] >= 0.5))
            for r in nat_rows
            if r["relabel_kind"] in STOP_TARGET_KINDS or r["relabel_kind"] == "keep_pixel"
        ]
        turn_rows = [
            dict(r, view_mostly_right=(r.get("slot_view_acc") is not None and r["slot_view_acc"] >= 0.5))
            for r in nat_rows
            if r["relabel_kind"] == "correct_turn"
        ]
        report["natural_association"] = {
            "stop_vs_progress (relevant)": _clustered_risk_difference(stop_rows, outcome="stop_decision_correct", condition="progress_correct"),
            "stop_vs_slot_views (specificity)": _clustered_risk_difference(stop_rows, outcome="stop_decision_correct", condition="view_mostly_right"),
            "turn_vs_slot_views (relevant)": _clustered_risk_difference(turn_rows, outcome="decision_correct", condition="view_mostly_right"),
            "turn_vs_progress (specificity)": _clustered_risk_difference(turn_rows, outcome="decision_correct", condition="progress_correct"),
        }
    return report


def merge(paths: list[Path], output: Path, passes: list[str]) -> None:
    states: list[dict[str, Any]] = []
    inputs = None
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        states.extend(payload["states"])
        inputs = inputs or payload.get("inputs")
    seen = {s["sample_key"] for s in states}
    if len(seen) != len(states):
        raise SystemExit("shards overlap: duplicate sample keys")
    report = summarise(states, passes)
    report.update({"schema": SCHEMA, "inputs": inputs, "merged_from": [str(p) for p in paths], "states_records": states})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "states_records"}, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merge", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="the arm's trained best.pth; omit to read the untrained parent")
    parser.add_argument("--parent-checkpoint", type=Path, default=None)
    parser.add_argument("--collection-root", type=Path, default=None)
    parser.add_argument("--oracle-views", type=Path, default=None)
    parser.add_argument("--reference-path-json", type=Path, default=None, help="overrides data.dagger_system2_sft.reference_path_json")
    parser.add_argument("--passes", default="natural,placeholder,no_pose")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--pose-noise-translation-m", type=float, default=0.0, help="EXP-15 noise on the input poses (all passes)")
    parser.add_argument("--pose-noise-rotation-deg", type=float, default=0.0)
    parser.add_argument("--pose-noise-no-drift", action="store_true", help="constant sigma instead of sqrt(1+age) drift")
    parser.add_argument("--pose-noise-seed", type=int, default=42)
    parser.add_argument("--max-states", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    passes = [p.strip() for p in args.passes.split(",") if p.strip()]

    if args.merge:
        merge(args.merge, args.output_json, passes)
        return
    for name in ("config", "parent_checkpoint", "collection_root", "oracle_views"):
        if getattr(args, name) is None:
            raise SystemExit(f"--{name.replace('_', '-')} is required unless --merge is used")

    import logging

    import torch
    from transformers import AutoProcessor

    from scripts.training.model_builder import build_model
    from scripts.training.pose_adaptation import load_pose_adaptation_initialization
    from scripts.training.utils import _load_normalized_state_dict, safe_torch_load
    from src.config_schema import load_and_validate_config
    from src.data.dagger_system2_sft import (
        DaggerSystem2SFTDataset,
        parse_cognition_prefix,
        placeholder_prefix,
    )
    from src.data.internnav_heatmap_control_collator import InternNavHeatmapControlCollator
    from src.data.trajectory_dagger_dataset import TrajectoryDaggerDataset

    logging.basicConfig(level=logging.INFO)
    cfg = load_and_validate_config(args.config)
    memory_cfg = cfg["model"].get("system2_memory") or {}
    if not memory_cfg.get("enabled", False):
        raise SystemExit("--config must be a System2 memory/geometry arm")
    mode = str(memory_cfg.get("mode", "memory"))
    sft_cfg = cfg["data"].get("dagger_system2_sft") or {}
    prefix_arm = bool(sft_cfg.get("cognition_prefix", False))
    reference_json = args.reference_path_json or sft_cfg.get("reference_path_json")
    if prefix_arm and not reference_json:
        raise SystemExit("a cognition-prefix arm needs --reference-path-json (or the config's reference_path_json)")
    if "placeholder" in passes and not prefix_arm:
        passes = [p for p in passes if p != "placeholder"]
    if "no_pose" in passes and mode != "geometry":
        passes = [p for p in passes if p != "no_pose"]
    if "natural" not in passes:
        raise SystemExit("the natural pass is mandatory")

    shard_paths = sorted(args.collection_root.glob("shard_*"))
    if not shard_paths:
        raise SystemExit(f"no shard_* directories under {args.collection_root}")
    fingerprints = {
        json.loads((shard / "collection_manifest.json").read_text(encoding="utf-8"))["contract"]["policy_fingerprint"]
        for shard in shard_paths
    }
    if len(fingerprints) != 1:
        raise SystemExit(f"shards disagree on the collecting policy: {sorted(fingerprints)}")

    reader = TrajectoryDaggerDataset(
        collection_roots=[str(shard) for shard in shard_paths],
        source_types=list(cfg["data"]["trajectory_dagger"]["source_types"]),
        num_history=int(cfg["data"]["trajectory_dagger"]["num_history"]),
        image_size=tuple(cfg["data"]["image_size"]),
        require_lookdown=True,
        expected_policy_mode="internnav_native",
        expected_policy_fingerprint=fingerprints.pop(),
    )
    dataset = DaggerSystem2SFTDataset(
        reader,
        oracle_views=args.oracle_views,
        max_turns=int(sft_cfg.get("max_turns", 4)),
        scene_split="val",
        val_scene_pct=int(sft_cfg.get("val_scene_pct", 25)),
        stop_supervision=bool(sft_cfg.get("stop_supervision", False)),
        stop_horizon_m=float(sft_cfg.get("stop_horizon_m", 1.0)),
        stop_oversample=1,
        cognition_prefix=prefix_arm,
        prefix_placeholder_fraction=0.0,
        reference_path_json=str(reference_json) if prefix_arm else None,
        prefix_distance_bins_m=list(sft_cfg.get("prefix_distance_bins_m", [2.0, 5.0])),
        prefix_progress_bins=int(sft_cfg.get("prefix_progress_bins", 4)),
    )
    print(f"val states: {len(dataset)} | {json.dumps(dataset.summary(), ensure_ascii=False)}", flush=True)

    processor = AutoProcessor.from_pretrained(
        cfg["model"]["llm"]["model_path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    tokenizer = processor.tokenizer
    num_tokens = int(memory_cfg.get("num_tokens", 8))
    collator = InternNavHeatmapControlCollator(
        processor,
        n_traj_query=0,
        max_seq_length=int(cfg["model"]["llm"].get("max_seq_length", 8192)),
        teacher_force_system2_answer=True,
        include_future_trajectory_targets=False,
        required_history_pose_provider=None,
        build_sft_labels=True,
        memory_token_count=num_tokens,
        memory_placeholder_position=str(memory_cfg.get("placeholder_position", "before_history")),
        internnav_conjunction="you can see ",
    )
    placeholder_ids = (
        tokenizer.encode(placeholder_prefix(num_tokens), add_special_tokens=False) if prefix_arm else None
    )

    device = torch.device("cuda:0")
    torch.manual_seed(42)
    model = build_model(cfg, verbose=True, device="cuda:0")
    if model.vlm_backbone.model is None:
        model.vlm_backbone._load_model()
    if bool((cfg["model"].get("heatmap") or {}).get("enable", False)):
        model._ensure_heatmap_vln()
        head_report = load_pose_adaptation_initialization(model, str(args.parent_checkpoint))
        print(f"frozen Past Head: {head_report['loaded_tensor_count']} tensors", flush=True)
    if args.checkpoint is not None:
        payload = safe_torch_load(str(args.checkpoint))
        trained = payload.get("trainable_state_dict") or {}
        if not trained:
            raise SystemExit(f"{args.checkpoint} carries no trainable_state_dict")
        _missing, unexpected, loaded = _load_normalized_state_dict(model, trained)
        if unexpected:
            raise SystemExit(f"arm checkpoint has tensors this model does not own: {unexpected[:8]}")
        print(f"trained arm: {loaded}/{len(trained)} tensors loaded", flush=True)
    else:
        print("no --checkpoint: reading the untrained arm (parent weights, fresh memory tokens)", flush=True)
    model.requires_grad_(False)
    model.eval()

    def memory_embeds_for(batch: dict[str, Any], sample: dict[str, Any], *, force_no_pose: bool) -> torch.Tensor:
        mask = batch["history_valid_mask"].to(device)
        if mode == "geometry":
            return model.system2_memory(
                None, mask, history_rel_poses=batch["history_rel_poses"].to(device), force_no_pose=force_no_pose
            )
        if mode == "constant":
            return model.system2_memory(None, None, batch_size=1)
        output = model(
            video_frames=None,
            instruction_text=[sample["text"]],
            current_observation=batch["current_frame"].to(device) if "current_frame" in batch else None,
            panoramic_inputs=batch.get("pano_inputs"),
            panoramic_num_histories=batch.get("pano_num_histories"),
            heatmap_single_view_inputs=batch.get("heatmap_single_view_inputs"),
            heatmap_single_view_num_histories=batch.get("heatmap_single_view_num_histories"),
            heatmap_control_history_mask=batch.get("heatmap_control_history_mask"),
            history_valid_mask=batch.get("history_valid_mask"),
            history_age_steps=batch.get("history_age_steps"),
            history_rel_poses=batch["history_rel_poses"].to(device),
            sample_trajectory=False,
            return_heatmaps=False,
            return_actions=False,
            return_future_heatmaps=False,
            return_history_memory=True,
            inject_system2_memory=False,
        )
        return model.system2_memory(output["history_memory"], output["history_memory_mask"])

    def prompt_inputs_from(batch: dict[str, Any]) -> dict[str, Any]:
        return prompt_inputs_from_batch(batch["pano_inputs"])

    indices = [i for i in range(len(dataset)) if i % args.shard_count == args.shard_index]
    budget = len(indices)
    if args.max_states > 0:
        budget = min(budget, max(1, args.max_states // max(1, args.shard_count)))

    states: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for visited, index in enumerate(indices):
            if len(states) >= budget:
                break
            sample = dataset[index]
            plan = dataset.plans[index]
            batch = collator([sample])
            if args.pose_noise_translation_m > 0.0 or args.pose_noise_rotation_deg > 0.0:
                noisy = perturb_rel_poses_np(
                    batch["history_rel_poses"].cpu().numpy(),
                    translation_m=args.pose_noise_translation_m,
                    rotation_deg=args.pose_noise_rotation_deg,
                    ages=batch["history_age_steps"].cpu().numpy() if batch.get("history_age_steps") is not None else None,
                    drift=not args.pose_noise_no_drift,
                    seed=state_noise_seed(args.pose_noise_seed, plan["sample_key"]),
                )
                batch["history_rel_poses"] = torch.from_numpy(noisy)
            prompt = prompt_inputs_from(batch)
            truth_fields = None
            if prefix_arm:
                truth_fields, _ = parse_cognition_prefix(sample["cognition_prefix_truth"])
            target = target_of(plan)
            record: dict[str, Any] = {
                "sample_key": plan["sample_key"],
                "scene_id": plan["scene_id"],
                "episode_key": str(sample.get("episode_key") or plan["sample_key"].split(":")[0]),
                "source_type": plan["source_type"],
                "relabel_kind": plan["kind"],
                "failure_tags": plan["failure_tags"],
                "target_texts": plan["target_texts"],
                "target_decision": target[0],
                "prefix_truth": sample.get("cognition_prefix_truth"),
                "passes": {},
            }
            for pass_name in passes:
                embeds = memory_embeds_for(batch, sample, force_no_pose=(pass_name == "no_pose"))
                forced = placeholder_ids if pass_name == "placeholder" else None
                new_ids = model.vlm_backbone.generate_with_sentinels(
                    prompt, memory_embeds=embeds, max_new_tokens=args.max_new_tokens, forced_prefix_ids=forced
                )
                text = tokenizer.decode(new_ids[0].tolist(), skip_special_tokens=True)
                if prefix_arm and pass_name != "placeholder":
                    fields, rest = parse_cognition_prefix(text)
                else:
                    fields, rest = None, text
                predicted = decision_of(rest)
                entry: dict[str, Any] = {
                    "generated_text": text,
                    "decision_text": rest.strip(),
                    "decision": predicted[0],
                    "decision_direction": predicted[1],
                    "decision_correct": decision_correct(predicted, target),
                    "predicted_is_stop": predicted[0] == "stop",
                    "predicted_nonpixel": predicted[0] in ("stop", "turn_left", "turn_right"),
                    "prefix_truth_present": bool(prefix_arm and pass_name != "placeholder"),
                }
                if prefix_arm and pass_name != "placeholder":
                    entry.update(score_prefix(fields, truth_fields))
                record["passes"][pass_name] = entry
            states.append(record)
            if len(states) % max(1, args.progress_every) == 0:
                rate = len(states) / (time.perf_counter() - started)
                print(f"  {len(states)}/{budget} scored ({visited + 1} visited, {rate:.2f} states/s)", flush=True)

    report = summarise(states, passes)
    report.update(
        {
            "schema": SCHEMA,
            "inputs": {
                "config": str(args.config),
                "checkpoint": str(args.checkpoint),
                "parent_checkpoint": str(args.parent_checkpoint),
                "collection_root": str(args.collection_root),
                "oracle_views": str(args.oracle_views),
                "mode": mode,
                "prefix_arm": prefix_arm,
                "passes": passes,
                "max_new_tokens": args.max_new_tokens,
                "shard": [args.shard_index, args.shard_count],
                "placeholder_fraction_trained": float(sft_cfg.get("prefix_placeholder_fraction", 0.0)),
                "pose_noise": {
                    "translation_m": args.pose_noise_translation_m,
                    "rotation_deg": args.pose_noise_rotation_deg,
                    "drift": not args.pose_noise_no_drift,
                    "seed": args.pose_noise_seed,
                },
            },
            "states_records": states,
            "states": states,
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("states_records", "states")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

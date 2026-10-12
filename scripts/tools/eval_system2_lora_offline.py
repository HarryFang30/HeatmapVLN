#!/usr/bin/env python3
"""EXP-23 offline gates for a System2 LoRA adapter, on held-out scenes (design §3).

Every call recorded in the held-out scenes is decoded the way the server decodes
it (greedy; turn 2 after a lookdown request), twice: once with the adapter on,
which is what EXP-23 would deploy, and once with it off, which is the released
model.

=====  ==========================================================  ===========
gate   quantity (adapter on)                                       line
=====  ==========================================================  ===========
G0     adapter off: answer == the recorded answer (replay)          >= 0.98
G1     agreeing states: answer == the model's recorded answer       >= 0.90
G2     calls with d > 3 m: share answered STOP                      <= 0.002
G3     calls with d <= R_stop: share answered STOP                  report only
G4     disagreeing states: answer of the oracle's class (turns:     report only
       same direction; pixels: back-projected through the stored
       lookdown depth to within 1 m of the oracle path)
=====  ==========================================================  ===========

G0 proper is ``train_system2_lora.py --parity-check``.  The adapter-off replay
here is the same measurement on these calls, so a gate report carries its own
input-alignment check.  Every metric is also reported with the adapter off, as
the reference.

The calls are sharded over cards under torchrun.  Rank 0 writes ``--output``
(json) and ``<output>.calls.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from scripts.train_system2_lora import (  # noqa: E402
    G0_THRESHOLD,
    _stable_order,
    adapter_off,
    base_model_identity,
    gather_rank_rows,
    greedy_native_decision,
    init_runtime,
    load_adapter,
    load_system2,
    provenance,
    sha256_file,
    shard_rows,
    shared_index,
    write_json,
)
from src.data.s2_records import CAMERA  # noqa: E402
from src.data.s2_sft_dataset import (  # noqa: E402
    DEFAULT_HOLDOUT_COUNT,
    DEFAULT_VLM_IMAGE_SIZE,
    HOLDOUT_SALT,
    CallRef,
    RecordCache,
    decision_class,
    holdout_scenes,
    index_calls,
    inference_messages,
    pixel_uv,
)

LOGGER = logging.getLogger("eval_system2_lora_offline")

GATES_SCHEMA = "heatmapvln-s2-lora-offline-gates-v1"
G1_THRESHOLD = 0.90
G2_THRESHOLD = 0.002
DEFAULT_R_STOP_M = 2.0
DEFAULT_FALSE_STOP_M = 3.0
DEFAULT_PATH_TOLERANCE_M = 1.0


# ---------------------------------------------------------------------------
# Geometry (G4 only)
# ---------------------------------------------------------------------------


def quat_wxyz_to_rot(q: Sequence[float]) -> np.ndarray:
    """3x3 rotation of a (w, x, y, z) quaternion, normalised first (as scripts/exp19/gt.py)."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def lookdown_pixel_to_world(
    row: int,
    col: int,
    depth_m: np.ndarray,
    position: Sequence[float],
    rotation_wxyz: Sequence[float],
    camera: Mapping[str, float] = CAMERA,
) -> np.ndarray | None:
    """World point under lookdown pixel (row, col), or None without a depth return.

    Habitat conventions (scripts/exp19/gt.py): y up, the camera looks along -z
    with x right, the sensor sits ``height_m`` above the body origin, and the
    lookdown is that camera pitched down by ``lookdown_pitch_deg`` (camera-to-body
    rotation R_x(-pitch)).  Depth is z-depth.  The image has square pixels with
    fx = (W / 2) / tan(HFOV / 2), and pixel (col, row) is centred at
    (col + 0.5, row + 0.5).
    """
    height, width = depth_m.shape[:2]
    if not (0 <= row < height and 0 <= col < width):
        return None
    depth = float(depth_m[row, col])
    if not math.isfinite(depth) or depth <= 0.0:
        return None
    fx = (float(camera["width"]) / 2.0) / math.tan(math.radians(float(camera["hfov_deg"])) / 2.0)
    # Depth stored at another resolution than the camera: scale the intrinsics with it.
    sx, sy = width / float(camera["width"]), height / float(camera["height"])
    u = (col + 0.5 - width / 2.0) / (fx * sx)
    v = (row + 0.5 - height / 2.0) / (fx * sy)
    point_cam = np.array([u * depth, -v * depth, -depth])
    pitch = math.radians(-float(camera["lookdown_pitch_deg"]))
    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(pitch), -math.sin(pitch)],
        [0.0, math.sin(pitch), math.cos(pitch)],
    ])
    body = quat_wxyz_to_rot(rotation_wxyz)
    origin = np.asarray(position, dtype=np.float64) + body @ np.array([0.0, float(camera["height_m"]), 0.0])
    return body @ rot_x @ point_cam + origin


def distance_to_polyline(point: np.ndarray, path: Sequence[Sequence[float]]) -> float:
    points = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    if len(points) == 1:
        return float(np.linalg.norm(point - points[0]))
    a, b = points[:-1], points[1:]
    ab = b - a
    denom = np.maximum((ab * ab).sum(axis=1), 1e-12)
    t = np.clip(((point - a) * ab).sum(axis=1) / denom, 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None] * ab - point, axis=1).min())


def call_pose(record, call: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    """Ground-truth body pose at the call: the call row's own, else its capture step's."""
    for source in (call, call.get("pose") or {}):
        if "position" in source and "rotation_wxyz" in source:
            return list(source["position"]), list(source["rotation_wxyz"])
    step = record.step(call["step_id"])
    return list(step["position"]), list(step["rotation_wxyz"])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_answer(
    turns: Sequence[str],
    ref: CallRef,
    *,
    pixel_world: Any,
    path_tolerance_m: float,
) -> dict[str, Any]:
    """One decoded answer against the record and the label.

    ``pixel_world(text)`` returns the world point of a pixel answer, or None when
    it cannot be placed (no depth return, no oracle path).
    """
    turns = tuple(turns)
    cls = decision_class(turns[-1])
    scored: dict[str, Any] = {
        "turns": list(turns),
        "class": cls,
        "stop": cls == "stop",
        "exact_recorded": turns == tuple(ref.recorded),
    }
    label = ref.label
    if label is None or label.agree is not False or label.oracle_turns is None:
        return scored
    oracle_class = decision_class(label.oracle_turns[-1])
    scored["oracle_class"] = oracle_class
    scored["oracle_class_match"] = cls == oracle_class
    if cls == "pixel" and oracle_class == "pixel":
        distance = None
        point = pixel_world(turns[-1])
        if point is not None and label.oracle_path:
            distance = distance_to_polyline(point, label.oracle_path)
        scored["pixel_to_oracle_path_m"] = None if distance is None else round(distance, 4)
        scored["corrected"] = None if distance is None else bool(distance <= path_tolerance_m)
    else:
        scored["corrected"] = bool(cls == oracle_class)
    return scored


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def summarize_mode(
    rows: Sequence[Mapping[str, Any]], mode: str, *, r_stop_m: float, false_stop_m: float, device: str
) -> dict[str, Any]:
    """Gate readings for one decode mode ("on" / "off").

    G0 replays only calls recorded on ``device`` (the platform this runs on): a CUDA
    answer replayed on an NPU differs by a few pixels about 5% of the time, which says
    nothing about whether the training inputs are aligned.  G1 asks whether the adapter
    keeps the answers that were already right, so it compares the adapter-on answer
    with the adapter-off answer on this same machine; the comparison with the recorded
    answer stays as a report-only reading.
    """
    def distance(row: Mapping[str, Any]) -> float | None:
        return row.get("distance_to_goal")

    every = [row[mode] for row in rows]
    same_device = [row[mode] for row in rows if row.get("record_device") == device]
    agree_rows = [row for row in rows if row.get("agree") is True]
    agree = [row[mode] for row in agree_rows]
    kept = sum(bool(row.get("on_off_identical")) for row in agree_rows) if mode == "on" else len(agree_rows)
    far = [row[mode] for row in rows if distance(row) is not None and distance(row) > false_stop_m]
    near = [row[mode] for row in rows if distance(row) is not None and distance(row) <= r_stop_m]
    disagree = [row[mode] for row in rows if row.get("agree") is False]
    scored = [a for a in disagree if a.get("corrected") is not None]

    def count(answers: Sequence[Mapping[str, Any]], flag: str) -> int:
        return sum(bool(a.get(flag)) for a in answers)

    return {
        "n": len(every),
        "replay_exact": {"n": len(every), "matched": count(every, "exact_recorded"),
                         "rate": _rate(count(every, "exact_recorded"), len(every))},
        "stop_rate": _rate(count(every, "stop"), len(every)),
        "replay_exact_same_device": {"device": device, "n": len(same_device),
                                     "matched": count(same_device, "exact_recorded"),
                                     "rate": _rate(count(same_device, "exact_recorded"), len(same_device))},
        "G1_keep": {"n": len(agree_rows), "matched": kept, "rate": _rate(kept, len(agree_rows)),
                    "compares": "adapter on vs adapter off, same machine"},
        "G1_keep_vs_recorded": {"n": len(agree), "matched": count(agree, "exact_recorded"),
                                "rate": _rate(count(agree, "exact_recorded"), len(agree))},
        "G2_false_stop": {"n": len(far), "stops": count(far, "stop"),
                          "rate": _rate(count(far, "stop"), len(far)), "distance_gt_m": false_stop_m},
        "G3_stop_recall": {"n": len(near), "stops": count(near, "stop"),
                           "rate": _rate(count(near, "stop"), len(near)), "distance_le_m": r_stop_m},
        "G4_correction": {
            "n": len(disagree),
            "class_match": count(disagree, "oracle_class_match"),
            "class_match_rate": _rate(count(disagree, "oracle_class_match"), len(disagree)),
            "scored": len(scored),
            "corrected": count(scored, "corrected"),
            "corrected_rate": _rate(count(scored, "corrected"), len(scored)),
            # pixel against pixel without a depth return or an oracle_path, or no oracle answer
            "unscored": len(disagree) - len(scored),
        },
        "calls_without_distance": sum(1 for row in rows if distance(row) is None),
    }


def gate_report(on: Mapping[str, Any], off: Mapping[str, Any]) -> dict[str, Any]:
    def gate(value: float | None, threshold: float, upper: bool) -> dict[str, Any]:
        passed = value is not None and (value <= threshold if upper else value >= threshold)
        return {"value": value, "threshold": threshold, "direction": "<=" if upper else ">=", "pass": bool(passed)}

    gates = {
        "G0_adapter_off_replay": gate(off["replay_exact_same_device"]["rate"], G0_THRESHOLD, upper=False),
        "G1_keep": gate(on["G1_keep"]["rate"], G1_THRESHOLD, upper=False),
        "G2_false_stop": gate(on["G2_false_stop"]["rate"], G2_THRESHOLD, upper=True),
    }
    gates["all_pass"] = all(g["pass"] for g in gates.values())
    gates["report_only"] = {"G3_stop_recall": on["G3_stop_recall"]["rate"], "G4_corrected": on["G4_correction"]["corrected_rate"]}
    return gates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _adapter_manifest(adapter_dir: Path) -> dict[str, Any]:
    path = adapter_dir.parent / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _holdout(args: argparse.Namespace, manifest: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    if args.holdout_scenes:
        return tuple(sorted(s.strip() for s in args.holdout_scenes.split(",") if s.strip())), "cli"
    scenes = (manifest.get("holdout") or {}).get("scenes")
    if scenes:
        return tuple(sorted(scenes)), "adapter manifest"
    return holdout_scenes(args.holdout_count, salt=args.holdout_salt), "salted hash"


def evaluate(args: argparse.Namespace) -> int:
    rt = init_runtime(args.device)
    output = Path(args.output)
    work_dir = output.with_name(output.stem + "_work")
    if rt.is_main:
        work_dir.mkdir(parents=True, exist_ok=True)
    rt.barrier()
    adapter_dir = Path(args.adapter)
    manifest = _adapter_manifest(adapter_dir)
    holdout, holdout_source = _holdout(args, manifest)
    refs, index_stats = shared_index(
        rt,
        work_dir / "eval_index.pkl",
        lambda: index_calls(args.record_root, args.label_root, split="holdout", holdout=holdout, require_labels=False),
    )
    by_uid = {ref.uid: ref for ref in refs}
    order = _stable_order(by_uid, args.seed)
    picks = [by_uid[k] for k in (order[: args.max_calls] if args.max_calls else order)]

    model, processor = load_system2(args.model_path, rt.device)
    model = load_adapter(model, adapter_dir)
    records = RecordCache(8)
    rows: list[dict[str, Any]] = []
    for ref in shard_rows(picks, rt):
        record, calls = records.get(ref.record_path)
        call = calls[ref.call_index]
        messages, images, lookdown = inference_messages(
            record, call, default_vlm_image_size=tuple(args.default_vlm_image_size)
        )
        started = time.time()
        on = greedy_native_decision(model, processor, messages, images, lookdown, rt.device)
        with adapter_off(model):
            off = greedy_native_decision(model, processor, messages, images, lookdown, rt.device)
        seconds = round(time.time() - started, 3)

        depth = record.lookdown_depth(ref.call_index) if record.has_lookdown_depth(ref.call_index) else None
        position, rotation = call_pose(record, call)

        def pixel_world(text: str, depth=depth, position=position, rotation=rotation):
            uv = pixel_uv(text)
            if depth is None or uv is None:
                return None
            return lookdown_pixel_to_world(uv[1], uv[0], depth, position, rotation)

        label = ref.label
        rows.append({
            "key": ref.key,
            "record_device": ref.record_device,
            "uid": ref.uid,
            "scene_id": ref.scene_id,
            "kind": call.get("kind"),
            "distance_to_goal": ref.distance_to_goal,
            "labelled": label is not None,
            "skip": bool(label.skip) if label is not None else None,
            "agree": label.agree if label is not None else None,
            "oracle_turns": list(label.oracle_turns) if label is not None and label.oracle_turns else None,
            "recorded": list(ref.recorded),
            "on": score_answer(on, ref, pixel_world=pixel_world, path_tolerance_m=args.path_tolerance),
            "off": score_answer(off, ref, pixel_world=pixel_world, path_tolerance_m=args.path_tolerance),
            "on_off_identical": tuple(on) == tuple(off),
            "seconds": seconds,
        })
    merged = gather_rank_rows(rows, rt, work_dir / "shards", "eval")
    if rt.is_main:
        summaries = {
            mode: summarize_mode(merged, mode, r_stop_m=args.r_stop, false_stop_m=args.false_stop_distance,
                                 device=rt.device.type)
            for mode in ("on", "off")
        }
        report = {
            "schema": GATES_SCHEMA,
            "created_unix": time.time(),
            "adapter": {
                "dir": str(adapter_dir.resolve()),
                "adapter_model_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
                "manifest_run_key": (manifest.get("training") or {}).get("run_key"),
            },
            "base_model": base_model_identity(args.model_path),
            "provenance": provenance(),
            "holdout": {"scenes": list(holdout), "source": holdout_source},
            "index": dict(index_stats),
            "calls_evaluated": len(merged),
            "labelled": sum(1 for r in merged if r["labelled"]),
            "skipped_by_label": sum(1 for r in merged if r["skip"]),
            "on_off_identical_rate": _rate(sum(r["on_off_identical"] for r in merged), len(merged)),
            "adapter_on": summaries["on"],
            "adapter_off": summaries["off"],
            "gates": gate_report(summaries["on"], summaries["off"]),
            "settings": {
                "r_stop_m": args.r_stop,
                "false_stop_distance_m": args.false_stop_distance,
                "path_tolerance_m": args.path_tolerance,
                "max_calls": args.max_calls,
                "seed": args.seed,
                "world_size": rt.world_size,
                "device": rt.device.type,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
        with open(output.with_suffix(".calls.jsonl"), "w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        print(json.dumps({"calls": len(merged), "gates": report["gates"],
                          "on_off_identical_rate": report["on_off_identical_rate"]}, ensure_ascii=False), flush=True)
    rt.barrier()
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record-root", action="append", required=True)
    parser.add_argument("--label-root", action="append", default=None)
    parser.add_argument("--adapter", required=True, help="PEFT directory (adapter_config.json + adapter_model.safetensors)")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True, help="gate report json")
    parser.add_argument("--holdout-scenes", default="", help="default: the adapter manifest's, else the salted hash")
    parser.add_argument("--holdout-count", type=int, default=DEFAULT_HOLDOUT_COUNT)
    parser.add_argument("--holdout-salt", default=HOLDOUT_SALT)
    parser.add_argument("--max-calls", type=int, default=0, help="seeded subset of the held-out calls (0 = all)")
    parser.add_argument("--r-stop", type=float, default=DEFAULT_R_STOP_M)
    parser.add_argument("--false-stop-distance", type=float, default=DEFAULT_FALSE_STOP_M)
    parser.add_argument("--path-tolerance", type=float, default=DEFAULT_PATH_TOLERANCE_M)
    parser.add_argument("--default-vlm-image-size", type=int, nargs=2, default=list(DEFAULT_VLM_IMAGE_SIZE))
    parser.add_argument("--device", choices=("auto", "npu", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    try:
        return evaluate(args)
    finally:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""EXP-25 probe B3: does the released System2 agree with InternData-N1's experts?

Every sample is built exactly as the EXP-23 replay builder builds it:
``scene_samples`` picks the frames and the expert target
(``src/data/interndata_replay.py``), ``sample_jpegs`` turns the frames into the
JPEG bytes the client would send (as ``build_s2_train_stream.py`` packs them),
and ``packed_inputs`` decodes them into the first-turn messages the trainer and
the server build (``build_native_messages``).  The base System2 -- loaded as the
server and the offline gates load it (``load_system2``: bf16, SDPA, no adapter)
-- answers with the server's greedy two-turn decode (``greedy_native_decision``).
Teacher-forced: the history is the expert's own trajectory, not the model's.

The answer's decision class (``decision_class`` of the final turn: ``pixel``,
``turn_left``, ``turn_right``, ``stop``; the model can also say ``forward``,
``lookdown`` or ``invalid``, which no expert target is) is compared with the
expert target's.  For pixel against pixel the distance between the two (u, v)
points on the 640x480 lookdown is reported too.

The pre-registered reading (docs/experiments/README.md, EXP-25 B3) is the
overall agreement: <= 0.93 GO, >= 0.97 NO-GO, in between undecided.

    python scripts/tools/teacher_forced_agreement.py \\
        --scene-root /workspace/exp23/data/interndata_n1/r2r --scenes 17DRP5sb8fy,1LXtFkjw3qL \\
        --model-path /workspace/InternNav_Model --output /workspace/exp23/b3/r2r_check.json \\
        --max-samples-per-scene 300

``--dry-run`` loads no model: it counts the samples and, with
``--check-encoding N``, runs the trainer's prompt-prefix check
(``encode_conversation``) on N of them with the processor alone.

Outputs: ``--output`` (summary json) and ``<output>.calls.jsonl`` (one row per
sample).  The rows file is also the resume log: a rerun skips the keys it holds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.data.interndata_replay import ReplaySample, sample_jpegs, scene_samples  # noqa: E402
from src.data.s2_sft_dataset import DEFAULT_VLM_IMAGE_SIZE, decision_class, pixel_uv  # noqa: E402

LOGGER = logging.getLogger("teacher_forced_agreement")

SCHEMA = "heatmapvln-s2-teacher-forced-agreement-v1"
# docs/experiments/README.md EXP-25, B3 (written 2026-10-15, before any result).
B3_GO_AT_MOST = 0.93
B3_NO_GO_AT_LEAST = 0.97
EXPERT_CLASSES = ("pixel", "turn_left", "turn_right", "stop")
PIXEL_RADII = (10, 25, 50, 100)
BOOTSTRAP_RESAMPLES = 2000


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


def sample_turns(sample: ReplaySample) -> tuple[str, ...]:
    return (sample.turn1,) + ((sample.turn2,) if sample.turn2 is not None else ())


def sample_key(sample: ReplaySample) -> str:
    """``build_s2_train_stream._replay_item``'s key."""
    digest = hashlib.sha1(sample.instruction.encode()).hexdigest()[:8]
    return f"replay:{sample.scene_id}_{sample.episode_index:06d}@{sample.frame_id}:{digest}"


def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def pick_samples(
    samples: Iterable[ReplaySample], *, max_per_scene: int, seed: int
) -> tuple[list[ReplaySample], dict[str, Any]]:
    """All samples, deduplicated by key, or a seeded uniform subset of each scene's."""
    by_scene: dict[str, dict[str, ReplaySample]] = defaultdict(dict)
    duplicates = 0
    for sample in samples:
        key = sample_key(sample)
        if key in by_scene[sample.scene_id]:
            duplicates += 1  # an instruction repeated within one episode's task string
            continue
        by_scene[sample.scene_id][key] = sample
    picks: list[ReplaySample] = []
    stats: dict[str, Any] = {"duplicate_keys": duplicates, "scenes": {}}
    for scene in sorted(by_scene):
        keyed = by_scene[scene]
        order = sorted(keyed, key=lambda k: _rank(seed, k))
        chosen = order[:max_per_scene] if max_per_scene else order
        picks.extend(keyed[k] for k in sorted(chosen))
        stats["scenes"][scene] = {
            "samples": len(keyed),
            "picked": len(chosen),
            "episodes": len({s.episode_index for s in keyed.values()}),
            "expert_classes": dict(Counter(decision_class(sample_turns(s)[-1]) for s in keyed.values())),
        }
    return picks, stats


def packed_sample(scene_dir: Path, sample: ReplaySample, vlm_image_size: tuple[int, int], jpeg_quality: int):
    """The ``PackedSample`` ``build_s2_train_stream._pack`` writes for this replay sample."""
    from src.data.s2_stream import PackedSample

    history, current, lookdown = sample_jpegs(
        scene_dir, sample, vlm_image_size=vlm_image_size, jpeg_quality=jpeg_quality
    )
    return PackedSample(
        key=sample_key(sample),
        source="replay",
        instruction=sample.instruction,
        turns=sample_turns(sample),
        vlm_image_size=tuple(vlm_image_size),
        history_jpegs=tuple(history),
        current_jpeg=current,
        lookdown_jpeg=lookdown,
    )


def model_inputs(scene_dir: Path, sample: ReplaySample, vlm_image_size: tuple[int, int], jpeg_quality: int):
    """(first-turn messages, prompt images, lookdown) exactly as the trainer reads a packed replay sample."""
    from src.data.s2_stream import packed_inputs

    return packed_inputs(packed_sample(scene_dir, sample, vlm_image_size, jpeg_quality))


def prefetched(fn, items: Sequence[Any], *, workers: int, window: int) -> Iterator[tuple[Any, Any]]:
    """``(item, fn(item))`` in order, with at most ``window`` results built ahead."""
    if workers <= 0:
        for item in items:
            yield item, fn(item)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: deque = deque()
        it = iter(items)
        for item in it:
            pending.append((item, pool.submit(fn, item)))
            if len(pending) >= window:
                break
        end = object()
        while pending:
            item, future = pending.popleft()
            nxt = next(it, end)
            if nxt is not end:
                pending.append((nxt, pool.submit(fn, nxt)))
            yield item, future.result()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score(expert: Sequence[str], answer: Sequence[str]) -> dict[str, Any]:
    expert_class = decision_class(expert[-1])
    model_class = decision_class(answer[-1])
    row: dict[str, Any] = {
        "expert": list(expert),
        "answer": list(answer),
        "expert_class": expert_class,
        "model_class": model_class,
        "class_match": expert_class == model_class,
        "exact": tuple(expert) == tuple(answer),
    }
    if expert_class == "pixel" and model_class == "pixel":
        e, m = pixel_uv(expert[-1]), pixel_uv(answer[-1])
        row["pixel_distance_px"] = round(math.hypot(m[0] - e[0], m[1] - e[1]), 3)
    return row


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def cluster_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0
) -> list[float] | None:
    """95% interval of the agreement rate, resampling whole episodes (samples of one episode are not independent)."""
    if not rows:
        return None
    clusters: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        c = clusters[(row["scene_id"], int(row["episode_index"]))]
        c[0] += int(bool(row["class_match"]))
        c[1] += 1
    hits = np.array([c[0] for c in clusters.values()], dtype=np.float64)
    counts = np.array([c[1] for c in clusters.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(hits), size=(resamples, len(hits)))
    rates = hits[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return [round(float(np.quantile(rates, 0.025)), 6), round(float(np.quantile(rates, 0.975)), 6)]


def b3_reading(rate: float | None) -> str:
    if rate is None:
        return "no data"
    if rate <= B3_GO_AT_MOST:
        return "GO"
    if rate >= B3_NO_GO_AT_LEAST:
        return "NO-GO"
    return "undecided"


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    agree = sum(bool(r["class_match"]) for r in rows)
    overall = _rate(agree, n)
    by_class: dict[str, Any] = {}
    for cls in sorted({r["expert_class"] for r in rows}, key=lambda c: (c not in EXPERT_CLASSES, c)):
        subset = [r for r in rows if r["expert_class"] == cls]
        hit = sum(bool(r["class_match"]) for r in subset)
        by_class[cls] = {
            "n": len(subset),
            "agree": hit,
            "rate": _rate(hit, len(subset)),
            "model_classes": dict(sorted(Counter(r["model_class"] for r in subset).items())),
            "exact": _rate(sum(bool(r["exact"]) for r in subset), len(subset)),
        }
    macro = [v["rate"] for v in by_class.values() if v["rate"] is not None]
    by_scene = {}
    for scene in sorted({r["scene_id"] for r in rows}):
        subset = [r for r in rows if r["scene_id"] == scene]
        hit = sum(bool(r["class_match"]) for r in subset)
        by_scene[scene] = {
            "n": len(subset),
            "episodes": len({r["episode_index"] for r in subset}),
            "rate": _rate(hit, len(subset)),
            "by_class": {
                cls: _rate(sum(bool(r["class_match"]) for r in subset if r["expert_class"] == cls),
                           sum(1 for r in subset if r["expert_class"] == cls))
                for cls in EXPERT_CLASSES
            },
        }
    distances = np.array([r["pixel_distance_px"] for r in rows if "pixel_distance_px" in r], dtype=np.float64)
    pixel: dict[str, Any] = {"n": int(distances.size)}
    if distances.size:
        pixel.update({
            "mean_px": round(float(distances.mean()), 3),
            "median_px": round(float(np.median(distances)), 3),
            "p75_px": round(float(np.quantile(distances, 0.75)), 3),
            "p90_px": round(float(np.quantile(distances, 0.90)), 3),
            **{f"within_{r}px": round(float((distances <= r).mean()), 6) for r in PIXEL_RADII},
        })
    seconds = [float(r["seconds"]) for r in rows if r.get("seconds") is not None]
    return {
        "n": n,
        "episodes": len({(r["scene_id"], r["episode_index"]) for r in rows}),
        "agree": agree,
        "overall_rate": overall,
        "overall_ci95_episode_bootstrap": cluster_bootstrap_ci(rows),
        "macro_rate_over_expert_classes": round(float(np.mean(macro)), 6) if macro else None,
        "exact_rate": _rate(sum(bool(r["exact"]) for r in rows), n),
        "by_expert_class": by_class,
        "by_scene": by_scene,
        "pixel_vs_pixel": pixel,
        "b3": {"rule": f"overall <= {B3_GO_AT_MOST} GO, >= {B3_NO_GO_AT_LEAST} NO-GO",
               "reading": b3_reading(overall)},
        "decode_seconds": {"total": round(sum(seconds), 1),
                           "mean": round(sum(seconds) / len(seconds), 4) if seconds else None},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _scene_dirs(args: argparse.Namespace) -> list[Path]:
    root = Path(args.scene_root)
    if args.scenes:
        dirs = [root / s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))
    missing = [str(d) for d in dirs if not (d / "meta" / "episodes.jsonl").is_file()]
    if missing:
        raise SystemExit(f"not extracted scene directories: {missing}")
    return dirs


def _read_rows(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["key"]] = row
    return rows


def run(args: argparse.Namespace) -> int:
    started = time.time()
    scene_dirs = _scene_dirs(args)
    dir_of = {d.name: d for d in scene_dirs}
    all_samples: list[ReplaySample] = []
    for scene_dir in scene_dirs:
        all_samples.extend(scene_samples(scene_dir, phase=args.frame_phase))
    picks, pick_stats = pick_samples(all_samples, max_per_scene=args.max_samples_per_scene, seed=args.seed)
    vlm_image_size = tuple(args.vlm_image_size)
    LOGGER.info("%d samples picked from %d scenes (%d built)", len(picks), len(scene_dirs), len(all_samples))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "scene_root": str(Path(args.scene_root).resolve()),
        "scenes": [d.name for d in scene_dirs],
        "selection": pick_stats,
        "settings": {
            "max_samples_per_scene": args.max_samples_per_scene,
            "frame_phase": args.frame_phase,
            "seed": args.seed,
            "vlm_image_size": list(vlm_image_size),
            "jpeg_quality": args.jpeg_quality,
            "attn_implementation": "sdpa",
            "decode": "greedy_native_decision (server two-turn, max_new_tokens 128, greedy)",
        },
    }

    def build(sample: ReplaySample):
        return model_inputs(dir_of[sample.scene_id], sample, vlm_image_size, args.jpeg_quality)

    if args.dry_run:
        checked = 0
        if args.check_encoding:
            from transformers import AutoProcessor

            from src.data.s2_sft_dataset import encode_conversation

            # Qwen2_5VLIntegration._load_model's processor.
            processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
            processor.tokenizer.padding_side = "left"
            for sample in picks[: args.check_encoding]:
                messages, images, lookdown = build(sample)
                encoded = encode_conversation(processor, messages, images, lookdown, sample_turns(sample))
                if int((encoded["labels"] != -100).sum()) == 0:
                    raise RuntimeError(f"{sample_key(sample)}: nothing supervised")
                checked += 1
        report.update({"dry_run": True, "encoding_checked": checked, "seconds": round(time.time() - started, 1)})
        output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"picked": len(picks), "encoding_checked": checked,
                          "scenes": pick_stats["scenes"]}, ensure_ascii=False), flush=True)
        return 0

    import torch

    from scripts.train_system2_lora import (
        base_model_identity,
        greedy_native_decision,
        init_runtime,
        load_adapter,
        load_system2,
        provenance,
        seed_everything,
    )

    rt = init_runtime(args.device)
    seed_everything(args.seed, rt.device)
    calls_path = output.with_suffix(".calls.jsonl")
    done = _read_rows(calls_path)
    todo = [s for s in picks if sample_key(s) not in done]
    LOGGER.info("%d already scored, %d to go", len(picks) - len(todo), len(todo))
    if todo:
        model, processor = load_system2(args.model_path, rt.device)
        if args.adapter:  # EXP-26: the same agreement for a fine-tuned System2 (PEFT dir)
            model = load_adapter(model, args.adapter)
        model.eval()
        with open(calls_path, "a", encoding="utf-8") as handle:
            for i, (sample, (messages, images, lookdown)) in enumerate(
                prefetched(build, todo, workers=args.workers, window=args.prefetch), 1
            ):
                t0 = time.time()
                with torch.no_grad():
                    answer = greedy_native_decision(model, processor, messages, images, lookdown, rt.device)
                row = {
                    "key": sample_key(sample),
                    "scene_id": sample.scene_id,
                    "episode_index": sample.episode_index,
                    "frame_id": sample.frame_id,
                    "kind": sample.kind,
                    "num_history": len(sample.history_frame_ids),
                    **score(sample_turns(sample), answer),
                    "seconds": round(time.time() - t0, 3),
                }
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                handle.flush()
                done[row["key"]] = row
                if i % args.log_every == 0 or i == len(todo):
                    rows = [done[sample_key(s)] for s in picks if sample_key(s) in done]
                    agree = sum(bool(r["class_match"]) for r in rows)
                    LOGGER.info("%d/%d scored; agreement %.4f over %d", i, len(todo), agree / len(rows), len(rows))
    rows = [done[sample_key(s)] for s in picks]
    report.update({
        "summary": summarize(rows),
        "base_model": base_model_identity(args.model_path),
        "adapter": str(Path(args.adapter).resolve()) if args.adapter else None,
        "provenance": provenance(),
        "device": str(rt.device),
        "wall_seconds": round(time.time() - started, 1),
        "calls_jsonl": str(calls_path),
    })
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(output)
    summary = report["summary"]
    print(json.dumps({
        "n": summary["n"],
        "overall_rate": summary["overall_rate"],
        "ci95": summary["overall_ci95_episode_bootstrap"],
        "by_expert_class": {k: (v["n"], v["rate"]) for k, v in summary["by_expert_class"].items()},
        "pixel_median_px": summary["pixel_vs_pixel"].get("median_px"),
        "b3": summary["b3"]["reading"],
    }, ensure_ascii=False), flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-root", required=True, help="directory of extracted InternData-N1 scene dirs")
    parser.add_argument("--scenes", default="", help="comma-separated scene dirs under --scene-root (default: all)")
    parser.add_argument("--model-path", required=True, help="the deployed InternNav_Model directory")
    parser.add_argument("--adapter", default="", help="a System2 LoRA (PEFT dir) to attach; empty: the base model")
    parser.add_argument("--output", required=True, help="summary json; rows go to <output>.calls.jsonl")
    parser.add_argument("--max-samples-per-scene", type=int, default=0, help="seeded uniform subset (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-phase", type=int, default=0,
                        help="sample frames 4n+PHASE instead of InternNav's training grid 4n (EXP-27 memory test)")
    parser.add_argument("--vlm-image-size", type=int, nargs=2, default=list(DEFAULT_VLM_IMAGE_SIZE))
    parser.add_argument("--jpeg-quality", type=int, default=90, help="the client's --rpc_jpeg_quality")
    parser.add_argument("--device", choices=("auto", "cuda", "npu", "cpu"), default="auto")
    parser.add_argument("--workers", type=int, default=4, help="threads building inputs ahead of the GPU")
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="count samples only; no model")
    parser.add_argument("--check-encoding", type=int, default=0,
                        help="with --dry-run: run encode_conversation's prompt-prefix check on N samples")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

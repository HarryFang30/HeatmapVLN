#!/usr/bin/env python3
"""EXP-37 gate G1: a DPO adapter's implicit-reward accuracy on held-out preference pairs.

The ledger's EXP-37 (written 2026-10-28 before any EXP-37 training): on the held-out pairs (R2R H and
HM3D-20 scenes, the base wrong, d > 3 m, neither side STOP), the share of pairs whose implicit reward
margin [s_theta(c) - s_ref(c)] - [s_theta(r) - s_ref(r)] is > 0 must be >= 0.60, and its 2.5% bound
from an episode-cluster bootstrap (2,000 draws, seed 0) must be > 0.50.

s(y) is the trainer's own score (``train_system2_lora.sequence_scores``): the mean log-prob of y's
supervised tokens, on the pair as the trainer encodes it (``s2_stream.encode_pair`` and
``collate_pairs``), with the model loaded as the trainer loads it (``load_system2``, bf16 / SDPA) and the
adapter attached as the server attaches it (``load_adapter``, eval mode, so no LoRA dropout).  The
reference is the same model in ``adapter_off``.  A pair's four numbers are therefore the trainer's.

Subcommands::

    score      NPU / GPU, one process per card (LOCAL_RANK picks the card).  --pairs is a pair stream
               (manifest ``pairs: true``); --shard i/n keeps the pairs whose key hashes to i.  Rows go
               to --output (jsonl), appended and flushed per pair; a rerun skips keys already there.
    summarize  No model.  Reads every --rows file and writes --output.  Exit 0: G1 passes; 1: it fails;
               2: invalid (no rows, a duplicate key, rows of two adapters or of another --adapter, a row
               without its episode, pairs of --pairs not scored).

    python scripts/exp37/eval_pairs.py score --pairs <pair stream dir> --adapter <run>/adapter \\
        --model-path <InternNav_Model> --shard 0/8 --device npu --output <dir>/heldout.rows.shard00of08.jsonl
    python scripts/exp37/eval_pairs.py summarize --rows '<dir>/heldout.rows.shard*.jsonl' \\
        [--adapter <run>/adapter] [--pairs <pair stream dir>] --output <dir>/heldout_summary.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.data.s2_stream import (  # noqa: E402
    PackedSample,
    is_pair_stream,
    read_manifest,
    read_shard,
    shard_digest_ok,
    shard_name,
)

LOGGER = logging.getLogger("eval_pairs")
ROW_SCHEMA = "heatmapvln-exp37-pair-score-v1"
SCHEMA = "heatmapvln-exp37-g1-v1"
G1_ACCURACY = 0.60  # reward accuracy >= this
G1_LOWER = 0.50  # and the episode-cluster bootstrap's 2.5% bound > this
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0
SCORES = ("s_pol_c", "s_pol_r", "s_ref_c", "s_ref_r")


def parse_shard(text: str) -> tuple[int, int]:
    index, count = (int(v) for v in str(text).split("/"))
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"--shard {text!r}: want i/n with 0 <= i < n")
    return index, count


def shard_of(key: str, count: int) -> int:
    """A stable shard for a pair key (the gates' hash: sha256 of the key, first 48 bits, mod n)."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16) % count


# --------------------------------------------------------------------------- #
# Pair stream
# --------------------------------------------------------------------------- #
def pair_stream(pairs_dir: str | Path) -> dict[str, Any]:
    manifest = read_manifest(pairs_dir)
    if not is_pair_stream(manifest):
        raise ValueError(f"{pairs_dir} is not a pair stream (its manifest has no pairs: true)")
    return manifest


def _shard_paths(pairs_dir: str | Path, manifest: Mapping[str, Any]) -> list[tuple[Path, str]]:
    entries = list(manifest["shards"])
    if len(entries) != int(manifest["num_shards"]):
        raise ValueError(f"{pairs_dir}: {len(entries)} shard entries for num_shards {manifest['num_shards']}")
    return [(Path(pairs_dir) / shard_name(i), str(entry["sha256"])) for i, entry in enumerate(entries)]


def pair_keys(pairs_dir: str | Path, manifest: Mapping[str, Any]) -> list[str]:
    """Every pair key of the stream, reading only the json members."""
    keys: list[str] = []
    for path, _ in _shard_paths(pairs_dir, manifest):
        with tarfile.open(path, mode="r:") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".json"):
                    keys.append(str(json.loads(tar.extractfile(member).read())["key"]))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{pairs_dir}: {len(keys) - len(set(keys))} duplicate pair keys")
    return keys


def iter_pairs(pairs_dir: str | Path, manifest: Mapping[str, Any]) -> Iterator[PackedSample]:
    """The stream's pairs in shard order; every shard's sha256 is checked against the manifest first."""
    for path, digest in _shard_paths(pairs_dir, manifest):
        if not shard_digest_ok(path, digest):
            raise ValueError(f"{path}: sha256 differs from the manifest's {digest}")
        for sample in read_shard(path):
            if sample.rejected_turns is None:
                raise ValueError(f"{path}: {sample.key} has no rejected_turns")
            yield sample


def read_rows(paths: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    """key -> row over every file; a key in two rows is an error."""
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema") != ROW_SCHEMA:
                    raise ValueError(f"{path}: row schema {row.get('schema')!r} != {ROW_SCHEMA}")
                if row["key"] in rows:
                    raise ValueError(f"{path}: pair {row['key']} is scored twice")
                rows[row["key"]] = row
    return rows


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def score_pair(model, processor, sample: PackedSample, device) -> dict[str, Any]:
    """The four scores of one pair, exactly as ``dpo_micro_batch_loss`` computes them, without a graph."""
    import torch

    from scripts.train_system2_lora import reference_scores, sequence_scores
    from src.data.s2_stream import collate_pairs, encode_pair

    batch = collate_pairs([encode_pair(processor, sample)], pad_token_id=int(processor.tokenizer.pad_token_id))
    with torch.no_grad():
        s_pol_c, n_c = sequence_scores(model, batch["chosen"], device)
        s_pol_r, n_r = sequence_scores(model, batch["rejected"], device)
    s_ref_c, _ = reference_scores(model, batch["chosen"], device)
    s_ref_r, _ = reference_scores(model, batch["rejected"], device)
    pair = sample.pair or {}
    return {
        "key": sample.key,
        "episode": pair.get("episode"),
        "source": pair.get("source"),
        "chosen_class": pair.get("chosen_class"),
        "rejected_class": pair.get("rejected_class"),
        "d": pair.get("d"),
        "s_pol_c": float(s_pol_c[0]), "s_pol_r": float(s_pol_r[0]),
        "s_ref_c": float(s_ref_c[0]), "s_ref_r": float(s_ref_r[0]),
        "n_tokens_c": int(n_c[0]), "n_tokens_r": int(n_r[0]),
    }


def load_scorer(args: argparse.Namespace):
    """(model with the adapter attached, processor, device), as the trainer and the server load them."""
    from scripts.train_system2_lora import init_runtime, load_adapter, load_system2, seed_everything

    rt = init_runtime(args.device)
    seed_everything(args.seed, rt.device)
    model, processor = load_system2(args.model_path, rt.device)
    model = load_adapter(model, args.adapter)
    model.eval()
    return model, processor, rt.device


def selected_keys(args: argparse.Namespace, manifest: Mapping[str, Any]) -> set[str] | None:
    """--max-pairs: the first N keys of the whole stream in sha256 order (the same set on every shard)."""
    if not args.max_pairs:
        return None
    keys = sorted(pair_keys(args.pairs, manifest), key=lambda k: hashlib.sha256(f"exp37-g1:{k}".encode()).hexdigest())
    return set(keys[: args.max_pairs])


def score(args: argparse.Namespace) -> int:
    from scripts.train_system2_lora import sha256_file

    started = time.time()
    manifest = pair_stream(args.pairs)
    index, count = parse_shard(args.shard)
    keep = selected_keys(args, manifest)
    adapter_sha = sha256_file(Path(args.adapter) / "adapter_model.safetensors")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = read_rows([output]) if output.is_file() else {}
    others = {r.get("adapter_sha256") for r in done.values()} - {adapter_sha}
    if others:
        raise SystemExit(f"{output} holds rows of another adapter {sorted(map(str, others))}; use another --output")
    loaded = None
    scored = mine = 0
    with open(output, "a", encoding="utf-8") as handle:
        for sample in iter_pairs(args.pairs, manifest):
            if shard_of(sample.key, count) != index or (keep is not None and sample.key not in keep):
                continue
            mine += 1
            if sample.key in done:
                continue
            if loaded is None:
                loaded = load_scorer(args)
            model, processor, device = loaded
            t0 = time.time()
            row = {"schema": ROW_SCHEMA, **score_pair(model, processor, sample, device),
                   "adapter_sha256": adapter_sha, "device": device.type, "seconds": round(time.time() - t0, 3)}
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            scored += 1
            if scored % args.log_every == 0:
                LOGGER.info("%d pairs scored (%.1f s/pair)", scored, (time.time() - started) / scored)
    print(json.dumps({"shard": f"{index}/{count}", "pairs": mine, "scored_now": scored, "rows": str(output),
                      "seconds": round(time.time() - started, 1)}), flush=True)
    return 0


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def margin(row: Mapping[str, Any]) -> float:
    return (float(row["s_pol_c"]) - float(row["s_ref_c"])) - (float(row["s_pol_r"]) - float(row["s_ref_r"]))


def cluster_bootstrap(correct: Sequence[bool], clusters: Sequence[str], *, draws: int = BOOTSTRAP_DRAWS,
                      seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """(2.5%, 97.5%) of the pooled accuracy when whole episodes are drawn with replacement.

    Episodes are taken in sorted order, each draw picks as many episodes as there are,
    and a draw's accuracy is its correct pairs over its pairs (numpy's default quantile).
    """
    names = sorted(set(clusters))
    at = {name: i for i, name in enumerate(names)}
    hits = np.zeros(len(names), dtype=np.float64)
    sizes = np.zeros(len(names), dtype=np.float64)
    for ok, name in zip(correct, clusters):
        hits[at[name]] += float(bool(ok))
        sizes[at[name]] += 1.0
    picks = np.random.default_rng(seed).integers(0, len(names), size=(int(draws), len(names)))
    acc = hits[picks].sum(axis=1) / sizes[picks].sum(axis=1)
    return float(np.quantile(acc, 0.025)), float(np.quantile(acc, 0.975))


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reward / policy-only / reference-only accuracy, the bootstrap bound and G1."""
    if not rows:
        raise ValueError("no rows")
    missing = [r["key"] for r in rows if not r.get("episode")]
    if missing:
        raise ValueError(f"{len(missing)} rows have no episode, e.g. {missing[:3]}")
    rows = sorted(rows, key=lambda r: r["key"])
    n = len(rows)
    margins = [margin(r) for r in rows]
    reward = [m > 0 for m in margins]
    policy = [float(r["s_pol_c"]) > float(r["s_pol_r"]) for r in rows]
    reference = [float(r["s_ref_c"]) > float(r["s_ref_r"]) for r in rows]
    episodes = [str(r["episode"]) for r in rows]
    accuracy = sum(reward) / n
    lower, upper = cluster_bootstrap(reward, episodes)
    by_source: dict[str, Any] = {}
    for source in sorted({str(r.get("source")) for r in rows}):
        idx = [i for i, r in enumerate(rows) if str(r.get("source")) == source]
        by_source[source] = {"pairs": len(idx), "episodes": len({episodes[i] for i in idx}),
                             "reward_accuracy": round(sum(reward[i] for i in idx) / len(idx), 6)}
    return {
        "schema": SCHEMA,
        "pairs": n,
        "episodes": len(set(episodes)),
        "reward_accuracy": round(accuracy, 6),
        "reward_accuracy_ci95": [round(lower, 6), round(upper, 6)],
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "cluster": "pair.episode"},
        "policy_accuracy": round(sum(policy) / n, 6),
        "reference_accuracy": round(sum(reference) / n, 6),
        "margin_mean": round(float(np.mean(margins)), 6),
        "margin_zero": sum(m == 0 for m in margins),
        "chosen_logratio_mean": round(float(np.mean([float(r["s_pol_c"]) - float(r["s_ref_c"]) for r in rows])), 6),
        "rejected_logratio_mean": round(float(np.mean([float(r["s_pol_r"]) - float(r["s_ref_r"]) for r in rows])), 6),
        "by_source": by_source,
        "by_rejected_class": dict(Counter(str(r.get("rejected_class")) for r in rows)),
        "g1": {"accuracy_line": G1_ACCURACY, "lower_line": G1_LOWER,
               "pass": bool(accuracy >= G1_ACCURACY and lower > G1_LOWER)},
    }


def summarize(args: argparse.Namespace) -> int:
    paths = sorted(glob.glob(args.rows))
    report: dict[str, Any] = {"schema": SCHEMA, "rows": paths}
    try:
        if not paths:
            raise ValueError(f"no files match {args.rows!r}")
        rows = read_rows(paths)
        shas = {str(r.get("adapter_sha256")) for r in rows.values()}
        if len(shas) != 1:
            raise ValueError(f"rows of {len(shas)} adapters: {sorted(shas)}")
        if args.adapter:
            from scripts.train_system2_lora import sha256_file

            want = sha256_file(Path(args.adapter) / "adapter_model.safetensors")
            if shas != {want}:
                raise ValueError(f"rows are of adapter {sorted(shas)}, not --adapter {want}")
        if args.pairs:
            manifest = pair_stream(args.pairs)
            expected = selected_keys(args, manifest) if args.max_pairs else set(pair_keys(args.pairs, manifest))
            if set(rows) != expected:
                raise ValueError(f"{len(expected - set(rows))} pairs of {args.pairs} not scored, "
                                 f"{len(set(rows) - expected)} scored pairs not in it")
        report.update(summarize_rows(list(rows.values())))
        report["adapter_sha256"] = shas.pop()
    except (ValueError, KeyError, OSError) as exc:
        report.update(valid=False, error=str(exc))
        _write(Path(args.output), report)
        print(json.dumps({"valid": False, "error": str(exc)}), flush=True)
        return 2
    report["valid"] = True
    _write(Path(args.output), report)
    print(json.dumps({k: report[k] for k in ("pairs", "episodes", "reward_accuracy", "reward_accuracy_ci95",
                                             "policy_accuracy", "reference_accuracy", "g1")}), flush=True)
    return 0 if report["g1"]["pass"] else 1


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sco = sub.add_parser("score", help="the four scores of every pair in this shard (NPU / GPU)")
    sco.add_argument("--pairs", required=True, help="a pair stream dir (stream_manifest.json with pairs: true)")
    sco.add_argument("--adapter", required=True, help="the candidate's saved adapter dir")
    sco.add_argument("--model-path", required=True, help="the deployed InternNav_Model directory")
    sco.add_argument("--shard", default="0/1", help="i/n: score the pairs whose key hashes to i")
    sco.add_argument("--output", required=True, help="rows (jsonl); an existing file is resumed")
    sco.add_argument("--device", choices=("auto", "cuda", "npu", "cpu"), default="auto")
    sco.add_argument("--seed", type=int, default=42)
    sco.add_argument("--max-pairs", type=int, default=0,
                     help="score only the first N pairs of the stream in sha256 order (0 = all)")
    sco.add_argument("--log-every", type=int, default=100)
    summ = sub.add_parser("summarize", help="reward accuracy, its episode-cluster bootstrap and G1 (no model)")
    summ.add_argument("--rows", required=True, help="glob of score rows (quote it)")
    summ.add_argument("--output", required=True)
    summ.add_argument("--adapter", default="", help="optional: every row must be this adapter's")
    summ.add_argument("--pairs", default="", help="optional: every pair of this stream must be scored")
    summ.add_argument("--max-pairs", type=int, default=0, help="the --max-pairs the rows were scored with")
    args = parser.parse_args(argv)
    if args.max_pairs < 0:
        parser.error("--max-pairs must be >= 0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args(argv)
    return {"score": score, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""EXP-28 offline gates O4 and O5 on the HM3D held-out base records, and the verdict over O1-O5.

The ledger's EXP-28 D (written 2026-10-15 19:35 UTC, before any HM3D result) gates a System2 LoRA
before it may take the 4090's val time.  O1-O3 are measured by the existing tools
(``teacher_forced_agreement.py``, ``eval_system2_lora_offline.py`` + ``stratified_gates.py --gate
exp28-o3``); this script measures O4 and O5 and reads all five into one verdict.

States.  The base model's own closed-loop records in the 20 held-out HM3D scenes, labelled with line B
(``label_route_corrections.py``, ``RouteLabelConfig`` defaults; the label files' ``labeler`` digest must
be that config's).  A state is one System2 call beyond 3 m of the goal (geodesic, the labeller's) among
the first 48 calls of its episode (training rule R1).  Its line-B kind is ``correction`` (the recorded
answer was judged wrong), ``anchor`` (judged right) or ``none`` (no oracle answer).

Decoding.  Teacher-forced: every state's prompt is the one the server built for the recorded call
(``inference_messages``: the recorded JPEG bytes, the recorded history), decoded greedily the way the
server decodes (``greedy_native_decision``), once with the adapter on and once with it off -- the base
model on the same machine, so both arms see the same calls.

====  ============================================================================  ==================
O4    correction states: share of answers line B judges agreeing, on minus off        >= +0.15
      anchor states: the adapter keeps the base model's decision class                 >= 0.92
      every state (d > 3 m): STOP rate, adapter on                                     <= 1.5 x base
O5    the same correction states with the instruction replaced by another episode's     d_shuf <= 0.3 d_corr
      of the same scene (seeded): d_shuf = on minus off agreeing share under the
      shuffled instruction, d_corr = the same with the episode's own instruction
====  ============================================================================  ==================

"Agreeing" is line B's own rule (``s2_route_labels.label_route_call``), applied to the decoded answer
in the recorded state exactly as the labeller applied it to the recorded answer: same pose, depth,
route and previously executed turn (``call_contexts`` follows ``label_route_corrections.
label_record_rows`` step for step, and ``score`` refuses to go on unless the recorded answers come out
as the label files say).  An unjudgeable answer (a pixel whose floor point cannot be snapped or reached)
does not count as agreeing.  Under a shuffled instruction the answer is still judged against the
episode's own route: agreeing there is following the route without the words.  Exact thresholds are
compared in whole counts, so a value exactly on its line passes.

Subcommands::

    decode   GPU.  Decodes the selected states (on, off; with --shuffle-instructions also both for every
             correction under its shuffled instruction).  --shard i/n keeps the states whose key hashes
             to i, so one process per card shares the work; <output stem>.decode.shardIIofNN.jsonl is
             the resume log.  --dry-run counts the states without a model.
    score    CPU, habitat_sim's PathFinder on <scenes-dir>/<scene>/<scene>.navmesh (no renderer).  Judges
             every decoded answer, writes --output and <output stem>.calls.jsonl.  Exit 0: O4 and O5
             pass; 1: not; 2: the measurement is invalid (a record without labels or without the navmesh
             sha256 it ran on, or not on this navmesh; states not decoded, rows of another adapter or of
             another shuffle donor; labels not reproduced; no correction or anchor state; O5 not decoded
             on every correction with a donor).
    verdict  No model.  O1 and O2 (candidate against the base on the same samples, >= base - 1.0 pt),
             O3, O4, O5 and the pre-registered retry: only O4's correction item fails -> retrain on the
             same stream for 2 epochs; O1, O2, O3 or O4's anchor / STOP item fails -> LoRA scale 0.5, no
             retraining, gate again; O5 fails -> kill; a second attempt that fails -> kill.  Exit 0 all
             pass, 1 a gate failed, 2 something could not be measured (next: invalid, not a retry).
             ``--rules exp28`` is the default and writes exactly what it wrote before EXP-32.

             ``--rules exp32`` (ledger EXP-32 第二部分 F, written before any part-1 reading) gates O1, O2, O3,
             O4's anchor and STOP items, L and R; O4's correction item and O5 are report-only.  L reads
             ``score``'s rows (``--o4-calls``, default <o4 stem>.calls.jsonl): on O5's states (corrections with a
             same-scene donor instruction), the share whose greedy answer under the donor's instruction is not
             ``same_answer`` (scripts/exp30/probe_judge.py: another class, a pixel more than 50 px away on
             either axis, a turn the other way) as under the episode's own; the candidate's share >= the base's
             - 5.0 pt in whole counts.  R is ``scripts/exp32/rft_gate.py score``'s gate (``--rgate``
             rft_gate.json: R, or M when R is report-only) and must be of the adapter O4 decoded.  Retry: L
             fails -> kill; O1, O2, O3, O4_anchor or O4_stop fails -> scale 0.5 if R passed, kill if not; only R
             fails -> retrain on the same stream for 2 epochs if M says not learnt (D_M < 0.15), kill if learnt;
             a second attempt that fails -> kill.  Same exit codes, its own schema.

    python scripts/exp28/hm3d_route_gate.py decode --record-root /workspace/exp28/gates/hm3d_heldout/records \\
        --label-root /workspace/exp28/gates/hm3d_heldout/labels --holdout-scenes-file heldout_scenes.txt \\
        --adapter <adapter dir> --model-path /workspace/InternNav_Model --output <dir>/o4/hm3d.json \\
        --shard 0/3 --shuffle-instructions --device cuda
    python scripts/exp28/hm3d_route_gate.py score <same selection flags> --scenes-dir \\
        /workspace/exp28/scene_datasets/hm3d --output <dir>/o4/hm3d.json

``decode --on-mode`` (ledger EXP-37 设置 部署): what the "on" answer is.  ``adapter`` (the default) is the
adapter's own greedy answer, as above.  ``base-stop`` is the answer the server's ``--system2_base_decides_stop``
mode gives (``base-stop-v1``): the base (adapter off, the "off" arm) decodes first; if turn 1's first generated
token is a STOP token (``STOP`` or `` STOP``, each checked to be a single token of the model's tokenizer) the
answer is ``STOP`` and the adapter does not run (``stop_source`` base); otherwise the adapter decodes with those
ids masked at turn 1's first generated position (``begin_suppress_tokens``; turn 2 as served), and an answer that
still ends in the STOP class -- a later STOP, or no action token at all, the server's fallback stop -- is
``stop_source`` adapter, anything else none.  Both arms keep the served decoding (``NATIVE_GENERATE_KWARGS`` =
``rpc_model_server.system2_generate_kwargs()`` greedy, and InternNav_Model's generation_config, whose repetition
penalty 1.05 applies to every generate call here as it does in the server).  The shuffled-instruction answers
use the same mode.  Every decode row names its mode; a resume log of another mode is refused, and ``score``
reads no rows of mixed modes (``join.on_mode`` says which it scored; rows without the field are ``adapter``).

EXP-37 (G2 and the report-only O4 correction, O5 and L of the X37 adapter, read by
``scripts/exp37/gate_verdict.py``).  O4/O5 so far always decoded on the 4090 (CUDA, ``run_offline_gates.sh``:
C1, X29V1, X32R1 with 6 shards, 4.7-5.5 s per state there); the deployed card of X37 is the 910B, so it decodes
there, 8 cards, one process per card (``PYTHONPATH`` must hold ``$R/rpc/src``); ``score`` needs habitat_sim's
PathFinder and the HM3D navmeshes, so it runs on the 4090's CPU.  The held-out records live on the 4090
(``hm3d_heldout/records`` links into ``collect_base/records``, about 1.8 GB), so they, their labels and the scene
list are copied to the 910B first, flat into ``$R/exp37/gate/hm3d_heldout/{records,labels,heldout_scenes.txt}``,
the layout ``scripts/exp37/run_gate_910b.sh`` reads (``X37_HM3D_DIR``) and which runs this decode (tar -h follows
the links; the label files find their records by scene and file name, and ``score`` joins the rows by key, so the
paths may differ).  The copy, from the operator's machine (literal paths: nothing is expanded there)::

    ssh 6024_fjl 'docker exec fjl-habitat tar -C /workspace/exp28/gates/hm3d_heldout -chf - records labels \\
        -C /workspace/exp28/episodes/splits heldout_scenes.txt' \\
      | ssh modelarts_job 'D=/mnt/sfs_turbo/zhr/zhr_1/exp37/gate/hm3d_heldout && mkdir -p $D && tar -x -C $D'

and the decode by hand, as run_gate_910b.sh's step 3 does (910B, R=/mnt/sfs_turbo/zhr/zhr_1, G=$R/exp37/gate,
H=$G/hm3d_heldout, SRC a clean checkout of the commit under test)::

    for i in 0 1 2 3 4 5 6 7; do
      ASCEND_RT_VISIBLE_DEVICES=$i PYTHONPATH=$SRC:$R/rpc/src nohup $R/envs/ppa/bin/python \\
        $SRC/scripts/exp28/hm3d_route_gate.py decode --record-root $H/records --label-root $H/labels \\
        --holdout-scenes-file $H/heldout_scenes.txt --adapter $R/exp37/train/X37/adapter \\
        --model-path $R/InternNav_Model --output $G/o4/hm3d.json \\
        --shard $i/8 --shuffle-instructions --on-mode base-stop --device npu > $G/o4_$i.log 2>&1 &
    done
    # 4090 container, after copying $G/o4/hm3d.decode.shard*of08.jsonl to /workspace/exp37/gate/o4/
    python scripts/exp28/hm3d_route_gate.py score --record-root /workspace/exp28/gates/hm3d_heldout/records \\
        --label-root /workspace/exp28/gates/hm3d_heldout/labels \\
        --holdout-scenes-file /workspace/exp28/episodes/splits/heldout_scenes.txt \\
        --scenes-dir /workspace/exp28/scene_datasets/hm3d --output /workspace/exp37/gate/o4/hm3d.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from scripts.exp30.probe_judge import same_answer  # noqa: E402
from scripts.tools.stratified_gates import decision_class  # noqa: E402
from scripts.tools import label_route_corrections as lr  # noqa: E402

REC, ORL, RL = lr.REC, lr.ORL, lr.RL

LOGGER = logging.getLogger("hm3d_route_gate")

SCHEMA = "heatmapvln-exp28-hm3d-route-gate-v1"
DECODE_SCHEMA = "heatmapvln-exp28-hm3d-route-gate-decode-v1"
VERDICT_SCHEMA = "heatmapvln-exp28-offline-gates-verdict-v1"

# Ledger EXP-28 D, written 2026-10-15 19:35 UTC before any HM3D result.
O1_REFERENCE = 0.859  # the base on B3's 5 ScaleVLN scenes (EXP-25 run record 10, 3,500 samples)
O2_REFERENCE = 0.975  # the base on r2r's 17DRP5sb8fy and 5LpN3gDmAk7 (600 samples)
O1_O2_MAX_DROP = Fraction(1, 100)
O4_CORRECTION_MIN_GAIN = Fraction(15, 100)
O4_ANCHOR_KEEP_MIN = Fraction(92, 100)
O4_FAR_STOP_MAX_RATIO = Fraction(3, 2)
O5_MAX_RATIO = Fraction(3, 10)
FAR_M = 3.0  # line B labels only beyond this; O4's STOP item counts the same states
MAX_CALL_INDEX = 48  # R1: the first 48 calls of an episode
R2_STOP_FREE_M = 5.0  # R2: a recorded STOP at 3-5 m gives no training sample (report-only subset)
BOOTSTRAP_RESAMPLES = 2000
STATE_KINDS = ("correction", "anchor", "none")
# scripts/evaluation/rpc_model_server.py: what a server serving a System2 LoRA advertises (records keep both)
LORA_FORMAT_PREFIX = "system2-lora-sha256:"
LORA_MODEL_TAG = "+system2-lora:"
# decode --on-mode (see the module doc): the adapter's own answer, or the server's base-stop-v1 answer (EXP-37)
ON_MODES = ("adapter", "base-stop")
STOP_TOKEN_TEXTS = ("STOP", " STOP")  # what base-stop reads and masks at turn 1's first generated position
BASE_STOP_ANSWER = "STOP"
GROUPS = ("O1", "O2", "O3", "O4_correction", "O4_anchor", "O4_stop", "O5")
RETRY_RULES = {
    "val": "every gate passed",
    "retrain_2_epochs": "only O4's correction item failed (nothing learnt): train the same stream for 2 epochs, once",
    "scale_0.5": "O1, O2, O3 or O4's anchor / STOP item failed: LoRA scale 0.5, no retraining, gate once more",
    "kill": "O5 failed (scaling does not change its ratio), or the one retry failed: the candidate does not go to val",
    "invalid": "a gate could not be measured (see notes: O4/O5 invalid or a smoke subset, O1/O2 not on the base's "
               "samples, O3 rows without records): fix the inputs and run this attempt again; it is not a retry",
}

# Ledger EXP-32 第二部分 F, written before any part-1 reading (verdict --rules exp32).
VERDICT_SCHEMA_EXP32 = "heatmapvln-exp32-offline-gates-verdict-v1"
RFT_GATE_SCHEMA = "heatmapvln-exp32-rft-gate-v1"  # scripts/exp32/rft_gate.py score's rft_gate.json
L_MAX_DROP = Fraction(5, 100)
HARM_GROUPS_EXP32 = ("O1", "O2", "O3", "O4_anchor", "O4_stop")
GROUPS_EXP32 = HARM_GROUPS_EXP32 + ("L", "R")
REPORT_ONLY_EXP32 = ("O4_correction", "O5")
LEDGER_SIZES_EXP32 = {"O1": 3500, "O2": 600, "O3": 3000}  # F: B3's 3,500 samples, r2r's 600, O3's 3,000 calls
RETRY_RULES_EXP32 = {
    "val": "every gated item passed (O1, O2, O3, O4_anchor, O4_stop, L, R): the candidate goes on to P, which runs "
           "beside these gates, and, unless P kills it, to val",
    "scale_0.5": "O1, O2, O3, O4_anchor or O4_stop failed and R passed: a copy with lora_B x 0.5 (EXP28_ATTEMPT=2 "
                 "EXP28_LORA_SCALE=0.5), every gate once more, R decoded on that copy against the same line",
    "retrain_2_epochs": "only R failed and M says not learnt (D_M < 0.15): train the same stream twice "
                        "(double_stream.py), once, and gate every item again",
    "kill": "L failed; or O1, O2, O3, O4_anchor or O4_stop failed together with R; or only R failed with D_M >= 0.15 "
            "(learnt, does not transfer); or the one retry failed: the candidate does not go to val",
    "invalid": "a gate could not be measured (see notes: O4/O5/L invalid or a smoke subset, O1/O2 not on the base's "
               "samples, O3 rows without records, rft_gate.json invalid or of another adapter or R waiting for B3's "
               "seed-2024 pass, M unmeasured where the retry type needs it): fix the inputs and run this attempt "
               "again; it is not a retry",
}


class GateInvalid(RuntimeError):
    """The measurement itself is broken; no pass or fail can be read from it."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _num(value: Fraction | None) -> float | None:
    return None if value is None else round(float(value), 6)


def scene_key(name: str) -> str:
    """The scene as records key it: ``00481-5jLhtVmWd5F`` (the HM3D directory) and ``5jLhtVmWd5F`` alike."""
    name = name.strip()
    head, sep, tail = name.partition("-")
    return tail if sep and head.isdigit() and tail else name


def read_scene_list(path: str | os.PathLike) -> list[str]:
    """One scene per line (``#`` comments and blank lines skipped)."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def holdout_scene_set(scenes: Iterable[str]) -> set[str]:
    return {scene_key(s) for s in scenes if s.strip()}


def _norm_text(text: Any) -> str:
    return " ".join(str(text).lower().split())


def answer_call(turns: Sequence[str]) -> dict[str, Any]:
    """A decoded answer in the shape of a recorded call row, for ``parse_model_answer``."""
    turns = [str(t) for t in turns]
    if len(turns) == 2:
        return {"llm_output": turns[1], "native_first_output": turns[0], "native_lookdown_turns": 1}
    if len(turns) != 1:
        raise ValueError(f"an answer has one or two turns, got {turns!r}")
    return {"llm_output": turns[0], "native_first_output": turns[0], "native_lookdown_turns": 0}


def _arm(turns: Sequence[str]) -> dict[str, Any]:
    turns = [str(t) for t in turns]
    return {"turns": turns, "class": decision_class(turns[-1])}


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def served_adapter(episode: Mapping[str, Any]) -> str | None:
    """The System2 LoRA the record's server served (``rpc_model_server``'s tags), None for the base model."""
    server = episode.get("server") or {}
    for item in server.get("supported_formats") or []:
        if str(item).startswith(LORA_FORMAT_PREFIX):
            return str(item)[len(LORA_FORMAT_PREFIX):]
    version = str(server.get("model_version") or "")
    # a mode suffix may follow the sha (EXP-37's "+base-stop-v1")
    return version.split(LORA_MODEL_TAG, 1)[1].split("+", 1)[0] if LORA_MODEL_TAG in version else None


def holdout_records(record_roots: Sequence[str | os.PathLike], scenes: Collection[str]) -> list[dict[str, Any]]:
    """Every committed record of a held-out scene, sorted by key; a record twice across roots is an error,
    and so is a record a LoRA served (O4's states are the base model's own, e.g. not proxy P's candidate arm)."""
    seen: dict[str, dict[str, Any]] = {}
    for root in record_roots:
        for path in REC.iter_record_paths(root):
            if path.parent.name not in scenes:
                continue
            record = REC.EpisodeRecord(path)
            if record.scene_id != path.parent.name:
                raise ValueError(f"{path}: scene_id {record.scene_id!r} is not its directory")
            adapter = served_adapter(record.episode)
            if adapter is not None:
                raise ValueError(f"{path} was served with System2 LoRA {adapter}: O4 reads the base model's records")
            key = REC.episode_key(record.scene_id, record.episode_id)
            if key in seen:
                raise ValueError(f"{key} is recorded twice: {seen[key]['path']} and {path}")
            seen[key] = {"key": key, "path": str(path), "scene_id": record.scene_id,
                         "episode_id": record.episode_id, "instruction": record.episode.get("instruction")}
    return [seen[k] for k in sorted(seen)]


def shuffle_donors(pool: Sequence[Mapping[str, Any]], seed: int) -> dict[str, dict[str, Any]]:
    """Per record key, whose instruction it gets under O5 (``donor``), or why none (``why``).

    Within each scene the records are put in a seeded order and each takes the instruction of the
    next one (the last the first's) -- a seeded shuffle in which no episode keeps its own words.  A
    donor whose instruction reads the same (case and spacing aside) is passed over for the one after.
    """
    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in pool:
        by_scene[str(item["scene_id"])].append(item)
    out: dict[str, dict[str, Any]] = {}
    for scene in sorted(by_scene):
        order = sorted(by_scene[scene], key=lambda it: _rank(seed, f"{scene}:{it['key']}"))
        n = len(order)
        for i, item in enumerate(order):
            donor = None
            for step in range(1, n):
                candidate = order[(i + step) % n]
                if isinstance(candidate["instruction"], str) and \
                        _norm_text(candidate["instruction"]) != _norm_text(item["instruction"]):
                    donor = candidate
                    break
            if donor is None:
                out[item["key"]] = {"donor": None, "instruction": None,
                                    "why": "single_record_in_scene" if n < 2 else "no_other_instruction_in_scene"}
            else:
                out[item["key"]] = {"donor": donor["key"], "instruction": donor["instruction"], "why": None}
    return out


def _turns(first: Any, second: Any) -> list[str] | None:
    if first is None:
        return None
    return [str(first)] + ([str(second)] if second is not None else [])


def select_states(
    record_roots: Sequence[str | os.PathLike],
    label_root: str | os.PathLike,
    *,
    scenes: Collection[str],
    config_digest: str,
    max_call_index: int = MAX_CALL_INDEX,
    far_m: float = FAR_M,
    max_episodes: int = 0,
    seed: int = 42,
    shuffle_seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The gate's states (see the module doc) and how they were chosen.

    The shuffle pool is every held-out record, so ``max_episodes`` (a seeded subset of episodes, for
    smoke runs) does not move any episode's shuffled instruction.
    """
    scenes = set(scenes)
    if not scenes:
        raise ValueError("no held-out scenes given")
    pool = holdout_records(record_roots, scenes)
    donors = shuffle_donors(pool, shuffle_seed)
    chosen = list(pool)
    if max_episodes and max_episodes < len(pool):
        chosen = sorted(sorted(pool, key=lambda it: _rank(seed, it["key"]))[:max_episodes], key=lambda it: it["key"])
    stats: Counter = Counter()
    states: list[dict[str, Any]] = []
    for item in chosen:
        path = Path(item["path"])
        label_file = REC.label_path_for(path, label_root)
        if not label_file.is_file():
            # Skipping it would shrink O4 without a trace (and decode and score could see different states).
            raise ValueError(f"{path} has no line-B labels ({label_file}); run label_route_corrections.py on the "
                             "held-out records first")
        rows = REC.read_labels(label_file)
        sidecar = lr.base._sidecar_digest(path)
        for row in rows:
            if row.get("labeler") != config_digest:
                raise ValueError(f"{label_file}: labelled by {row.get('labeler')!r}, not line B's defaults "
                                 f"({config_digest}); relabel with label_route_corrections.py")
            if row.get("record_sha256") != sidecar:
                raise ValueError(f"{label_file}: labels of another version of {path.name}; relabel")
        by_index = {int(r["call_index"]): r for r in rows}
        record = REC.EpisodeRecord(path)
        stats["records"] += 1
        for call in sorted(record.calls, key=lambda c: int(c["call_index"])):
            index = int(call["call_index"])
            stats["calls"] += 1
            if index >= max_call_index:
                stats["calls_beyond_window"] += 1
                continue
            row = by_index.get(index)
            if row is None:
                raise ValueError(f"{label_file}: no row for call {index}; relabel")
            d = row.get("distance_to_goal")
            if row.get("skip") or d is None or not float(d) > far_m:
                stats["calls_within_radius_or_skipped"] += 1
                continue
            kind = str(row.get("kind"))
            if kind not in STATE_KINDS:
                raise ValueError(f"{label_file} call {index}: kind {kind!r} is not a line-B row kind")
            recorded = _turns(row.get("model_turn1"), row.get("model_turn2"))
            if recorded is None:
                raise ValueError(f"{label_file} call {index}: no recorded answer (model_turn1)")
            state = {
                "key": f"{item['key']}#{index}",
                "uid": f"{path}#{index}",
                "record_path": str(path),
                "record_key": item["key"],
                "scene_id": item["scene_id"],
                "episode_id": int(item["episode_id"]),
                "call_index": index,
                "distance_to_goal": float(d),
                "label_kind": kind,
                "label_agree": row.get("agree"),
                "label_reason": row.get("reason"),
                "recorded": recorded,
                "recorded_model_kind": row.get("model_kind"),
                "previous_turn_direction": int(row.get("previous_turn_direction") or 0),
                "label_target": _turns(row.get("target_turn1"), row.get("target_turn2")),
            }
            if kind == "correction":
                state["shuffle"] = dict(donors[item["key"]])
            states.append(state)
            stats[f"states_{kind}"] += 1
    stats["states"] = len(states)
    with_records = {it["scene_id"] for it in pool}
    meta = {
        "scenes": sorted(scenes),
        "scenes_with_records": sorted(with_records),
        "scenes_without_records": sorted(scenes - with_records),
        "records_in_pool": len(pool),
        "episodes_selected": len(chosen),
        "stats": dict(stats),
        "settings": {"max_call_index": max_call_index, "far_m": far_m, "max_episodes": max_episodes, "seed": seed,
                     "shuffle_seed": shuffle_seed, "labeler": config_digest},
        "shuffle_without_donor": dict(Counter(d["why"] for d in donors.values() if d["donor"] is None)),
    }
    return states, meta


# --------------------------------------------------------------------------- #
# Line B's judgement of any answer in a recorded state
# --------------------------------------------------------------------------- #
def call_contexts(
    record: Any,
    geometry: Any,
    *,
    dataset_entry: Mapping[str, Any] | None,
    config: Any,
    camera: Any = None,
    only: Collection[int] | None = None,
) -> dict[int, dict[str, Any]]:
    """Per call index, every argument ``label_route_call`` gets in ``label_record_rows`` except ``model``.

    The same steps in the same order as ``label_route_corrections.label_record_rows`` (the tests pin
    the two together); ``only`` limits the route and depth work to the calls that will be judged,
    while the previous executed answer is still taken from every call.
    """
    base = lr.base
    camera = camera if camera is not None else ORL.LookdownCamera()
    ref, goal, start = base.resolve_episode_geometry(record.episode, dataset_entry)
    waypoints = ORL.route_waypoints(ref, goal)
    remaining = segments = None

    steps = sorted(record.steps, key=lambda s: int(s["step_id"]))
    step_ids = [int(s["step_id"]) for s in steps]
    positions = [s["position"] for s in steps]
    if start is not None:
        step_ids, positions = [-1] + step_ids, [start] + positions
    reached = ORL.last_reached_by_step(
        np.asarray(positions, dtype=np.float64).reshape(-1, 3), waypoints, geometry.geodesic, config.reached_radius_m
    )
    wanted = None if only is None else {int(i) for i in only}
    contexts: dict[int, dict[str, Any]] = {}
    previous = None
    for call in sorted(record.calls, key=lambda c: int(c["call_index"])):
        index = int(call["call_index"])
        model = ORL.parse_model_answer(call)
        if wanted is not None and index not in wanted:
            previous = model
            continue
        position, rotation = base.call_pose(record, call)
        p = np.asarray(position, dtype=np.float64)
        d = geometry.geodesic(p, goal)
        route, depth = None, None
        if math.isfinite(d) and d > config.route_min_distance_m:
            if remaining is None:
                remaining = ORL.remaining_reference_m(waypoints, geometry.geodesic)
                segments = [geometry.find_path(waypoints[k], waypoints[k + 1]) for k in range(len(waypoints) - 1)]
            upto = int(np.searchsorted(step_ids, int(call["step_id"]), side="right")) - 1
            last_reached = max(
                int(reached[upto]) if upto >= 0 else -1,
                int(ORL.last_reached_by_step(p[None], waypoints, geometry.geodesic, config.reached_radius_m)[0]),
            )
            last_reached = RL.rejoin_after(last_reached, len(waypoints), d, config)
            k_star, _ = ORL.choose_waypoint(
                p, waypoints, last_reached, remaining, geometry.geodesic, slack_m=config.rejoin_slack_m
            )
            if k_star is not None:
                route = ORL.route_points(p, waypoints, k_star, geometry.find_path, segments)
            depth = record.lookdown_depth(index) if record.has_lookdown_depth(index) else None
        contexts[index] = {
            "previous_model": previous,
            "position": position,
            "rotation_wxyz": rotation,
            "distance_to_goal": d,
            "goal": goal,
            "route": route,
            "depth_m": depth,
            "geodesic": geometry.geodesic,
            "snap": geometry.snap,
            "config": config,
            "camera": camera,
        }
        previous = model
    return contexts


def judge(context: Mapping[str, Any], turns: Sequence[str]) -> dict[str, Any]:
    """Line B on ``turns`` in the state ``context`` describes: agree (True / False / None), kind, reason."""
    row = RL.label_route_call(model=ORL.parse_model_answer(answer_call(turns)), **context)
    return {"agree": row["agree"], "kind": row["kind"], "reason": row["reason"], "model_kind": row["model_kind"]}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def read_decode_rows(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Decoded rows by key; a later row of a key replaces an earlier one (files in name order)."""
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["key"]] = row
    return rows


def row_on_mode(row: Mapping[str, Any]) -> str:
    """A decode row's ``--on-mode`` (rows written before the option are the adapter's own answers)."""
    return str(row.get("on_mode") or "adapter")


def _needs_shuffle(state: Mapping[str, Any]) -> bool:
    return state["label_kind"] == "correction" and (state.get("shuffle") or {}).get("donor") is not None


def score_states(
    states: Sequence[Mapping[str, Any]],
    decoded: Mapping[str, Mapping[str, Any]],
    contexts_for: Callable[[Any, Collection[int]], Mapping[int, Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """States joined with their decoded answers and line B's judgement of each answer.

    ``contexts_for(record, call_indices)`` gives ``call_contexts`` for one record.  Raises
    ``GateInvalid`` when a state was not decoded, rows of different adapters or ``--on-mode`` s are mixed,
    or line B does not reproduce the label file on the recorded answer.
    """
    missing = [s["key"] for s in states if s["key"] not in decoded]
    if missing:
        raise GateInvalid(f"{len(missing)} of {len(states)} states were not decoded (first: {missing[:3]})")
    adapters = {decoded[s["key"]].get("adapter_sha256") for s in states}
    if len(adapters) != 1:
        raise GateInvalid(f"the decoded rows come from {len(adapters)} adapters: {sorted(map(str, adapters))}")
    modes = {row_on_mode(decoded[s["key"]]) for s in states}
    if len(modes) != 1 or not modes <= set(ON_MODES):
        raise GateInvalid(f"the decoded rows come from {len(modes)} on-modes: {sorted(modes)} (one of {ON_MODES})")
    by_record: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for state in states:
        by_record[state["record_path"]].append(state)
    rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    donor_mismatches: list[str] = []
    stats: Counter = Counter()
    for record_path in sorted(by_record):
        group = by_record[record_path]
        judged = {s["call_index"] for s in group if s["label_kind"] in ("correction", "anchor")}
        contexts = contexts_for(REC.EpisodeRecord(record_path), sorted(judged)) if judged else {}
        for state in group:
            row = dict(state)
            got = decoded[state["key"]]
            row.update({"on": dict(got["on"]), "off": dict(got["off"]), "record_device": got.get("record_device"),
                        "device": got.get("device"), "shuffled": None})
            if state["call_index"] in judged:
                context = contexts[state["call_index"]]
                check = judge(context, state["recorded"])
                if (check["kind"], check["agree"]) != (state["label_kind"], state["label_agree"]):
                    mismatches.append(f"{state['key']}: label {state['label_kind']}/{state['label_agree']}, "
                                      f"rejudged {check['kind']}/{check['agree']} ({check['reason']})")
                for arm in ("on", "off"):
                    verdict = judge(context, row[arm]["turns"])
                    row[arm].update(agree=verdict["agree"], judge_reason=verdict["reason"])
                shuffled = got.get("shuffled")
                if _needs_shuffle(state) and shuffled:
                    if shuffled.get("donor") != state["shuffle"]["donor"]:
                        # The held-out pool changed after decoding: that answer read another episode's words.
                        donor_mismatches.append(f"{state['key']}: decoded with {shuffled.get('donor')}, "
                                                f"the selection gives {state['shuffle']['donor']}")
                    row["shuffled"] = {"donor": shuffled.get("donor")}
                    for arm in ("on", "off"):
                        verdict = judge(context, shuffled[arm]["turns"])
                        row["shuffled"][arm] = {**shuffled[arm], "agree": verdict["agree"],
                                                "judge_reason": verdict["reason"]}
                    stats["shuffled_judged"] += 1
                stats["judged"] += 1
            rows.append(row)
    if mismatches:
        raise GateInvalid(f"line B does not reproduce {len(mismatches)} label rows on the recorded answer "
                          f"(wrong navmesh, labels or code?): {mismatches[:3]}")
    if donor_mismatches:
        raise GateInvalid(f"{len(donor_mismatches)} shuffled answers were decoded with another donor; decode again "
                          f"into a fresh --output: {donor_mismatches[:3]}")
    stats["rows"] = len(rows)
    return rows, {"adapter_sha256": next(iter(adapters)), "on_mode": next(iter(modes)), "stats": dict(stats)}


def _criterion(value: Fraction | None, threshold: Fraction | None, direction: str, ok: bool,
               **extra: Any) -> dict[str, Any]:
    return {"value": _num(value), "threshold": _num(threshold), "direction": direction, "pass": bool(ok), **extra}


def _agreeing(rows: Sequence[Mapping[str, Any]], arm: str) -> int:
    return sum(1 for r in rows if r[arm].get("agree") is True)


def _unjudged(rows: Sequence[Mapping[str, Any]], arm: str) -> int:
    return sum(1 for r in rows if r[arm].get("agree") is None)


def _shuffled_agreeing(rows: Sequence[Mapping[str, Any]], arm: str) -> int:
    return sum(1 for r in rows if r["shuffled"][arm].get("agree") is True)


def episode_bootstrap(rows: Sequence[Mapping[str, Any]], diff: Callable[[Mapping[str, Any]], float], *,
                      resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0) -> list[float] | None:
    """95% interval of the mean of ``diff`` over rows, resampling whole episodes."""
    if not rows:
        return None
    clusters: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        c = clusters[row["record_key"]]
        c[0] += float(diff(row))
        c[1] += 1.0
    sums = np.array([c[0] for c in clusters.values()])
    counts = np.array([c[1] for c in clusters.values()])
    draws = np.random.default_rng(seed).integers(0, len(sums), size=(resamples, len(sums)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return [round(float(np.quantile(means, 0.025)), 6), round(float(np.quantile(means, 0.975)), 6)]


def _correction_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    on, off = _agreeing(rows, "on"), _agreeing(rows, "off")
    return {"n": n, "on_agree": on, "base_agree": off, "on_rate": _rate(on, n), "base_rate": _rate(off, n),
            "gain": _num(Fraction(on - off, n)) if n else None}


def _r6_reversal(row: Mapping[str, Any]) -> bool:
    target = row.get("label_target")
    if not target:
        return False
    direction = RL.answer_direction(target[0], target[1] if len(target) > 1 else None)
    return direction != 0 and direction == -int(row.get("previous_turn_direction") or 0)


def summarize(rows: Sequence[Mapping[str, Any]], *, shuffle_requested: bool) -> dict[str, Any]:
    """O4 and O5 (with their report-only breakdowns) from scored rows."""
    corrections = [r for r in rows if r["label_kind"] == "correction"]
    anchors = [r for r in rows if r["label_kind"] == "anchor"]

    n = len(corrections)
    on, off = _agreeing(corrections, "on"), _agreeing(corrections, "off")
    gain = Fraction(on - off, n) if n else None
    correction = _criterion(
        gain, O4_CORRECTION_MIN_GAIN, ">=", bool(n) and gain >= O4_CORRECTION_MIN_GAIN,
        n=n, on_rate=_rate(on, n), base_rate=_rate(off, n), on_agree=on, base_agree=off,
        on_unjudged=_unjudged(corrections, "on"), base_unjudged=_unjudged(corrections, "off"),
        recorded_rate=0.0 if n else None, ci95_episode_bootstrap=episode_bootstrap(
            corrections, lambda r: (r["on"].get("agree") is True) - (r["off"].get("agree") is True)),
    )
    m = len(anchors)
    kept = sum(1 for r in anchors if r["on"]["class"] == r["off"]["class"])
    keep = Fraction(kept, m) if m else None
    anchor = _criterion(keep, O4_ANCHOR_KEEP_MIN, ">=", bool(m) and keep >= O4_ANCHOR_KEEP_MIN, n=m, kept=kept)
    total = len(rows)
    on_stops = sum(1 for r in rows if r["on"]["class"] == "stop")
    off_stops = sum(1 for r in rows if r["off"]["class"] == "stop")
    far_stop = _criterion(
        Fraction(on_stops, total) if total else None,
        O4_FAR_STOP_MAX_RATIO * Fraction(off_stops, total) if total else None,
        "<=", bool(total) and on_stops <= O4_FAR_STOP_MAX_RATIO * off_stops,
        n=total, on_stops=on_stops, base_stops=off_stops, base_rate=_rate(off_stops, total),
        ratio_max=float(O4_FAR_STOP_MAX_RATIO), distance_gt_m=FAR_M,
    )
    o4 = {"correction_agree_gain": correction, "anchor_class_keep": anchor, "far_stop_vs_base": far_stop}

    with_donor = [r for r in corrections if (r.get("shuffle") or {}).get("donor") is not None]
    measured = [r for r in with_donor if r.get("shuffled") and "on" in r["shuffled"] and "off" in r["shuffled"]]
    k = len(measured)
    complete = shuffle_requested and k == len(with_donor) and k > 0
    if k:
        c_on, c_off = _agreeing(measured, "on"), _agreeing(measured, "off")
        s_on, s_off = _shuffled_agreeing(measured, "on"), _shuffled_agreeing(measured, "off")
        d_corr, d_shuf = Fraction(c_on - c_off, k), Fraction(s_on - s_off, k)
        threshold = O5_MAX_RATIO * d_corr
        o5_ok = complete and d_shuf <= threshold
        o5 = _criterion(
            d_shuf, threshold, "<=", o5_ok, n=k, delta_corr=_num(d_corr), ratio_max=float(O5_MAX_RATIO),
            ratio=_num(d_shuf / d_corr) if d_corr else None, delta_corr_nonpositive=d_corr <= 0,
            on_rate_shuffled=_rate(s_on, k), base_rate_shuffled=_rate(s_off, k),
            on_rate_own=_rate(c_on, k), base_rate_own=_rate(c_off, k),
            # the other reading of the ledger's wording, against the base with the episode's own words
            delta_shuf_vs_base_own_instruction=_num(Fraction(s_on - c_off, k)),
            ci95_delta_shuf=episode_bootstrap(measured, lambda r: (r["shuffled"]["on"].get("agree") is True)
                                              - (r["shuffled"]["off"].get("agree") is True)),
            corrections_without_donor=n - len(with_donor), corrections_not_decoded_shuffled=len(with_donor) - k,
            measured_every_correction_with_donor=complete,
        )
    else:
        o5 = _criterion(None, None, "<=", False, n=0, corrections_without_donor=n - len(with_donor),
                        corrections_not_decoded_shuffled=len(with_donor),
                        why="not measured: decode without --shuffle-instructions" if not shuffle_requested
                        else "no correction state has a shuffled answer")

    eligible = [r for r in corrections
                if not (r.get("recorded_model_kind") == "stop" and r["distance_to_goal"] <= R2_STOP_FREE_M)
                and not _r6_reversal(r)]
    same_device = [r for r in rows if r.get("record_device") and r.get("record_device") == r.get("device")]
    replayed = sum(1 for r in same_device if list(r["off"]["turns"]) == list(r["recorded"]))
    report = {
        "corrections_by_recorded_kind": {
            kind: _correction_table([r for r in corrections if r.get("recorded_model_kind") == kind])
            for kind in sorted({str(r.get("recorded_model_kind")) for r in corrections})
        },
        "corrections_by_distance": {
            "3-6": _correction_table([r for r in corrections if r["distance_to_goal"] <= 6.0]),
            "gt6": _correction_table([r for r in corrections if r["distance_to_goal"] > 6.0]),
        },
        # The corrections training could have taught: no recorded STOP at 3-5 m (R2), no target that
        # reverses the executed turn (R6).
        "corrections_training_eligible_r2_r6": _correction_table(eligible),
        "corrections_by_scene": {
            scene: _correction_table([r for r in corrections if r["scene_id"] == scene])
            for scene in sorted({r["scene_id"] for r in corrections})
        },
        "anchor_line_b_agree": {"n": m, "on_rate": _rate(_agreeing(anchors, "on"), m),
                                "base_rate": _rate(_agreeing(anchors, "off"), m)},
        "classes": {
            kind: {arm: dict(sorted(Counter(r[arm]["class"] for r in rows if r["label_kind"] == kind).items()))
                   for arm in ("on", "off")}
            for kind in STATE_KINDS
        },
        "on_off_identical": _rate(sum(1 for r in rows if r["on"]["turns"] == r["off"]["turns"]), total),
        "base_replays_recorded_same_device": {"n": len(same_device), "matched": replayed,
                                              "rate": _rate(replayed, len(same_device))},
        "states": dict(Counter(r["label_kind"] for r in rows)),
        "episodes": len({r["record_key"] for r in rows}),
    }
    if any("stop_source" in r["on"] for r in rows):  # --on-mode base-stop: who decided each on answer's STOP
        report["on_stop_source"] = {
            kind: dict(sorted(Counter(str(r["on"].get("stop_source")) for r in rows
                                      if kind == "all" or r["label_kind"] == kind).items()))
            for kind in ("all",) + STATE_KINDS
        }
    return {
        "O4": {"criteria": o4, "all_pass": all(c["pass"] for c in o4.values())},
        "O5": {"criteria": {"shuffle_vs_own_instruction": o5}, "all_pass": bool(o5["pass"])},
        "report_only": report,
    }


def measurement_problems(summary: Mapping[str, Any]) -> list[str]:
    """Why O4 / O5 cannot be read from ``summarize``'s result (empty: they can).

    An item with nothing to measure fails in ``summarize``; as a gate result that FAIL would send a
    candidate to a retry or to kill for a broken run (no labels, a decode without
    --shuffle-instructions), so ``score`` reports it as invalid instead.
    """
    problems = []
    criteria = summary["O4"]["criteria"]
    if not criteria["correction_agree_gain"]["n"]:
        problems.append("O4: no correction state")
    if not criteria["anchor_class_keep"]["n"]:
        problems.append("O4: no anchor state")
    o5 = summary["O5"]["criteria"]["shuffle_vs_own_instruction"]
    if not o5.get("measured_every_correction_with_donor"):
        problems.append("O5 not measured on every correction with a donor: " + str(
            o5.get("why") or f"{o5.get('corrections_not_decoded_shuffled')} of them have no shuffled answer"))
    return problems


# --------------------------------------------------------------------------- #
# Verdict over O1-O5
# --------------------------------------------------------------------------- #
def agreement_rows(summary_path: str | os.PathLike) -> dict[str, bool]:
    """``teacher_forced_agreement.py``'s rows (``<output>.calls.jsonl``): key -> decision class matched."""
    rows_path = Path(summary_path).with_suffix(".calls.jsonl")
    out: dict[str, bool] = {}
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["key"]] = bool(row["class_match"])
    return out


def compare_agreement(candidate: Mapping[str, bool], base: Mapping[str, bool], *, reference: float) -> dict[str, Any]:
    """O1 / O2: the candidate's agreement >= the base's - 1.0 pt, on the very same samples."""
    same = set(candidate) == set(base)
    keys = sorted(set(candidate) & set(base))
    n = len(keys)
    hit_c = sum(1 for k in keys if candidate[k])
    hit_b = sum(1 for k in keys if base[k])
    delta = Fraction(hit_c - hit_b, n) if n else None
    ok = same and bool(n) and delta >= -O1_O2_MAX_DROP
    base_rate = _rate(hit_b, n)
    return {
        "value": _rate(hit_c, n),
        "threshold": _num(Fraction(hit_b, n) - O1_O2_MAX_DROP) if n else None,
        "direction": ">=",
        "pass": bool(ok),
        "n": n,
        "base_rate": base_rate,
        "delta": _num(delta),
        "same_samples": same,
        "only_candidate": len(set(candidate) - set(base)),
        "only_base": len(set(base) - set(candidate)),
        "ledger_base": reference,
        "base_reproduces_ledger": base_rate is not None and abs(base_rate - reference) < 0.001,
    }


def next_step(groups: Mapping[str, bool], attempt: int) -> str:
    """The pre-registered consequence of one gate pass (ledger EXP-28 D, "重试只有一次")."""
    if all(groups[g] for g in GROUPS):
        return "val"
    if attempt >= 2 or not groups["O5"]:
        return "kill"
    if not all(groups[g] for g in ("O1", "O2", "O3", "O4_anchor", "O4_stop")):
        return "scale_0.5"
    return "retrain_2_epochs"


def _o5_criterion(o4: Mapping[str, Any]) -> Mapping[str, Any]:
    return ((o4.get("O5") or {}).get("criteria") or {}).get("shuffle_vs_own_instruction") or {}


def gate_groups(o1: Mapping[str, Any], o2: Mapping[str, Any], o3: Mapping[str, Any],
                o4: Mapping[str, Any]) -> dict[str, bool]:
    """O1-O5 pass / fail, in ``GROUPS`` order, as both rule sets read them (an invalid O4/O5 passes nothing)."""
    if "O3_exp28" not in o3:
        raise ValueError("the O3 result has no O3_exp28 block: run stratified_gates.py from this commit")
    valid = bool(o4.get("valid"))
    criteria4 = (o4.get("O4") or {}).get("criteria") or {}
    return {
        "O1": bool(o1["pass"]),
        "O2": bool(o2["pass"]),
        "O3": bool(o3["O3_exp28"]["all_pass"]),
        "O4_correction": valid and bool((criteria4.get("correction_agree_gain") or {}).get("pass")),
        "O4_anchor": valid and bool((criteria4.get("anchor_class_keep") or {}).get("pass")),
        "O4_stop": valid and bool((criteria4.get("far_stop_vs_base") or {}).get("pass")),
        "O5": valid and bool(_o5_criterion(o4).get("pass")),
    }


def input_problems(o1: Mapping[str, Any], o2: Mapping[str, Any], o3: Mapping[str, Any],
                   o4: Mapping[str, Any]) -> list[str]:
    """Why O1-O5 cannot be read (empty: they can), for both rule sets.

    Anything here makes the pass unreadable ("invalid": fix and run the attempt again), never a FAIL that
    would spend the one retry or kill the candidate.
    """
    problems = []
    if not bool(o4.get("valid")):
        problems.append(f"O4/O5 measurement invalid: {o4.get('invalid')}")
    subset = (((o4.get("selection") or {}).get("settings") or {}).get("max_episodes"))
    if subset:
        problems.append(f"O4/O5 ran on a seeded subset of {subset} episodes (--max-episodes): a smoke run, "
                        "not the gate")
    for name, result in (("O1", o1), ("O2", o2)):
        if not result.get("same_samples"):
            problems.append(f"{name}: candidate and base were not scored on the same samples")
        elif not result.get("n"):
            problems.append(f"{name}: no samples")
    inputs3 = o3.get("inputs") or {}
    rows3, join3 = inputs3.get("rows_in_file"), inputs3.get("join") or {}
    if not rows3 or join3.get("rows") != rows3:
        problems.append(f"O3: {join3.get('rows')} of {rows3} decoded calls joined their records (records moved or "
                        "unreadable: stratified_gates.py drops those rows)")
    return problems


def _reference_notes(o1: Mapping[str, Any], o2: Mapping[str, Any], o4: Mapping[str, Any]) -> list[str]:
    notes = []
    for name, result in (("O1", o1), ("O2", o2)):
        if result.get("same_samples") and result.get("n") and not result.get("base_reproduces_ledger"):
            notes.append(f"{name}: the base reference {result.get('base_rate')} is not the ledger's "
                         f"{result.get('ledger_base')}")
    missing = (o4.get("selection") or {}).get("scenes_without_records")
    if missing:
        notes.append(f"O4/O5: {len(missing)} held-out scenes have no base records: {missing}")
    return notes


def verdict(*, o1: Mapping[str, Any], o2: Mapping[str, Any], o3: Mapping[str, Any], o4: Mapping[str, Any],
            attempt: int) -> dict[str, Any]:
    """``o1`` / ``o2``: ``compare_agreement`` results; ``o3``: a stratified result; ``o4``: this script's ``score``."""
    groups = gate_groups(o1, o2, o3, o4)
    problems = input_problems(o1, o2, o3, o4)
    measured = not problems
    step = next_step(groups, attempt) if measured else "invalid"
    notes = list(problems)
    if _o5_criterion(o4).get("delta_corr_nonpositive"):
        notes.append("O5 compares against delta_corr <= 0: its ratio says nothing about language; the ledger's "
                     "rule order still applies")
    notes += _reference_notes(o1, o2, o4)
    return {
        "schema": VERDICT_SCHEMA,
        "attempt": attempt,
        "groups": groups,
        "all_pass": measured and all(groups.values()),
        "measured": measured,
        "next": step,
        "next_rule": RETRY_RULES[step],
        "O1": o1,
        "O2": o2,
        "O3": o3["O3_exp28"],
        "O4": o4.get("O4"),
        "O5": o4.get("O5"),
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# Verdict under EXP-32's rules: L and R gate, O4's correction item and O5 report
# --------------------------------------------------------------------------- #
def read_rows(path: str | os.PathLike) -> list[dict[str, Any]]:
    """``score``'s rows (``<output stem>.calls.jsonl``)."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _o5_states(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """O5's states among ``score``'s rows: the corrections with a same-scene donor (``shuffle_donors``)."""
    return [r for r in rows
            if r.get("label_kind") == "correction" and (r.get("shuffle") or {}).get("donor") is not None]


def _has_shuffled(row: Mapping[str, Any]) -> bool:
    shuffled = row.get("shuffled") or {}
    return "on" in shuffled and "off" in shuffled


def instruction_sensitivity(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    """L's reading of one arm (``on``: the candidate, ``off``: the base) on ``score``'s rows: among the corrections
    with a shuffled answer, those whose answer under the donor's instruction is not ``same_answer`` as under the
    episode's own (another class -- a turn the other way is one -- or a pixel more than 50 px away on either axis)."""
    states = [r for r in rows if r.get("label_kind") == "correction" and _has_shuffled(r)]
    how: Counter = Counter()
    for row in states:
        own, other = row[arm]["turns"], row["shuffled"][arm]["turns"]
        if not same_answer(own, other):
            same_class = bool(own) and bool(other) and decision_class(own[-1]) == decision_class(other[-1])
            how["pixel_over_50px" if same_class else "class"] += 1
    different = sum(how.values())
    return {"n": len(states), "different": different, "rate": _rate(different, len(states)),
            "different_by": dict(sorted(how.items()))}


def language_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """L (ledger EXP-32 F): on O5's states, the candidate's instruction sensitivity >= the base's - 5.0 pt, compared
    in whole counts (a value exactly on the line passes).  It is read only when every correction with a donor has
    both shuffled answers; otherwise it fails here and ``verdict_exp32`` calls the pass invalid."""
    with_donor = _o5_states(rows)
    measured = [r for r in with_donor if _has_shuffled(r)]
    n = len(measured)
    on, off = instruction_sensitivity(measured, "on"), instruction_sensitivity(measured, "off")
    complete = bool(n) and n == len(with_donor)
    value = Fraction(on["different"], n) if n else None
    base = Fraction(off["different"], n) if n else None
    threshold = base - L_MAX_DROP if n else None
    return _criterion(
        value, threshold, ">=", complete and value >= threshold, n=n, on_different=on["different"],
        base_different=off["different"], base_rate=_num(base), delta=_num(value - base) if n else None,
        max_drop=float(L_MAX_DROP), on_different_by=on["different_by"], base_different_by=off["different_by"],
        corrections_without_donor=sum(1 for r in rows if r.get("label_kind") == "correction") - len(with_donor),
        corrections_not_decoded_shuffled=len(with_donor) - n, measured_every_correction_with_donor=complete,
    )


def _l_problems(lang: Mapping[str, Any], o4: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Why L cannot be read: an O5 state without its shuffled answers, or rows that are not the ones ``o4`` was
    scored from (row count, O5's state count and its shuffled agreeing rates must all be ``o4``'s)."""
    problems = []
    if not lang["measured_every_correction_with_donor"]:
        problems.append(f"L not measured on every correction with a donor: {lang['corrections_not_decoded_shuffled']} "
                        f"of them have no shuffled answer" if lang["n"] else "L: no correction with a shuffled answer")
    measured = [r for r in _o5_states(rows) if _has_shuffled(r)]
    k = len(measured)
    o5 = _o5_criterion(o4)
    found = (len(rows), k, _rate(_shuffled_agreeing(measured, "on"), k), _rate(_shuffled_agreeing(measured, "off"), k))
    scored = (((o4.get("join") or {}).get("stats") or {}).get("rows"), o5.get("n"), o5.get("on_rate_shuffled"),
              o5.get("base_rate_shuffled"))
    if found != scored:
        problems.append(f"L: the rows file (rows, O5 states, shuffled agreeing rates on/off {list(found)}) is not the "
                        f"one --o4 was scored from ({list(scored)})")
    return problems


def next_step_exp32(groups: Mapping[str, bool], attempt: int, *, m_learnt: bool | None) -> str:
    """The pre-registered consequence of one EXP-32 gate pass (ledger EXP-32 F, "重试只有一次，类型预先定").

    ``m_learnt``: rft_gate.json's M (D_M >= 0.15; None: not measured), read only when R alone failed; None
    there gives "invalid", since the retry type cannot be read.
    """
    if all(groups[g] for g in GROUPS_EXP32):
        return "val"
    if attempt >= 2 or not groups["L"]:
        return "kill"
    if not all(groups[g] for g in HARM_GROUPS_EXP32):
        return "scale_0.5" if groups["R"] else "kill"
    if m_learnt is None:
        return "invalid"
    return "kill" if m_learnt else "retrain_2_epochs"


def verdict_exp32(*, o1: Mapping[str, Any], o2: Mapping[str, Any], o3: Mapping[str, Any], o4: Mapping[str, Any],
                  o4_rows: Sequence[Mapping[str, Any]], rgate: Mapping[str, Any], attempt: int) -> dict[str, Any]:
    """EXP-32's verdict.  ``o1``-``o4`` as for ``verdict``; ``o4_rows``: ``score``'s rows beside ``o4`` (L);
    ``rgate``: ``scripts/exp32/rft_gate.py score``'s rft_gate.json (R: its ``gate``, which is M when R is
    report-only; its ``M.learnt`` picks the retry after R alone fails)."""
    if rgate.get("schema") != RFT_GATE_SCHEMA:
        raise ValueError(f"the R input is not scripts/exp32/rft_gate.py score's rft_gate.json "
                         f"(schema {rgate.get('schema')!r})")
    base = gate_groups(o1, o2, o3, o4)
    valid4 = bool(o4.get("valid"))
    lang = language_gate(o4_rows)
    gate = rgate.get("gate") or {}
    r_readable = bool(rgate.get("valid")) and gate.get("uses") in ("R", "M") and gate.get("pass") is not None
    groups = {name: base[name] for name in HARM_GROUPS_EXP32}
    groups["L"] = valid4 and bool(lang["pass"])
    groups["R"] = r_readable and bool(gate["pass"])
    m = rgate.get("M") or {}
    m_learnt = m.get("learnt") if m.get("measured") else None
    adapters = {"O4": (o4.get("join") or {}).get("adapter_sha256"), "R": rgate.get("adapter_sha256")}

    problems = input_problems(o1, o2, o3, o4)
    if valid4:
        problems += _l_problems(lang, o4, o4_rows)
    if not rgate.get("valid"):
        problems.append(f"R/M measurement invalid (rft_gate.json): {rgate.get('invalid')}")
    elif not r_readable:
        problems.append(f"R cannot be read: gate {dict(gate)}")
    elif valid4 and adapters["O4"] != adapters["R"]:
        problems.append(f"R/M were decoded with adapter {adapters['R']}, O4 with {adapters['O4']}: R must read the "
                        "adapter O1-O4 gate (after scale_0.5, the scaled copy)")
    step = next_step_exp32(groups, attempt, m_learnt=m_learnt) if not problems else "invalid"
    if step == "invalid" and not problems:
        problems.append("only R failed and M was not measured (rft_gate.py score without --m-states): the retry type "
                        "cannot be read")
    measured = not problems
    notes = list(problems)
    if r_readable and gate.get("uses") == "M":
        notes.append(f"R is report-only ({(rgate.get('R') or {}).get('divergent')} divergent states with B3's "
                     "seed-2024 pass, under 150): M (D_M >= 0.15) stands in for it")
    if _o5_criterion(o4).get("delta_corr_nonpositive"):
        notes.append("O5 (report-only) compares against delta_corr <= 0: its ratio says nothing about language; L "
                     "gates language")
    sizes = {"O1": o1.get("n"), "O2": o2.get("n"), "O3": (o3.get("inputs") or {}).get("rows_in_file")}
    for name, size in sizes.items():
        if size != LEDGER_SIZES_EXP32[name]:
            notes.append(f"{name}: {size} {'calls' if name == 'O3' else 'samples'}, the ledger's "
                         f"{LEDGER_SIZES_EXP32[name]}")
    notes += _reference_notes(o1, o2, o4)
    notes += [f"rft_gate: {note}" for note in rgate.get("notes") or []]
    return {
        "schema": VERDICT_SCHEMA_EXP32,
        "rules": "ledger EXP-32 第二部分 F, written before any part-1 reading",
        "attempt": attempt,
        "groups": groups,
        "report_only_groups": {name: base[name] for name in REPORT_ONLY_EXP32},
        "all_pass": measured and all(groups.values()),
        "measured": measured,
        "next": step,
        "next_rule": RETRY_RULES_EXP32[step],
        "O1": o1,
        "O2": o2,
        "O3": o3["O3_exp28"],
        "O4": o4.get("O4"),
        "O5": o4.get("O5"),
        "L": lang,
        "R": {key: rgate.get(key) for key in ("valid", "invalid", "gate", "R", "M", "R_fail", "delta_R_minus_R_fail")},
        "G_A": o3.get("G_A"),
        "adapter_sha256": adapters,
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #
def parse_shard(text: str) -> tuple[int, int]:
    index, _, count = str(text).partition("/")
    i, n = int(index), int(count or 1)
    if not 0 <= i < n:
        raise ValueError(f"--shard {text!r}: need i/n with 0 <= i < n")
    return i, n


def shard_of(key: str, count: int) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16) % count


def decode_path(output: Path, index: int, count: int) -> Path:
    return output.with_name(f"{output.stem}.decode.shard{index:02d}of{count:02d}.jsonl")


class WithInstruction:
    """A record that reads as another instruction: ``inference_messages`` then builds the shuffled prompt.

    Everything else (frames, lookdown, image size) is the record's own, byte for byte.
    """

    def __init__(self, record: Any, instruction: str):
        self._record = record
        self.episode = {**record.episode, "instruction": instruction}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._record, name)


def stop_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """The ids of ``STOP_TOKEN_TEXTS`` in this tokenizer, each required to be one token (a fragment would let a
    STOP through base-stop's first-position mask, and the base's STOP would go unread)."""
    ids = []
    for text in STOP_TOKEN_TEXTS:
        got = [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]
        if len(got) != 1:
            raise ValueError(f"{text!r} is not a single token in this tokenizer ({got}): base-stop cannot read or "
                             "mask it at the first position")
        ids.append(got[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"{STOP_TOKEN_TEXTS} share a token id ({ids})")
    return tuple(ids)


def greedy_first_token_decision(
    model: Any, processor: Any, messages: list[dict], images: list, lookdown: Any, device: Any, *,
    suppress_first: Sequence[int] = (),
) -> tuple[tuple[str, ...], int | None]:
    """``train_system2_lora.greedy_native_decision`` with the id of turn 1's first generated token beside the text.

    ``suppress_first``: ids masked at turn 1's first generated position only (``begin_suppress_tokens``, the
    server's base-stop decode); turn 2 decodes as served.  Empty: the very same generate calls as
    ``greedy_native_decision`` (the tests pin the two together).
    """
    import torch

    from scripts.train_system2_lora import NATIVE_GENERATE_KWARGS, _served_inputs
    from src.models.heatmap.native_internnav_exact import append_native_lookdown_turn, native_requests_lookdown

    kwargs = dict(NATIVE_GENERATE_KWARGS)
    if suppress_first:
        kwargs["begin_suppress_tokens"] = [int(t) for t in suppress_first]
    with torch.no_grad():
        inputs = _served_inputs(processor, messages, device)
        prompt_len = inputs["input_ids"].shape[1]
        output = model.generate(**inputs, **kwargs).sequences
        first_id = int(output[0][prompt_len]) if output.shape[1] > prompt_len else None
        first = processor.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        if not native_requests_lookdown(first):
            return (first,), first_id
        messages, _ = append_native_lookdown_turn(messages, images, first, lookdown)
        inputs = _served_inputs(processor, messages, device)
        prompt_len = inputs["input_ids"].shape[1]
        output = model.generate(**inputs, **NATIVE_GENERATE_KWARGS).sequences
        return (first, processor.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)), first_id


def base_stop_arms(decide: Callable[[bool, Sequence[int]], tuple[Sequence[str], int | None]],
                   stop_ids: Sequence[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(on, off) of one prompt under ``--on-mode base-stop`` (the module doc).

    ``decide(adapter_on, suppress_first)`` -> (turns, turn 1's first generated token id).  The base decodes
    first, unmasked: that is the "off" arm, and its first token decides STOP.  The adapter runs only when it
    does not.
    """
    stop_ids = tuple(int(t) for t in stop_ids)
    off_turns, off_first = decide(False, ())
    off = {**_arm(off_turns), "first_token_id": off_first}
    if off_first is not None and int(off_first) in stop_ids:
        on = {**_arm([BASE_STOP_ANSWER]), "first_token_id": None, "stop_source": "base"}
    else:
        on_turns, on_first = decide(True, stop_ids)
        on = {**_arm(on_turns), "first_token_id": on_first}
        # the server's stop or fallback_stop despite the mask: a STOP later in turn 1, or no action token at all
        on["stop_source"] = "adapter" if on["class"] == "stop" else "none"
    return on, off


def _selection(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scenes = list(args.scenes.split(",")) if args.scenes else []
    if args.holdout_scenes_file:
        scenes += read_scene_list(args.holdout_scenes_file)
    held = holdout_scene_set(scenes)
    if not held:
        raise SystemExit("give the held-out scenes (--holdout-scenes-file or --scenes): training scenes must not enter O4")
    return select_states(
        args.record_root, args.label_root, scenes=held, config_digest=RL.RouteLabelConfig().digest(),
        max_call_index=args.max_call_index, max_episodes=args.max_episodes, seed=args.seed,
        shuffle_seed=args.shuffle_seed,
    )


def _complete(row: Mapping[str, Any], state: Mapping[str, Any], shuffle: bool) -> bool:
    """A resumed row counts only if it has every answer this run wants, the shuffled one under today's donor."""
    if "on" not in row or "off" not in row:
        return False
    if not (shuffle and _needs_shuffle(state)):
        return True
    return (row.get("shuffled") or {}).get("donor") == state["shuffle"]["donor"]


def decode(args: argparse.Namespace) -> int:
    started = time.time()
    states, meta = _selection(args)
    index, count = parse_shard(args.shard)
    mine = [s for s in states if shard_of(s["key"], count) == index]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_path = decode_path(output, index, count)
    LOGGER.info("%d states (%s), %d in shard %d/%d", len(states), meta["stats"], len(mine), index, count)
    if args.dry_run:
        shuffle = Counter("donor" if _needs_shuffle(s) else "none" for s in states if s["label_kind"] == "correction")
        print(json.dumps({"states": len(states), "shard": len(mine), "selection": meta,
                          "corrections_shuffle": dict(shuffle), "on_mode": args.on_mode},
                         ensure_ascii=False), flush=True)
        return 0

    import torch

    from scripts.train_system2_lora import (
        adapter_off,
        greedy_native_decision,
        init_runtime,
        load_adapter,
        load_system2,
        seed_everything,
        sha256_file,
    )
    from src.data.s2_sft_dataset import RecordCache, inference_messages, record_device

    adapter_sha = sha256_file(Path(args.adapter) / "adapter_model.safetensors")
    done = read_decode_rows([rows_path]) if rows_path.is_file() else {}
    others = {r.get("adapter_sha256") for r in done.values()} - {adapter_sha}
    if others:
        raise SystemExit(f"{rows_path} holds rows of another adapter {sorted(map(str, others))}; use another --output")
    other_modes = {row_on_mode(r) for r in done.values()} - {args.on_mode}
    if other_modes:
        raise SystemExit(f"{rows_path} holds rows of --on-mode {sorted(other_modes)}, not {args.on_mode}; use another "
                         "--output")
    todo = [s for s in mine if not (s["key"] in done and _complete(done[s["key"]], s, args.shuffle_instructions))]
    LOGGER.info("%d already decoded, %d to go (on-mode %s)", len(mine) - len(todo), len(todo), args.on_mode)
    if todo:
        rt = init_runtime(args.device)
        seed_everything(args.seed, rt.device)
        model, processor = load_system2(args.model_path, rt.device)
        model = load_adapter(model, args.adapter)
        model.eval()
        records = RecordCache(8)
        size = tuple(args.default_vlm_image_size)
        stop_ids = stop_token_ids(processor.tokenizer) if args.on_mode == "base-stop" else ()
        if stop_ids:
            LOGGER.info("base-stop: STOP token ids %s (%s)", list(stop_ids), list(STOP_TOKEN_TEXTS))

        def both(record_like, call):
            messages, images, lookdown = inference_messages(record_like, call, default_vlm_image_size=size)
            if args.on_mode == "base-stop":
                def decide(adapter_on, suppress):
                    if adapter_on:
                        return greedy_first_token_decision(model, processor, messages, images, lookdown, rt.device,
                                                           suppress_first=suppress)
                    with adapter_off(model):
                        return greedy_first_token_decision(model, processor, messages, images, lookdown, rt.device,
                                                           suppress_first=suppress)

                return base_stop_arms(decide, stop_ids)
            with torch.no_grad():
                on = greedy_native_decision(model, processor, messages, images, lookdown, rt.device)
                with adapter_off(model):
                    off = greedy_native_decision(model, processor, messages, images, lookdown, rt.device)
            return _arm(on), _arm(off)

        with open(rows_path, "a", encoding="utf-8") as handle:
            for i, state in enumerate(todo, 1):
                t0 = time.time()
                record, calls = records.get(state["record_path"])
                call = calls[state["call_index"]]
                on, off = both(record, call)
                row = {"schema": DECODE_SCHEMA, "key": state["key"], "uid": state["uid"], "adapter_sha256": adapter_sha,
                       "device": rt.device.type, "record_device": record_device(record), "on": on, "off": off,
                       "shuffle_instructions": bool(args.shuffle_instructions), "on_mode": args.on_mode}
                if stop_ids:
                    row["stop_token_ids"] = list(stop_ids)
                if args.shuffle_instructions and _needs_shuffle(state):
                    s_on, s_off = both(WithInstruction(record, state["shuffle"]["instruction"]), call)
                    row["shuffled"] = {"donor": state["shuffle"]["donor"], "on": s_on, "off": s_off}
                row["seconds"] = round(time.time() - t0, 3)
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                handle.flush()
                if i % args.log_every == 0 or i == len(todo):
                    LOGGER.info("%d/%d decoded (%.1f s/state)", i, len(todo), (time.time() - started) / i)
    print(json.dumps({"shard": f"{index}/{count}", "states": len(mine), "decoded_now": len(todo),
                      "rows": str(rows_path), "seconds": round(time.time() - started, 1)}), flush=True)
    return 0


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def navmesh_contexts(scenes_dir: str | os.PathLike, dataset_index: Mapping | None, config: Any):
    """``contexts_for`` on the scenes' navmeshes; a record not collected on that very navmesh is refused."""
    geometries: dict[str, Any] = {}
    digests: dict[str, str] = {}

    def contexts_for(record: Any, indices: Collection[int]):
        scene = record.scene_id
        navmesh = Path(scenes_dir) / scene / f"{scene}.navmesh"
        if scene not in geometries:
            geometries[scene] = lr.RouteNavmeshGeometry(navmesh)
            digests[scene] = _sha256(navmesh)
        # EXP-28's client writes it into every HM3D record (r2r_val_unseen._episode_navmesh_sha256); without
        # it nothing ties the geodesics judged here to the navmesh the episode ran on.
        expected = record.episode.get("navmesh_sha256")
        if expected is None:
            raise GateInvalid(f"{record.path}: episode.json has no navmesh_sha256, so it was not collected by "
                              "EXP-28's client on the rebaked navmesh")
        if expected != digests[scene]:
            raise GateInvalid(f"{record.path}: collected on navmesh {expected}, {navmesh} is {digests[scene]}")
        entry = None
        if dataset_index is not None:
            entry = dataset_index.get((scene, record.episode_id))
            if entry is None:
                raise GateInvalid(f"{scene}/{record.episode_id} is not in --episodes")
        return call_contexts(record, geometries[scene], dataset_entry=entry, config=config, only=indices)

    return contexts_for


def score(args: argparse.Namespace) -> int:
    started = time.time()
    output = Path(args.output)
    report: dict[str, Any] = {"schema": SCHEMA, "valid": False}
    meta: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    paths = sorted(output.parent.glob(f"{output.stem}.decode.shard*of*.jsonl"))
    try:
        states, meta = _selection(args)
        decoded = read_decode_rows(paths)
        config = RL.RouteLabelConfig()
        dataset_index = lr.base.load_episode_index(args.episodes) if args.episodes else None
        shuffle_requested = any(r.get("shuffle_instructions") or r.get("shuffled") for r in decoded.values())
        rows, joined = score_states(states, decoded, navmesh_contexts(args.scenes_dir, dataset_index, config))
        summary = summarize(rows, shuffle_requested=shuffle_requested)
        report.update(summary)
        report.update({"join": joined, "shuffle_requested": shuffle_requested})
        problems = measurement_problems(summary)
        if problems:  # the numbers stay in the report for a look; no pass or fail is read from them
            report["invalid"] = "GateInvalid: " + "; ".join(problems)
        else:
            report["valid"] = True
    except (RuntimeError, ValueError, KeyError, OSError) as err:
        # Stale labels, a navmesh that is missing or will not load (GateInvalid is a RuntimeError),
        # undecoded states: no pass or fail can be read (exit 2, not 1).
        report["invalid"] = f"{type(err).__name__}: {err}"
        rows = []
    report.update({
        "selection": meta,
        "inputs": {"record_roots": [str(r) for r in args.record_root], "label_root": str(args.label_root),
                   "scenes_dir": str(args.scenes_dir), "episodes": list(args.episodes or []),
                   "decode_files": [str(p) for p in paths]},
        "rules": "ledger EXP-28 D, written 2026-10-15 19:35 UTC before any HM3D result",
        "seconds": round(time.time() - started, 1),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output.with_suffix(".calls.jsonl"), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(output)
    if not report["valid"]:
        print(f"INVALID O4/O5: {report['invalid']}", flush=True)
        return 2
    o4, o5 = report["O4"], report["O5"]
    parts = [f"{k}={v['value']}{'' if v['pass'] else '(FAIL)'}" for k, v in
             {**o4["criteria"], **o5["criteria"]}.items()]
    if report["join"]["on_mode"] != "adapter":
        parts.append(f"(on-mode {report['join']['on_mode']})")
    print(("PASS" if o4["all_pass"] and o5["all_pass"] else "FAIL") + " O4/O5 " + " ".join(parts), flush=True)
    return 0 if o4["all_pass"] and o5["all_pass"] else 1


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def run_verdict(args: argparse.Namespace) -> int:
    def load(path: str) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    exp32 = args.rules == "exp32"
    title = "EXP-32" if exp32 else "EXP-28"
    try:
        o1 = compare_agreement(agreement_rows(args.o1), agreement_rows(args.o1_base), reference=O1_REFERENCE)
        o2 = compare_agreement(agreement_rows(args.o2), agreement_rows(args.o2_base), reference=O2_REFERENCE)
        if exp32:
            result = verdict_exp32(o1=o1, o2=o2, o3=load(args.o3), o4=load(args.o4), o4_rows=read_rows(args.o4_calls),
                                   rgate=load(args.rgate), attempt=args.attempt)
        else:
            result = verdict(o1=o1, o2=o2, o3=load(args.o3), o4=load(args.o4), attempt=args.attempt)
    except (ValueError, KeyError, OSError) as err:  # json errors are ValueErrors
        # A missing or foreign input: nothing was judged (exit 2, never the FAIL of exit 1).
        print(f"INVALID {title} offline gates attempt {args.attempt}: {type(err).__name__}: {err}", flush=True)
        return 2
    keys = ("o1", "o1_base", "o2", "o2_base", "o3", "o4") + (("o4_calls", "rgate") if exp32 else ())
    result["inputs"] = {k: str(Path(getattr(args, k)).resolve()) for k in keys}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    groups = " ".join(f"{k}={'pass' if v else 'FAIL'}" for k, v in result["groups"].items())
    if exp32 and (result["R"].get("gate") or {}).get("uses") == "M":
        groups += " (R read from M: R is report-only)"
    head = "PASS" if result["all_pass"] else ("FAIL" if result["measured"] else "INVALID")
    print(f"{head} {title} offline gates attempt {args.attempt}: {groups} -> next: {result['next']}", flush=True)
    for note in result["notes"]:
        print(f"[verdict] {note}", file=sys.stderr)
    return 0 if result["all_pass"] else (1 if result["measured"] else 2)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--record-root", action="append", required=True, help="committed base records (repeatable)")
    parser.add_argument("--label-root", required=True, help="their line-B labels (label_route_corrections.py)")
    parser.add_argument("--holdout-scenes-file", default="", help="held-out scenes, one per line (ID or 00xxx-ID)")
    parser.add_argument("--scenes", default="", help="comma-separated held-out scenes (added to the file's)")
    parser.add_argument("--output", required=True, help="summary json; rows beside it (see the module doc)")
    parser.add_argument("--max-call-index", type=int, default=MAX_CALL_INDEX, help="R1: calls 0..N-1 of an episode")
    parser.add_argument("--max-episodes", type=int, default=0, help="seeded subset of episodes, for smoke runs (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=42, help="O5's within-scene shuffle")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    dec = sub.add_parser("decode", help="decode the states (GPU)")
    _selection_args(dec)
    dec.add_argument("--adapter", required=True, help="PEFT directory (adapter_config.json + adapter_model.safetensors)")
    dec.add_argument("--model-path", required=True, help="the deployed InternNav_Model directory")
    dec.add_argument("--shard", default="0/1", help="i/n: decode the states whose key hashes to i")
    dec.add_argument("--shuffle-instructions", action="store_true",
                     help="O5: also decode every correction under another episode's instruction of its scene")
    dec.add_argument("--on-mode", choices=ON_MODES, default="adapter",
                     help="the on answer: the adapter's own (default), or base-stop: the server's "
                          "--system2_base_decides_stop answer (STOP iff the base's first token is STOP, else the "
                          "adapter with STOP masked at the first position; EXP-37)")
    dec.add_argument("--default-vlm-image-size", type=int, nargs=2, default=[384, 384])
    dec.add_argument("--device", choices=("auto", "cuda", "npu", "cpu"), default="auto")
    dec.add_argument("--log-every", type=int, default=100)
    dec.add_argument("--dry-run", action="store_true", help="count the states only; no model")

    sco = sub.add_parser("score", help="judge the decoded answers with line B (CPU, habitat_sim PathFinder)")
    _selection_args(sco)
    sco.add_argument("--scenes-dir", required=True, help="holds <scene>/<scene>.navmesh (the labeller's)")
    sco.add_argument("--episodes", nargs="*", default=[], help="R2R-CE-format split json.gz, as given to the labeller")

    ver = sub.add_parser("verdict", help="O1-O5 (or EXP-32's O1-O4, L, R) and the pre-registered next step")
    ver.add_argument("--o1", required=True, help="teacher_forced_agreement.py summary, candidate, B3's 5 scenes")
    ver.add_argument("--o1-base", required=True, help="the same samples, base model")
    ver.add_argument("--o2", required=True, help="teacher_forced_agreement.py summary, candidate, r2r held-out")
    ver.add_argument("--o2-base", required=True)
    ver.add_argument("--o3", required=True, help="stratified_gates.py output (with the O3_exp28 block)")
    ver.add_argument("--o4", required=True, help="this script's score output (O4 and O5)")
    ver.add_argument("--attempt", type=int, choices=(1, 2), default=1, help="2: the one retry")
    ver.add_argument("--output", required=True)
    ver.add_argument("--rules", choices=("exp28", "exp32"), default="exp28",
                     help="exp28: ledger EXP-28 D (O1-O5); exp32: ledger EXP-32 第二部分 F (O1-O4 without O4's "
                          "correction item, L, R; needs --rgate)")
    ver.add_argument("--o4-calls", default="", help="exp32: score's rows for L (default: <o4 stem>.calls.jsonl)")
    ver.add_argument("--rgate", default="", help="exp32: scripts/exp32/rft_gate.py score's rft_gate.json")
    args = parser.parse_args(argv)
    if args.command == "verdict":
        if args.rules == "exp32" and not args.rgate:
            ver.error("--rules exp32 needs --rgate (scripts/exp32/rft_gate.py score's rft_gate.json)")
        if args.rules == "exp28" and (args.rgate or args.o4_calls):
            ver.error("--rgate and --o4-calls are read only under --rules exp32")
        if args.rules == "exp32" and not args.o4_calls:
            args.o4_calls = str(Path(args.o4).with_suffix(".calls.jsonl"))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    return {"decode": decode, "score": score, "verdict": run_verdict}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

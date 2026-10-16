#!/usr/bin/env python3
"""EXP-28 HM3D episodes: StreamVLN's ScaleVLN annotation -> R2R-CE-schema episodes, scene split and cohort.

The rules are the pre-registered ones (docs/experiments/README.md, EXP-28 "设置"); this script only applies them.

Input: ``scalevln_subset_150k.json.gz`` (HF ``cywan/StreamVLN-Trajectory-Data``, ``ScaleVLN/``: 155,098 VLN-CE
episodes over 795 HM3D train scenes, ``scene_id`` ``hm3d/00xxx-<ID>/<ID>.basis.glb`` without ``train/``,
sha256 ``23e31a05...81abdf``).  Its positions are used as they are: Habitat world coordinates of the ``.basis.glb``.

1. **Scenes.**  Used are the 300 InternData-N1 ScaleVLN scene dirs on the 4090 (``--interndata-root``, those with
   ``meta/episodes.jsonl``) plus B3's five (``--b3-root``).  Held out: B3's five plus the 15 of the 300 with the
   smallest ``sha256("exp28-hm3d-heldout:" + <scene dir>)``; the other 285 train.  A scene of the 300 that the
   annotation lacks cannot be used and is reported (it would shift the hash pick to the next scene).
2. **Episodes** (only those of the used scenes) in the schema of R2R_VLNCE_v1-3_preprocessed train (checked key by
   key against ``--r2r-train``): top-level keys other than ``episodes`` (``instruction_vocab``) copied from it;
   ``scene_id`` ``hm3d/<ID>/<ID>.glb`` (the scene tree of ``scripts/exp28/prepare_hm3d_scenes.py``: the client keys
   episodes by ``Path(scene_id).stem`` and by the parent dir, which must agree); ``episode_id`` and
   ``trajectory_id`` kept (as ints); ``instruction_tokens`` re-indexed into the copied vocabulary with
   habitat-lab's ``VocabDict.tokenize_and_index`` rule (mirrored here) and padded like R2R's; ``info`` also
   carries ``navmesh_sha256``, the navmesh the episode was filtered on, which ``r2r_val_unseen.py`` checks against
   the navmesh Habitat loads after each reset.  Asserted: every ``episode_id`` of the annotation is unique, an
   HM3D ID names one scene dir, no HM3D ID is one of the MP3D IDs (``--mp3d-dir`` plus R2R train's scenes), the
   episodes of a route (scene, ``trajectory_id``) share start, goal and path, and every used ``start_rotation`` is
   a unit yaw quaternion in R2R-CE's ``[x, y, z, w]`` order (x = z = 0): another order would start every agent
   facing the wrong way and only V0 would see it.
3. **Filter** on the rebaked navmesh the client loads (``<scenes-dir>/hm3d/<ID>/<ID>.navmesh``, a bare
   ``habitat_sim.PathFinder`` through the line-B labeller's own geometry class: no renderer, no GPU): start, goal
   and every reference-path point snap within 0.5 m; the start-goal geodesic is finite and within max(1 m, 15%) of
   ``info.geodesic_distance``.  The annotation's positions are used as they are, as Habitat world coordinates
   (y up) of the ``.basis.glb``, which the tree links unchanged; D0 and V0 are what test that.  D0: more than 25%
   of the used scenes' episodes dropped means coordinates or navmeshes are wrong; so does a used scene that loses
   every episode (habitat-sim 0.1.7's ``is_loaded`` does not prove a navmesh usable, and such a scene would
   otherwise just vanish from the cohort) -- nothing is written but ``<out-dir>.d0_failed.json``, exit 3.  An
   episode without ``info.geodesic_distance`` is dropped (``--missing-info-geodesic fill`` keeps it on the finite
   check alone and writes the navmesh's geodesic).
4. **One instruction per route**, chosen by a seeded hash; routes in a seeded hash order
   (``build_r2r_train_cohorts.select_routes``, the R2R cohorts' rule).  A route is a path: routes of one scene
   whose start, goal and reference path agree at 1 cm under different ``trajectory_id``s are one route, the first
   in draw order (the others' ids are listed beside it in ``routes.json`` and never drawn).
5. **Cohort** (``heatmapvln-r2r-train-cohorts-v1``, split train, ``run_s2_dagger_collect_cuda.sh`` takes it as is):
   held-out shards first (20 scenes x the first 20 routes, one scene per shard), then train pass 1 (285 scenes x
   routes 1-6, one scene per shard), then pass 2 (routes 7-12).  Scenes in a seeded hash order inside each part.
   The manifest's ``parts`` give each part's shard range for ``PPA_COLLECT_SHARDS``; routes no part took are C2's.
6. **Overlap with InternData**: a json episode overlaps when InternData has the same scene and instruction, both
   through ``interndata_replay.client_instruction`` after collapsing whitespace, lower case (the key V0 pairs
   episodes by, ``verify_hm3d_render.match_key``; the client's rule alone and a words-only key are reported too).
   Reported per stage; if the annotation's overlap rate in the used scenes is < 80% on that key, held-out routes
   are drawn only from routes none of whose instructions InternData has, on the words-only key (the widest of the
   three, so a punctuation difference cannot let an InternData route into the held-out cohort); training is not
   filtered.

Output (``--out-dir``, created whole or not at all)::

    pool.json.gz              every filtered route of the used scenes, one instruction each (R2R-CE split file;
                              the labeller's --episodes)
    cohorts/                  the cohort: manifest.json, shard_NNN.json, dataset_shard_NNN.json.gz
    splits/heldout_scenes.txt the 20 held-out scene dirs (merge_streams.py --holdout-scenes-file)
    splits/train_scenes.txt   the 285 training scene dirs
    splits/splits.json        both, with IDs, hash ranks and why
    routes.json               per scene, its routes in draw order and the cohort part that took each (null: unused,
                              C2's; trajectory_ids merged into a route by path are listed beside it, never drawn)
    overlap_keys.json         [ID, episode_id] of the pool episodes InternData has (phase0_report.py --overlap-keys)
    filter_report.json        per-scene drops by reason, snap/geodesic quantiles, navmesh sha256, dropped examples
    manifest.json             inputs with sha256, rules, counts, D0, overlap, tokenizer check, files with sha256

Example (4090 container, CPU only, after prepare_hm3d_scenes.py)::

    /opt/conda/bin/python scripts/exp28/build_hm3d_episodes.py \\
        --source /workspace/exp28/data/scalevln_subset_150k.json.gz \\
        --r2r-train /workspace/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz \\
        --mp3d-dir /dataset/mp3d --scenes-dir /workspace/exp28/scene_datasets \\
        --interndata-root /workspace/exp23/data/interndata_n1/scalevln_train \\
        --b3-root /workspace/exp23/data/interndata_n1/scalevln \\
        --out-dir /workspace/exp28/episodes --workers 12

``--dry-run`` stops before the navmeshes: it prints the scene split and the overlap and writes nothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools import label_route_corrections as route_labeller  # noqa: E402  (numpy-only loader of src/data)
from scripts.tools.build_r2r_train_cohorts import (  # noqa: E402
    SCHEMA as COHORT_SCHEMA,
    CohortError,
    _digest,
    deal_shards,
    deterministic_gzip_json,
    scene_name,
    select_routes,
    sha256_file,
)

# src/data/interndata_replay.py without src/data/__init__.py (torch, pydantic), as the labellers load src/data.
IR = importlib.import_module(route_labeller.ORL.__name__.rsplit(".", 1)[0] + ".interndata_replay")
INSTRUCTION_SEP, client_instruction = IR.INSTRUCTION_SEP, IR.client_instruction

SCHEMA = "heatmapvln-exp28-hm3d-episodes-v1"
SOURCE_SHA256 = "23e31a057858691e239d019a7605c622bfe266aa6e0f9a68ddcdd7830d81abdf"
B3_SCENES = ("00481-5jLhtVmWd5F", "00393-dD37vuCa4FE", "00396-wuKkTq5GJbi", "00560-gjhYih4upQ9", "00770-NBg5UqG3di3")
HELDOUT_SALT = "exp28-hm3d-heldout:"
OVERLAP_RULE_THRESHOLD = 0.80
MAX_DROP_RATE = 0.25  # D0

SOURCE_SCENE = re.compile(r"^hm3d/(?:train/)?(?P<dir>\d{5}-(?P<id>[A-Za-z0-9]{11}))/(?P=id)\.basis\.glb$")
SCENE_DIR = re.compile(r"^\d{5}-(?P<id>[A-Za-z0-9]{11})$")
# The keys this script writes; the R2R train file must have exactly these (checked, not assumed).
EPISODE_KEYS = (
    "episode_id", "goals", "info", "instruction", "reference_path", "scene_id", "start_position", "start_rotation",
    "trajectory_id",
)
INSTRUCTION_KEYS = ("instruction_text", "instruction_tokens")
GOAL_KEYS = ("position", "radius")
INFO_KEYS = ("geodesic_distance",)


class D0Failed(CohortError):
    """More episodes failed the geometric filter than D0 allows; nothing but the report is written."""


@dataclasses.dataclass(frozen=True)
class FilterRule:
    """EXP-28 "设置 / 集 / 过滤".  ``missing_info_geodesic``: what an episode without ``info.geodesic_distance``
    gets -- ``drop`` (the pre-registered rule cannot be applied) or ``fill`` (kept on the finite-geodesic check
    alone, ``info.geodesic_distance`` set to the navmesh's; counted in the manifest)."""

    snap_m: float = 0.5
    geodesic_abs_m: float = 1.0
    geodesic_rel: float = 0.15
    missing_info_geodesic: str = "drop"


# --------------------------------------------------------------------------- #
# Scenes
# --------------------------------------------------------------------------- #
def parse_source_scene(scene_id: Any) -> tuple[str, str]:
    """``hm3d/00xxx-<ID>/<ID>.basis.glb`` (``train/`` optional) -> (``00xxx-<ID>``, ``<ID>``)."""
    match = SOURCE_SCENE.match(str(scene_id))
    if match is None:
        raise CohortError(f"scene_id {scene_id!r} is not hm3d/00xxx-<ID>/<ID>.basis.glb")
    return match["dir"], match["id"]


def hm3d_scene_id(hm3d_id: str) -> str:
    """The scene id the episodes carry: relative to the EXP-28 scene tree, stem == parent dir == ID."""
    return f"hm3d/{hm3d_id}/{hm3d_id}.glb"


def dir_id(scene_dir: str) -> str:
    match = SCENE_DIR.match(scene_dir)
    if match is None:
        raise CohortError(f"scene dir {scene_dir!r} is not 00xxx-<11-character HM3D ID>")
    return match["id"]


def heldout_rank(scene_dir: str) -> str:
    return hashlib.sha256(f"{HELDOUT_SALT}{scene_dir}".encode("utf-8")).hexdigest()


def interndata_scene_dirs(root: Path) -> dict[str, Path]:
    """InternData-N1 scene dirs under ``root`` (those with ``meta/episodes.jsonl``), by dir name."""
    if not root.is_dir():
        raise CohortError(f"{root} is not a directory")
    out = {}
    for path in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if (path / "meta" / "episodes.jsonl").is_file():
            dir_id(path.name)
            out[path.name] = path
    return out


def split_scenes(
    interndata_dirs: Iterable[str], b3: Sequence[str], annotated: set[str], *, from_hash: int
) -> dict[str, Any]:
    """Held out: B3 plus the ``from_hash`` smallest-ranked usable InternData scenes; the rest of those trains."""
    pool = sorted(set(interndata_dirs))
    b3 = list(b3)
    if len(set(b3)) != len(b3):
        raise CohortError(f"B3 scenes repeat: {b3}")
    if set(b3) & set(pool):
        raise CohortError(f"B3 scenes are among the InternData train scenes: {sorted(set(b3) & set(pool))}")
    missing_b3 = [s for s in b3 if s not in annotated]
    if missing_b3:
        raise CohortError(f"the annotation has no episode in B3 scenes {missing_b3}")
    unusable = [s for s in pool if s not in annotated]
    ranked = sorted((s for s in pool if s in annotated), key=lambda s: (heldout_rank(s), s))
    if len(ranked) <= from_hash:
        raise CohortError(f"only {len(ranked)} usable InternData scenes, {from_hash} are to be held out")
    plain = sorted(pool, key=lambda s: (heldout_rank(s), s))[:from_hash]
    return {
        "heldout": [*b3, *ranked[:from_hash]],
        "heldout_from_b3": b3,
        "heldout_from_hash": ranked[:from_hash],
        "train": sorted(ranked[from_hash:]),
        "interndata_scenes_without_episodes": unusable,
        # Scenes the plain rule ("the 15 smallest of the 300") would have held out: the pick moved past them.
        "heldout_hash_skipped": sorted(s for s in plain if s not in annotated),
    }


def mp3d_scene_ids(mp3d_dir: Path) -> set[str]:
    if not mp3d_dir.is_dir():
        raise CohortError(f"{mp3d_dir} is not a directory")
    return {p.name for p in mp3d_dir.iterdir() if p.is_dir() and (p / f"{p.name}.glb").exists()}


# --------------------------------------------------------------------------- #
# Source annotation
# --------------------------------------------------------------------------- #
def _drop_source_tokens(obj: dict) -> dict:
    # The source's token ids index its own vocabulary and are re-made below; dropping them while
    # parsing keeps 155k x 200 ints out of memory.
    if "instruction_text" in obj:
        obj.pop("instruction_tokens", None)
    return obj


def load_json_gz(path: Path, *, drop_instruction_tokens: bool = False) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle, object_hook=_drop_source_tokens if drop_instruction_tokens else None)
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list) or not payload["episodes"]:
        raise CohortError(f"{path} has no 'episodes' array")
    return payload


def index_source(episodes: Sequence[Any]) -> dict[str, list[dict[str, Any]]]:
    """Source episodes by scene dir, after the global checks (scene path, IDs, unique episode_id)."""
    by_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dir_of_id: dict[str, str] = {}
    seen: set[int] = set()
    repeats: list[int] = []
    scene_keys: set[tuple[str, int]] = set()
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise CohortError(f"source episode[{index}] is not an object")
        try:
            scene_dir, hm3d_id = parse_source_scene(episode["scene_id"])
            episode_id = int(episode["episode_id"])
            int(episode["trajectory_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CohortError(
                f"source episode[{index}] lacks a usable scene_id/episode_id/trajectory_id: {exc}"
            ) from exc
        if dir_of_id.setdefault(hm3d_id, scene_dir) != scene_dir:
            raise CohortError(f"HM3D ID {hm3d_id} names two scene dirs: {dir_of_id[hm3d_id]}, {scene_dir}")
        if episode_id in seen:
            repeats.append(episode_id)
        seen.add(episode_id)
        scene_keys.add((scene_dir, episode_id))
        by_dir[scene_dir].append(episode)
    if repeats:
        per_scene = "unique" if len(scene_keys) == len(episodes) else "not unique either"
        raise CohortError(
            f"episode_id is not globally unique: {len(repeats)} repeats, e.g. {sorted(set(repeats))[:5]} "
            f"((scene, episode_id) is {per_scene})"
        )
    return dict(by_dir)


# --------------------------------------------------------------------------- #
# R2R reference: schema, vocabulary, tokens
# --------------------------------------------------------------------------- #
SENTENCE_SPLIT_REGEX = re.compile(r"([^\w-]+)")


class VocabTokenizer:
    """habitat-lab's ``tokenize`` rule over R2R train's ``word_list``, padded to R2R's fixed length.

    The sentence is lowered, ``'s`` split off, ``,`` and ``?`` removed, then split on runs of characters other
    than word characters and ``-``; unknown words map to ``<unk>`` (put first if the list lacked it, which R2R's
    does not).  Whether this reproduces R2R's own tokens is measured on the R2R file and reported, not assumed:
    the deployed client sends ``instruction_text`` and never reads the tokens.
    """

    def __init__(self, word_list: Sequence[str], length: int | None):
        words = list(word_list)
        if "<unk>" not in words:
            words = ["<unk>", *words]
        self.index = {word: position for position, word in enumerate(words)}
        self.unk = self.index["<unk>"]
        self.pad = self.index.get("<pad>", 0)
        self.length = length

    @staticmethod
    def tokenize(sentence: str) -> list[str]:
        sentence = sentence.lower()
        for token in ("'s",):
            sentence = sentence.replace(token, " " + token)
        for token in (",", "?"):
            sentence = sentence.replace(token, "")
        return [t.strip() for t in SENTENCE_SPLIT_REGEX.split(sentence) if t.strip()]

    def __call__(self, sentence: str) -> list[int]:
        ids = [self.index.get(word, self.unk) for word in self.tokenize(sentence)]
        if self.length is None:
            return ids
        return ids[: self.length] + [self.pad] * max(0, self.length - len(ids))


@dataclasses.dataclass(frozen=True)
class R2RReference:
    top_level: dict[str, Any]  # every top-level key but "episodes", copied into each output file
    token_length: int | None
    scenes: frozenset[str]
    tokenizer_reproduction: float
    yaw_quaternions: float  # share of R2R train's start_rotations is_yaw_quaternion_xyzw accepts


def load_r2r_reference(path: Path) -> R2RReference:
    payload = load_json_gz(path)
    vocab = payload.get("instruction_vocab")
    if not isinstance(vocab, dict) or not isinstance(vocab.get("word_list"), list):
        raise CohortError(f"{path} has no instruction_vocab.word_list")
    shapes: Counter = Counter()
    lengths: Counter = Counter()
    for episode in payload["episodes"]:
        instruction, goal = episode.get("instruction") or {}, (episode.get("goals") or [{}])[0]
        shapes[(tuple(sorted(episode)), tuple(sorted(instruction)), tuple(sorted(goal)),
                tuple(sorted(episode.get("info") or {})))] += 1
        lengths[len(instruction.get("instruction_tokens") or ())] += 1
    expected = (tuple(sorted(EPISODE_KEYS)), tuple(sorted(INSTRUCTION_KEYS)), tuple(sorted(GOAL_KEYS)),
                tuple(sorted(INFO_KEYS)))
    if set(shapes) != {expected}:
        raise CohortError(
            f"{path}: episode schema {sorted(shapes)} is not the one this script writes {expected}; "
            "the output would not match R2R's"
        )
    token_length = next(iter(lengths)) if len(lengths) == 1 else None
    tokenizer = VocabTokenizer(vocab["word_list"], token_length)
    same = sum(
        tokenizer(e["instruction"]["instruction_text"]) == list(e["instruction"]["instruction_tokens"])
        for e in payload["episodes"]
    )
    return R2RReference(
        top_level={k: v for k, v in payload.items() if k != "episodes"},
        token_length=token_length,
        scenes=frozenset(scene_name(e["scene_id"]) for e in payload["episodes"]),
        tokenizer_reproduction=same / len(payload["episodes"]),
        yaw_quaternions=sum(is_yaw_quaternion_xyzw(e["start_rotation"]) for e in payload["episodes"])
        / len(payload["episodes"]),
    )


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def _vec(value: Any, n: int, what: str) -> list[float]:
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise CohortError(f"{what} is not a list of numbers: {value!r}") from exc
    if len(out) != n or not all(math.isfinite(v) for v in out):
        raise CohortError(f"{what} must be {n} finite numbers, got {value!r}")
    return out


def convert_episode(source: Mapping[str, Any], tokenizer: Callable[[str], list[int]]) -> dict[str, Any]:
    """One source episode in the R2R-CE train schema; ``info.geodesic_distance`` is None when the source lacks it."""
    episode_id = int(source["episode_id"])
    what = f"episode {episode_id}"
    _, hm3d_id = parse_source_scene(source["scene_id"])
    instruction = source.get("instruction")
    text = instruction.get("instruction_text") if isinstance(instruction, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise CohortError(f"{what}: no instruction_text")
    goals = source.get("goals")
    if not isinstance(goals, list) or len(goals) != 1 or not isinstance(goals[0], dict):
        raise CohortError(f"{what}: goals must hold exactly one goal, got {goals!r}")
    path = source.get("reference_path")
    if not isinstance(path, list) or not path:
        raise CohortError(f"{what}: empty reference_path")
    info = source.get("info") if isinstance(source.get("info"), dict) else {}
    geodesic = info.get("geodesic_distance")
    return {
        "episode_id": episode_id,
        "trajectory_id": int(source["trajectory_id"]),
        "scene_id": hm3d_scene_id(hm3d_id),
        "start_position": _vec(source.get("start_position"), 3, f"{what} start_position"),
        "start_rotation": _vec(source.get("start_rotation"), 4, f"{what} start_rotation"),
        "info": {"geodesic_distance": None if geodesic is None else float(geodesic)},
        "goals": [{
            "position": _vec(goals[0].get("position"), 3, f"{what} goal"),
            "radius": float(goals[0].get("radius", 3.0)),
        }],
        "instruction": {"instruction_text": text, "instruction_tokens": tokenizer(text)},
        "reference_path": [_vec(p, 3, f"{what} reference_path point") for p in path],
    }


def _close(a: Sequence[float], b: Sequence[float], tol: float = 1e-3) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def check_route_geometry(routes: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]]) -> None:
    """The episodes of one (scene, trajectory_id) are one route: same start, goal and reference path."""
    bad = []
    for key, episodes in routes.items():
        first = episodes[0]
        for other in episodes[1:]:
            same = (
                _close(first["start_position"], other["start_position"])
                and _close(first["goals"][0]["position"], other["goals"][0]["position"])
                and len(first["reference_path"]) == len(other["reference_path"])
                and all(_close(p, q) for p, q in zip(first["reference_path"], other["reference_path"]))
            )
            if not same:
                bad.append(key)
                break
    if bad:
        raise CohortError(
            f"{len(bad)} (scene, trajectory_id) groups hold different paths, e.g. {sorted(bad)[:5]}: "
            "trajectory_id does not identify a route in this annotation"
        )


Route = tuple[tuple[str, int], list[dict[str, Any]]]


def path_signature(episode: Mapping[str, Any]) -> tuple:
    """Start, goal and reference path at 1 cm: a route's geometry, whatever its trajectory_id."""
    points = [episode["start_position"], episode["goals"][0]["position"], *episode["reference_path"]]
    return tuple(round(float(v), 2) + 0.0 for p in points for v in p)  # + 0.0: -0.0 and 0.0 alike


def routes_sharing_a_path(routes: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]]) -> int:
    """Routes whose start, goal and reference path (at 1 cm) another trajectory_id of the scene repeats
    (``merge_same_path_routes`` makes them one route before anything is drawn)."""
    groups: Counter = Counter()
    for (hm3d_id, _), episodes in routes.items():
        groups[(hm3d_id, path_signature(episodes[0]))] += 1
    return sum(n for n in groups.values() if n > 1)


def merge_same_path_routes(drawn: Sequence[Route]) -> tuple[list[Route], dict[tuple[str, int], list[Route]]]:
    """(routes in draw order, one per path; {kept route: [(route, episodes) of the same path drawn later]}).

    One instruction per route means one per path: two trajectory_ids over one path would otherwise give the
    same route twice (pass 1 and pass 2, or a pass and C2) under two instructions.
    """
    owner_of: dict[tuple, tuple[str, int]] = {}
    kept, merged = [], defaultdict(list)
    for key, episodes in drawn:
        owner = owner_of.setdefault((key[0], path_signature(episodes[0])), key)
        if owner == key:
            kept.append((key, episodes))
        else:
            merged[owner].append((key, episodes))
    return kept, dict(merged)


def is_yaw_quaternion_xyzw(rotation: Sequence[float], *, tol: float = 0.02) -> bool:
    """A unit rotation about +y in habitat's ``[x, y, z, w]`` order: R2R-CE's start_rotation convention.

    The same rotation written ``[w, x, y, z]`` has its first component nonzero (unless the yaw is 180 degrees),
    so an annotation in that order fails here on nearly every episode instead of turning every agent around.
    """
    x, y, z, w = (float(v) for v in rotation)
    return abs(math.sqrt(x * x + y * y + z * z + w * w) - 1.0) <= 1e-3 and abs(x) <= tol and abs(z) <= tol


# --------------------------------------------------------------------------- #
# Geometric filter
# --------------------------------------------------------------------------- #
class NavmeshGeometry(route_labeller.RouteNavmeshGeometry):
    """The line-B labeller's own navmesh geometry (``habitat_sim.PathFinder``, no renderer: the same snap and
    geodesic queries ``label_route_corrections.py`` makes on this tree), plus the navmesh's sha256."""

    def __init__(self, navmesh_path: str | os.PathLike):
        if not Path(navmesh_path).is_file():
            raise FileNotFoundError(f"navmesh not found: {navmesh_path}")
        self.sha256 = sha256_file(Path(navmesh_path))
        try:
            super().__init__(navmesh_path)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} (a v2 navmesh? rebake it with scripts/exp28/prepare_hm3d_scenes.py)") from exc


@dataclasses.dataclass(frozen=True)
class SceneTree:
    """``<root>/hm3d/<ID>/<ID>.glb`` and its rebaked ``.navmesh`` -- what the client loads."""

    root: Path

    def __call__(self, hm3d_id: str) -> NavmeshGeometry:
        glb = self.root / "hm3d" / hm3d_id / f"{hm3d_id}.glb"
        if not glb.exists():
            raise FileNotFoundError(
                f"{glb} is missing: build the scene tree first (scripts/exp28/prepare_hm3d_scenes.py)"
            )
        return NavmeshGeometry(glb.with_suffix(".navmesh"))


def _snap_distance(geometry: Any, point: Sequence[float]) -> float:
    snapped = geometry.snap(point)
    return math.inf if snapped is None else math.dist(snapped, point)


def check_geometry(
    start: Sequence[float],
    goal: Sequence[float],
    path: Sequence[Sequence[float]],
    info_geodesic: float | None,
    geometry: Any,
    rule: FilterRule = FilterRule(),
) -> tuple[list[str], dict[str, float | None]]:
    """(reasons to drop -- empty keeps it, measurements) for one episode."""
    start_snap = _snap_distance(geometry, start)
    goal_snap = _snap_distance(geometry, goal)
    path_snap = max(_snap_distance(geometry, p) for p in path)
    geodesic = geometry.geodesic(start, goal)
    reasons = [name for name, d in (("start_snap", start_snap), ("goal_snap", goal_snap), ("path_snap", path_snap))
               if not d <= rule.snap_m]
    if not math.isfinite(geodesic):
        reasons.append("geodesic_not_finite")
    elif info_geodesic is None:
        if rule.missing_info_geodesic != "fill":
            reasons.append("no_info_geodesic")
    elif not abs(geodesic - info_geodesic) <= max(rule.geodesic_abs_m, rule.geodesic_rel * info_geodesic):
        reasons.append("geodesic_mismatch")
    return reasons, {
        "start_snap_m": start_snap,
        "goal_snap_m": goal_snap,
        "path_snap_max_m": path_snap,
        "geodesic_m": geodesic,
        "info_geodesic_m": info_geodesic,
    }


def _filter_scene(task: tuple) -> tuple[str, str | None, str | None, list]:
    hm3d_id, items, factory, rule = task
    try:
        geometry = factory(hm3d_id)
    except (OSError, RuntimeError) as exc:
        raise CohortError(f"scene {hm3d_id}: {exc}") from None
    rows = []
    for episode_id, start, goal, path, info_geodesic in items:
        reasons, stats = check_geometry(start, goal, path, info_geodesic, geometry, rule)
        rows.append((episode_id, reasons, stats))
    path = getattr(geometry, "path", None)
    return hm3d_id, None if path is None else str(path), getattr(geometry, "sha256", None), rows


def filter_episodes(
    episodes_by_scene: Mapping[str, Sequence[Mapping[str, Any]]],
    factory: Callable[[str], Any],
    *,
    rule: FilterRule,
    workers: int,
) -> dict[str, tuple[str | None, str | None, list]]:
    """Per HM3D ID: (navmesh path, navmesh sha256, [(episode_id, reasons, stats)])."""
    tasks = [
        (scene, [(int(e["episode_id"]), e["start_position"], e["goals"][0]["position"], e["reference_path"],
                  e["info"]["geodesic_distance"]) for e in episodes], factory, rule)
        for scene, episodes in sorted(episodes_by_scene.items())
    ]
    if workers <= 1 or len(tasks) <= 1:
        results = [_filter_scene(task) for task in tasks]
    else:
        with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
            results = list(pool.imap_unordered(_filter_scene, tasks))
    return {scene: (path, digest, rows) for scene, path, digest, rows in results}


def _finite(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def _quantiles(values: Sequence[float], qs: Sequence[int] = (50, 90, 95, 99, 100)) -> dict[str, float | None]:
    finite = sorted(v for v in values if v is not None and math.isfinite(v))
    if not finite:
        return {f"p{q}": None for q in qs}
    return {f"p{q}": round(finite[max(0, min(len(finite), math.ceil(q / 100 * len(finite))) - 1)], 4) for q in qs}


def summarize_filter(
    filtered: Mapping[str, tuple[str | None, str | None, list]],
    episodes: Sequence[Mapping[str, Any]],
    split_of_scene: Mapping[str, str],
    *,
    rule: FilterRule,
    max_drop_rate: float,
) -> tuple[dict[int, list[str]], dict[str, Any]]:
    """(reasons per episode_id, the filter report with its D0 verdict) from ``filter_episodes``' rows."""
    verdict: dict[int, list[str]] = {}
    stats_of: dict[int, dict] = {}
    per_scene, reasons_total = {}, Counter()
    for hm3d_id in sorted(filtered):
        navmesh, navmesh_sha, rows = filtered[hm3d_id]
        reasons = Counter()
        for episode_id, why, stats in rows:
            verdict[episode_id] = why
            stats_of[episode_id] = stats
            reasons.update(why)
        reasons_total.update(reasons)
        per_scene[hm3d_id] = {
            "split": split_of_scene[hm3d_id], "episodes": len(rows), "dropped": sum(bool(why) for _, why, _ in rows),
            "reasons": dict(sorted(reasons.items())), "navmesh": navmesh, "navmesh_sha256": navmesh_sha,
        }
    dropped = [e for e in episodes if verdict[e["episode_id"]]]
    routes: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for e in episodes:
        routes[(scene_name(e["scene_id"]), int(e["trajectory_id"]))].append(bool(verdict[e["episode_id"]]))
    drop_rate = len(dropped) / len(episodes)
    stats_all = [stats_of[e["episode_id"]] for e in episodes]
    stats_kept = [stats_of[e["episode_id"]] for e in episodes if not verdict[e["episode_id"]]]
    snaps = ("start_snap_m", "goal_snap_m", "path_snap_max_m")
    # A used scene that keeps no episode has no usable navmesh in the frame of its episodes, whatever
    # is_loaded said; under the 25% line alone a dozen such scenes would just drop out of the cohort.
    dead = sorted(s for s, row in per_scene.items() if row["episodes"] and row["dropped"] == row["episodes"])
    d0 = {
        "dropped_episodes": len(dropped), "episodes": len(episodes), "drop_rate": round(drop_rate, 6),
        "dropped_routes": sum(all(v) for v in routes.values()), "routes": len(routes),
        "max_drop_rate": max_drop_rate, "scenes_all_dropped": dead,
        "pass": drop_rate <= max_drop_rate and not dead,
    }
    report = {
        "rule": dataclasses.asdict(rule),
        "d0": d0,
        "reasons": dict(sorted(reasons_total.items())),
        "scene_drop_rate": _quantiles([r["dropped"] / r["episodes"] for r in per_scene.values() if r["episodes"]]),
        "quantiles_all": {key: _quantiles([s[key] for s in stats_all]) for key in snaps},
        "quantiles_kept": {key: _quantiles([s[key] for s in stats_kept]) for key in snaps},
        "geodesic_relative_error_all": _quantiles([
            abs(s["geodesic_m"] - s["info_geodesic_m"]) / max(s["info_geodesic_m"], 1e-6) for s in stats_all
            if s["info_geodesic_m"] is not None and math.isfinite(s["geodesic_m"])
        ]),
        "dropped_examples": [
            {"scene": scene_name(e["scene_id"]), "episode_id": e["episode_id"], "reasons": verdict[e["episode_id"]],
             **{k: _finite(v) for k, v in stats_of[e["episode_id"]].items()}}
            for e in dropped[:50]
        ],
        "scenes": per_scene,
    }
    return verdict, report


# --------------------------------------------------------------------------- #
# Route assignment
# --------------------------------------------------------------------------- #
def assign_routes(
    routes_of: Mapping[str, Sequence[Route]],
    heldout: Sequence[str],
    train: Sequence[str],
    route_overlaps: Mapping[tuple[str, int], bool],
    *,
    heldout_nonoverlap: bool,
    heldout_per_scene: int,
    routes_per_pass: int,
    train_passes: int,
) -> tuple[dict[tuple[str, int], str], dict[str, dict[str, int]]]:
    """(route -> cohort part, scenes short of routes per part with what they got), in each scene's draw order.

    Held-out scenes give their first ``heldout_per_scene`` routes (only routes InternData has no instruction of,
    ``route_overlaps`` false, when ``heldout_nonoverlap``); training scenes give routes ``[p*n, (p+1)*n)`` to pass
    ``p + 1``.
    """
    assignment: dict[tuple[str, int], str] = {}
    shortfall: dict[str, dict[str, int]] = defaultdict(dict)
    for hm3d_id in heldout:
        eligible = [key for key, _ in routes_of.get(hm3d_id, ()) if not (heldout_nonoverlap and route_overlaps[key])]
        for key in eligible[:heldout_per_scene]:
            assignment[key] = "heldout"
        if len(eligible) < heldout_per_scene:
            shortfall["heldout"][hm3d_id] = len(eligible)
    for hm3d_id in train:
        keys = [key for key, _ in routes_of.get(hm3d_id, ())]
        for p in range(train_passes):
            got = keys[p * routes_per_pass: (p + 1) * routes_per_pass]
            for key in got:
                assignment[key] = f"train_pass{p + 1}"
            if len(got) < routes_per_pass:
                shortfall[f"train_pass{p + 1}"][hm3d_id] = len(got)
    return assignment, {part: dict(sorted(v.items())) for part, v in shortfall.items()}


def make_parts(
    routes_of: Mapping[str, Sequence[Route]],
    chosen: Mapping[tuple[str, int], dict[str, Any]],
    assignment: Mapping[tuple[str, int], str],
    *,
    part_scenes: Mapping[str, Sequence[str]],
    seed: int,
    heldout_shards_per_scene: int,
) -> list[tuple[str, list[list[dict[str, Any]]]]]:
    """Per part (in ``part_scenes`` order): its shards, scenes in a seeded hash order, one scene per shard
    (held-out: ``heldout_shards_per_scene``, dealt by ``deal_shards``), episodes in draw order."""
    parts = []
    for part, scenes in part_scenes.items():
        shards = []
        for hm3d_id in sorted(scenes, key=lambda s: (_digest(seed, "scene", s), s)):
            routes = [(key, [chosen[key]]) for key, _ in routes_of.get(hm3d_id, ()) if assignment.get(key) == part]
            if routes:
                per_scene = heldout_shards_per_scene if part == "heldout" else 1
                shards.extend(deal_shards(routes, num_shards=min(per_scene, len(routes)), shard_size=None))
        parts.append((part, shards))
    return parts


# --------------------------------------------------------------------------- #
# InternData overlap
# --------------------------------------------------------------------------- #
def interndata_instructions(scene_dir: Path) -> tuple[int, list[str]]:
    """(episodes, every raw instruction) of one InternData-N1 scene dir, split as ``interndata_replay`` splits."""
    episodes, texts = 0, []
    with open(scene_dir / "meta" / "episodes.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            episode = json.loads(line)
            episodes += 1
            texts.extend(str(episode["tasks"][0]).split(INSTRUCTION_SEP))
    return episodes, texts


def instruction_key(text: str) -> str:
    """The overlap key (with the scene): ``interndata_replay.client_instruction`` after collapsing whitespace,
    lower case -- ``verify_hm3d_render.match_key``, so V0's json-InternData pairs and this overlap agree.
    Collapsing first keeps an R2R-style trailing ``". "`` from hiding the final period from the client's rule."""
    return client_instruction(" ".join(str(text).split())).lower()


def strict_key(text: str) -> str:
    """Only reported: the client's normalisation alone, the bytes the model would see."""
    return client_instruction(str(text))


def loose_key(text: str) -> str:
    """Only reported: lowercase words and digits."""
    return " ".join(re.findall(r"[0-9a-z]+", str(text).lower()))


OVERLAP_KEYS: dict[str, Callable[[str], str]] = {
    "normalized": instruction_key, "strict": strict_key, "loose": loose_key,
}


def _rate(hits: int, total: int) -> dict[str, Any]:
    return {"hits": hits, "total": total, "rate": round(hits / total, 6) if total else None}


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _ranges(indices: Sequence[int]) -> str:
    """[0, 1, 2, 5] -> "0-2,5" (PPA_COLLECT_SHARDS syntax)."""
    parts, start = [], None
    for position, value in enumerate(indices):
        if start is None:
            start = value
        if position + 1 == len(indices) or indices[position + 1] != value + 1:
            parts.append(f"{start}" if start == value else f"{start}-{value}")
            start = None
    return ",".join(parts)


def write_cohort(
    out: Path,
    parts: Sequence[tuple[str, list[list[dict[str, Any]]]]],
    *,
    template: Mapping[str, Any],
    pool_path: Path,
    pool_sha256: str,
    seed: int,
) -> dict[str, Any]:
    """One train cohort whose shards are ``parts`` in order; the manifest's ``parts`` give each one's shards."""
    shards = [(name, shard) for name, part in parts for shard in part]
    out.mkdir(parents=True)
    records, part_info, owner = [], {}, {}
    for index, (part, shard) in enumerate(shards):
        name, dataset_name = f"shard_{index:03d}.json", f"dataset_shard_{index:03d}.json.gz"
        entries = [{"scene_id": scene_name(e["scene_id"]), "episode_id": int(e["episode_id"]),
                    "trajectory_id": int(e["trajectory_id"])} for e in shard]
        for e in entries:
            if owner.setdefault((e["scene_id"], e["trajectory_id"]), index) != index:
                raise CohortError(f"internal error: route {(e['scene_id'], e['trajectory_id'])} is in two shards")
        _write_json(out / name, {
            "cohort_name": f"exp28_hm3d_{part}_{index:03d}_of_{len(shards):03d}",
            "split": "train",
            "dataset_sha256": pool_sha256,
            "num_shards": len(shards),
            "shard_index": index,
            "part": part,
            "episodes": entries,
        })
        (out / dataset_name).write_bytes(deterministic_gzip_json({**template, "episodes": list(shard)}))
        scenes = sorted({e["scene_id"] for e in entries})
        records.append({
            "index": index,
            "path": name,
            "sha256": sha256_file(out / name),
            "dataset_path": dataset_name,
            "dataset_sha256": sha256_file(out / dataset_name),
            "episodes": len(entries),
            "routes": len({(e["scene_id"], e["trajectory_id"]) for e in entries}),
            "scenes": scenes,
            "part": part,
        })
        info = part_info.setdefault(part, {"shard_indices": [], "episodes": 0, "scenes": set()})
        info["shard_indices"].append(index)
        info["episodes"] += len(entries)
        info["scenes"].update(scenes)
    manifest = {
        "schema": COHORT_SCHEMA,
        "split": "train",
        "dataset": str(pool_path),
        "dataset_sha256": pool_sha256,
        "seed": int(seed),
        "episodes_per_route": 1,
        "max_routes": None,
        "exclude_scenes": [],
        "num_shards": len(shards),
        "total_episodes": sum(r["episodes"] for r in records),
        "total_routes": len(owner),
        "scenes": sorted({scene for scene, _ in owner}),
        "route_disjoint": True,
        "shards": records,
        "parts": {
            part: {"shards": _ranges(info["shard_indices"]), "num_shards": len(info["shard_indices"]),
                   "episodes": info["episodes"], "scenes": sorted(info["scenes"])}
            for part, info in part_info.items()
        },
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(args: argparse.Namespace, *, geometry_factory: Callable[[str], Any] | None = None) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    if not args.dry_run and out_dir.exists() and any(out_dir.iterdir()):
        raise CohortError(f"refusing to write into a non-empty {out_dir}")
    source_path, r2r_path = Path(args.source).resolve(), Path(args.r2r_train).resolve()
    source_sha = sha256_file(source_path)
    if args.source_sha256 and source_sha != args.source_sha256:
        raise CohortError(f"{source_path} has sha256 {source_sha}, not the expected {args.source_sha256}")

    # Inputs and the global checks.
    reference = load_r2r_reference(r2r_path)
    tokenizer = VocabTokenizer(reference.top_level["instruction_vocab"]["word_list"], reference.token_length)
    mp3d = mp3d_scene_ids(Path(args.mp3d_dir)) | set(reference.scenes)
    if args.expected_mp3d_scenes and len(mp3d) != args.expected_mp3d_scenes:
        raise CohortError(
            f"{len(mp3d)} MP3D scene IDs (--mp3d-dir and R2R train), expected {args.expected_mp3d_scenes} "
            "(--expected-mp3d-scenes 0 skips the count; the clash check runs on the IDs found)"
        )
    source = load_json_gz(source_path, drop_instruction_tokens=True)
    by_dir = index_source(source["episodes"])
    clash = sorted({dir_id(d) for d in by_dir} & mp3d)
    if clash:
        raise CohortError(f"HM3D IDs equal to MP3D IDs: {clash}")

    interndata = interndata_scene_dirs(Path(args.interndata_root))
    if args.expected_interndata_scenes and len(interndata) != args.expected_interndata_scenes:
        raise CohortError(
            f"{len(interndata)} InternData scene dirs under {args.interndata_root}, "
            f"expected {args.expected_interndata_scenes}"
        )
    b3_root = Path(args.b3_root)
    b3_dirs = {name: b3_root / name for name in args.b3_scenes}
    missing = [n for n, p in b3_dirs.items() if not (p / "meta" / "episodes.jsonl").is_file()]
    if missing:
        raise CohortError(f"B3 scenes without meta/episodes.jsonl under {b3_root}: {missing}")
    split = split_scenes(interndata, args.b3_scenes, set(by_dir), from_hash=args.heldout_from_hash)
    used_dirs = [*split["heldout"], *split["train"]]
    split_of = {d: "heldout" for d in split["heldout"]} | {d: "train" for d in split["train"]}
    id_of = {d: dir_id(d) for d in used_dirs}

    # Convert the used scenes' episodes; routes must be routes.
    converted: dict[str, list[dict[str, Any]]] = {}
    for scene_dir in used_dirs:
        converted[id_of[scene_dir]] = [convert_episode(e, tokenizer) for e in by_dir[scene_dir]]
    routes_all: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for hm3d_id, episodes in converted.items():
        for e in episodes:
            routes_all[(hm3d_id, e["trajectory_id"])].append(e)
    check_route_geometry(routes_all)
    bad_rotation = sorted(e["episode_id"] for episodes in converted.values() for e in episodes
                          if not is_yaw_quaternion_xyzw(e["start_rotation"]))
    if bad_rotation:
        raise CohortError(
            f"{len(bad_rotation)} used episodes' start_rotation is not a unit yaw quaternion [x, y, z, w] "
            f"(x = z = 0), e.g. episode_id {bad_rotation[:5]} (R2R train: {reference.yaw_quaternions:.4f} of its "
            "episodes are); another component order would start every agent facing the wrong way"
        )
    if split["heldout_hash_skipped"]:
        print(f"WARNING: the annotation has no episode in {split['heldout_hash_skipped']}, which the hash rule "
              "would have held out; the next-ranked scenes are held out instead", file=sys.stderr)

    # InternData overlap: (scene dir, instruction key), the decision on the "normalized" key.
    interndata_keys: dict[str, dict[str, set[str]]] = {name: {} for name in OVERLAP_KEYS}
    interndata_texts: dict[str, list[str]] = {}
    interndata_episodes = 0
    for scene_dir in used_dirs:
        count, texts = interndata_instructions(b3_dirs.get(scene_dir) or interndata[scene_dir])
        interndata_episodes += count
        interndata_texts[scene_dir] = texts
        for name, key_of in OVERLAP_KEYS.items():
            interndata_keys[name][scene_dir] = {key_of(t) for t in texts}
    dir_of_id = {v: k for k, v in id_of.items()}

    def in_interndata(e: Mapping[str, Any], key: str = "normalized") -> bool:
        scene_dir = dir_of_id[scene_name(e["scene_id"])]
        return OVERLAP_KEYS[key](e["instruction"]["instruction_text"]) in interndata_keys[key][scene_dir]

    def rates(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {key: _rate(sum(in_interndata(e, key) for e in episodes), len(episodes)) for key in OVERLAP_KEYS}

    all_used = [e for episodes in converted.values() for e in episodes]
    source_overlap = rates(all_used)
    json_keys = {d: {instruction_key(e["instruction"]["instruction_text"]) for e in converted[id_of[d]]}
                 for d in used_dirs}
    reverse = _rate(sum(instruction_key(t) in json_keys[d] for d in used_dirs for t in interndata_texts[d]),
                    sum(len(interndata_texts[d]) for d in used_dirs))
    decision_rate = source_overlap["normalized"]["rate"]
    heldout_nonoverlap = decision_rate is not None and decision_rate < args.overlap_rule_threshold
    sides = {key: r["rate"] < args.overlap_rule_threshold for key, r in source_overlap.items() if r["rate"] is not None}
    keys_disagree = len(set(sides.values())) > 1
    if keys_disagree:
        print(f"WARNING: the overlap keys fall on both sides of {args.overlap_rule_threshold} "
              f"({ {k: v['rate'] for k, v in source_overlap.items()} }); the rule follows the normalized key, "
              "InternData's text may differ from the json's in more than whitespace, case and the final period",
              file=sys.stderr)
    no_info = sum(e["info"]["geodesic_distance"] is None for e in all_used)
    summary: dict[str, Any] = {
        "scenes": {"heldout": len(split["heldout"]), "train": len(split["train"]),
                   "interndata_scenes_without_episodes": split["interndata_scenes_without_episodes"],
                   "heldout_hash_skipped": split["heldout_hash_skipped"]},
        "source_episodes": len(source["episodes"]),
        "source_scenes": len(by_dir),
        "used_episodes": len(all_used),
        "used_routes": len(routes_all),
        "used_routes_sharing_a_path": routes_sharing_a_path(routes_all),
        "used_episodes_without_info_geodesic": no_info,
        "used_goal_radius": dict(sorted(Counter(
            str((e.get("goals") or [{}])[0].get("radius")) for d in used_dirs for e in by_dir[d]).items())),
        "overlap_source_episodes": source_overlap,
        "overlap_keys_disagree_on_rule": keys_disagree,
        "heldout_nonoverlap_rule": heldout_nonoverlap,
    }
    if args.dry_run:
        summary["split"] = {"heldout": split["heldout"], "train": split["train"]}
        return summary

    # Geometric filter (D0).
    rule = FilterRule(missing_info_geodesic=args.missing_info_geodesic)
    factory = geometry_factory or SceneTree(Path(args.scenes_dir).resolve())
    filtered = filter_episodes(converted, factory, rule=rule, workers=args.workers)
    verdict, filter_report = summarize_filter(
        filtered, all_used, {dir_id(d): split_of[d] for d in used_dirs}, rule=rule, max_drop_rate=args.max_drop_rate
    )
    for hm3d_id, row in filter_report["scenes"].items():
        row["dir"] = dir_of_id[hm3d_id]
    d0 = filter_report["d0"]
    if not d0["pass"]:
        report_path = out_dir.with_name(out_dir.name + ".d0_failed.json")
        _write_json(report_path, {"summary": summary, **filter_report})
        hint = ""
        if filter_report["reasons"].get("no_info_geodesic"):
            hint = (f" {filter_report['reasons']['no_info_geodesic']} of them lack info.geodesic_distance "
                    "(--missing-info-geodesic fill keeps those on the finite-geodesic check alone).")
        dead = d0["scenes_all_dropped"]
        if dead:
            hint += (f" {len(dead)} used scenes keep no episode at all, e.g. {dead[:5]}: their navmesh is not usable "
                     "in the episodes' frame.")
        raise D0Failed(
            f"D0: {d0['dropped_episodes']}/{d0['episodes']} episodes ({d0['drop_rate']:.1%}) fail the filter "
            f"(limit {args.max_drop_rate:.0%}, and no used scene may lose all); coordinates or navmeshes are "
            f"wrong.{hint} Report: {report_path}"
        )

    # The kept episodes carry the navmesh they were filtered on (r2r_val_unseen.py checks it after each reset)
    # and, under --missing-info-geodesic fill, that navmesh's geodesic.
    navmeshes = {hm3d_id: digest for hm3d_id, (_, digest, _) in filtered.items()}
    unhashed = sorted(hm3d_id for hm3d_id, digest in navmeshes.items() if not digest)
    if unhashed:
        raise CohortError(f"no navmesh sha256 for scenes {unhashed[:5]} ({len(unhashed)})")
    geodesic_of = {episode_id: stats["geodesic_m"] for _, _, rows in filtered.values() for episode_id, _, stats in rows}
    kept, filled = [], 0
    for e in all_used:
        if verdict[e["episode_id"]]:
            continue
        hm3d_id = scene_name(e["scene_id"])
        if e["info"]["geodesic_distance"] is None:
            e["info"]["geodesic_distance"] = round(geodesic_of[e["episode_id"]], 4)
            filled += 1
        e["info"]["navmesh_sha256"] = navmeshes[hm3d_id]
        kept.append(e)
    summary["kept_episodes_info_geodesic_filled"] = filled

    # One instruction per route (per path: trajectory_ids over one path are merged), routes in draw order.
    drawn, same_path = merge_same_path_routes(
        select_routes(kept, seed=args.seed, episodes_per_route=0, exclude_scenes=set(), max_routes=None))
    routes_of: dict[str, list[tuple[tuple[str, int], list[dict[str, Any]]]]] = defaultdict(list)
    for key, episodes in drawn:
        routes_of[key[0]].append((key, episodes))
    chosen = {key: episodes[0] for key, episodes in drawn}
    every_instruction = {key: [*episodes, *(e for _, more in same_path.get(key, ()) for e in more)]
                         for key, episodes in drawn}
    route_overlaps = {key: any(in_interndata(e) for e in episodes) for key, episodes in every_instruction.items()}
    # Held-out exclusion (the < 80% branch) on the widest key: no InternData route may pass as unseen.
    route_overlaps_loose = {key: any(in_interndata(e, "loose") for e in episodes)
                            for key, episodes in every_instruction.items()}
    summary["kept_routes_merged_into_a_same_path_route"] = sum(len(v) for v in same_path.values())

    # The cohort parts.
    heldout_ids = [id_of[d] for d in split["heldout"]]
    train_ids = [id_of[d] for d in split["train"]]
    assignment, shortfall = assign_routes(
        routes_of, heldout_ids, train_ids, route_overlaps_loose, heldout_nonoverlap=heldout_nonoverlap,
        heldout_per_scene=args.heldout_per_scene, routes_per_pass=args.routes_per_pass, train_passes=args.train_passes,
    )
    part_names = ["heldout", *(f"train_pass{p + 1}" for p in range(args.train_passes))]
    parts = make_parts(
        routes_of, chosen, assignment,
        part_scenes={p: heldout_ids if p == "heldout" else train_ids for p in part_names},
        seed=args.seed, heldout_shards_per_scene=args.heldout_shards_per_scene,
    )

    # Everything on disk at once.
    pool_episodes = [chosen[key] for hm3d_id in sorted(routes_of) for key, _ in routes_of[hm3d_id]]
    template = dict(reference.top_level)
    overlap_keys = sorted([scene_name(e["scene_id"]), e["episode_id"]] for e in pool_episodes if in_interndata(e))

    def part_rate(part: str | None) -> dict[str, Any]:
        return rates([chosen[k] for k in chosen if part is None or assignment.get(k) == part])

    overlap = {
        "keys": {
            "normalized": "(scene dir, client_instruction(' '.join(text.split())).lower()) -- decides the rule; "
                          "verify_hm3d_render.match_key",
            "strict": "(scene dir, interndata_replay.client_instruction(text))",
            "loose": "(scene dir, lowercase words and digits)",
        },
        "source_episodes_used_scenes": source_overlap,
        "kept_episodes": rates(kept),
        "routes_with_any_instruction_in_interndata": _rate(sum(route_overlaps.values()), len(route_overlaps)),
        "pool": part_rate(None),
        **{f"cohort_{part}": part_rate(part) for part in part_names},
        "interndata_instructions_in_json": reverse,
        "interndata_episodes_used_scenes": interndata_episodes,
        "rule": f"heldout from routes without any InternData instruction (loose key, every instruction of the path) "
                f"iff the normalized source rate < {args.overlap_rule_threshold}; training is not filtered",
        "heldout_nonoverlap_rule": heldout_nonoverlap,
    }
    words = [VocabTokenizer.tokenize(e["instruction"]["instruction_text"]) for e in pool_episodes]
    unk = sum(w not in tokenizer.index for ws in words for w in ws)
    total_words = sum(len(ws) for ws in words)
    tokens = {
        "rule": "habitat-lab VocabDict.tokenize_and_index on R2R train's word_list, padded to R2R's length",
        "token_length": reference.token_length,
        "r2r_train_reproduction": round(reference.tokenizer_reproduction, 6),
        "pool_unk_rate": round(unk / total_words, 6) if total_words else None,
        "pool_truncated": sum(len(ws) > reference.token_length for ws in words) if reference.token_length else 0,
    }

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        (staging / "pool.json.gz").write_bytes(deterministic_gzip_json({**template, "episodes": pool_episodes}))
        pool_sha = sha256_file(staging / "pool.json.gz")
        cohort = write_cohort(staging / "cohorts", parts, template=template, pool_path=out_dir / "pool.json.gz",
                              pool_sha256=pool_sha, seed=args.seed)
        written = [(s["scene_id"], s["episode_id"]) for r in cohort["shards"]
                   for s in json.loads((staging / "cohorts" / r["path"]).read_text())["episodes"]]
        expected = sorted((k[0], chosen[k]["episode_id"]) for k in assignment)
        if sorted(written) != expected or len(set(written)) != len(written):
            raise CohortError("internal error: the cohort does not hold each assigned route exactly once")
        cohort_scenes = {part: set(info["scenes"]) for part, info in cohort["parts"].items()}
        for part, scenes in cohort_scenes.items():
            allowed = set(heldout_ids) if part == "heldout" else set(train_ids)
            if not scenes <= allowed:
                raise CohortError(f"internal error: {part} holds scenes {sorted(scenes - allowed)} of the other split")

        lines = "# EXP-28 {} scenes (scripts/exp28/build_hm3d_episodes.py); scene dir 00xxx-<ID>, one per line\n"
        (staging / "splits").mkdir()
        (staging / "splits" / "heldout_scenes.txt").write_text(
            lines.format("held-out") + "".join(f"{d}\n" for d in split["heldout"]), encoding="utf-8")
        (staging / "splits" / "train_scenes.txt").write_text(
            lines.format("training") + "".join(f"{d}\n" for d in split["train"]), encoding="utf-8")
        _write_json(staging / "splits" / "splits.json", {
            "rule": f"held out: B3 + the {args.heldout_from_hash} usable InternData scenes with the smallest "
                    f"sha256('{HELDOUT_SALT}' + scene dir); train: the rest",
            "heldout": [{"dir": d, "id": id_of[d], "source": "b3" if d in split["heldout_from_b3"] else "hash",
                         "rank": heldout_rank(d)} for d in split["heldout"]],
            "train": [{"dir": d, "id": id_of[d], "rank": heldout_rank(d)} for d in split["train"]],
            "interndata_scenes_without_episodes": split["interndata_scenes_without_episodes"],
            "heldout_hash_skipped": split["heldout_hash_skipped"],
        })
        _write_json(staging / "routes.json", {
            hm3d_id: {
                "dir": dir_of_id[hm3d_id],
                "split": split_of[dir_of_id[hm3d_id]],
                "routes": [{"trajectory_id": key[1], "episode_id": chosen[key]["episode_id"],
                            "part": assignment.get(key), "instruction_in_interndata": in_interndata(chosen[key]),
                            "route_in_interndata": route_overlaps[key],
                            "route_in_interndata_loose": route_overlaps_loose[key],
                            "instructions": len(every_instruction[key]),
                            "same_path_trajectory_ids": sorted(k[1] for k, _ in same_path.get(key, ()))}
                           for key, episodes in routes_of[hm3d_id]],
            }
            for hm3d_id in sorted(routes_of)
        })
        _write_json(staging / "overlap_keys.json", overlap_keys)
        _write_json(staging / "filter_report.json", filter_report)
        files = {}
        for path in sorted(p for p in staging.rglob("*") if p.is_file()):
            files[str(path.relative_to(staging))] = sha256_file(path)
        summary.update({
            "d0": d0,
            "pool_routes": len(pool_episodes),
            "cohort": {part: {k: v for k, v in info.items() if k != "scenes"} | {"scenes": len(info["scenes"])}
                       for part, info in cohort["parts"].items()},
            "shortfall": shortfall,
            "overlap_pool": overlap["pool"]["normalized"],
        })
        manifest = {
            "schema": SCHEMA,
            "inputs": {
                "source": str(source_path), "source_sha256": source_sha, "source_sha256_expected": args.source_sha256,
                "r2r_train": str(r2r_path), "r2r_train_sha256": sha256_file(r2r_path),
                "mp3d_dir": str(Path(args.mp3d_dir).resolve()), "mp3d_scene_ids": len(mp3d),
                "scenes_dir": str(Path(args.scenes_dir).resolve()),
                "interndata_root": str(Path(args.interndata_root).resolve()),
                "b3_root": str(b3_root.resolve()),
            },
            "rules": {
                "seed": args.seed,
                "scene_id": "hm3d/<ID>/<ID>.glb",
                "route": "(scene, trajectory_id); one instruction per route, chosen by build_r2r_train_cohorts' hash",
                "filter": dataclasses.asdict(rule),
                "max_drop_rate": args.max_drop_rate,
                "heldout_per_scene": args.heldout_per_scene,
                "heldout_from_hash": args.heldout_from_hash,
                "heldout_shards_per_scene": args.heldout_shards_per_scene,
                "b3_scenes": list(args.b3_scenes),
                "routes_per_pass": args.routes_per_pass,
                "train_passes": args.train_passes,
                "overlap_rule_threshold": args.overlap_rule_threshold,
            },
            "navmesh_sha256": dict(sorted(navmeshes.items())),
            "summary": summary,
            "overlap": overlap,
            "tokens": tokens,
            "files": files,
        }
        _write_json(staging / "manifest.json", manifest)
        if out_dir.exists():
            out_dir.rmdir()  # empty, checked above
        os.replace(staging, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    summary["out_dir"] = str(out_dir)
    summary["tokens"] = tokens
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="scalevln_subset_150k.json.gz")
    parser.add_argument("--source-sha256", default=SOURCE_SHA256, help="expected sha256 of --source ('' skips)")
    parser.add_argument("--r2r-train", required=True, help="R2R_VLNCE_v1-3_preprocessed train.json.gz (schema, vocab)")
    parser.add_argument("--mp3d-dir", required=True, help="holds <ID>/<ID>.glb for the MP3D scenes")
    parser.add_argument("--expected-mp3d-scenes", type=int, default=90, help="0 skips the count check")
    parser.add_argument("--scenes-dir", required=True, help="the EXP-28 scene tree: hm3d/<ID>/<ID>.glb and .navmesh")
    parser.add_argument("--interndata-root", required=True, help="InternData-N1 ScaleVLN scene dirs (the 300)")
    parser.add_argument("--expected-interndata-scenes", type=int, default=300, help="0 skips the count check")
    parser.add_argument("--b3-root", required=True, help="InternData-N1 ScaleVLN dirs of B3's scenes")
    parser.add_argument("--b3-scenes", type=lambda s: tuple(x for x in s.split(",") if x), default=B3_SCENES,
                        help="comma-separated B3 scene dirs (default: the five of EXP-25)")
    parser.add_argument("--out-dir", required=True, help="new (or empty) directory")
    parser.add_argument("--seed", type=int, default=28, help="route draw, instruction choice, shard order")
    parser.add_argument("--heldout-from-hash", type=int, default=15)
    parser.add_argument("--heldout-per-scene", type=int, default=20)
    parser.add_argument("--heldout-shards-per-scene", type=int, default=1)
    parser.add_argument("--routes-per-pass", type=int, default=6)
    parser.add_argument("--train-passes", type=int, default=2)
    parser.add_argument("--max-drop-rate", type=float, default=MAX_DROP_RATE, help="D0")
    parser.add_argument("--overlap-rule-threshold", type=float, default=OVERLAP_RULE_THRESHOLD)
    parser.add_argument("--missing-info-geodesic", choices=("drop", "fill"), default="drop",
                        help="an episode without info.geodesic_distance: drop it (the pre-registered rule) or keep it "
                             "on the finite-geodesic check alone and fill in the navmesh's geodesic")
    parser.add_argument("--workers", type=int, default=8, help="processes for the navmesh filter")
    parser.add_argument("--dry-run", action="store_true", help="split and overlap only: no navmesh, nothing written")
    args = parser.parse_args(argv)
    for name in ("heldout_per_scene", "heldout_shards_per_scene", "routes_per_pass", "train_passes"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    return args


def main(argv: list[str] | None = None, *, geometry_factory: Callable[[str], Any] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build(args, geometry_factory=geometry_factory)
    except D0Failed as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    except CohortError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

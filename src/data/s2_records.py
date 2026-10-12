"""On-policy System2 call records: one uncompressed tar per closed-loop episode.

EXP-23 fine-tunes the deployed System2 on states the deployed model actually
visits.  Ledger pitfall #1 (and EXP-14's 58% token preservation) say the
training input must be the serving input byte for byte, so a record keeps the
exact JPEG bytes the client put on the wire, not a re-encode of them:

* ``front/{step:05d}.jpg`` -- the front view captured at primitive step
  ``step``.  The client JPEG-encodes the same array at the same quality every
  time that frame is sent (as ``current/front`` once, then as
  ``history/i/front``), and the encoder is deterministic, so one copy per step
  is exact.  The writer checks that by hashing every resend.
* ``lookdown/{call:04d}.jpg`` -- the 640x480 conversational lookdown of call
  ``call``, as sent.
* ``lookdown_depth/{call:04d}.png`` -- metric depth of that lookdown in
  millimetres (uint16, 0 = no return), for the offline occlusion test.

Plus ``episode.json`` (identity, instruction, goal, reference path, outcome),
``steps.jsonl`` (ground-truth agent pose at every primitive step) and
``calls.jsonl`` (one row per System2 call: which frames the prompt used and
what the server answered).  Oracle labels are *not* stored here; the labeller
writes them next to the tar so they can be recomputed without touching it.

Only stdlib, numpy and PIL are imported, so collector, labeller and trainer can
all share this module.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image


RECORD_SCHEMA = "heatmapvln-s2-record-v1"
LABEL_SCHEMA = "heatmapvln-s2-labels-v1"
RECORD_SUFFIX = ".s2rec.tar"
LABEL_SUFFIX = ".s2labels.jsonl"
DEPTH_UNITS_PER_METRE = 1000.0
LOOKDOWN_SIZE = (640, 480)

# Habitat camera of the deployed client (r2r_val_unseen.py build_habitat_config):
# RGB and depth share it.  The lookdown is two 15 degree LOOKDOWN actions.
CAMERA = {
    "width": 640,
    "height": 480,
    "hfov_deg": 79.0,
    "height_m": 1.25,
    "lookdown_pitch_deg": 30.0,
    "depth_max_m": 10.0,
}

_REQUIRED_CALL_FIELDS = (
    "call_index",
    "step_id",
    "history_step_ids",
    "llm_output",
    "kind",
)


def episode_key(scene_id: str, episode_id: int | str) -> str:
    return f"{scene_id}_{int(episode_id):06d}"


def record_path(root: str | os.PathLike, scene_id: str, episode_id: int | str) -> Path:
    return Path(root) / scene_id / f"{episode_key(scene_id, episode_id)}{RECORD_SUFFIX}"


def label_path_for(record: str | os.PathLike, label_root: str | os.PathLike) -> Path:
    record = Path(record)
    stem = record.name[: -len(RECORD_SUFFIX)]
    return Path(label_root) / record.parent.name / f"{stem}{LABEL_SUFFIX}"


def depth_to_png_bytes(depth_m: np.ndarray) -> bytes:
    """Metric depth (H, W) float -> 16-bit PNG in millimetres, lossless."""
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")
    if not np.all(np.isfinite(depth)):
        raise ValueError("depth has non-finite values")
    mm = np.clip(np.rint(depth * DEPTH_UNITS_PER_METRE), 0, np.iinfo(np.uint16).max)
    buf = io.BytesIO()
    Image.fromarray(mm.astype(np.uint16)).save(buf, format="PNG")
    return buf.getvalue()


def depth_from_png_bytes(data: bytes) -> np.ndarray:
    """Inverse of :func:`depth_to_png_bytes`: float32 metres, 0 = no return."""
    arr = np.asarray(Image.open(io.BytesIO(data)), dtype=np.float32)
    return arr / DEPTH_UNITS_PER_METRE


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for row in rows
    )


@dataclass
class EpisodeRecordWriter:
    """Accumulates one episode in memory and commits it as a single tar.

    The tar is written to a sibling temp file, fsynced, then renamed, so a
    reader never sees a half-written record.  A ``.sha256`` sidecar is
    written last and is the commit marker.
    """

    root: Path
    scene_id: str
    episode_id: int
    episode: dict[str, Any]
    steps: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    _front: dict[int, bytes] = field(default_factory=dict)
    _lookdown: dict[int, bytes] = field(default_factory=dict)
    _depth: dict[int, bytes] = field(default_factory=dict)

    def add_step(self, step_id: int, position: Sequence[float], rotation_wxyz: Sequence[float], **extra: Any) -> None:
        row = {
            "step_id": int(step_id),
            "position": [float(v) for v in position],
            "rotation_wxyz": [float(v) for v in rotation_wxyz],
        }
        row.update(extra)
        self.steps.append(row)

    def add_front(self, step_id: int, jpeg: bytes) -> None:
        """Store the front-view bytes of ``step_id``; a resend must be identical."""
        step_id = int(step_id)
        previous = self._front.get(step_id)
        if previous is None:
            self._front[step_id] = bytes(jpeg)
        elif previous != jpeg:
            raise ValueError(
                f"front view of step {step_id} was resent with different bytes "
                f"({hashlib.sha256(previous).hexdigest()[:12]} vs "
                f"{hashlib.sha256(jpeg).hexdigest()[:12]}); the record could not "
                "reproduce the prompt exactly"
            )

    def add_call(
        self,
        call: Mapping[str, Any],
        *,
        lookdown_jpeg: bytes,
        lookdown_depth_m: np.ndarray | None,
    ) -> None:
        missing = [name for name in _REQUIRED_CALL_FIELDS if name not in call]
        if missing:
            raise ValueError(f"System2 call row lacks {missing}")
        index = int(call["call_index"])
        if index in self._lookdown:
            raise ValueError(f"call {index} recorded twice")
        for step in list(call["history_step_ids"]) + [call["step_id"]]:
            if int(step) not in self._front:
                raise ValueError(
                    f"call {index} references step {step} whose front view was never stored"
                )
        self._lookdown[index] = bytes(lookdown_jpeg)
        if lookdown_depth_m is not None:
            self._depth[index] = depth_to_png_bytes(lookdown_depth_m)
        self.calls.append(dict(call))

    def commit(self) -> Path:
        referenced = {int(c["step_id"]) for c in self.calls}
        for c in self.calls:
            referenced.update(int(s) for s in c["history_step_ids"])
        episode = dict(self.episode)
        episode.update(
            {
                "schema": RECORD_SCHEMA,
                "scene_id": self.scene_id,
                "episode_id": int(self.episode_id),
                "num_steps": len(self.steps),
                "num_calls": len(self.calls),
                "camera": dict(CAMERA),
                "committed_unix": time.time(),
            }
        )
        members: list[tuple[str, bytes]] = [
            ("episode.json", _json_bytes(episode)),
            ("steps.jsonl", _jsonl_bytes(self.steps)),
            ("calls.jsonl", _jsonl_bytes(sorted(self.calls, key=lambda c: int(c["call_index"])))),
        ]
        # Unreferenced fronts are dropped: only prompt inputs are training inputs.
        for step_id in sorted(referenced):
            members.append((f"front/{step_id:05d}.jpg", self._front[step_id]))
        for index in sorted(self._lookdown):
            members.append((f"lookdown/{index:04d}.jpg", self._lookdown[index]))
        for index in sorted(self._depth):
            members.append((f"lookdown_depth/{index:04d}.png", self._depth[index]))

        path = record_path(self.root, self.scene_id, self.episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        digest = hashlib.sha256()
        with open(tmp, "wb") as raw:
            with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                for name, data in members:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))
            raw.flush()
            os.fsync(raw.fileno())
        with open(tmp, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        os.replace(tmp, path)
        sidecar = path.with_name(path.name + ".sha256")
        sidecar.write_text(f"{digest.hexdigest()}  {path.name}\n", encoding="utf-8")
        return path


class EpisodeRecord:
    """Read-only view of a committed record tar (members read lazily)."""

    def __init__(self, path: str | os.PathLike, *, verify: bool = False):
        self.path = Path(path)
        sidecar = self.path.with_name(self.path.name + ".sha256")
        if not sidecar.is_file():
            raise FileNotFoundError(f"{self.path} has no commit marker {sidecar.name}")
        if verify:
            expected = sidecar.read_text(encoding="utf-8").split()[0]
            digest = hashlib.sha256()
            with open(self.path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"{self.path} does not match its sha256 commit marker")
        with tarfile.open(self.path, mode="r:") as tar:
            self._index = {m.name: (m.offset_data, m.size) for m in tar.getmembers()}
        self.episode: dict[str, Any] = json.loads(self._read("episode.json"))
        if self.episode.get("schema") != RECORD_SCHEMA:
            raise ValueError(f"{self.path}: schema {self.episode.get('schema')!r} != {RECORD_SCHEMA}")
        self.steps: list[dict[str, Any]] = [
            json.loads(line) for line in self._read("steps.jsonl").decode("utf-8").splitlines() if line
        ]
        self.calls: list[dict[str, Any]] = [
            json.loads(line) for line in self._read("calls.jsonl").decode("utf-8").splitlines() if line
        ]
        self._steps_by_id = {int(s["step_id"]): s for s in self.steps}

    def _read(self, name: str) -> bytes:
        try:
            offset, size = self._index[name]
        except KeyError as err:
            raise KeyError(f"{self.path} has no member {name}") from err
        with open(self.path, "rb") as handle:
            handle.seek(offset)
            data = handle.read(size)
        if len(data) != size:
            raise IOError(f"short read of {name} in {self.path}")
        return data

    @property
    def scene_id(self) -> str:
        return str(self.episode["scene_id"])

    @property
    def episode_id(self) -> int:
        return int(self.episode["episode_id"])

    def step(self, step_id: int) -> dict[str, Any]:
        return self._steps_by_id[int(step_id)]

    def front_jpeg(self, step_id: int) -> bytes:
        return self._read(f"front/{int(step_id):05d}.jpg")

    def lookdown_jpeg(self, call_index: int) -> bytes:
        return self._read(f"lookdown/{int(call_index):04d}.jpg")

    def has_lookdown_depth(self, call_index: int) -> bool:
        return f"lookdown_depth/{int(call_index):04d}.png" in self._index

    def lookdown_depth(self, call_index: int) -> np.ndarray:
        return depth_from_png_bytes(self._read(f"lookdown_depth/{int(call_index):04d}.png"))


def iter_record_paths(root: str | os.PathLike) -> Iterator[Path]:
    """Committed records under ``root`` (a record without its marker is skipped)."""
    for path in sorted(Path(root).glob(f"*/*{RECORD_SUFFIX}")):
        if path.with_name(path.name + ".sha256").is_file():
            yield path


def read_labels(path: str | os.PathLike) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        if row.get("schema") != LABEL_SCHEMA:
            raise ValueError(f"{path}: label schema {row.get('schema')!r} != {LABEL_SCHEMA}")
    return rows


__all__ = [
    "CAMERA",
    "DEPTH_UNITS_PER_METRE",
    "EpisodeRecord",
    "EpisodeRecordWriter",
    "LABEL_SCHEMA",
    "LABEL_SUFFIX",
    "LOOKDOWN_SIZE",
    "RECORD_SCHEMA",
    "RECORD_SUFFIX",
    "depth_from_png_bytes",
    "depth_to_png_bytes",
    "episode_key",
    "iter_record_paths",
    "label_path_for",
    "read_labels",
    "record_path",
]

"""EXP-23 training stream: packed sample shards, written on the 4090, read on the 910B.

The 910B keeps only weights and a bounded buffer (design §4), so the training
data never lives there as a dataset.  The 4090 draws the whole training order
once -- the same weighted draw ``WeightedDistributedSampler`` makes, plus the
replay mix -- and packs every drawn sample with the bytes it needs into
fixed-size shards.  The shards are pushed through the ssh tunnel into a buffer
directory on the 910B (``scripts/tools/stream_recv.py`` enforces its size cap),
and :class:`StreamShardDataset` reads them in order and deletes each one once
every rank has read its slice.

Shard ``stream-{index:06d}.tar`` (uncompressed, written atomically) holds, per
sample ``i`` (zero-padded to 6 digits):

* ``{i}.json``  -- ``key``, ``source`` ("dagger" | "replay"), ``instruction``,
  ``turns`` (the supervised assistant turns), ``vlm_image_size``,
  ``num_history``;
* ``{i}.h{k:02d}.jpg`` for ``k < num_history``, ``{i}.cur.jpg`` -- the front
  views as the client sent them;
* ``{i}.look.jpg`` -- the 640x480 lookdown as sent.

A DAgger sample carries the record's bytes verbatim; a replay sample carries
the client-path encoding of InternData-N1 frames
(``interndata_replay.sample_jpegs``).  Either way the trainer decodes them with
the server's own decoder, so a stream sample and a record give the same input.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.models.heatmap.native_internnav_exact import NATIVE_LOOKDOWN_SIZE, build_native_messages


STREAM_SCHEMA = "heatmapvln-s2-stream-v1"
SHARD_PATTERN = "stream-{:06d}.tar"
MANIFEST_NAME = "stream_manifest.json"


def shard_name(index: int) -> str:
    return SHARD_PATTERN.format(int(index))


@dataclass(frozen=True)
class PackedSample:
    key: str
    source: str
    instruction: str
    turns: tuple[str, ...]
    vlm_image_size: tuple[int, int]
    history_jpegs: tuple[bytes, ...]
    current_jpeg: bytes
    lookdown_jpeg: bytes


def write_shard(path: str | os.PathLike, samples: Sequence[PackedSample]) -> str:
    """Write one shard atomically; returns its sha256."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as raw:
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for i, sample in enumerate(samples):
                meta = {
                    "key": sample.key,
                    "source": sample.source,
                    "instruction": sample.instruction,
                    "turns": list(sample.turns),
                    "vlm_image_size": list(sample.vlm_image_size),
                    "num_history": len(sample.history_jpegs),
                }
                members = [(f"{i:06d}.json", json.dumps(meta, ensure_ascii=False).encode("utf-8"))]
                members += [(f"{i:06d}.h{k:02d}.jpg", data) for k, data in enumerate(sample.history_jpegs)]
                members += [(f"{i:06d}.cur.jpg", sample.current_jpeg), (f"{i:06d}.look.jpg", sample.lookdown_jpeg)]
                for name, data in members:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))
        raw.flush()
        os.fsync(raw.fileno())
    digest = hashlib.sha256()
    with open(tmp, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    os.replace(tmp, path)
    return digest.hexdigest()


def read_shard(path: str | os.PathLike) -> list[PackedSample]:
    with tarfile.open(path, mode="r:") as tar:
        blobs = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    samples = []
    i = 0
    while f"{i:06d}.json" in blobs:
        meta = json.loads(blobs[f"{i:06d}.json"])
        samples.append(
            PackedSample(
                key=str(meta["key"]),
                source=str(meta["source"]),
                instruction=str(meta["instruction"]),
                turns=tuple(str(t) for t in meta["turns"]),
                vlm_image_size=tuple(int(v) for v in meta["vlm_image_size"]),  # type: ignore[arg-type]
                history_jpegs=tuple(blobs[f"{i:06d}.h{k:02d}.jpg"] for k in range(int(meta["num_history"]))),
                current_jpeg=blobs[f"{i:06d}.cur.jpg"],
                lookdown_jpeg=blobs[f"{i:06d}.look.jpg"],
            )
        )
        i += 1
    return samples


def packed_inputs(sample: PackedSample):
    """(first-turn messages, prompt images, lookdown) exactly as the server builds them."""
    from src.data.s2_sft_dataset import S2SFTError, pil_from_jpeg

    size = tuple(sample.vlm_image_size)
    history = [pil_from_jpeg(data, size) for data in sample.history_jpegs]
    current = pil_from_jpeg(sample.current_jpeg, size)
    lookdown = pil_from_jpeg(sample.lookdown_jpeg)
    if lookdown.size != NATIVE_LOOKDOWN_SIZE:
        raise S2SFTError(f"{sample.key}: lookdown is {lookdown.size}, the native protocol requires {NATIVE_LOOKDOWN_SIZE}")
    messages, images = build_native_messages(sample.instruction, history, current)
    return messages, images, lookdown


def draw_order(weights: Sequence[float], total: int, seed: int) -> np.ndarray:
    """The global training order: ``total`` weighted draws with replacement."""
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0 or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    return np.random.default_rng(seed).choice(w.size, size=int(total), replace=True, p=w / w.sum())


def done_marker(stream_dir: Path, index: int, rank: int) -> Path:
    return stream_dir / f"{shard_name(index)}.done.r{rank}"


class StreamShardDataset(IterableDataset):
    """Tokenised samples from the shard buffer, in shard order, deleting as it goes.

    Shard ``s`` is read by DataLoader worker ``s % num_workers`` of every rank;
    rank ``r`` takes samples ``r, r + world, ...`` of it (the same split as
    ``WeightedDistributedSampler``), so each rank yields ``shard_size / world``
    samples per shard.  When a rank finishes a shard it leaves a marker; whoever
    finds all ``world`` markers deletes the shard and the markers.  A shard that
    has not arrived yet is waited for (the 4090 pushes ahead of the reader).
    """

    def __init__(
        self,
        stream_dir: str | os.PathLike,
        processor: Any,
        *,
        rank: int,
        world_size: int,
        start_shard: int,
        num_shards: int,
        shard_size: int,
        poll_s: float = 5.0,
        wait_timeout_s: float = 6 * 3600.0,
        delete_consumed: bool = True,
    ):
        if shard_size % world_size:
            raise ValueError(f"shard size {shard_size} is not a multiple of world size {world_size}")
        self.stream_dir = Path(stream_dir)
        self.processor = processor
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_shard = int(start_shard)
        self.num_shards = int(num_shards)
        self.shard_size = int(shard_size)
        self.poll_s = float(poll_s)
        self.wait_timeout_s = float(wait_timeout_s)
        self.delete_consumed = bool(delete_consumed)

    def _wait_for(self, path: Path) -> None:
        waited = 0.0
        while not path.is_file():
            if waited >= self.wait_timeout_s:
                raise TimeoutError(f"{path} did not arrive within {self.wait_timeout_s:.0f}s")
            time.sleep(self.poll_s)
            waited += self.poll_s

    def _finish(self, index: int) -> None:
        done_marker(self.stream_dir, index, self.rank).touch()
        if not self.delete_consumed:
            return
        markers = [done_marker(self.stream_dir, index, r) for r in range(self.world_size)]
        if all(m.exists() for m in markers):
            for path in [self.stream_dir / shard_name(index), *markers]:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass  # another rank got there first

    def __iter__(self) -> Iterator[dict[str, Any]]:
        from src.data.s2_sft_dataset import encode_conversation

        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info is not None else (0, 1)
        for index in range(self.start_shard, self.num_shards):
            if (index - self.start_shard) % workers != worker:
                continue
            path = self.stream_dir / shard_name(index)
            self._wait_for(path)
            samples = read_shard(path)
            if len(samples) != self.shard_size:
                raise ValueError(f"{path} holds {len(samples)} samples, expected {self.shard_size}")
            for sample in samples[self.rank :: self.world_size]:
                messages, images, lookdown = packed_inputs(sample)
                encoded = encode_conversation(self.processor, messages, images, lookdown, sample.turns)
                encoded["key"] = sample.key
                yield encoded
            self._finish(index)


def read_manifest(stream_dir: str | os.PathLike) -> dict[str, Any]:
    manifest = json.loads((Path(stream_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("schema") != STREAM_SCHEMA:
        raise ValueError(f"{stream_dir}: stream schema {manifest.get('schema')!r} != {STREAM_SCHEMA}")
    return manifest


def shard_digest_ok(path: Path, expected: str) -> bool:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


__all__ = [
    "MANIFEST_NAME",
    "PackedSample",
    "STREAM_SCHEMA",
    "StreamShardDataset",
    "done_marker",
    "draw_order",
    "packed_inputs",
    "read_manifest",
    "read_shard",
    "shard_digest_ok",
    "shard_name",
    "write_shard",
]

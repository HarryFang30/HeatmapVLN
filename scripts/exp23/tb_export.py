#!/usr/bin/env python3
"""Mirror an EXP-23 round's training log, gates and status into TensorBoard event files.

Runs on the 910B beside the trainer, with the ``ppa`` env's tensorboard and no torch
import, writing only under ``--logdir``:

    tb_export.py --round-dir ~/work/zhr/zhr_1/exp23/round1 --logdir ~/work/zhr/zhr_1/exp23/tb/round1
    tensorboard --logdir ~/work/zhr/zhr_1/exp23/tb --host 127.0.0.1 --port <port>

It polls ``lora/train_log.jsonl`` (the trainer's JSONL), ``lora/gates_*.json`` and
``STATUS`` every ``--interval`` seconds.  The event files are rebuilt from scratch on
start, so a restart never doubles a point.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from tensorboard.compat.proto import tensor_pb2, tensor_shape_pb2, types_pb2
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.plugins.text import metadata as text_metadata
from tensorboard.summary.writer.event_file_writer import EventFileWriter

# TensorBoard sorts groups by name; the numbers put the loss first.
STEP_TAGS = {
    "loss": "1_train/loss",
    "grad_norm": "1_train/grad_norm",
    "lr": "1_train/lr",
    "supervised_tokens": "1_train/supervised_tokens",
    "s_per_step": "4_speed/s_per_step",
    "samples_per_s": "4_speed/samples_per_s",
    "max_allocated_gib": "5_memory/max_allocated_gib",
    "max_reserved_gib": "5_memory/max_reserved_gib",
}


class Writer:
    def __init__(self, logdir: Path):
        self._w = EventFileWriter(str(logdir))

    def scalar(self, tag: str, value: float, step: int) -> None:
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])
        self._w.add_event(Event(wall_time=time.time(), step=int(step), summary=summary))

    def text(self, tag: str, text: str, step: int) -> None:
        tensor = tensor_pb2.TensorProto(
            dtype=types_pb2.DT_STRING,
            string_val=[text.encode("utf-8")],
            tensor_shape=tensor_shape_pb2.TensorShapeProto(dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)]),
        )
        metadata = text_metadata.create_summary_metadata(display_name=tag, description="")
        summary = Summary(value=[Summary.Value(tag=tag, metadata=metadata, tensor=tensor)])
        self._w.add_event(Event(wall_time=time.time(), step=int(step), summary=summary))

    def flush(self) -> None:
        self._w.flush()


def _numbers(payload: dict, prefix: str):
    for key, value in payload.items():
        if isinstance(value, bool):
            yield f"{prefix}/{key}", float(value)
        elif isinstance(value, (int, float)):
            yield f"{prefix}/{key}", float(value)
        elif isinstance(value, dict):
            yield from _numbers(value, f"{prefix}/{key}")


class Exporter:
    def __init__(self, round_dir: Path, writer: Writer):
        self.round_dir = round_dir
        self.writer = writer
        self.offset = 0
        self.step = 0
        self.total_steps = 0
        self.status = ""
        self.gates_seen: set[str] = set()

    def poll(self) -> None:
        self._train_log()
        self._gates()
        self._status()
        self.writer.flush()

    def _train_log(self) -> None:
        path = self.round_dir / "lora" / "train_log.jsonl"
        if not path.is_file():
            return
        with open(path, "rb") as handle:
            handle.seek(self.offset)
            data = handle.read()
        end = data.rfind(b"\n") + 1  # a half-written last line waits for the next poll
        self.offset += end
        for raw in data[:end].splitlines():
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            self._row(row)

    def _row(self, row: dict) -> None:
        event = row.get("event", "")
        if event == "data":
            self.total_steps = int(row.get("total_steps") or 0)
            self.writer.text("data/selection", "```\n" + json.dumps(row, indent=1, sort_keys=True) + "\n```", 0)
            return
        if event == "step":
            self.step = int(row["step"])
            for key, tag in STEP_TAGS.items():
                if isinstance(row.get(key), (int, float)):
                    self.writer.scalar(tag, row[key], self.step)
            if self.total_steps:
                self.writer.scalar("2_progress/percent", 100.0 * self.step / self.total_steps, self.step)
                if isinstance(row.get("s_per_step"), (int, float)):
                    eta_h = (self.total_steps - self.step) * float(row["s_per_step"]) / 3600.0
                    self.writer.scalar("2_progress/eta_hours", eta_h, self.step)
            return
        step = int(row.get("step", self.step))
        for tag, value in _numbers({k: v for k, v in row.items() if k not in ("event", "step")}, f"6_{event or 'other'}"):
            self.writer.scalar(tag, value, step)

    def _gates(self) -> None:
        for path in sorted((self.round_dir / "lora").glob("gates_*.json")):
            if path.name in self.gates_seen:
                continue
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue  # still being written
            self.gates_seen.add(path.name)
            step = self.total_steps or self.step
            gates = report.get("gates", {})
            for name, gate in gates.items():
                if isinstance(gate, dict):
                    for key in ("value", "pass"):
                        if isinstance(gate.get(key), (int, float, bool)):
                            self.writer.scalar(f"3_gates/{name}/{key}", float(gate[key]), step)
                elif isinstance(gate, (int, float, bool)):
                    self.writer.scalar(f"3_gates/{name}", float(gate), step)
            self.writer.text(f"gates/{path.stem}", "```\n" + json.dumps(gates, indent=1, sort_keys=True) + "\n```", step)

    def _status(self) -> None:
        try:
            status = (self.round_dir / "STATUS").read_text(encoding="utf-8").strip()
        except OSError:
            return
        if status != self.status:
            self.status = status
            self.writer.text("status", status, self.step)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    logdir = Path(args.logdir).expanduser()
    shutil.rmtree(logdir, ignore_errors=True)
    logdir.mkdir(parents=True)
    exporter = Exporter(Path(args.round_dir).expanduser(), Writer(logdir))
    while True:
        exporter.poll()
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

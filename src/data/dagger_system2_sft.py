"""Oracle-relabelled System2 targets for sealed DAgger states.

EXP-12 measured that in recovery states native System2 answers "front" almost
without exception, and that in ``wrong_branch`` states the oracle disagrees with
it 91.7% of the time.  Training System2 to fix exactly those states is what this
module supervises.

The relabelling rule is deliberately minimal, because the arm has to be able to
say *what changed*:

``keep``
    the frozen policy's own answer, reproduced verbatim.  Self-distillation on
    every state where native already agrees with the oracle direction, which is
    most of them.  Without this the fine-tune would drift the whole policy and
    no closed-loop number could be attributed to the memory.
``correct``
    the oracle's leading turn, written in the native arrow protocol.  Emitted
    only where the oracle's own first move is a turn *and* the native proposal
    points at a different canonical view.
``correct_stop`` (EXP-14; only when ``stop_supervision`` is set)
    the answer ``STOP`` where the oracle's route ends at the goal within
    ``stop_horizon_m`` metres of the current pose and native kept walking.
    The collector never writes STOP into ``oracle.actions``: it sets
    ``oracle.terminal`` and records ``travelled_m``, the length of the shadow
    rollout to the goal, so "the oracle would stop here" is read from those.
    Native STOP states do not exist in the sealed collection (the collector
    discards every state whose native answer bypasses System1), so the
    opposite correction -- an early native stop -- cannot be supervised from
    this data and is left to the decision-level false-alarm metric.

All three shapes are already legal native outputs, so a policy trained this
way is deployable through the released RPC path with no evaluator change: an
answer containing digits is a pixel goal, an answer of arrows or ``STOP`` is
executed as primitives.

Directions come from EXP-12's ``d1_per_state.jsonl``.  That file is the single
implementation of "which canonical view is the oracle in", shared with D1, D2
and the EXP-13 feature cache, so no two experiments can drift apart on it.

Rows that cannot be labelled without guessing are dropped at construction and
counted, never silently defaulted.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

# Native System2 primitive vocabulary; the collector stored the same codes.
TURN_TEXT: dict[int, str] = {2: "←", 3: "→"}
FORWARD_CODE = 1
STOP_CODE = 0
LOOKDOWN_TEXT = "↓"
STOP_TEXT = "STOP"
_INTEGER = re.compile(r"\d+")

RELABEL_KINDS = (
    "keep_pixel",
    "keep_turn",
    "keep_stop",
    "correct_turn",
    "correct_stop",
)
CORRECTED_KINDS = ("correct_turn", "correct_stop")
STOP_KINDS = ("correct_stop", "keep_stop")


class DaggerSystem2SFTError(ValueError):
    """The relabelling contract was violated by the inputs."""


def leading_turn_run(actions: Sequence[Any]) -> tuple[int, int | None]:
    """Length and code of the oracle's opening same-direction turn run."""
    if not actions:
        return 0, None
    try:
        first = int(actions[0])
    except (TypeError, ValueError):
        return 0, None
    if first not in TURN_TEXT:
        return 0, None
    count = 0
    for value in actions:
        try:
            code = int(value)
        except (TypeError, ValueError):
            break
        if code != first:
            break
        count += 1
    return count, first


def native_pixel_answer(pixel_goal: Sequence[Any]) -> str:
    """Reproduce the native "v u" answer from the stored [u, v] pixel goal."""
    return f"{int(pixel_goal[1])} {int(pixel_goal[0])}"


def oracle_stops_within(
    oracle: dict[str, Any], horizon_m: float
) -> tuple[bool, float | None]:
    """Whether the oracle's shadow route ends at the goal within ``horizon_m``.

    ``terminal`` is the collector's own verdict that the rollout reached the
    final goal; ``travelled_m`` is how far it walked to get there.  A route that
    is terminal but longer than the horizon is a state where walking on is still
    the right answer, so it is not a stop.  Anything malformed is "not a stop":
    a guessed STOP ends an episode, a guessed pixel goal does not.
    """
    if not isinstance(oracle, dict) or not bool(oracle.get("terminal")):
        return False, None
    try:
        travelled = float(oracle.get("travelled_m"))
    except (TypeError, ValueError):
        return False, None
    if not math.isfinite(travelled) or travelled < 0.0:
        return False, None
    return travelled <= float(horizon_m), travelled


def scene_bucket(scene_id: str) -> int:
    """Stable 0-99 bucket for a scene, shared with the EXP-13 readout probe."""
    return int(hashlib.md5(str(scene_id).encode("utf-8")).hexdigest()[:8], 16) % 100


def load_oracle_views(path: str | Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("sample_key")
            if key:
                index[str(key)] = row
    return index


def plan_for_sample(
    sample: dict[str, Any],
    oracle_row: dict[str, Any] | None,
    *,
    max_turns: int = 4,
    stop_supervision: bool = False,
    stop_horizon_m: float = 1.0,
) -> dict[str, Any]:
    """Decide this state's System2 target without opening the episode tar."""
    native = sample.get("native") if isinstance(sample.get("native"), dict) else {}
    oracle = sample.get("oracle") if isinstance(sample.get("oracle"), dict) else {}
    actions = oracle.get("actions") if isinstance(oracle.get("actions"), list) else []
    turn_run, turn_code = leading_turn_run(actions)

    oracle_view = None if oracle_row is None else oracle_row.get("oracle_view")
    native_view = None if oracle_row is None else oracle_row.get("native_view")
    disagrees = (
        oracle_view is not None
        and native_view is not None
        and int(oracle_view) != int(native_view)
    )

    native_actions = native.get("actions") if isinstance(native.get("actions"), list) else []
    try:
        native_stopped = bool(native_actions) and int(native_actions[0]) == STOP_CODE
    except (TypeError, ValueError):
        native_stopped = False

    # The stop decision outranks a turn: within the horizon the goal is here,
    # and a STOP inside R2R's 3 m radius is a success whichever way one faces.
    if stop_supervision:
        stops, remaining_m = oracle_stops_within(oracle, stop_horizon_m)
        if stops and not native_stopped:
            return {
                "kind": "correct_stop",
                "target_texts": [STOP_TEXT],
                "drop_pixel_goal": True,
                "oracle_view": None if oracle_view is None else int(oracle_view),
                "native_view": None if native_view is None else int(native_view),
                "oracle_remaining_m": float(remaining_m),
                "oracle_kind": str(oracle.get("kind") or ""),
            }

    if turn_run >= 1 and disagrees:
        emitted = min(int(turn_run), int(max_turns))
        return {
            "kind": "correct_turn",
            "target_texts": [TURN_TEXT[int(turn_code)] * emitted],
            "drop_pixel_goal": True,
            "oracle_view": int(oracle_view),
            "native_view": int(native_view),
            "turn_run": int(turn_run),
            "emitted_turns": emitted,
        }

    pixel_goal = native.get("pixel_goal")
    if isinstance(pixel_goal, (list, tuple)) and len(pixel_goal) == 2:
        answer = native_pixel_answer(pixel_goal)
        recorded = [int(value) for value in _INTEGER.findall(str(native.get("llm_output") or ""))]
        if recorded != [int(pixel_goal[1]), int(pixel_goal[0])]:
            return {
                "kind": None,
                "reason": "native pixel goal disagrees with the recorded answer",
            }
        return {
            "kind": "keep_pixel",
            "target_texts": [LOOKDOWN_TEXT, answer],
            "drop_pixel_goal": False,
            "oracle_view": None if oracle_view is None else int(oracle_view),
            "native_view": None if native_view is None else int(native_view),
        }

    native_turns, native_code = leading_turn_run(native_actions)
    if native_turns >= 1:
        return {
            "kind": "keep_turn",
            "target_texts": [TURN_TEXT[int(native_code)] * min(native_turns, int(max_turns))],
            "drop_pixel_goal": True,
            "oracle_view": None if oracle_view is None else int(oracle_view),
            "native_view": None if native_view is None else int(native_view),
        }

    if native_stopped:
        return {
            "kind": "keep_stop",
            "target_texts": [STOP_TEXT],
            "drop_pixel_goal": True,
            "oracle_view": None if oracle_view is None else int(oracle_view),
            "native_view": None if native_view is None else int(native_view),
        }

    return {"kind": None, "reason": "no pixel goal, turn or stop to reproduce"}


class DaggerSystem2SFTDataset(Dataset):
    """A ``TrajectoryDaggerDataset`` view carrying System2 supervision."""

    def __init__(
        self,
        dataset: Any,
        *,
        oracle_views: str | Path | dict[str, dict[str, Any]],
        max_turns: int = 4,
        require_oracle_row: bool = True,
        scene_split: str = "all",
        val_scene_pct: int = 25,
        stop_supervision: bool = False,
        stop_horizon_m: float = 1.0,
        stop_oversample: int = 1,
    ) -> None:
        if not hasattr(dataset, "sample_metadata"):
            raise DaggerSystem2SFTError(
                "the wrapped dataset must expose sample_metadata(index)"
            )
        if scene_split not in ("all", "train", "val"):
            raise DaggerSystem2SFTError(
                f"scene_split must be all/train/val, got {scene_split!r}"
            )
        if not 0 < int(val_scene_pct) < 100:
            raise DaggerSystem2SFTError("val_scene_pct must be in (0, 100)")
        if float(stop_horizon_m) <= 0.0:
            raise DaggerSystem2SFTError("stop_horizon_m must be > 0")
        if int(stop_oversample) < 1:
            raise DaggerSystem2SFTError("stop_oversample must be >= 1")
        if int(stop_oversample) > 1 and not stop_supervision:
            raise DaggerSystem2SFTError("stop_oversample > 1 requires stop_supervision")
        self.dataset = dataset
        self.max_turns = int(max_turns)
        self.require_oracle_row = bool(require_oracle_row)
        self.scene_split = str(scene_split)
        self.val_scene_pct = int(val_scene_pct)
        self.stop_supervision = bool(stop_supervision)
        self.stop_horizon_m = float(stop_horizon_m)
        # Oversampling is a training device.  The val slice is where the
        # decision-level metrics are read and must stay one row per state.
        self.stop_oversample = int(stop_oversample) if self.scene_split != "val" else 1
        self.oracle_views = (
            oracle_views
            if isinstance(oracle_views, dict)
            else load_oracle_views(oracle_views)
        )

        # Observation-contract passthrough: the enclosing trainer picks its
        # collator from these, so a wrapper that hid them would silently switch
        # the model to a different input path.
        self._is_panoramic = bool(getattr(dataset, "_is_panoramic", False))
        self.single_view_rgb_input = bool(
            getattr(dataset, "single_view_rgb_input", False)
        )
        self.dynamic_sampling_enabled = bool(
            getattr(dataset, "dynamic_sampling_enabled", False)
        )

        self.indices: list[int] = []
        self.plans: list[dict[str, Any]] = []
        self.dropped: dict[str, int] = {}
        for index in range(len(dataset)):
            sample = dataset.sample_metadata(index)
            key = str(sample.get("key") or "")
            scene_id = str(sample.get("scene_id") or "")
            if self.scene_split != "all":
                in_val = scene_bucket(scene_id) < self.val_scene_pct
                if (self.scene_split == "val") != in_val:
                    continue
            row = self.oracle_views.get(key)
            plan = plan_for_sample(
                sample,
                row,
                max_turns=self.max_turns,
                stop_supervision=self.stop_supervision,
                stop_horizon_m=self.stop_horizon_m,
            )
            # A direction row is what keeps a turn from being guessed; a stop
            # needs no direction.  The states with no row are exactly the ones
            # already inside the oracle's goal tolerance (empty oracle.actions,
            # so EXP-12 D1 had no first move to project), which under stop
            # supervision are the clearest STOP targets there are.
            if (
                row is None
                and self.require_oracle_row
                and plan.get("kind") not in STOP_KINDS
            ):
                self.dropped["no_oracle_row"] = self.dropped.get("no_oracle_row", 0) + 1
                continue
            if plan.get("kind") is None:
                reason = str(plan.get("reason") or "unlabelled")
                self.dropped[reason] = self.dropped.get(reason, 0) + 1
                continue
            plan["sample_key"] = key
            plan["source_type"] = str(sample.get("source_type") or "")
            plan["scene_id"] = scene_id
            plan["failure_tags"] = list(sample.get("failure_tags") or [])
            self.indices.append(index)
            self.plans.append(plan)

        if not self.indices:
            raise DaggerSystem2SFTError(
                "no DAgger state could be relabelled; check oracle_views and "
                f"scene_split={self.scene_split!r}"
            )

        # Repeat the correct_stop rows for training.  Copies are appended after
        # the unique rows and flagged, so summary() and any index below
        # unique_states still mean one state each.
        self.unique_states = len(self.indices)
        self.oversampled_copies = 0
        if self.stop_oversample > 1:
            extra_indices: list[int] = []
            extra_plans: list[dict[str, Any]] = []
            for index, plan in zip(self.indices, self.plans):
                if plan["kind"] != "correct_stop":
                    continue
                for _ in range(self.stop_oversample - 1):
                    extra_indices.append(index)
                    extra_plans.append(dict(plan, oversampled=True))
            self.indices.extend(extra_indices)
            self.plans.extend(extra_plans)
            self.oversampled_copies = len(extra_indices)

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.dataset, "set_epoch", None)
        if callable(setter):
            setter(int(epoch))

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        by_source: dict[str, dict[str, int]] = {}
        for plan in self.plans:
            if plan.get("oversampled"):
                continue
            kind = str(plan["kind"])
            kinds[kind] = kinds.get(kind, 0) + 1
            bucket = by_source.setdefault(plan["source_type"], {})
            bucket[kind] = bucket.get(kind, 0) + 1
        corrected_turn = kinds.get("correct_turn", 0)
        corrected_stop = kinds.get("correct_stop", 0)
        unique = self.unique_states
        return {
            "states": len(self.indices),
            "unique_states": unique,
            "kinds": kinds,
            "by_source_type": by_source,
            # Unchanged for the EXP-13 arms: with stop supervision off this is
            # exactly correct_turn / states, the number their ledger entry reads.
            "corrected_fraction": (corrected_turn + corrected_stop) / unique,
            "corrected_turn_fraction": corrected_turn / unique,
            "corrected_stop_fraction": corrected_stop / unique,
            "dropped": dict(self.dropped),
            "max_turns": self.max_turns,
            "stop_supervision": self.stop_supervision,
            "stop_horizon_m": self.stop_horizon_m,
            "stop_oversample": self.stop_oversample,
            "oversampled_copies": self.oversampled_copies,
            "scene_split": self.scene_split,
            "val_scene_pct": self.val_scene_pct,
            "scenes": len({plan["scene_id"] for plan in self.plans}),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        plan = self.plans[index]
        sample = self.dataset[self.indices[index]]
        if str(sample.get("sample_key") or "") != plan["sample_key"]:
            raise DaggerSystem2SFTError(
                "relabelling plan and dataset row disagree on sample_key; the "
                "wrapped dataset reordered itself after construction"
            )
        if plan["drop_pixel_goal"]:
            for key in ("pixel_goal", "pano_pixel_goal", "pano_sample_kind", "pano_view_id"):
                sample.pop(key, None)
        sample["system2_target_texts"] = list(plan["target_texts"])
        sample["system2_relabel_kind"] = plan["kind"]
        return sample


__all__ = [
    "CORRECTED_KINDS",
    "DaggerSystem2SFTDataset",
    "DaggerSystem2SFTError",
    "RELABEL_KINDS",
    "STOP_KINDS",
    "leading_turn_run",
    "load_oracle_views",
    "native_pixel_answer",
    "oracle_stops_within",
    "plan_for_sample",
    "scene_bucket",
]

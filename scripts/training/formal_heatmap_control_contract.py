"""Fail-closed contract for the formal three-epoch control training loop.

Navigation evaluation is intentionally outside ``scripts/train.py`` for this
recipe.  The launcher owns exactly one sealed 8-GPU val_unseen evaluation
after the complete ``epoch_003.pth`` checkpoint has been produced.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


FORMAL_STAGE_NAME = "heatmap_system1_control"


class FormalHeatmapControlContractError(RuntimeError):
    """The formal recipe could run validation or select a best checkpoint."""


def is_formal_heatmap_control_recipe(cfg: Mapping[str, Any]) -> bool:
    training = cfg.get("training")
    if not isinstance(training, Mapping):
        return False
    stages = training.get("stages")
    return (
        isinstance(stages, (list, tuple))
        and len(stages) == 1
        and isinstance(stages[0], Mapping)
        and stages[0].get("name") == FORMAL_STAGE_NAME
    )


def assert_formal_heatmap_control_no_training_eval(
    cfg: Mapping[str, Any],
    *,
    require_formal_recipe: bool = False,
) -> dict[str, Any] | None:
    """Validate that train.py can only train/save, never evaluate/select-best."""

    if not is_formal_heatmap_control_recipe(cfg):
        if require_formal_recipe:
            raise FormalHeatmapControlContractError(
                f"formal launcher requires exactly one {FORMAL_STAGE_NAME!r} stage"
            )
        return None

    violations: list[str] = []
    training = cfg["training"]
    stage = training["stages"][0]
    if type(stage.get("epochs")) is not int or stage.get("epochs") != 3:
        violations.append("training.stages[0].epochs must be exactly 3")

    validation = cfg.get("validation")
    if not isinstance(validation, Mapping):
        violations.append("validation must be a mapping")
        validation = {}
    expected_validation = {
        "enabled": False,
        "eval_every_epochs": 0,
        "evaluate_before_training": False,
        "baseline_as_best_threshold": False,
        "best_selection_enabled": False,
        "val_inference_batches": 0,
    }
    for key, expected in expected_validation.items():
        actual = validation.get(key)
        if actual != expected or type(actual) is not type(expected):
            violations.append(
                f"validation.{key} must be exactly {expected!r}, got {actual!r}"
            )

    log_cfg = cfg.get("log")
    if not isinstance(log_cfg, Mapping):
        violations.append("log must be a mapping")
        log_cfg = {}
    if type(log_cfg.get("save_every_epochs")) is not int or log_cfg.get(
        "save_every_epochs"
    ) != 1:
        violations.append(
            "log.save_every_epochs must be exactly 1 so every epoch is resumable"
        )
    if type(log_cfg.get("val_vis_batches")) is not int or log_cfg.get(
        "val_vis_batches"
    ) != 0:
        violations.append("log.val_vis_batches must be exactly 0")

    data = cfg.get("data")
    if not isinstance(data, Mapping):
        violations.append("data must be a mapping")
        data = {}
    if data.get("val_root") not in (None, ""):
        violations.append("data.val_root must be absent for train-side no-eval")
    dagger = data.get("trajectory_dagger")
    if isinstance(dagger, Mapping):
        for key in ("val_collection_root", "val_collection_roots"):
            if dagger.get(key) not in (None, "", [], ()):
                violations.append(
                    f"data.trajectory_dagger.{key} must be absent for train-side no-eval"
                )

    if violations:
        raise FormalHeatmapControlContractError(
            "formal heatmap-control train-side evaluation contract violated: "
            + "; ".join(violations)
        )
    return {
        "schema": "heatmap-control-post-training-eval-only-v1",
        "stage_name": FORMAL_STAGE_NAME,
        "epochs": 3,
        "per_epoch_validation": False,
        "pre_training_validation": False,
        "best_checkpoint_selection": False,
        "save_every_epochs": 1,
        "external_eval_checkpoint": "epoch_003.pth",
    }


__all__ = [
    "FORMAL_STAGE_NAME",
    "FormalHeatmapControlContractError",
    "assert_formal_heatmap_control_no_training_eval",
    "is_formal_heatmap_control_recipe",
]

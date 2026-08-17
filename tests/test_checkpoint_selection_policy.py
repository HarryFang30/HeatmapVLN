import pytest

from scripts.training.selection import BestCheckpointSelector


def _metrics(
    *,
    macro: float,
    overall: float,
    back: float,
    loss: float,
):
    return {
        "val_heatmap_macro_joint_pck8": macro,
        "val_heatmap_joint_pck8": overall,
        "val_heatmap_back_pck8": back,
        "val_loss": loss,
    }


def _constrained_selector():
    return BestCheckpointSelector(
        primary_metric="val_heatmap_macro_joint_pck8",
        primary_mode="max",
        baseline_as_incumbent=True,
        overall_metric="val_heatmap_joint_pck8",
        overall_tolerance=0.02,
        back_metric="val_heatmap_back_pck8",
        back_tolerance=0.03,
        loss_metric="val_loss",
    )


def _all_direction_selector():
    return BestCheckpointSelector(
        primary_metric="val_heatmap_macro_joint_pck8",
        primary_mode="max",
        baseline_as_incumbent=True,
        overall_metric="val_heatmap_joint_pck8",
        overall_tolerance=0.02,
        back_metric="val_heatmap_back_pck8",
        back_tolerance=0.03,
        direction_metrics={
            direction: f"val_heatmap_{direction}_pck8"
            for direction in ("front", "right", "back", "left")
        },
        direction_tolerance=0.03,
        loss_metric="val_loss",
    )


def test_default_policy_preserves_strict_single_metric_min_behavior():
    selector = BestCheckpointSelector()

    assert selector.consider({"val_total_loss": 2.0}, epoch=1)[
        "accepted_as_best"
    ]
    equal = selector.consider({"val_total_loss": 2.0}, epoch=2)
    assert not equal["accepted_as_best"]
    assert equal["reason_codes"] == ["not_better_than_incumbent"]
    assert selector.consider({"val_total_loss": 1.5}, epoch=3)[
        "accepted_as_best"
    ]
    assert selector.incumbent_epoch == 3


def test_constrained_macro_policy_rejects_step0_and_natural_quality_regressions():
    selector = _constrained_selector()
    baseline = _metrics(macro=0.50, overall=0.70, back=0.80, loss=1.0)
    record = selector.set_baseline(baseline)
    assert record["baseline_as_incumbent"] is True
    assert selector.incumbent_source == "step0_baseline"

    no_macro_gain = selector.consider(
        _metrics(macro=0.50, overall=0.75, back=0.85, loss=0.9),
        epoch=1,
    )
    assert not no_macro_gain["accepted_as_best"]
    assert "primary_not_above_step0" in no_macro_gain["reason_codes"]

    overall_regression = selector.consider(
        _metrics(macro=0.60, overall=0.679, back=0.80, loss=0.9),
        epoch=2,
    )
    assert not overall_regression["eligible"]
    assert (
        "overall_below_step0_tolerance"
        in overall_regression["reason_codes"]
    )

    back_regression = selector.consider(
        _metrics(macro=0.60, overall=0.70, back=0.769, loss=0.9),
        epoch=3,
    )
    assert not back_regression["eligible"]
    assert "back_below_step0_tolerance" in back_regression["reason_codes"]

    accepted = selector.consider(
        _metrics(macro=0.60, overall=0.69, back=0.78, loss=0.9),
        epoch=4,
    )
    assert accepted["eligible"]
    assert accepted["accepted_as_best"]
    assert selector.incumbent_epoch == 4


def test_constrained_policy_uses_overall_then_loss_as_tie_breakers():
    selector = _constrained_selector()
    selector.set_baseline(
        _metrics(macro=0.50, overall=0.70, back=0.80, loss=1.0)
    )
    assert selector.consider(
        _metrics(macro=0.60, overall=0.69, back=0.79, loss=0.9),
        epoch=1,
    )["accepted_as_best"]

    # Equal macro, improved natural-distribution metric.
    assert selector.consider(
        _metrics(macro=0.60, overall=0.71, back=0.79, loss=1.1),
        epoch=2,
    )["accepted_as_best"]

    # Equal macro and overall, lower validation loss.
    assert selector.consider(
        _metrics(macro=0.60, overall=0.71, back=0.79, loss=0.8),
        epoch=3,
    )["accepted_as_best"]

    worse_loss = selector.consider(
        _metrics(macro=0.60, overall=0.71, back=0.80, loss=0.9),
        epoch=4,
    )
    assert not worse_loss["accepted_as_best"]
    assert worse_loss["reason_codes"] == ["not_better_than_incumbent"]
    assert selector.incumbent_epoch == 3


def test_selector_state_round_trip_preserves_baseline_and_incumbent():
    selector = _constrained_selector()
    selector.set_baseline(
        _metrics(macro=0.50, overall=0.70, back=0.80, loss=1.0)
    )
    selector.consider(
        _metrics(macro=0.60, overall=0.70, back=0.79, loss=0.9),
        epoch=2,
    )

    restored = _constrained_selector()
    restored.load_state_dict(selector.state_dict())
    assert restored.state_dict() == selector.state_dict()

    incompatible = BestCheckpointSelector(
        primary_metric="val_heatmap_macro_joint_pck8",
        primary_mode="max",
        baseline_as_incumbent=True,
        overall_tolerance=0.01,
    )
    with pytest.raises(ValueError, match="config changed"):
        incompatible.load_state_dict(selector.state_dict())


def test_constrained_policy_fails_closed_without_a_finite_step0_baseline():
    selector = _constrained_selector()
    with pytest.raises(ValueError, match="lacks finite"):
        selector.set_baseline(
            {
                "val_heatmap_macro_joint_pck8": 0.5,
                "val_heatmap_joint_pck8": 0.7,
            }
        )
    with pytest.raises(RuntimeError, match=r"set_baseline\(\)"):
        selector.consider(
            _metrics(macro=0.6, overall=0.7, back=0.8, loss=0.9),
            epoch=1,
        )


def test_optional_all_direction_gate_catches_a_macro_hidden_collapse():
    selector = _all_direction_selector()
    baseline = {
        **_metrics(macro=0.50, overall=0.70, back=0.80, loss=1.0),
        "val_heatmap_front_pck8": 0.70,
        "val_heatmap_right_pck8": 0.60,
        "val_heatmap_left_pck8": 0.50,
    }
    record = selector.set_baseline(baseline)
    assert record["direction_values"]["right"] == pytest.approx(0.60)

    candidate = {
        **_metrics(macro=0.60, overall=0.70, back=0.80, loss=0.9),
        "val_heatmap_front_pck8": 0.80,
        # Other gains make macro better, but right has collapsed by 0.04.
        "val_heatmap_right_pck8": 0.56,
        "val_heatmap_left_pck8": 0.70,
    }
    decision = selector.consider(candidate, epoch=1)
    assert not decision["eligible"]
    assert not decision["accepted_as_best"]
    assert (
        "direction_right_below_step0_tolerance"
        in decision["reason_codes"]
    )
    assert decision["candidate_directions"]["right"] == pytest.approx(0.56)


def test_legacy_resume_incumbent_preserves_default_loss_selection():
    selector = BestCheckpointSelector()
    selector.set_incumbent(
        {"val_total_loss": 1.0},
        epoch=5,
        source="legacy_resume",
    )
    assert not selector.consider(
        {"val_total_loss": 1.1},
        epoch=6,
    )["accepted_as_best"]
    assert selector.consider(
        {"val_total_loss": 0.9},
        epoch=7,
    )["accepted_as_best"]

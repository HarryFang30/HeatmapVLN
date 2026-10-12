"""--stop_at_step_budget in the RPC client: an episode that runs out of steps without calling STOP
scores 0 however close it ends, so with the flag the last action of the budget is STOP.  The
episode still takes --max_steps_per_episode actions, every action before the last is the one
the run without the flag took, and episodes that stop on their own are untouched.  Runs the
client end to end on the fakes of ``test_s2_record_collection``."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from tests import test_s2_record_collection as harness
from tests.test_s2_record_collection import client  # noqa: F401  (fixture)

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "run_ppa_r2r_val_unseen_cuda.sh"
BUDGET = 6
FLAG_FIELDS = ("stop_at_step_budget", "step_budget_stop")


def _actions_by_episode(run) -> dict:
    actions = defaultdict(list)
    for key, action, _position, _rotation in run.env.primitive:
        actions[key].append(action)
    return actions


def _row_key(row) -> tuple:
    return (row["scene_id"], int(row["episode_id"]))


def _without_flag(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in FLAG_FIELDS}


@pytest.mark.parametrize("mode", ["vo", "native"])
def test_out_of_budget_episodes_end_with_stop_and_nothing_before_it_changes(client, monkeypatch, tmp_path, mode):
    budget = ("--max_steps_per_episode", str(BUDGET))
    off = harness._run(client, monkeypatch, tmp_path, mode=mode, record=False, tag="off", extra_argv=budget)
    on = harness._run(
        client, monkeypatch, tmp_path, mode=mode, record=False, tag="on",
        extra_argv=budget + ("--stop_at_step_budget",),
    )
    off_actions, on_actions = _actions_by_episode(off), _actions_by_episode(on)
    off_rows = {_row_key(row): row for row in off.progress}
    assert len(on.progress) == len(off.progress) == len(harness.EPISODES)

    ran_out = 0
    for row in on.progress:
        key = _row_key(row)
        before = off_rows[key]
        assert row["stop_at_step_budget"] is True
        if before["steps"] >= BUDGET and off_actions[key][-1] != 0:  # this episode ran out of steps
            ran_out += 1
            assert before["success"] == 0.0
            assert on_actions[key] == off_actions[key][: BUDGET - 1] + [0]
            assert row["steps"] == BUDGET and row["step_budget_stop"] is True
            assert row["success"] == float(row["ne"] <= 3.0)
        else:
            assert on_actions[key] == off_actions[key]
            assert row["step_budget_stop"] is False and _without_flag(row) == before
    assert ran_out > 0  # the budget is small enough that the fakes run out of it


def test_with_room_to_spare_the_flag_changes_no_action(client, monkeypatch, tmp_path):
    off = harness._run(client, monkeypatch, tmp_path, mode="vo", record=False, tag="off")
    on = harness._run(
        client, monkeypatch, tmp_path, mode="vo", record=False, tag="on", extra_argv=("--stop_at_step_budget",)
    )
    assert on.env.trace == off.env.trace
    assert [r["payload"] for r in on.server.requests] == [r["payload"] for r in off.server.requests]
    assert all(row["step_budget_stop"] is False for row in on.progress)
    assert [_without_flag(row) for row in on.progress] == off.progress
    assert all(not set(FLAG_FIELDS) & set(row) for row in off.progress)  # off keeps the old row schema


def test_resume_refuses_rows_written_without_the_flag(client, monkeypatch, tmp_path):
    budget = ("--max_steps_per_episode", str(BUDGET))
    harness._run(client, monkeypatch, tmp_path, mode="vo", record=False, tag="mixed", extra_argv=budget)
    with pytest.raises(ValueError, match="contract mismatch"):
        harness._run(
            client, monkeypatch, tmp_path, mode="vo", record=False, tag="mixed",
            extra_argv=budget + ("--stop_at_step_budget", "--resume"),
        )


def test_launcher_passes_the_flag_only_when_asked():
    sh = LAUNCHER.read_text(encoding="utf-8")
    assert 'STOP_AT_STEP_BUDGET="${PPA_EVAL_STOP_AT_STEP_BUDGET:-0}"' in sh
    assert "CLIENT_EXTRA=(--stop_at_step_budget)" in sh
    assert '"${CLIENT_EXTRA[@]}"' in sh and "declare -a CLIENT_EXTRA=()" in sh

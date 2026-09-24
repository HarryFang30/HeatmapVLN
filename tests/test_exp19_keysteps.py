"""EXP-19 key-moment rule (scripts/exp19/keysteps.py): every branch of the pre-registered 关键时刻.

Pure Python, no numpy / torch.
"""
from __future__ import annotations

import itertools
import random

import pytest

from scripts.exp19.keysteps import (
    K2_MIN_TURN_DEG,
    closest_approach_step,
    net_turn_deg,
    select_key_steps,
)

F, L, R, S = 1, 2, 3, 0


def calls(*specs):
    """specs: (step, executed_actions) in call order; call_index = 10 * position (not contiguous on purpose)."""
    return [{"call_index": 10 * i, "step": step, "executed_actions": list(acts)} for i, (step, acts) in enumerate(specs)]


def picks(keys):
    return {k["label"]: k["ready_index"] for k in keys}


def branches(keys):
    return {k["label"]: k["branch"] for k in keys}


def test_net_turn_is_left_positive_15_degrees_per_turn():
    assert net_turn_deg([L, L, F]) == 30.0
    assert net_turn_deg([R, R, R, L]) == -30.0
    assert net_turn_deg([F, F, S]) == 0.0
    assert net_turn_deg([]) == 0.0


def test_no_ready_calls_gives_no_key_steps():
    assert select_key_steps([], category="T1", episode_steps=50) == []


@pytest.mark.parametrize("n", [1, 2, 3])
def test_fewer_than_four_ready_calls_draws_all_in_call_order(n):
    ready = calls(*[(20 + 4 * i, [L, L, L, L]) for i in range(n)])
    keys = select_key_steps(ready, category="F1", episode_steps=60, closest_step=None)
    assert [k["label"] for k in keys] == ["K1", "K2", "K3", "K4"][:n]
    assert [k["ready_index"] for k in keys] == list(range(n))
    assert [k["call_index"] for k in keys] == [10 * i for i in range(n)]
    assert all(k["branch"] == "all_lt4" for k in keys)


def test_k1_first_k2_largest_turn_k3_two_thirds_k4_last():
    ready = calls((20, [L, L, L, L]),  # K1 even though it turns most
                  (24, [F, F, F, F]),
                  (28, [R, R, R, F]),  # |-45| largest among the others -> K2
                  (32, [L, L, F, F]),
                  (40, [F, F, F, F]),  # 2/3 of 60 = 40 -> K3
                  (44, [F, F, F, F]),
                  (48, [F, F, S]))     # last -> K4
    keys = select_key_steps(ready, category="T1", episode_steps=60)
    assert picks(keys) == {"K1": 0, "K2": 2, "K3": 4, "K4": 6}
    assert branches(keys) == {"K1": "K1_first", "K2": "K2_turn", "K3": "K3_two_thirds", "K4": "K4_last"}
    assert keys[1]["net_turn_deg"] == -45.0
    assert [k["label"] for k in keys] == ["K1", "K2", "K3", "K4"]


def test_k2_turn_ties_take_the_earliest_and_sign_does_not_matter():
    ready = calls((20, [F]), (24, [R, R, R]), (28, [L, L, L]), (32, [F]), (36, [F]))
    keys = select_key_steps(ready, category="T2", episode_steps=60)
    assert picks(keys)["K2"] == 1
    assert "tied" in keys[1]["rule"]


def test_k2_threshold_is_inclusive_at_30_degrees():
    ready = calls((20, [F]), (24, [F]), (28, [L, L, F, F]), (32, [F]), (36, [F]), (40, [F]), (44, [F]))
    keys = select_key_steps(ready, category="T3", episode_steps=66)
    assert K2_MIN_TURN_DEG == 30.0
    assert picks(keys)["K2"] == 2 and branches(keys)["K2"] == "K2_turn"


def test_k2_uses_the_executed_chunk_net_angle_not_the_turn_count():
    # L, R, L, R has four turns but nets to 0; a plain 30 deg chunk wins
    ready = calls((20, [F]), (24, [L, R, L, R]), (28, [F]), (32, [R, R]), (36, [F]))
    keys = select_key_steps(ready, category="T1", episode_steps=60)
    assert picks(keys)["K2"] == 3


@pytest.mark.parametrize("n,expected", [(4, 1), (5, 1), (6, 1), (7, 2), (10, 3), (13, 4)])
def test_k2_fallback_is_floor_n_minus_1_over_3_when_nothing_turns_30(n, expected):
    ready = calls(*[(20 + 4 * i, [L, F, F, F] if i % 2 else [F, F, F, F]) for i in range(n)])  # max 15 deg
    keys = select_key_steps(ready, category="F2", episode_steps=500)
    assert picks(keys)["K2"] == expected
    assert branches(keys)["K2"] == "K2_fallback"
    assert "max 15 deg" in keys[1]["rule"]


def test_k1_turn_does_not_count_for_k2():
    ready = calls((20, [L, L, L, L, L, L]), (24, [F]), (28, [F]), (32, [F]), (36, [F]))
    keys = select_key_steps(ready, category="T1", episode_steps=60)
    assert branches(keys)["K2"] == "K2_fallback" and picks(keys)["K2"] == 1


def test_k3_two_thirds_ties_take_the_earlier_and_skip_chosen():
    # 2/3 of 60 = 40: steps 38 and 42 tie -> 38.  K2 (big turn) sits at exactly 40, so it is excluded.
    ready = calls((20, [F]), (38, [F]), (40, [R, R, R]), (42, [F]), (50, [F]))
    keys = select_key_steps(ready, category="T1", episode_steps=60)
    assert picks(keys)["K2"] == 2
    assert picks(keys)["K3"] == 1
    assert "tied" in keys[2]["rule"]


def test_k3_two_thirds_uses_exact_integer_distances():
    # steps 61 -> target 40.67: |3*27 - 122| = 41 vs |3*41 - 122| = 1 -> step 41
    ready = calls((20, [F]), (27, [F]), (41, [F]), (55, [F]), (59, [F]))
    keys = select_key_steps(ready, category="T3", episode_steps=61)
    assert keys[2]["step"] == 41


def test_k3_f1_last_ready_call_at_or_before_the_closest_approach():
    ready = calls((20, [F]), (30, [F]), (40, [F]), (50, [F]), (60, [F]), (70, [F]))
    # K2 fallback = floor(5/3) = 1 (step 30); closest approach at step 50 -> index 3 (inclusive), not index 4
    keys = select_key_steps(ready, category="F1", episode_steps=90, closest_step=50)
    assert picks(keys) == {"K1": 0, "K2": 1, "K3": 3, "K4": 5}
    assert branches(keys)["K3"] == "K3_f1_closest"
    keys = select_key_steps(ready, category="F1", episode_steps=90, closest_step=59)
    assert picks(keys)["K3"] == 3


def test_k3_f1_skips_chosen_calls_before_the_closest_approach():
    # the call right before the closest approach is K2 (big turn) -> the one before it
    ready = calls((20, [F]), (30, [F]), (40, [L, L, L]), (50, [F]), (60, [F]))
    keys = select_key_steps(ready, category="F1", episode_steps=80, closest_step=45)
    assert picks(keys)["K2"] == 2 and picks(keys)["K3"] == 1


def test_k3_f1_fallback_when_nothing_is_before_the_closest_approach():
    ready = calls((20, [F]), (30, [F]), (40, [F]), (50, [F]))
    keys = select_key_steps(ready, category="F1", episode_steps=80, closest_step=22)
    # K1 = 0 (step 20 is the only call at or before 22, already chosen); K2 fallback = 1 -> K3 = first remaining after
    assert picks(keys)["K3"] == 2
    assert branches(keys)["K3"] == "K3_f1_fallback_after"
    assert "not pre-registered" in keys[2]["rule"]


def test_k4_takes_the_nearest_earlier_unchosen_when_the_last_is_taken():
    ready = calls((20, [F]), (24, [F]), (28, [F]), (32, [F]), (36, [L, L, L]))  # K2 = last call
    keys = select_key_steps(ready, category="T1", episode_steps=40)  # K3: 2/3*40=26.7 -> step 28 (index 2)
    assert picks(keys) == {"K1": 0, "K2": 4, "K3": 2, "K4": 3}
    assert branches(keys)["K4"] == "K4_shifted"
    # last two taken (K2 last, K3 second-to-last) -> K4 steps back twice
    keys = select_key_steps(ready, category="T1", episode_steps=48)  # 2/3*48 = 32 -> index 3
    assert picks(keys) == {"K1": 0, "K2": 4, "K3": 3, "K4": 2}


def test_four_ready_calls_give_four_distinct_labels_on_every_branch():
    rnd = random.Random(0)
    for _ in range(2000):
        n = rnd.randint(4, 30)
        steps = sorted(rnd.sample(range(20, 400), n))
        ready = calls(*[(s, [rnd.choice([F, F, L, R, S]) for _ in range(rnd.randint(1, 4))]) for s in steps])
        cat = rnd.choice(["T1", "T2", "T3", "F1", "F2"])
        keys = select_key_steps(ready, category=cat, episode_steps=steps[-1] + rnd.randint(1, 20),
                                closest_step=rnd.randint(0, 420) if cat == "F1" else None)
        assert [k["label"] for k in keys] == ["K1", "K2", "K3", "K4"]
        assert len({k["ready_index"] for k in keys}) == 4
        assert keys[0]["ready_index"] == 0
        assert all(0 <= k["ready_index"] < n for k in keys)


def test_input_validation():
    with pytest.raises(ValueError, match="strictly increasing"):
        select_key_steps([{"call_index": 3, "step": 20}, {"call_index": 3, "step": 24}], category="T1", episode_steps=50)
    with pytest.raises(ValueError, match="strictly increasing"):
        select_key_steps([{"call_index": 5, "step": 20}, {"call_index": 4, "step": 24}], category="T1", episode_steps=50)
    with pytest.raises(ValueError, match="unknown category"):
        select_key_steps(calls((20, [F])), category="S1", episode_steps=50)
    with pytest.raises(ValueError, match="closest_step"):
        select_key_steps(calls(*[(20 + i, [F]) for i in range(5)]), category="F1", episode_steps=50)


def test_closest_approach_step_takes_the_earliest_minimum():
    assert closest_approach_step([0, 1, 2, 3, 4], [5.0, 3.0, 1.0, 1.0, 2.0]) == 2
    assert closest_approach_step([10, 11], [0.5, 0.7]) == 10
    with pytest.raises(ValueError):
        closest_approach_step([0, 1], [1.0])
    with pytest.raises(ValueError):
        closest_approach_step([], [])


def test_every_branch_is_reachable():
    seen = set()
    for spec in itertools.product([0, 1], repeat=3):
        turn, f1, late = spec
        ready = calls((20, [F]), (30, [L, L] if turn else [F]), (40, [F]), (50, [F]), (60, [F]))
        keys = select_key_steps(ready, category="F1" if f1 else "T1", episode_steps=90,
                                closest_step=25 if late else 55)
        seen |= {k["branch"] for k in keys}
    keys = select_key_steps(calls((20, [F]), (24, [F]), (28, [F]), (32, [F]), (36, [L, L])), category="T1", episode_steps=40)
    seen |= {k["branch"] for k in keys}
    seen |= {k["branch"] for k in select_key_steps(calls((20, [F])), category="T1", episode_steps=40)}
    assert seen == {"K1_first", "K2_turn", "K2_fallback", "K3_two_thirds", "K3_f1_closest", "K3_f1_fallback_after",
                    "K4_last", "K4_shifted", "all_lt4"}

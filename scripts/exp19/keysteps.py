"""EXP-19 key moments: the pre-registered 关键时刻 rule as a pure function.

docs/experiments/README.md, EXP-19 "关键时刻（写死）", verbatim:

  只从**就绪调用**里取——``kind = trajectory`` 且 ``ppa_applied``，按调用序排。每集 4 个：
  - K1 = 第一个就绪调用；
  - K2 = 其余就绪调用中，**实际执行的动作块**净转角绝对值最大且 ≥ 30° 者（左 +15°、右 −15°），
    并列取最早；没有 ≥ 30° 的，取就绪调用序列 ⌊(n−1)/3⌋ 处（与 K1 重复则顺延）；
  - K3 = T1/T2/T3/F2：其余就绪调用中步号最接近全集步数 2/3 者（并列取较早）；
    F1：距目标（欧氏，真值位置）最近那一步及其之前的最后一个其余就绪调用；
  - K4 = 最后一个就绪调用（已被选则取之前最近的未选者）。
  就绪调用不足 4 个的集全部画出。

Readings fixed here (the ledger leaves them implicit):

* "其余" = ready calls not already chosen by an earlier label (K1 before K2 before K3 before K4).
* "实际执行的动作块" = the primitive actions the client actually executed from that call's chunk
  (the response chunk cut at its first STOP, or cut by the episode end), not the response list.
  Net turn = 15 deg x (#LEFT - #RIGHT); FORWARD / STOP contribute 0.
* ⌊(n−1)/3⌋ indexes the ready-call sequence 0-based (>= 1 whenever n >= 4, so it never repeats K1).
* "全集步数" = the rerun episode's step count (steps.jsonl episode_end ``steps``).  Distances to
  the 2/3 target are compared as integers |3*step - 2*steps| so ties are exact.
* F1 "距目标最近那一步" = the earliest recorded step with the minimum Euclidean distance between the
  GT agent position and the goal position; "及其之前" includes that step.  If no remaining ready
  call is at or before it, the first remaining ready call after it is taken and the rule text says
  so (fallback, not pre-registered; the only branch the ledger does not define).
* Fewer than 4 ready calls: every ready call is drawn, labelled K1.. in call order.

Pure Python (no numpy); Python 3.8 compatible.
"""
from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence

STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
TURN_STEP_DEG = 15.0
K2_MIN_TURN_DEG = 30.0
LABELS = ("K1", "K2", "K3", "K4")
TWO_THIRDS_CATEGORIES = ("T1", "T2", "T3", "F2")
CLOSEST_APPROACH_CATEGORIES = ("F1",)
CATEGORIES = TWO_THIRDS_CATEGORIES + CLOSEST_APPROACH_CATEGORIES


def net_turn_deg(actions: Iterable[int]) -> float:
    """Net turn of an executed chunk, degrees, left-positive (LEFT +15, RIGHT -15)."""
    acts = [int(a) for a in actions]
    return TURN_STEP_DEG * (acts.count(LEFT) - acts.count(RIGHT))


def closest_approach_step(steps: Sequence[int], distances: Sequence[float]) -> int:
    """Earliest step with the minimum distance to the goal (F1's anchor for K3)."""
    if len(steps) != len(distances) or not steps:
        raise ValueError("steps and distances must be non-empty and the same length")
    order = sorted(range(len(steps)), key=lambda i: (float(distances[i]), int(steps[i])))
    return int(steps[order[0]])


def _check_ready(ready: Sequence[Mapping]) -> None:
    prev = None
    for r in ready:
        idx = int(r["call_index"])
        if prev is not None and idx <= prev:
            raise ValueError(f"ready calls must be in strictly increasing call order, got {prev} then {idx}")
        prev = idx


def _entry(label: str, i: int, ready: Sequence[Mapping], branch: str, rule: str) -> dict:
    r = ready[i]
    return {"label": label, "ready_index": i, "call_index": int(r["call_index"]), "step": int(r["step"]),
            "net_turn_deg": net_turn_deg(r.get("executed_actions") or []), "branch": branch, "rule": rule}


def select_key_steps(ready: Sequence[Mapping], *, category: str, episode_steps: int,
                     closest_step: Optional[int] = None) -> List[dict]:
    """Apply the pre-registered rule to one episode.

    ready: the episode's ready calls in call order, each a mapping with ``call_index``,
    ``step`` (the call's capture step) and ``executed_actions`` (list of action codes).
    category: the episode's case category (T1, T2, T3, F1, F2).
    episode_steps: steps of the rerun episode (for the 2/3 target).
    closest_step: F1 only, ``closest_approach_step`` of the rerun.

    Returns the key steps in label order (K1..K4), each with ``branch`` (machine tag of the rule
    branch taken) and ``rule`` (human-readable reason).
    """
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; expected one of {CATEGORIES}")
    _check_ready(ready)
    n = len(ready)
    if n == 0:
        return []
    if n < 4:
        return [_entry(LABELS[i], i, ready, "all_lt4", f"all ready calls drawn (n = {n} < 4)") for i in range(n)]

    chosen: List[int] = []
    out: List[dict] = []

    # K1
    chosen.append(0)
    out.append(_entry("K1", 0, ready, "K1_first", "first ready call"))

    # K2
    rest = [i for i in range(n) if i not in chosen]
    turns = {i: abs(net_turn_deg(ready[i].get("executed_actions") or [])) for i in rest}
    best = max(turns.values())
    if best >= K2_MIN_TURN_DEG:
        i2 = min(i for i in rest if turns[i] == best)  # ties -> earliest
        ties = sum(1 for i in rest if turns[i] == best)
        out.append(_entry("K2", i2, ready, "K2_turn",
                          f"largest |net turn| of the executed chunk among the other ready calls: {best:.0f} deg "
                          f">= {K2_MIN_TURN_DEG:.0f} deg" + (f" ({ties} tied, earliest taken)" if ties > 1 else "")))
    else:
        # floor((n-1)/3) >= 1 for n >= 4, so the ledger's "与 K1 重复则顺延" never fires here
        # (it only could for n < 4, where every ready call is drawn anyway).
        i2 = (n - 1) // 3
        out.append(_entry("K2", i2, ready, "K2_fallback",
                          f"no other ready call turns >= {K2_MIN_TURN_DEG:.0f} deg (max {best:.0f} deg): "
                          f"ready index floor((n-1)/3) = {i2}"))
    chosen.append(i2)

    # K3
    rest = [i for i in range(n) if i not in chosen]
    if category in TWO_THIRDS_CATEGORIES:
        steps = int(episode_steps)
        dist = {i: abs(3 * int(ready[i]["step"]) - 2 * steps) for i in rest}
        i3 = min(rest, key=lambda i: (dist[i], i))  # ties -> earlier
        ties = sum(1 for i in rest if dist[i] == dist[i3])
        out.append(_entry("K3", i3, ready, "K3_two_thirds",
                          f"step closest to 2/3 of the episode's {steps} steps (target {2 * steps / 3:.1f}, "
                          f"|diff| {dist[i3] / 3:.2f})" + (f" ({ties} tied, earlier taken)" if ties > 1 else "")))
    else:
        if closest_step is None:
            raise ValueError("F1 needs closest_step (closest_approach_step of the rerun)")
        before = [i for i in rest if int(ready[i]["step"]) <= int(closest_step)]
        if before:
            i3 = max(before)
            out.append(_entry("K3", i3, ready, "K3_f1_closest",
                              f"last other ready call at or before the closest approach to the goal (step {closest_step})"))
        else:
            i3 = min(rest)
            out.append(_entry("K3", i3, ready, "K3_f1_fallback_after",
                              f"no other ready call at or before the closest approach (step {closest_step}); "
                              "took the first one after it (fallback, not pre-registered)"))
    chosen.append(i3)

    # K4
    i4 = n - 1
    while i4 in chosen:
        i4 -= 1
    out.append(_entry("K4", i4, ready, "K4_last" if i4 == n - 1 else "K4_shifted",
                      "last ready call" if i4 == n - 1 else
                      f"last ready call (index {n - 1}) already chosen; nearest earlier unchosen is {i4}"))
    return out

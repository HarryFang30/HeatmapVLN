from collections import Counter
from types import SimpleNamespace

import pytest

from scripts.tools.build_replay_stream import select


def _scene(n, stops=1):
    return [SimpleNamespace(kind="stop" if i < stops else "pixel", i=i) for i in range(n)]


def test_per_scene_cap_spreads_the_draw_and_stops_are_upsampled():
    pools = {"scalevln": {"big": _scene(100), "small": _scene(10)}, "r2r": {"a": _scene(20, stops=2)}}
    chosen, counts = select(pools, samples={"scalevln": 20, "r2r": 20}, max_per_scene={"scalevln": 10},
                            stop_upsample=5, seed=0)
    assert counts["scalevln_pool"] == 20  # 10 from "big" (capped) + 10 from "small"
    scalevln = [x for root, x in chosen if root == "scalevln"]
    assert len({id(x) for x in scalevln}) == 20  # every capped sample drawn once...
    stops = sum(x.kind == "stop" for x in scalevln)
    assert stops == 5 * len({id(x) for x in scalevln if x.kind == "stop"})  # ...and each STOP five times
    r2r = Counter(x.kind for root, x in chosen if root == "r2r")
    assert r2r == {"pixel": 18, "stop": 10}
    assert counts["r2r"] == 28 and counts["r2r_target_stop"] == 10


def test_the_draw_is_seeded_and_an_oversized_ask_fails():
    pools = {"x": {"s": _scene(50)}}
    a, _ = select(pools, samples={"x": 30}, max_per_scene={}, stop_upsample=1, seed=3)
    b, _ = select(pools, samples={"x": 30}, max_per_scene={}, stop_upsample=1, seed=3)
    assert [x.i for _, x in a] == [x.i for _, x in b]
    with pytest.raises(SystemExit, match="asked for 60"):
        select(pools, samples={"x": 60}, max_per_scene={}, stop_upsample=1, seed=3)

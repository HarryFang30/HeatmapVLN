from scripts.tools.regate_s2_lora import regate


def _row(agree, off, on, d, off_turns=None, on_turns=None):
    return {"agree": agree, "distance_to_goal": d,
            "off": {"class": off, "turns": off_turns}, "on": {"class": on, "turns": on_turns}}


REPORT = {"settings": {"false_stop_distance_m": 3.0},
          "gates": {"G0_adapter_off_replay": {"value": 0.998, "threshold": 0.98, "pass": True}}}


def test_class_keep_ignores_pixel_jitter_and_false_stops_are_judged_against_the_base():
    rows = [_row(True, "pixel", "pixel", 5.0, ["↓", "100 200"], ["↓", "102 200"]) for _ in range(9)]
    rows.append(_row(True, "pixel", "stop", 5.0, ["↓", "100 200"], ["STOP"]))  # 9/10 classes kept
    rows += [_row(False, "turn_left", "turn_left", 6.0) for _ in range(88)]
    rows += [_row(False, "stop", "stop", 6.0), _row(False, "turn_left", "stop", 6.0)]
    out = regate(REPORT, rows)["gates"]
    assert out["G1_class_keep"]["value"] == 0.9 and out["G1_class_keep"]["pass"]
    # beyond 3 m: base 1 stop / 100 calls, adapter 3 / 100 -> 3x the base, over 1.5x
    assert out["G2_false_stop_vs_base"]["base_rate"] == 0.01 and out["G2_false_stop_vs_base"]["value"] == 0.03
    assert not out["G2_false_stop_vs_base"]["pass"] and not out["all_pass"]


def test_a_skipped_call_does_not_count():
    rows = [_row(True, "pixel", "pixel", 5.0, ["↓", "1 1"], ["↓", "1 1"]),
            dict(_row(True, "pixel", "stop", 5.0), skip=True)]
    out = regate(REPORT, rows)["gates"]
    assert out["G1_class_keep"]["n"] == 1 and out["all_pass"]

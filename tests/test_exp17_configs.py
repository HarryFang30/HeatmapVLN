"""The EXP-17 arms differ from EXP-14 and from each other only where registered.

exp17a vs exp14a is "the same relabelled fine-tune with pose tokens instead of
the Past Head memory"; exp17b vs exp17a is "the same arm plus the cognition
prefix".  The substantive (non-comment) line diffs are pinned so a stray edit
cannot make a result unattributable.
"""

from __future__ import annotations

import difflib
from pathlib import Path

_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "ablation"


def _substantive_lines(name: str) -> list[str]:
    text = (_CONFIGS / name).read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _changed(a: str, b: str) -> list[str]:
    diff = difflib.unified_diff(_substantive_lines(a), _substantive_lines(b), lineterm="", n=0)
    return [
        line
        for line in diff
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("+++")
        and not line.startswith("---")
    ]


def test_exp17a_is_exp14a_with_pose_tokens_after_the_current_view() -> None:
    changed = _changed(
        "exp14a_system2_memory_stop_lora_8gpu.yaml",
        "exp17a_c1_geometry_stop_lora_8gpu.yaml",
    )
    assert sorted(changed) == sorted(
        [
            "-    mode: memory",
            "+    mode: geometry",
            "+    placeholder_position: after_current",
            "+    pose_dropout: 0.0",
            "-    - name: exp14a_system2_memory_stop",
            "+    - name: exp17a_c1_geometry_stop",
        ]
    )


def test_exp17b_is_exp17a_plus_the_cognition_prefix() -> None:
    changed = _changed(
        "exp17a_c1_geometry_stop_lora_8gpu.yaml",
        "exp17b_c3_geometry_prefix_stop_lora_8gpu.yaml",
    )
    assert sorted(changed) == sorted(
        [
            "+    cognition_prefix: true",
            "+    prefix_placeholder_fraction: 0.2",
            "+    reference_path_json: $R2R_TRAIN_JSON",
            "-    - name: exp17a_c1_geometry_stop",
            "+    - name: exp17b_c3_geometry_prefix_stop",
        ]
    )


def test_placeholder_fraction_is_the_registered_value() -> None:
    lines = _substantive_lines("exp17b_c3_geometry_prefix_stop_lora_8gpu.yaml")
    assert "    prefix_placeholder_fraction: 0.2" in lines

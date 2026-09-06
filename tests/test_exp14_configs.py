"""The EXP-14 arms differ from EXP-13 and from each other only where registered.

The ledger's attribution rests on two diffs: exp14a vs exp13a is "the same arm
plus stop supervision", and exp14a vs exp14b is "the same data, memory tokens
that do or do not depend on M_t".  A stray edit in either config would make a
result unattributable without anyone noticing, so the substantive (non-comment)
line diffs are pinned here.
"""

from __future__ import annotations

import difflib
from pathlib import Path

_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "ablation"


def _substantive_lines(name: str) -> list[str]:
    text = (_CONFIGS / name).read_text(encoding="utf-8")
    return [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _changed(a: str, b: str) -> list[str]:
    diff = difflib.unified_diff(_substantive_lines(a), _substantive_lines(b), lineterm="", n=0)
    return [
        line
        for line in diff
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("+++")
        and not line.startswith("---")
    ]


def test_exp14a_is_exp13a_plus_stop_supervision_and_a_stage_name() -> None:
    changed = _changed(
        "exp13a_system2_memory_lora_8gpu.yaml",
        "exp14a_system2_memory_stop_lora_8gpu.yaml",
    )
    assert sorted(changed) == sorted(
        [
            "-    - name: exp13a_system2_memory",
            "+    - name: exp14a_system2_memory_stop",
            "+    stop_supervision: true",
            "+    stop_horizon_m: 1.0",
            "+    stop_oversample: 1",
        ]
    )


def test_exp14b_is_exp13b_plus_the_same_stop_keys() -> None:
    changed = _changed(
        "exp13b_system2_constant_lora_8gpu.yaml",
        "exp14b_system2_constant_stop_lora_8gpu.yaml",
    )
    assert sorted(changed) == sorted(
        [
            "-    - name: exp13b_system2_constant",
            "+    - name: exp14b_system2_constant_stop",
            "+    stop_supervision: true",
            "+    stop_horizon_m: 1.0",
            "+    stop_oversample: 1",
        ]
    )


def test_the_two_exp14_arms_differ_only_in_memory_mode_and_stage_name() -> None:
    changed = _changed(
        "exp14a_system2_memory_stop_lora_8gpu.yaml",
        "exp14b_system2_constant_stop_lora_8gpu.yaml",
    )
    assert sorted(changed) == sorted(
        [
            "-    mode: memory",
            "+    mode: constant",
            "-    - name: exp14a_system2_memory_stop",
            "+    - name: exp14b_system2_constant_stop",
        ]
    )


# The 4-GPU variants (2026-09-06) must be the same optimisation as the 8-GPU
# arms: halving the ranks doubles the accumulation, so each optimizer step still
# averages one 16-sample window.  Anything else that differs would silently
# change the protocol the ledger pre-registered.
_FOUR_VS_EIGHT = [
    "-  devices: [0, 1, 2, 3, 4, 5, 6, 7]",
    "+  devices: [0, 1, 2, 3]",
    "-  grad_accum_steps: 2",
    "+  grad_accum_steps: 4",
]


def test_exp14a_4gpu_differs_from_8gpu_only_in_devices_and_accumulation() -> None:
    changed = _changed(
        "exp14a_system2_memory_stop_lora_8gpu.yaml",
        "exp14a_system2_memory_stop_lora_4gpu.yaml",
    )
    assert sorted(changed) == sorted(_FOUR_VS_EIGHT)


def test_exp14b_4gpu_differs_from_8gpu_only_in_devices_and_accumulation() -> None:
    changed = _changed(
        "exp14b_system2_constant_stop_lora_8gpu.yaml",
        "exp14b_system2_constant_stop_lora_4gpu.yaml",
    )
    assert sorted(changed) == sorted(_FOUR_VS_EIGHT)


def test_the_two_4gpu_arms_differ_only_in_memory_mode_and_stage_name() -> None:
    changed = _changed(
        "exp14a_system2_memory_stop_lora_4gpu.yaml",
        "exp14b_system2_constant_stop_lora_4gpu.yaml",
    )
    assert sorted(changed) == sorted(
        [
            "-    mode: memory",
            "+    mode: constant",
            "-    - name: exp14a_system2_memory_stop",
            "+    - name: exp14b_system2_constant_stop",
        ]
    )


def test_effective_global_batch_is_16_in_every_exp14_config() -> None:
    import re

    for name in (
        "exp14a_system2_memory_stop_lora_8gpu.yaml",
        "exp14b_system2_constant_stop_lora_8gpu.yaml",
        "exp14a_system2_memory_stop_lora_4gpu.yaml",
        "exp14b_system2_constant_stop_lora_4gpu.yaml",
    ):
        text = (_CONFIGS / name).read_text(encoding="utf-8")
        devices = re.search(r"^  devices: \[([0-9, ]+)\]$", text, re.M).group(1).split(",")
        batch = int(re.search(r"^  batch_size: (\d+)$", text, re.M).group(1))
        accum = int(re.search(r"^  grad_accum_steps: (\d+)$", text, re.M).group(1))
        assert len(devices) * batch * accum == 16, name

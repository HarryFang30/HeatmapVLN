"""``scripts/train.py --validate-only`` contract."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import scripts.train as train


def test_validate_only_requires_pre_training_validation() -> None:
    with pytest.raises(ValueError, match="evaluate_before_training=true"):
        train._check_validate_only_flags(
            validate_only=True, evaluate_before_training=False
        )
    train._check_validate_only_flags(validate_only=True, evaluate_before_training=True)
    train._check_validate_only_flags(validate_only=False, evaluate_before_training=False)


def test_validate_only_exits_before_the_training_loop() -> None:
    source = Path(train.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    flag_added = any(
        isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "add_argument"
        and node.args
        and getattr(node.args[0], "value", None) == "--validate-only"
        for node in ast.walk(main)
    )
    assert flag_added, "--validate-only must be a train.py argument"
    # The early return must come before the epoch loop is entered.
    exit_index = source.index('"record_type": "validate_only_complete"')
    loop_index = source.index("for epoch in range(start_epoch, total_epochs + 1):")
    assert exit_index < loop_index

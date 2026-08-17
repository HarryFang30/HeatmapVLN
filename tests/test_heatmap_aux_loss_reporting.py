import ast
from pathlib import Path


def _declared_component_keys(path: str) -> tuple[str, ...]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name)
            and target.id == "_HEATMAP_COMPONENT_KEYS"
            for target in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"_HEATMAP_COMPONENT_KEYS not declared in {path}")


def test_train_and_validation_report_every_heatmap_auxiliary_loss():
    train_component_keys = _declared_component_keys(
        "scripts/training/train_loop.py"
    )
    val_component_keys = _declared_component_keys(
        "scripts/training/validate.py"
    )
    assert train_component_keys == val_component_keys
    assert {
        "view_macro_loss",
        "direction_macro_loss",
        "panoramic_view_loss",
    }.issubset(train_component_keys)

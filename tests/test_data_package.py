"""Regression tests for lightweight src.data package imports."""

import importlib
import sys


def test_src_data_uses_lazy_dataset_imports():
    for module_name in [
        "src.data",
        "src.data.sliding_window_dataset",
        "src.data.trajectory_dataset",
    ]:
        sys.modules.pop(module_name, None)

    data_pkg = importlib.import_module("src.data")

    assert "src.data.sliding_window_dataset" not in sys.modules
    assert "src.data.trajectory_dataset" not in sys.modules

    _ = data_pkg.build_dataset
    assert "src.data.sliding_window_dataset" not in sys.modules

    _ = data_pkg.VLNSlidingWindowDataset
    assert "src.data.sliding_window_dataset" in sys.modules

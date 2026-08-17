from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts/visualization/visualize_system2_future_heatmap_amb3r.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_future_amb3r_vis_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_native_internnav_row_col_is_converted_to_uv() -> None:
    module = _load_module()
    assert module.native_pixel_yx_to_uv((123, 515), (640, 480)) == (515, 123)
    assert module.native_pixel_yx_to_uv((308, 216), (640, 480)) == (216, 308)


def test_native_pixel_contract_rejects_out_of_frame_values() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="outside"):
        module.native_pixel_yx_to_uv((515, 123), (640, 480))


def test_only_nested_metric_prediction_is_accepted() -> None:
    module = _load_module()
    assert module._resolve_metric_flag(1)
    assert not module._resolve_metric_flag(0)


def test_rgb_only_source_has_no_gt_depth_loader() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "depth_front_down" not in source
    assert "waypoint_dist_m" not in source
    assert "label_clip_frames" not in source
    assert '"gt_depth_read": False' in source

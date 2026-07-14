from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from scripts.tools import export_pose_free_lora_wrapper as exporter


def _lora_state() -> dict[str, torch.Tensor]:
    return {
        "qwen2_5_vl.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.arange(
            6, dtype=torch.float32
        ).reshape(2, 3),
        "qwen2_5_vl.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.arange(
            8, dtype=torch.bfloat16
        ).reshape(4, 2),
    }


def _source_payload() -> dict:
    lora = _lora_state()
    head = {"projection.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    return {
        "schema": exporter.SOURCE_SCHEMA,
        "protocol": exporter.SOURCE_PROTOCOL,
        "train_mode": "lora-heatmap-control",
        "step": 1024,
        "expected_lora_tensors": len(lora),
        "lora_state_dict": lora,
        "head_state_dict": head,
        "lora_state_sha256": exporter.tensor_state_sha256(lora),
        "head_state_sha256": exporter.tensor_state_sha256(head),
    }


def _write_source(path: Path, payload: dict | None = None) -> dict:
    source = _source_payload() if payload is None else payload
    torch.save(source, path)
    return source


def test_exports_stage_loader_wrapper_and_json_summary(tmp_path: Path):
    source_path = tmp_path / "task38.pth"
    source = _write_source(source_path)
    output_path = tmp_path / "stage2_base.pth"

    summary = exporter.export_pose_free_lora_wrapper(
        source_path,
        output_path,
        expected_lora_tensors=2,
        expected_source_sha256=exporter.file_sha256(source_path),
        expected_lora_sha256=source["lora_state_sha256"],
    )

    summary_path = tmp_path / "stage2_base.pth.summary.json"
    wrapper = torch.load(output_path, map_location="cpu", weights_only=True)
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert set(wrapper["trainable_state_dict"]) == set(source["lora_state_dict"])
    assert all(tensor.device.type == "cpu" for tensor in wrapper["trainable_state_dict"].values())
    assert wrapper["schema"] == exporter.WRAPPER_SCHEMA
    assert wrapper["protocol"] == exporter.WRAPPER_PROTOCOL
    assert wrapper["source_path"] == str(source_path.resolve())
    assert wrapper["source_file_sha256"] == exporter.file_sha256(source_path)
    assert wrapper["source_schema"] == exporter.SOURCE_SCHEMA
    assert wrapper["source_protocol"] == exporter.SOURCE_PROTOCOL
    assert wrapper["source_train_mode"] == "lora-heatmap-control"
    assert wrapper["source_step"] == 1024
    assert wrapper["source_lora_state_sha256"] == source["lora_state_sha256"]
    assert wrapper["source_head_state_sha256"] == source["head_state_sha256"]
    assert summary == written_summary
    assert summary["roundtrip_verified"] is True
    assert summary["output_file_sha256"] == exporter.file_sha256(output_path)


@pytest.mark.parametrize("missing_key", ["lora_state_dict", "head_state_dict", "lora_state_sha256"])
def test_rejects_missing_required_source_key(tmp_path: Path, missing_key: str):
    source = _source_payload()
    del source[missing_key]
    source_path = tmp_path / "missing.pth"
    _write_source(source_path, source)

    with pytest.raises(exporter.CheckpointExportError, match=missing_key):
        exporter.export_pose_free_lora_wrapper(
            source_path,
            tmp_path / "output.pth",
            expected_lora_tensors=2,
        )


@pytest.mark.parametrize("state_key", ["lora_state_dict", "head_state_dict"])
def test_rejects_non_tensor_state_value(tmp_path: Path, state_key: str):
    source = _source_payload()
    source[state_key][next(iter(source[state_key]))] = "not-a-tensor"
    source_path = tmp_path / "non_tensor.pth"
    _write_source(source_path, source)

    with pytest.raises(exporter.CheckpointExportError, match="is not a tensor"):
        exporter.export_pose_free_lora_wrapper(
            source_path,
            tmp_path / "output.pth",
            expected_lora_tensors=2,
        )


def test_rejects_wrong_expected_lora_count(tmp_path: Path):
    source_path = tmp_path / "task38.pth"
    _write_source(source_path)

    with pytest.raises(exporter.CheckpointExportError, match="LoRA tensor count mismatch"):
        exporter.export_pose_free_lora_wrapper(
            source_path,
            tmp_path / "output.pth",
            expected_lora_tensors=224,
        )


def test_default_overwrite_protection_preserves_existing_outputs(tmp_path: Path):
    source_path = tmp_path / "task38.pth"
    _write_source(source_path)
    output_path = tmp_path / "output.pth"
    exporter.export_pose_free_lora_wrapper(source_path, output_path, expected_lora_tensors=2)
    output_sha = exporter.file_sha256(output_path)
    summary_path = tmp_path / "output.pth.summary.json"
    summary_text = summary_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        exporter.export_pose_free_lora_wrapper(source_path, output_path, expected_lora_tensors=2)

    assert exporter.file_sha256(output_path) == output_sha
    assert summary_path.read_text(encoding="utf-8") == summary_text


def test_roundtrip_tensor_state_hash_matches_source_metadata(tmp_path: Path):
    source_path = tmp_path / "task38.pth"
    source = _write_source(source_path)
    output_path = tmp_path / "output.pth"
    exporter.export_pose_free_lora_wrapper(source_path, output_path, expected_lora_tensors=2)

    wrapper = torch.load(output_path, map_location="cpu", weights_only=True)
    roundtrip_sha = exporter.tensor_state_sha256(wrapper["trainable_state_dict"])

    assert roundtrip_sha == source["lora_state_sha256"]
    assert roundtrip_sha == wrapper["source_lora_state_sha256"]
    assert roundtrip_sha == wrapper["trainable_state_sha256"]


def test_rejects_source_tensor_hash_metadata_mismatch(tmp_path: Path):
    source = _source_payload()
    source["lora_state_sha256"] = "0" * 64
    source_path = tmp_path / "bad_hash.pth"
    _write_source(source_path, source)

    with pytest.raises(exporter.CheckpointExportError, match="strong hash mismatch"):
        exporter.export_pose_free_lora_wrapper(
            source_path,
            tmp_path / "output.pth",
            expected_lora_tensors=2,
        )


def test_rejects_expected_source_file_sha256_pin_mismatch(tmp_path: Path):
    source_path = tmp_path / "task38.pth"
    _write_source(source_path)

    with pytest.raises(exporter.CheckpointExportError, match="Source file SHA-256 pin mismatch"):
        exporter.export_pose_free_lora_wrapper(
            source_path,
            tmp_path / "output.pth",
            expected_lora_tensors=2,
            expected_source_sha256="0" * 64,
        )


def test_rejects_expected_lora_sha256_pin_mismatch(tmp_path: Path):
    source_path = tmp_path / "task38.pth"
    _write_source(source_path)

    with pytest.raises(exporter.CheckpointExportError, match="Source LoRA SHA-256 pin mismatch"):
        exporter.export_pose_free_lora_wrapper(
            source_path,
            tmp_path / "output.pth",
            expected_lora_tensors=2,
            expected_lora_sha256="0" * 64,
        )

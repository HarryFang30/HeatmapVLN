"""
Pydantic v2 configuration schema for HeatmapVLN training configs.

Validates YAML configuration at startup, catching typos and type errors
before they cause cryptic failures deep in the training loop.

Usage::

    from src.config_schema import (
        validate_config,
        load_and_validate_config,
        prepare_config_for_use,
    )

    cfg = validate_config(raw_dict)          # dict -> TrainConfig
    cfg = load_and_validate_config("x.yaml")  # path -> validated dict

Optional top-level ``paths:`` (``dataset_root``, ``val_root``, ``log_out_dir``,
``tensorboard_dir``, ``llm_model_path``, ``internnav_model_path``) is merged
into ``data`` / ``log`` / ``model`` after expanding ``$VAR`` in all strings.
If the environment sets ``INTERNNAV_MODEL_PATH`` or
``HEATMAPVLN_INTERNNAV_MODEL_PATH``, that value overrides both the backbone and
Stage2 System1 source.  Legacy ``INTERNNAV_BACKBONE`` /
``HEATMAPVLN_LLM_MODEL_PATH`` still override only ``model.llm.model_path``.
Use :func:`prepare_config_for_use` when loading YAML without schema validation.
"""

from __future__ import annotations

import copy
import os
import re
import warnings
from pathlib import Path
from typing import Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class _Strict(BaseModel):
    """Base with extra='forbid' to catch misspelled keys."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class _Lenient(BaseModel):
    """Base with extra='allow' for sections where users add ad-hoc keys."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())


# --- Data -------------------------------------------------------------------



SINGLE_VIEW_HISTORY_CONFIG_KEY = "load_single_view_history_frames"
DEPRECATED_HISTORY_CONFIG_KEY = "load_history_frames"


def migrate_single_view_history_config(
    data: Any,
    *,
    section_path: str,
    warn_deprecated: bool = True,
) -> Any:
    """Normalize the ambiguous history-frame key without changing behavior.

    ``load_single_view_history_frames`` controls only the single-view
    ``history_frames`` tensor.  It does not control panoramic
    ``history_panoramas``.  The old key remains readable during migration,
    but conflicting old/new values fail closed.
    """
    if not isinstance(data, dict):
        return data

    normalized = dict(data)
    if DEPRECATED_HISTORY_CONFIG_KEY not in normalized:
        return normalized

    legacy_value = normalized.pop(DEPRECATED_HISTORY_CONFIG_KEY)
    if (
        SINGLE_VIEW_HISTORY_CONFIG_KEY in normalized
        and normalized[SINGLE_VIEW_HISTORY_CONFIG_KEY] != legacy_value
    ):
        raise ValueError(
            f"Conflicting history settings: "
            f"{section_path}.{SINGLE_VIEW_HISTORY_CONFIG_KEY}="
            f"{normalized[SINGLE_VIEW_HISTORY_CONFIG_KEY]!r}, but "
            f"{section_path}.{DEPRECATED_HISTORY_CONFIG_KEY}={legacy_value!r}. "
            "The setting controls only the single-view history_frames tensor; "
            "it does not control history_panoramas."
        )

    normalized.setdefault(SINGLE_VIEW_HISTORY_CONFIG_KEY, legacy_value)
    if warn_deprecated:
        warnings.warn(
            f"{section_path}.{DEPRECATED_HISTORY_CONFIG_KEY} is deprecated; use "
            f"{section_path}.{SINGLE_VIEW_HISTORY_CONFIG_KEY}. This setting controls "
            "only the single-view history_frames tensor; panoramic "
            "history_panoramas are independent.",
            FutureWarning,
            stacklevel=3,
        )
    return normalized


class SlidingWindowConfig(_Lenient):
    min_history: int = 5
    num_history_sample: int = 8
    load_depth: bool = True
    cache_poses: bool = True
    sample_stride: int = 1
    enable_augmentation: bool = False
    clip_level_sampling: bool = True
    samples_per_clip: int = 2
    val_samples_per_clip: int = 2
    defer_heatmap_to_gpu: bool = False
    # Controls only the single-view ``history_frames`` tensor. Panoramic
    # ``history_panoramas`` are loaded independently when available.
    load_single_view_history_frames: bool = True
    # Keep panoramic pose/depth supervision while loading only front RGB.
    single_view_rgb_input: bool = False
    # Optional input-only sidecar.  When set, history_rel_poses comes 100%
    # from causal AMB3R-VO; GT c2w remains available only to build targets.
    amb3r_pose_cache_root: str | None = None
    require_amb3r_pose_cache: bool = False
    amb3r_pose_cache_max_clips: int = 16

    @model_validator(mode="before")
    @classmethod
    def _migrate_history_frame_key(cls, data: Any) -> Any:
        return migrate_single_view_history_config(
            data,
            section_path="data.sliding_window",
        )

    @model_validator(mode="after")
    def _check_single_view_rgb_contract(self):
        if self.single_view_rgb_input and not self.load_single_view_history_frames:
            raise ValueError(
                "single_view_rgb_input requires "
                "load_single_view_history_frames=true"
            )
        if self.require_amb3r_pose_cache and not (
            isinstance(self.amb3r_pose_cache_root, str)
            and self.amb3r_pose_cache_root.strip()
        ):
            raise ValueError(
                "require_amb3r_pose_cache=true requires amb3r_pose_cache_root"
            )
        if self.amb3r_pose_cache_root and not self.require_amb3r_pose_cache:
            raise ValueError(
                "amb3r_pose_cache_root requires require_amb3r_pose_cache=true; "
                "an optional fallback to GT poses is forbidden"
            )
        if self.require_amb3r_pose_cache and not self.single_view_rgb_input:
            raise ValueError(
                "AMB3R pose-cache domain adaptation requires single_view_rgb_input=true"
            )
        if self.amb3r_pose_cache_max_clips < 1:
            raise ValueError("amb3r_pose_cache_max_clips must be >= 1")
        return self


class FutureTrajectoryHeatmapDataConfig(_Strict):
    enabled: bool = False
    heatmap_size: list[int] = [64, 64]

    @field_validator("heatmap_size")
    @classmethod
    def _check_fixed_heatmap_size(cls, value: list[int]) -> list[int]:
        if value != [64, 64]:
            raise ValueError(
                "Future trajectory heatmap_size must be exactly [64,64]"
            )
        return value


class TrajectoryConfig(_Lenient):
    min_history: int = 5
    num_history_sample: int = 8
    load_depth: bool = True
    cache_poses: bool = True
    sample_stride: int = 1
    clip_level_sampling: bool = True
    samples_per_clip: int = 8
    val_samples_per_clip: int = 2
    random_subsequence: bool = False
    min_subsequence_length: int = 30
    subsequence_samples_per_clip: int = 3
    predict_horizon: int = 24
    action_scale: float = 4.0
    enable_augmentation: bool = True
    enable_trajectory_augmentation: bool = True
    load_traj_images: bool = False
    # Controls only the single-view ``history_frames`` tensor. Panoramic
    # ``history_panoramas`` are loaded independently when enabled.
    load_single_view_history_frames: bool = True
    single_view_rgb_input: bool = False
    amb3r_pose_cache_root: str | None = None
    require_amb3r_pose_cache: bool = False
    amb3r_pose_cache_max_clips: int = 16
    traj_image_size: list[int] = [224, 224]
    compute_pixel_goal: bool = False
    compute_pano_view_pixel_goal: bool | None = None
    compute_aligned_native_pixel_goal: bool = False
    pano_max_side_dist_m: float = 6.0
    load_lookdown_for_system2: bool = False
    system2_sft_protocol: str = "direct"
    structured_pano_output: bool = True
    pixel_goal_direction: str = "front"
    load_history_heatmap: bool = True
    require_sft_target: bool = False
    sft_include_turns: bool = True
    sft_include_forward: bool = False
    sft_num_future_steps: int = 4
    system2_sample_step: int = 4
    system2_min_pixel_goal_len: int = 3
    system2_stop_oversample: int = 5
    include_stop_samples_random_subsequence: bool = False
    panoramic_vlm_input: bool = True
    trajectory_target_convention: str = "legacy_pitched_camera"
    future_heatmap: FutureTrajectoryHeatmapDataConfig = (
        FutureTrajectoryHeatmapDataConfig()
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_subinstruction_keys(cls, data: Any) -> Any:
        data = migrate_single_view_history_config(
            data,
            section_path="data.trajectory",
        )
        if isinstance(data, dict):
            removed = {"use_subinstruction", "fgr2r_subinstr_path"} & set(data)
            if removed:
                keys = ", ".join(sorted(removed))
                raise ValueError(
                    f"FGR2R/subinstruction support has been removed; delete trajectory config key(s): {keys}"
                )
        return data

    @field_validator("system2_sft_protocol")
    @classmethod
    def _check_system2_sft_protocol(cls, v: str) -> str:
        allowed = {"direct", "internnav"}
        if v not in allowed:
            raise ValueError(f"system2_sft_protocol must be one of {allowed}, got '{v}'")
        return v

    @field_validator("pixel_goal_direction")
    @classmethod
    def _check_pixel_goal_direction(cls, v: str) -> str:
        allowed = {"front", "right", "back", "left", "front_down"}
        if v not in allowed:
            raise ValueError(f"pixel_goal_direction must be one of {allowed}, got '{v}'")
        return v

    @field_validator("trajectory_target_convention")
    @classmethod
    def _check_trajectory_target_convention(cls, v: str) -> str:
        allowed = {"legacy_pitched_camera", "internnav_habitat"}
        if v not in allowed:
            raise ValueError(f"trajectory_target_convention must be one of {allowed}, got {v!r}")
        return v

    @field_validator("sft_num_future_steps")
    @classmethod
    def _check_sft_num_future_steps(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"sft_num_future_steps must be >= 1, got {v}")
        return v

    @field_validator("system2_sample_step", "system2_min_pixel_goal_len")
    @classmethod
    def _check_positive_system2_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"System2 InternNav alignment parameter must be >= 1, got {v}")
        return v

    @field_validator("system2_stop_oversample")
    @classmethod
    def _check_nonnegative_system2_stop_oversample(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"system2_stop_oversample must be >= 0, got {v}")
        return v

    @model_validator(mode="after")
    def _check_amb3r_future_contract(self):
        if self.single_view_rgb_input and not self.load_single_view_history_frames:
            raise ValueError(
                "single_view_rgb_input requires "
                "load_single_view_history_frames=true"
            )
        if self.require_amb3r_pose_cache and not (
            isinstance(self.amb3r_pose_cache_root, str)
            and self.amb3r_pose_cache_root.strip()
        ):
            raise ValueError(
                "require_amb3r_pose_cache=true requires amb3r_pose_cache_root"
            )
        if self.amb3r_pose_cache_root and not self.require_amb3r_pose_cache:
            raise ValueError(
                "amb3r_pose_cache_root requires "
                "require_amb3r_pose_cache=true; GT fallback is forbidden"
            )
        if self.require_amb3r_pose_cache and not self.single_view_rgb_input:
            raise ValueError(
                "trajectory AMB3R cache requires single_view_rgb_input=true"
            )
        if self.require_amb3r_pose_cache and self.random_subsequence:
            raise ValueError(
                "endpoint-v2 AMB3R cache requires random_subsequence=false"
            )
        if self.amb3r_pose_cache_max_clips < 1:
            raise ValueError("amb3r_pose_cache_max_clips must be >= 1")

        if self.future_heatmap.enabled:
            if self.predict_horizon != 32:
                raise ValueError(
                    "Future trajectory heatmaps require predict_horizon=32"
                )
            if self.trajectory_target_convention != "internnav_habitat":
                raise ValueError(
                    "Future trajectory heatmaps require "
                    "trajectory_target_convention='internnav_habitat'"
                )
            if not self.load_traj_images or not self.require_sft_target:
                raise ValueError(
                    "Future trajectory heatmaps require load_traj_images=true "
                    "and require_sft_target=true"
                )
            if self.enable_trajectory_augmentation:
                raise ValueError(
                    "Future trajectory heatmaps require "
                    "enable_trajectory_augmentation=false so the map and "
                    "System-1 action target remain identical"
                )
        return self


class TrajectoryDaggerConfig(_Strict):
    """Typed input contract for sealed online-correction trajectories."""

    collection_roots: list[str] | str | None = None
    collection_root: str | None = None
    val_collection_roots: list[str] | str | None = None
    val_collection_root: str | None = None
    allow_unsealed_debug: bool = False
    source_types: list[Literal["dagger_normal", "dagger_hard"]] | None = None
    num_history: int = 8
    verify_tar_sha256: bool = False
    require_lookdown: bool = True
    expected_policy_mode: str | None = None
    expected_policy_fingerprint: str | None = None

    @field_validator("num_history")
    @classmethod
    def _check_num_history(cls, value: int) -> int:
        if value < 1:
            raise ValueError("trajectory_dagger.num_history must be >= 1")
        return value

    @field_validator("collection_root", "val_collection_root")
    @classmethod
    def _check_single_roots(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("trajectory_dagger collection roots cannot be blank")
        return value

    @field_validator("source_types")
    @classmethod
    def _check_source_types(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and not value:
            raise ValueError("trajectory_dagger.source_types cannot be empty")
        return value

    @field_validator("collection_roots", "val_collection_roots")
    @classmethod
    def _check_root_lists(
        cls, value: list[str] | str | None
    ) -> list[str] | str | None:
        if isinstance(value, list) and not value:
            raise ValueError("trajectory_dagger collection root lists cannot be empty")
        if isinstance(value, list) and any(not str(root).strip() for root in value):
            raise ValueError("trajectory_dagger collection roots cannot be blank")
        if isinstance(value, str) and not value.strip():
            raise ValueError("trajectory_dagger collection roots cannot be blank")
        return value

    @model_validator(mode="after")
    def _check_provenance_and_roots(self):
        if (self.collection_roots is None) == (self.collection_root is None):
            raise ValueError(
                "set exactly one of trajectory_dagger.collection_roots or collection_root"
            )
        if (
            self.val_collection_roots is not None
            and self.val_collection_root is not None
        ):
            raise ValueError(
                "set at most one of trajectory_dagger.val_collection_roots or val_collection_root"
            )
        if self.expected_policy_fingerprint and not self.expected_policy_mode:
            raise ValueError(
                "expected_policy_fingerprint requires expected_policy_mode"
            )
        if self.expected_policy_mode == "internnav_native":
            fingerprint = self.expected_policy_fingerprint or ""
            if re.fullmatch(r"internnav-native-v1:[0-9a-f]{64}", fingerprint) is None:
                raise ValueError(
                    "internnav_native requires expected_policy_fingerprint="
                    "internnav-native-v1:<64 lowercase hex chars>"
                )
        return self


class ExpertDaggerMixtureConfig(_Strict):
    """Sampling contract for expert/normal-DAgger/hard-DAgger data."""

    profile: str | None = "expert50_normal20_hard30"
    weights: dict[str, float] | None = None
    epoch_size: int | None = None
    seed: int = 42

    @model_validator(mode="after")
    def _check_selection(self):
        if self.profile is None and self.weights is None:
            raise ValueError("mixture requires either profile or weights")
        if self.profile is not None and self.weights is not None:
            raise ValueError("mixture accepts either profile or weights, not both")
        if self.weights is not None:
            expected = {"expert", "dagger_normal", "dagger_hard"}
            if set(self.weights) != expected:
                raise ValueError(
                    f"mixture.weights must contain exactly {sorted(expected)}"
                )
            if any(weight < 0 for weight in self.weights.values()):
                raise ValueError("mixture weights must be non-negative")
            if sum(self.weights.values()) <= 0:
                raise ValueError("mixture weights must have a positive total")
        if self.epoch_size is not None and self.epoch_size < 1:
            raise ValueError("mixture.epoch_size must be >= 1")
        return self


class DaggerSystem2SFTConfig(_Strict):
    """Where the oracle directions come from when relabelling DAgger rows."""

    enabled: bool = False
    oracle_views_jsonl: str | None = None
    val_oracle_views_jsonl: str | None = None
    max_turns: int = 4
    # The DAgger collection is one pool, so train/val are cut by scene with the
    # same hash the EXP-13 readout probe uses.  Keeping the two experiments on
    # the same held-out scenes is what makes their conclusions comparable.
    val_scene_pct: int = 25

    @model_validator(mode="after")
    def _check_source(self):
        if self.enabled and not (self.oracle_views_jsonl or "").strip():
            raise ValueError(
                "dagger_system2_sft.enabled requires oracle_views_jsonl "
                "(EXP-12 d1_per_state.jsonl)"
            )
        if self.max_turns < 1:
            raise ValueError("dagger_system2_sft.max_turns must be >= 1")
        if not 0 < self.val_scene_pct < 100:
            raise ValueError("dagger_system2_sft.val_scene_pct must be in (0, 100)")
        return self


class DataConfig(_Strict):
    root: str | None = None
    train_split: str = "train"
    val_root: str | None = None
    val_split: str = "val"
    image_size: list[int]
    init_hm_size: list[int]
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    dataset_type: str = "sliding_window"
    sliding_window: SlidingWindowConfig | None = None
    trajectory: TrajectoryConfig | None = None
    dagger_system2_sft: DaggerSystem2SFTConfig | None = None
    trajectory_dagger: TrajectoryDaggerConfig | None = None
    mixture: ExpertDaggerMixtureConfig | None = None
    use_worker_tokenized_collator: bool | None = None
    # ``True`` is mandatory for heatmap-control mixture training so that a
    # deterministic sampler remains exactly replayable across mid-epoch resume.
    # ``None`` preserves the legacy policy for unrelated training recipes.
    in_order: bool | None = None
    shm_bypass: Union[bool, str] = "auto"
    shm_bypass_min_gb: float = 8.0

    @field_validator("image_size", "init_hm_size")
    @classmethod
    def _check_size_pair(cls, v: list[int]) -> list[int]:
        if len(v) != 2:
            raise ValueError(f"Expected [W, H], got {v}")
        return v

    @field_validator("dataset_type")
    @classmethod
    def _check_dataset_type(cls, v: str) -> str:
        allowed = {
            "sliding_window",
            "trajectory",
            "trajectory_dagger",
            "expert_dagger_mixture",
        }
        if v not in allowed:
            raise ValueError(f"dataset_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("shm_bypass")
    @classmethod
    def _check_shm_bypass(cls, v: Union[bool, str]) -> Union[bool, str]:
        if isinstance(v, bool):
            return v
        normalized = str(v).strip().lower()
        allowed = {"auto", "1", "0", "true", "false", "yes", "no", "on", "off"}
        if normalized not in allowed:
            raise ValueError(f"shm_bypass must be boolean-like or 'auto', got {v}")
        return normalized

    @field_validator("shm_bypass_min_gb")
    @classmethod
    def _check_shm_bypass_min_gb(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"shm_bypass_min_gb must be >= 0, got {v}")
        return v

    @model_validator(mode="after")
    def _check_dataset_sections(self):
        if self.dataset_type != "trajectory_dagger" and not (
            isinstance(self.root, str) and self.root.strip()
        ):
            raise ValueError(f"data.root is required for {self.dataset_type}")
        if self.dataset_type == "trajectory" and self.trajectory is None:
            raise ValueError("data.trajectory is required for trajectory datasets")
        if self.dataset_type == "trajectory_dagger" and self.trajectory_dagger is None:
            raise ValueError(
                "data.trajectory_dagger is required for trajectory_dagger datasets"
            )
        if self.dataset_type == "expert_dagger_mixture":
            if self.trajectory is None:
                raise ValueError("data.trajectory is required for expert_dagger_mixture")
            if self.trajectory_dagger is None:
                raise ValueError(
                    "data.trajectory_dagger is required for expert_dagger_mixture"
                )
            if self.mixture is None:
                raise ValueError("data.mixture is required for expert_dagger_mixture")
        return self


# --- Model -------------------------------------------------------------------


class LLMConfig(_Lenient):
    model_path: str = "./models/internnav_backbone"
    backbone_type: str = "qwen2_5_vl"
    hidden_dim: int = 3584
    token_dim: int = 896
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    max_video_frames: int = -1
    gradient_checkpointing: bool = False
    enable_packing: bool = False
    max_seq_length: int = 8192
    spatial_merge_size: int = 2
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_num_layers: int = 4
    lora_layer_indices: list[int] | None = None
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    frozen_traj_inference_mode: bool = False
    traj_last_hidden_state_only: bool = False


class HeatmapTrajectoryConfig(_Lenient):
    enable: bool = False
    num_freqs: int = 16
    d_attn: int = 256
    num_heads: int = 4
    num_layers: int = 2
    max_spatial_range: float = 10.0


class HeatmapPoseFreeConfig(_Strict):
    match_dim: int = 64
    visibility_hidden_dim: int = 16
    logit_temperature: float = 10.0
    heatmap_size: tuple[int, int] = (64, 64)
    history_query_source: Literal[
        "text_anchor",
        "history_visual_equal_view_mean_v1",
    ] = "text_anchor"


class HeatmapModelConfig(_Lenient):
    enable: bool = True
    c_vit: int = 1280
    c_llm: int = 3584
    c_fused: int = 256
    vit_layer_indices: list[int] = [7, 15, 23, 31]
    llm_layer_indices: list[int] = [6, 13, 20]
    heatmap_size: list[int] = [64, 64]
    image_size: int = 256
    lambda_vis: float = 1.0
    lambda_coord: float = 0.2
    lambda_kl: float = 0.0
    lambda_peak: float = 1.0
    heatmap_trains_backbone: bool = False
    decoder_mode: str = "legacy"
    restore_vit_spatial_layout: bool = False
    coarse_logit_residual: bool = False
    joint_panorama_inference: bool = False
    input_mode: Literal["panoramic", "internnav_single_view"] = "panoramic"
    feature_source: str = "vit_and_llm"
    architecture_id: str = ""
    output_direction_order: list[str] = ["front", "right", "back", "left"]
    history_pose_convention: str = ""
    conditioner_global_context: bool = True
    pose_free: HeatmapPoseFreeConfig | None = None
    trajectory: HeatmapTrajectoryConfig | None = None

    @field_validator("decoder_mode")
    @classmethod
    def _valid_decoder_mode(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"legacy", "pose_free_matcher"}:
            raise ValueError(f"decoder_mode must be 'legacy' or 'pose_free_matcher', got {value!r}")
        return normalized

    @model_validator(mode="after")
    def _check_single_view_heatmap_contract(self):
        if self.input_mode != "internnav_single_view":
            return self
        expected_architecture = (
            "internnav_single_view_vision_only_four_direction_v2"
        )
        expected_pose = (
            "habitat_c2w_minus_z__forward_left_cos_yaw_sin_yaw__v1"
        )
        if self.feature_source != "vit_only":
            raise ValueError("internnav_single_view requires feature_source=vit_only")
        if (self.c_vit, self.c_llm, self.c_fused) != (1280, 3584, 256):
            raise ValueError(
                "internnav_single_view v2 requires c_vit/c_llm/c_fused="
                "1280/3584/256"
            )
        if self.vit_layer_indices != [7, 15, 23, 31]:
            raise ValueError(
                "internnav_single_view v2 requires ViT layers [7,15,23,31]"
            )
        if self.architecture_id != expected_architecture:
            raise ValueError(
                "internnav_single_view architecture_id must be "
                f"{expected_architecture!r}"
            )
        if tuple(self.output_direction_order) != (
            "front",
            "right",
            "back",
            "left",
        ):
            raise ValueError(
                "internnav_single_view direction order must be "
                "front/right/back/left"
            )
        if self.history_pose_convention != expected_pose:
            raise ValueError(
                "internnav_single_view history_pose_convention must be "
                f"{expected_pose!r}"
            )
        if self.heatmap_trains_backbone:
            raise ValueError("internnav_single_view must freeze the backbone")
        if not self.restore_vit_spatial_layout:
            raise ValueError(
                "internnav_single_view requires restore_vit_spatial_layout=true"
            )
        if self.decoder_mode != "legacy":
            raise ValueError("internnav_single_view requires decoder_mode=legacy")
        if self.trajectory is None or not self.trajectory.enable:
            raise ValueError(
                "internnav_single_view requires heatmap.trajectory.enable=true"
            )
        trajectory_contract = (
            self.trajectory.num_freqs,
            self.trajectory.d_attn,
            self.trajectory.num_heads,
            self.trajectory.num_layers,
            self.trajectory.max_spatial_range,
        )
        if trajectory_contract != (16, 256, 4, 2, 10.0):
            raise ValueError(
                "internnav_single_view v2 requires trajectory "
                "num_freqs/d_attn/num_heads/num_layers/max_spatial_range="
                "16/256/4/2/10.0 for the audited weight migration"
            )
        return self


class HeatmapControlConfig(_Strict):
    """Frozen heatmap producer and trainable System1 control contract."""

    enabled: bool = False
    schema_version: Literal["heatmap-control-v1"] = "heatmap-control-v1"
    token_dim: int = 128
    control_dim: int = 128
    num_heads: int = 4
    coarse_size: int = 8
    temporal_layers: int = 1
    temporal_heads: int = 4
    temporal_ffn_dim: int = 512
    dropout: float = 0.0
    age_normalizer_steps: float = 32.0
    heatmap_checkpoint_path: str = ""
    heatmap_checkpoint_sha256: str = ""

    @model_validator(mode="after")
    def _check_architecture_contract(self):
        positive_dimensions = {
            "token_dim": self.token_dim,
            "control_dim": self.control_dim,
            "num_heads": self.num_heads,
            "coarse_size": self.coarse_size,
            "temporal_layers": self.temporal_layers,
            "temporal_heads": self.temporal_heads,
            "temporal_ffn_dim": self.temporal_ffn_dim,
        }
        invalid = [name for name, value in positive_dimensions.items() if value < 1]
        if invalid:
            raise ValueError(
                "heatmap_control dimensions must be positive: " + ", ".join(invalid)
            )
        if self.token_dim != self.control_dim:
            raise ValueError("heatmap_control requires token_dim == control_dim")
        if self.control_dim % self.num_heads != 0:
            raise ValueError("control_dim must be divisible by num_heads")
        if self.token_dim % self.temporal_heads != 0:
            raise ValueError("token_dim must be divisible by temporal_heads")
        architecture = (
            self.token_dim,
            self.control_dim,
            self.num_heads,
            self.coarse_size,
            self.temporal_layers,
            self.temporal_heads,
        )
        if architecture != (128, 128, 4, 8, 1, 4):
            raise ValueError(
                "heatmap-control-v1 requires token/control/heads/coarse/"
                "temporal_layers/temporal_heads=128/128/4/8/1/4"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("heatmap_control.dropout must be in [0, 1)")
        if self.age_normalizer_steps <= 0:
            raise ValueError("age_normalizer_steps must be > 0")
        if self.enabled:
            if not self.heatmap_checkpoint_path.strip():
                raise ValueError(
                    "enabled heatmap_control requires heatmap_checkpoint_path"
                )
            if re.fullmatch(
                r"[0-9a-f]{64}", self.heatmap_checkpoint_sha256
            ) is None:
                raise ValueError(
                    "enabled heatmap_control requires a 64-character lowercase "
                    "SHA-256 heatmap_checkpoint_sha256"
                )
        return self


class NextDiTConfig(_Lenient):
    enabled: bool = False
    vlm_hidden_dim: int = 3584
    latent_emb_size: int = 768
    n_query: int = 4
    dit_dim: int = 384
    dit_layers: int = 12
    dit_heads: int = 6
    dit_kv_heads: int = 6
    dit_ffn_dim_multiplier: float = 2 / 3
    predict_steps: int = 32
    action_dim: int = 3
    num_inference_steps: int = 10
    guidance_scale: float = 1.0
    num_sample_trajs: int = 32
    dav2_ckpt_path: str = ""
    enable_gradient_checkpointing: bool = True
    internnav_system1_path: str = ""
    internnav_model_path: str = ""
    pretrained_system1_path: str | None = None
    warmup_steps: int = 0
    heatmap_control: HeatmapControlConfig | None = None


class ActionHeadConfig(_Lenient):
    enable: bool = True
    nextdit: NextDiTConfig | None = None


class PastPlanActionConfig(_Strict):
    """Directed latent-chain v1; decoded maps are never injected into NextDiT."""

    enabled: bool = False
    schema_version: Literal["past-plan-action-v1"] = "past-plan-action-v1"
    plan_dim: int = 768
    memory_dim: int = 256
    bridge_heads: int = 8
    # Hard trust region on the bridge residual: per-token ||delta|| is capped
    # at this fraction of the native Plan token norm in training AND
    # deployment.  ``None`` keeps the legacy unconstrained bridge.
    max_delta_ratio: float | None = None

    @model_validator(mode="after")
    def _check_dimensions(self):
        if (self.plan_dim, self.memory_dim, self.bridge_heads) != (768, 256, 8):
            raise ValueError(
                "past-plan-action-v1 requires plan_dim/memory_dim/bridge_heads="
                "768/256/8"
            )
        if self.max_delta_ratio is not None and not (
            0.0 < self.max_delta_ratio <= 1.0
        ):
            raise ValueError(
                "past_plan_action.max_delta_ratio must be in (0, 1] or null"
            )
        return self


class System2MemoryConfig(_Strict):
    """History memory injected as extra System2 prompt tokens.

    The alternative injection point to the Plan bridge: EXP-05/EXP-07 showed
    the bridge can only modulate execution, while EXP-12 located the failures
    in System2's own decision.  ``mode`` selects the arm; ``constant`` is the
    control that holds token count and parameter budget fixed while removing
    all dependence on the memory.
    """

    enabled: bool = False
    schema_version: Literal["system2-memory-v1"] = "system2-memory-v1"
    mode: Literal["memory", "constant", "off"] = "memory"
    num_tokens: int = 8
    memory_dim: int = 256

    @model_validator(mode="after")
    def _check_shape(self):
        if self.enabled and self.mode == "off":
            raise ValueError(
                "system2_memory.enabled=true with mode='off' is contradictory; "
                "disable the block instead"
            )
        if self.num_tokens < 1:
            raise ValueError("system2_memory.num_tokens must be >= 1")
        if self.memory_dim != 256:
            raise ValueError(
                "system2_memory.memory_dim must match the Past Head bottleneck (256)"
            )
        return self


class ModelConfig(_Lenient):
    type: str = "vln_pipeline"
    device: str = "cuda"
    llm: LLMConfig | None = None
    heatmap: HeatmapModelConfig | None = None
    action_head: ActionHeadConfig | None = None
    past_plan_action: PastPlanActionConfig | None = None
    system2_memory: System2MemoryConfig | None = None


# --- Optim -------------------------------------------------------------------


class OptimConfig(_Lenient):
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    heatmap_lr: float = 2e-4
    heatmap_tokenizer_lr: float = 1e-4
    heatmap_control_lr: float = 5e-5
    heatmap_gate_lr: float = 1e-4
    system2_memory_lr: float = 1e-4
    heatmap_vit_lr: float | None = None
    heatmap_fine_lr: float | None = None
    heatmap_llm_lr: float | None = None
    heatmap_coarse_lr: float | None = None
    heatmap_proj_traj_lr: float | None = None
    heatmap_new_lr: float | None = None
    vis_head_lr: float | None = None
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: str = "bf16"
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    min_lr: float = 1e-6
    batch_size: int = 4
    grad_accum_steps: int = 1

    @field_validator("batch_size")
    @classmethod
    def _positive_batch(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"batch_size must be >= 1, got {v}")
        return v


# --- Loss --------------------------------------------------------------------


class TemperatureScheduleConfig(_Lenient):
    enabled: bool = False
    mode: str = "cosine"
    start: float = 1.0
    end: float = 1.0


class HeatmapVLNLossConfig(_Lenient):
    lambda_vis: float = 1.0
    lambda_coord: float = 0.2
    lambda_kl: float = 0.0
    lambda_peak: float = 1.0
    lambda_neg: float = 0.0
    temperature: float = 1.0
    vis_pos_weight: float = 1.0
    lambda_view_macro: float = 0.0
    lambda_direction_macro: float = 0.0
    lambda_panoramic_view: float = 0.0
    panoramic_detach_visibility: bool = False
    coord_smooth_l1_beta: float = 0.1
    allow_probability_fallback: bool = True
    temperature_schedule: TemperatureScheduleConfig | None = None


class LossConfig(_Lenient):
    heatmap_loss_type: str = "heatmap_vln"
    heatmap_vln: HeatmapVLNLossConfig | None = None
    heatmap_weight: float = 1.0
    trajectory_weight: float = 0.0
    lm_weight: float = 1.0
    history_weight: float = 0.3
    future_weight: float = 0.3
    preserve_weight: float = 0.5
    delta_z_weight: float = 0.01
    # Report the PPA delta penalty as the scale-free per-token ratio
    # ||delta||^2/||plan_z0||^2 instead of the absolute per-element mean.
    delta_z_relative: bool = False
    # Advantage-weighted PPA action loss: scale each sample's velocity MSE by
    # clamp(native_mse/reference, max=max_weight) under shared noise, so the
    # bridge is trained only where frozen native System1 is actually wrong.
    action_advantage_enabled: bool = False
    action_advantage_reference_mse: float = 0.125
    action_advantage_max_weight: float = 4.0
    future_heatmap: dict[str, Any] = {}

    @model_validator(mode="after")
    def _check_action_advantage(self):
        if self.action_advantage_reference_mse <= 0:
            raise ValueError("action_advantage_reference_mse must be positive")
        if self.action_advantage_max_weight < 1.0:
            raise ValueError("action_advantage_max_weight must be >= 1")
        return self


# --- Training ----------------------------------------------------------------


class HeatmapWarmstartContractConfig(_Strict):
    """Fail-closed warm-start contract for HeatmapVLN training."""

    policy: Literal[
        "spatial_reset_v1",
        "full_head_v1",
        "internnav_single_view_head_v2",
    ] = "spatial_reset_v1"
    expected_lora_tensors: int = 224
    expected_vit_dpt_tensors: int = 12
    expected_llm_dpt_tensors: int = 10
    expected_coarse_tensors: int = 37
    expected_fine_tensors: int = 6
    require_metadata: bool = True

    @field_validator(
        "expected_lora_tensors",
        "expected_vit_dpt_tensors",
        "expected_llm_dpt_tensors",
        "expected_coarse_tensors",
        "expected_fine_tensors",
    )
    @classmethod
    def _positive_expected_tensor_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("warm-start tensor counts must be >= 0")
        return value

    @model_validator(mode="after")
    def _check_policy_specific_counts(self):
        counts = {
            "expected_lora_tensors": self.expected_lora_tensors,
            "expected_vit_dpt_tensors": self.expected_vit_dpt_tensors,
            "expected_llm_dpt_tensors": self.expected_llm_dpt_tensors,
            "expected_coarse_tensors": self.expected_coarse_tensors,
            "expected_fine_tensors": self.expected_fine_tensors,
        }
        if self.policy == "internnav_single_view_head_v2":
            expected = {
                "expected_lora_tensors": 0,
                "expected_vit_dpt_tensors": 12,
                "expected_llm_dpt_tensors": 0,
                "expected_coarse_tensors": 35,
                "expected_fine_tensors": 6,
            }
            if counts != expected or not self.require_metadata:
                raise ValueError(
                    "internnav_single_view_head_v2 requires exact counts "
                    f"{expected} and require_metadata=true"
                )
        elif any(value < 1 for value in counts.values()):
            raise ValueError(
                f"{self.policy} warm-start tensor counts must all be >= 1"
            )
        return self


class TrainingStageConfig(_Lenient):
    name: str
    epochs: int
    hm_size: list[int] = [64, 64]
    heatmap_loss_type: str = "heatmap_vln"
    train_heatmap: bool | None = None
    train_history: bool | None = None
    train_future: bool | None = None
    train_lm: bool | None = None
    train_system2_sft: bool | None = None
    train_action: bool = True
    strict_trainable_modules: bool = False
    bridge_only: bool = False
    past_plan_action_bridge_only: bool = False
    requires_base_checkpoint: bool = False
    require_complete_internnav_system1: bool | None = None
    base_checkpoint_lora_only: bool = False
    merge_frozen_lora: bool = False
    heatmap_warmstart_contract: HeatmapWarmstartContractConfig | None = None
    retain_raw_panoramic_views: bool = True
    compute_pano_text_anchor_positions: bool = True
    retain_history_rel_poses: bool = True
    sft_include_turns: bool | None = None
    sft_include_forward: bool | None = None
    system2_sft_protocol: str | None = None
    lm_weight: float | None = None
    trainable_modules: list[str] = []
    frozen_modules: list[str] = []
    heatmap_trainable_parameter_prefixes: list[str] = []
    heatmap_pose_adaptation_init: bool = False
    # Load exactly the 79-tensor Past Head from --load-weights and freeze it.
    # Unlike heatmap_pose_adaptation_init this makes no claim about *training*
    # the head, so it does not drag in the AMB3R adaptation whitelist.
    load_frozen_past_head: bool = False
    required_history_pose_provider: str | None = None
    past_plan_action_stage: str | None = None
    # Bridge-only refinement normally warm-starts the trained Stage-2 bridge.
    # Setting this retrains the bridge from its exact-zero fresh state while
    # still loading the frozen Heatmap and Future heads from the base.
    past_plan_action_reset_bridge: bool = False
    trajectory_sequence_mode: str = "all"

    @field_validator("epochs")
    @classmethod
    def _positive_epochs(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"epochs must be >= 1, got {v}")
        return v

    @field_validator("system2_sft_protocol")
    @classmethod
    def _check_stage_system2_protocol(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"direct", "internnav"}
        if v not in allowed:
            raise ValueError(f"system2_sft_protocol must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    def _check_heatmap_pose_adaptation_scope(self):
        prefixes = self.heatmap_trainable_parameter_prefixes
        if self.past_plan_action_stage is not None:
            if prefixes:
                raise ValueError(
                    "Past->Plan->Action uses its audited shared-map scope and "
                    "must not reuse the four-prefix pose-adaptation whitelist"
                )
            if not self.heatmap_pose_adaptation_init:
                raise ValueError(
                    "Past->Plan->Action requires exact initialization from the "
                    "complete 79-parameter single-view Heatmap Head"
                )
            if self.required_history_pose_provider != "amb3r_vo_cache":
                raise ValueError(
                    "Past->Plan->Action requires "
                    "required_history_pose_provider='amb3r_vo_cache'"
                )
            if (
                self.past_plan_action_reset_bridge
                and not self.past_plan_action_bridge_only
            ):
                raise ValueError(
                    "past_plan_action_reset_bridge is valid only for "
                    "bridge-only action refinement"
                )
            return self
        if self.past_plan_action_reset_bridge:
            raise ValueError(
                "past_plan_action_reset_bridge requires a Past->Plan->Action stage"
            )
        if self.load_frozen_past_head and self.heatmap_pose_adaptation_init:
            raise ValueError(
                "load_frozen_past_head and heatmap_pose_adaptation_init are "
                "mutually exclusive"
            )
        if not prefixes and not self.heatmap_pose_adaptation_init:
            if self.required_history_pose_provider is not None:
                raise ValueError(
                    "required_history_pose_provider is only valid for pose adaptation"
                )
            return self
        expected = {
            "heatmap_vln.coarse.proj_traj.",
            "heatmap_vln.coarse.self_attn.",
            "heatmap_vln.coarse.vis_head.",
            "heatmap_vln.coarse.heatmap_head.",
        }
        if len(prefixes) != 4 or set(prefixes) != expected:
            raise ValueError(
                "AMB3R pose adaptation requires exactly the four audited "
                "heatmap_trainable_parameter_prefixes"
            )
        if self.trainable_modules != ["heatmap_vln"]:
            raise ValueError(
                "AMB3R pose adaptation requires trainable_modules=['heatmap_vln']"
            )
        if not self.strict_trainable_modules:
            raise ValueError(
                "AMB3R pose adaptation requires strict_trainable_modules=true"
            )
        if not self.heatmap_pose_adaptation_init:
            raise ValueError(
                "heatmap_trainable_parameter_prefixes requires "
                "heatmap_pose_adaptation_init=true"
            )
        if self.required_history_pose_provider != "amb3r_vo_cache":
            raise ValueError(
                "AMB3R pose adaptation requires "
                "required_history_pose_provider='amb3r_vo_cache'"
            )
        return self


class TrainingConfig(_Strict):
    stages: list[TrainingStageConfig]


# --- GPU ---------------------------------------------------------------------


class MultiGPUConfig(_Lenient):
    enabled: bool = False
    find_unused_parameters: bool = False


class GPUConfig(_Lenient):
    devices: list[int] = [0]
    backend: str = "nccl"
    multi_gpu: MultiGPUConfig | None = None


# --- Log / Notify ------------------------------------------------------------


class NotifyConfig(_Lenient):
    enabled: bool = False
    platform: str = "feishu"
    webhook_url: str = ""


class LogConfig(_Lenient):
    out_dir: str
    save_every_epochs: int = 1
    vis_every_steps: int = 500
    val_vis_batches: int = 2
    max_ckpts: int = 3
    use_tensorboard: bool = False
    tensorboard_dir: str | None = None
    enable_timing: bool = False
    log_interval: int = 10
    show_gpu_memory: bool = False
    tqdm_ncols: int = 120
    log_level: str = "INFO"
    diag_interval: int = 100
    mid_epoch_save_every: int = 500
    epoch_boundary_cooldown_s: float | None = None
    notify: NotifyConfig | None = None


# --- Validation --------------------------------------------------------------


class ValidationConfig(_Lenient):
    enabled: bool = True
    eval_every_epochs: int = 1
    best_selection_enabled: bool = True
    save_best_metric: str = "val_total_loss"
    save_best_mode: Literal["min", "max"] = "min"
    evaluate_before_training: bool = False
    baseline_as_best_threshold: bool = False
    baseline_overall_metric: str = "val_heatmap_joint_pck8"
    baseline_overall_tolerance: float = 0.02
    baseline_back_metric: str = "val_heatmap_back_pck8"
    baseline_back_tolerance: float = 0.03
    baseline_direction_metrics: dict[str, str] | None = None
    baseline_direction_tolerance: float = 0.03
    save_best_loss_tiebreak_metric: str = "val_loss"
    patience: int = 5
    val_inference_batches: int = 10
    # Per-rank number of PPA validation batches that additionally run the real
    # sampler (bridged vs native under shared noise) and score the deployment
    # post-processing.  0 disables sampled-rollout validation.
    val_rollout_batches: int = 0

    @field_validator("val_rollout_batches")
    @classmethod
    def _non_negative_rollout_batches(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"val_rollout_batches must be >= 0, got {v}")
        return v


# --- Top-level ---------------------------------------------------------------


class TrainConfig(_Lenient):
    """Top-level training configuration schema.

    Uses ``extra='allow'`` at the top level so that ``seed`` and other
    ad-hoc root keys don't cause validation failures, while all nested
    sub-configs use strict or lenient validation as appropriate.
    """

    seed: int = 42
    data: DataConfig
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    loss: LossConfig = LossConfig()
    training: TrainingConfig
    gpu: GPUConfig = GPUConfig()
    log: LogConfig
    validation: ValidationConfig = ValidationConfig()

    @model_validator(mode="after")
    def _check_native_single_view_training_contract(self):
        heatmap = self.model.heatmap
        if heatmap is None or heatmap.input_mode != "internnav_single_view":
            return self
        action_head = self.model.action_head
        nextdit = action_head.nextdit if action_head is not None else None
        control = nextdit.heatmap_control if nextdit is not None else None
        past_plan_action = self.model.past_plan_action
        system2_memory = self.model.system2_memory
        llm = self.model.llm
        if control is not None and control.enabled:
            if not heatmap.enable:
                raise ValueError("heatmap_control requires model.heatmap.enable=true")
            if llm is None:
                raise ValueError("heatmap_control requires the original InternNav LLM")
            if llm.use_lora:
                raise ValueError("heatmap_control forbids System2 LoRA")
            if llm.gradient_checkpointing:
                raise ValueError(
                    "heatmap_control freezes System2 and forbids its gradient checkpointing"
                )
            if action_head is None or not action_head.enable:
                raise ValueError("heatmap_control requires model.action_head.enable=true")
            if nextdit is None or not nextdit.enabled:
                raise ValueError("heatmap_control requires the original NextDiT System1")

            llm_path = os.path.normpath(llm.model_path.strip())
            system1_path = os.path.normpath(nextdit.internnav_model_path.strip())
            if not llm.model_path.strip() or not nextdit.internnav_model_path.strip():
                raise ValueError(
                    "heatmap_control requires one non-empty unified InternNav model path"
                )
            if llm_path != system1_path:
                raise ValueError(
                    "heatmap_control requires System2 and System1 to use the same "
                    "original InternNav model path"
                )
            if nextdit.internnav_system1_path.strip():
                raise ValueError("heatmap_control forbids internnav_system1_path overrides")
            if nextdit.pretrained_system1_path not in (None, ""):
                raise ValueError("heatmap_control forbids pretrained_system1_path overrides")
            if nextdit.dav2_ckpt_path.strip():
                raise ValueError("heatmap_control forbids external DAV2/System1 checkpoints")
            if nextdit.warmup_steps != 0:
                raise ValueError("heatmap_control requires NextDiT warmup_steps=0")

            pano_adapter = getattr(nextdit, "pano_latent_adapter", None)
            if isinstance(pano_adapter, dict):
                pano_adapter_enabled = bool(pano_adapter.get("enabled", False))
            else:
                pano_adapter_enabled = bool(
                    getattr(pano_adapter, "enabled", False)
                )
            if pano_adapter_enabled or bool(
                getattr(nextdit, "pano_latent_adapter_enabled", False)
            ):
                raise ValueError("heatmap_control forbids the panoramic latent adapter")

            if self.data.dataset_type not in {
                "trajectory_dagger",
                "expert_dagger_mixture",
            }:
                raise ValueError(
                    "heatmap_control requires trajectory_dagger or expert_dagger_mixture data"
                )
            if self.data.dataset_type == "expert_dagger_mixture":
                trajectory_data = self.data.trajectory
                if trajectory_data is None:  # guarded by DataConfig; keeps this fail-closed
                    raise ValueError(
                        "expert_dagger_mixture requires trajectory data settings"
                    )
                if trajectory_data.trajectory_target_convention != "internnav_habitat":
                    raise ValueError(
                        "heatmap_control expert data requires "
                        "trajectory_target_convention=internnav_habitat"
                    )
                if not trajectory_data.load_single_view_history_frames:
                    raise ValueError(
                        "heatmap_control expert data requires single-view history RGB"
                    )
                if not trajectory_data.load_traj_images:
                    raise ValueError(
                        "heatmap_control expert data requires load_traj_images=true"
                    )
                if trajectory_data.panoramic_vlm_input:
                    raise ValueError(
                        "heatmap_control requires the original front/lookdown "
                        "InternNav System2 path (panoramic_vlm_input=false)"
                    )
                if trajectory_data.pixel_goal_direction != "front_down":
                    raise ValueError(
                        "heatmap_control requires pixel_goal_direction=front_down"
                    )
                if trajectory_data.predict_horizon != nextdit.predict_steps:
                    raise ValueError(
                        "expert trajectory predict_horizon must equal NextDiT predict_steps"
                    )
                mixture = self.data.mixture
                if mixture is None or mixture.epoch_size is None:
                    raise ValueError(
                        "heatmap_control expert_dagger_mixture requires an explicit "
                        "mixture.epoch_size"
                    )
                if self.data.in_order is not True:
                    raise ValueError(
                        "heatmap_control expert_dagger_mixture requires "
                        "data.in_order=true for exact mid-epoch resume"
                    )
                configured_world_size = max(1, len(self.gpu.devices))
                full_accumulation_batch = (
                    configured_world_size
                    * self.optim.batch_size
                    * self.optim.grad_accum_steps
                )
                if mixture.epoch_size % full_accumulation_batch != 0:
                    raise ValueError(
                        "heatmap_control mixture.epoch_size must contain only "
                        "full DDP gradient-accumulation windows; expected a "
                        f"multiple of {full_accumulation_batch}"
                    )
                if (
                    self.log.mid_epoch_save_every > 0
                    and self.log.mid_epoch_save_every
                    % self.optim.grad_accum_steps
                    != 0
                ):
                    raise ValueError(
                        "heatmap_control log.mid_epoch_save_every must align "
                        "with optim.grad_accum_steps"
                    )
            for stage in self.training.stages:
                if stage.trainable_modules != [
                    "heatmap_tokenizer",
                    "heatmap_control",
                ]:
                    raise ValueError(
                        "heatmap_control must train exactly "
                        "['heatmap_tokenizer', 'heatmap_control']"
                    )
                if not stage.strict_trainable_modules:
                    raise ValueError(
                        "heatmap_control requires strict_trainable_modules=true"
                    )
                if not stage.train_action:
                    raise ValueError("heatmap_control requires train_action=true")
                if stage.train_heatmap is not False:
                    raise ValueError("heatmap_control requires train_heatmap=false")
                if stage.train_history is not False or stage.train_future is not False:
                    raise ValueError(
                        "heatmap_control requires train_history=false and train_future=false"
                    )
                if stage.train_lm is not False or bool(stage.train_system2_sft):
                    raise ValueError(
                        "heatmap_control requires train_lm=false and no System2 SFT"
                    )
            if self.loss.trajectory_weight <= 0:
                raise ValueError("heatmap_control requires trajectory_weight > 0")
            if self.loss.heatmap_weight != 0 or self.loss.lm_weight != 0:
                raise ValueError(
                    "heatmap_control requires heatmap_weight=0 and lm_weight=0"
                )
            return self

        if past_plan_action is not None and past_plan_action.enabled:
            if not heatmap.enable:
                raise ValueError("Past->Plan->Action requires Heatmap Head")
            if llm is None or llm.use_lora or llm.gradient_checkpointing:
                raise ValueError(
                    "Past->Plan->Action requires a completely frozen native Qwen"
                )
            if action_head is None or not action_head.enable or nextdit is None or not nextdit.enabled:
                raise ValueError(
                    "Past->Plan->Action requires the released NextDiT System1"
                )
            if control is not None and control.enabled:
                raise ValueError(
                    "Past->Plan->Action forbids legacy heatmap control"
                )
            if nextdit.warmup_steps != 0:
                raise ValueError(
                    "Past->Plan->Action requires NextDiT warmup_steps=0 so the "
                    "frozen native cond_projector cannot be re-enabled"
                )
            llm_path = os.path.normpath(llm.model_path.strip())
            system1_path = os.path.normpath(nextdit.internnav_model_path.strip())
            if not llm.model_path.strip() or not nextdit.internnav_model_path.strip():
                raise ValueError(
                    "Past->Plan->Action requires one complete native InternNav model path"
                )
            if llm_path != system1_path:
                raise ValueError(
                    "Past->Plan->Action requires Qwen and NextDiT from the same "
                    "native InternNav model"
                )
            if nextdit.internnav_system1_path.strip():
                raise ValueError(
                    "Past->Plan->Action forbids internnav_system1_path overrides"
                )
            if nextdit.pretrained_system1_path not in (None, ""):
                raise ValueError(
                    "Past->Plan->Action forbids pretrained_system1_path overrides"
                )
            if nextdit.dav2_ckpt_path.strip():
                raise ValueError(
                    "Past->Plan->Action forbids external DAV2/System1 checkpoints"
                )
            pano_adapter = getattr(nextdit, "pano_latent_adapter", None)
            if isinstance(pano_adapter, dict):
                pano_adapter_enabled = bool(pano_adapter.get("enabled", False))
            else:
                pano_adapter_enabled = bool(
                    getattr(pano_adapter, "enabled", False)
                )
            if pano_adapter_enabled or bool(
                getattr(nextdit, "pano_latent_adapter_enabled", False)
            ):
                raise ValueError(
                    "Past->Plan->Action forbids the panoramic latent adapter"
                )
            if (nextdit.n_query, nextdit.latent_emb_size, nextdit.predict_steps) != (
                4,
                768,
                32,
            ):
                raise ValueError(
                    "Past->Plan->Action v1 requires n_query/latent/predict_steps="
                    "4/768/32"
                )
            if self.data.dataset_type != "trajectory":
                raise ValueError(
                    "Past->Plan->Action v1 trains on the expert trajectory dataset; "
                    "DAgger rows need a separate AMB3R cache before they may be enabled"
                )
            trajectory_data = self.data.trajectory
            if trajectory_data is None:
                raise ValueError("Past->Plan->Action requires data.trajectory")
            if trajectory_data.trajectory_target_convention != "internnav_habitat":
                raise ValueError(
                    "Past->Plan->Action requires internnav_habitat trajectory targets"
                )
            if trajectory_data.predict_horizon != 32:
                raise ValueError("Past->Plan->Action requires a 32-step expert target")
            if trajectory_data.enable_trajectory_augmentation:
                raise ValueError(
                    "Past->Plan->Action requires enable_trajectory_augmentation=false"
                )
            if not trajectory_data.require_amb3r_pose_cache:
                raise ValueError(
                    "Past->Plan->Action requires endpoint-v2 AMB3R history poses; "
                    "GT fallback is forbidden"
                )
            if not trajectory_data.single_view_rgb_input:
                raise ValueError(
                    "Past->Plan->Action requires single_view_rgb_input=true"
                )
            if not trajectory_data.future_heatmap.enabled:
                raise ValueError(
                    "Past->Plan->Action requires the no-depth Future target renderer"
                )
            if not trajectory_data.load_single_view_history_frames:
                raise ValueError("Past->Plan->Action requires front history RGB")
            if not trajectory_data.load_traj_images:
                raise ValueError("Past->Plan->Action requires native trajectory images")
            if trajectory_data.panoramic_vlm_input:
                raise ValueError(
                    "Past->Plan->Action requires the native front/lookdown System2 path"
                )
            for stage in self.training.stages:
                if stage.past_plan_action_stage not in {
                    "stage1_map_pretrain",
                    "stage2_joint",
                }:
                    raise ValueError(
                        "Past->Plan->Action stage must be stage1_map_pretrain or "
                        "stage2_joint"
                    )
                if not stage.train_future:
                    raise ValueError(
                        "Past->Plan->Action stages must supervise Future maps"
                    )
                if not stage.train_history:
                    raise ValueError(
                        "Past->Plan->Action stages must retain History supervision"
                    )
                if stage.train_lm or bool(stage.train_system2_sft):
                    raise ValueError(
                        "Past->Plan->Action keeps System2 frozen and forbids LM loss"
                    )
                if stage.trajectory_sequence_mode != "first_only":
                    raise ValueError(
                        "Past->Plan->Action requires trajectory_sequence_mode=first_only"
                    )
                if stage.require_complete_internnav_system1 is not True:
                    raise ValueError(
                        "Past->Plan->Action requires complete native InternNav System1"
                    )
                if stage.trainable_modules != ["past_plan_action", "heatmap_vln"]:
                    raise ValueError(
                        "Past->Plan->Action must train exactly "
                        "['past_plan_action','heatmap_vln']"
                    )
                if (
                    stage.past_plan_action_bridge_only
                    and stage.past_plan_action_stage != "stage2_joint"
                ):
                    raise ValueError(
                        "past_plan_action_bridge_only is valid only for "
                        "stage2_joint action refinement"
                    )
                if not stage.strict_trainable_modules:
                    raise ValueError(
                        "Past->Plan->Action requires strict_trainable_modules=true"
                    )
                if not stage.heatmap_pose_adaptation_init:
                    raise ValueError(
                        "Past->Plan->Action requires exact 79-parameter Heatmap Head "
                        "initialization"
                    )
                if stage.required_history_pose_provider != "amb3r_vo_cache":
                    raise ValueError(
                        "Past->Plan->Action requires AMB3R pose provider fail-closed"
                    )
                if stage.past_plan_action_stage == "stage1_map_pretrain":
                    if stage.train_action:
                        raise ValueError(
                            "Past->Plan->Action Stage 1 must keep Action loss disabled"
                        )
                elif not stage.train_action:
                    raise ValueError(
                        "Past->Plan->Action Stage 2 requires native Action supervision"
                    )
            if self.loss.history_weight <= 0 or self.loss.future_weight <= 0:
                raise ValueError(
                    "Past->Plan->Action requires positive History and Future weights"
                )
            if self.loss.lm_weight != 0:
                raise ValueError("Past->Plan->Action requires lm_weight=0")
            if any(
                stage.past_plan_action_stage == "stage2_joint"
                for stage in self.training.stages
            ):
                if self.loss.trajectory_weight <= 0:
                    raise ValueError(
                        "Past->Plan->Action Stage 2 requires trajectory_weight > 0"
                    )
                if self.loss.preserve_weight <= 0:
                    raise ValueError(
                        "Past->Plan->Action Stage 2 requires preserve_weight > 0"
                    )
                if self.loss.delta_z_weight <= 0:
                    raise ValueError(
                        "Past->Plan->Action Stage 2 requires delta_z_weight > 0"
                    )
            validation = self.validation
            if (
                validation is not None
                and str(validation.save_best_metric).startswith("val_rollout")
                and int(getattr(validation, "val_rollout_batches", 0)) <= 0
            ):
                raise ValueError(
                    "save_best_metric=val_rollout_* requires "
                    "validation.val_rollout_batches > 0"
                )
            return self

        if system2_memory is not None and system2_memory.enabled:
            if not heatmap.enable:
                raise ValueError(
                    "System2 memory tokens require the Past Head that produces them"
                )
            if past_plan_action is not None and past_plan_action.enabled:
                raise ValueError(
                    "the Plan bridge and System2 memory tokens are two injection "
                    "points for the same memory; enable exactly one"
                )
            if control is not None and control.enabled:
                raise ValueError(
                    "System2 memory tokens forbid legacy heatmap control"
                )
            if llm is None or not llm.use_lora:
                raise ValueError(
                    "the System2 memory arm exists to train System2: "
                    "model.llm.use_lora=true"
                )
            if action_head is not None and action_head.enable:
                raise ValueError(
                    "the System2 memory arm supervises System2 text only: "
                    "model.action_head.enable=false"
                )
            if self.data.dataset_type != "trajectory_dagger":
                raise ValueError(
                    "System2 memory training reads sealed DAgger recovery states "
                    "(data.dataset_type=trajectory_dagger)"
                )
            dagger_sft = self.data.dagger_system2_sft
            if dagger_sft is None or not dagger_sft.enabled:
                raise ValueError(
                    "System2 memory training requires "
                    "data.dagger_system2_sft.enabled=true so its targets are "
                    "oracle-relabelled rather than guessed"
                )
            dagger = self.data.trajectory_dagger
            if dagger is None:
                raise ValueError("data.trajectory_dagger is required")
            if dagger.num_history != system2_memory.num_tokens:
                raise ValueError(
                    "one memory token per history slot: "
                    f"num_history={dagger.num_history} vs "
                    f"num_tokens={system2_memory.num_tokens}"
                )
            if self.loss.lm_weight <= 0:
                raise ValueError("System2 memory training requires loss.lm_weight > 0")
            if self.loss.heatmap_weight != 0 or self.loss.trajectory_weight != 0:
                raise ValueError(
                    "System2 memory training supervises the language model only: "
                    "heatmap_weight=0 and trajectory_weight=0"
                )
            for stage in self.training.stages:
                if not stage.strict_trainable_modules:
                    raise ValueError(
                        "System2 memory training requires strict_trainable_modules=true"
                    )
                if sorted(stage.trainable_modules) != ["lora", "system2_memory"]:
                    raise ValueError(
                        "System2 memory training must train exactly "
                        "['lora','system2_memory']"
                    )
                if stage.train_action:
                    raise ValueError(
                        "System2 memory training keeps System1 frozen: train_action=false"
                    )
                if (
                    stage.train_heatmap is True
                    or stage.train_history is True
                    or stage.train_future is True
                ):
                    raise ValueError(
                        "System2 memory training keeps both cognition heads frozen"
                    )
                if not (stage.train_lm or stage.train_system2_sft):
                    raise ValueError(
                        "System2 memory training requires the System2 SFT loss"
                    )
                if (stage.system2_sft_protocol or "internnav") != "internnav":
                    raise ValueError(
                        "System2 memory training uses the released InternNav protocol"
                    )
                if getattr(stage, "teacher_force_system2_answer", False) is not True:
                    raise ValueError(
                        "System2 memory training teacher-forces the answer it supervises"
                    )
                if not stage.load_frozen_past_head:
                    raise ValueError(
                        "System2 memory training needs the deployed Past Head: "
                        "load_frozen_past_head=true"
                    )
                if stage.heatmap_pose_adaptation_init:
                    raise ValueError(
                        "System2 memory training does not adapt the Past Head; "
                        "use load_frozen_past_head instead"
                    )
            validation = self.validation
            if (
                validation is not None
                and validation.enabled
                and str(validation.save_best_metric) != "val_lm_loss"
            ):
                raise ValueError(
                    "System2 memory training selects on val_lm_loss over the "
                    "held-out scenes"
                )
            return self

        if llm is None or llm.use_lora:
            raise ValueError(
                "internnav_single_view requires model.llm.use_lora=false"
            )
        if llm.gradient_checkpointing:
            raise ValueError(
                "internnav_single_view frozen vision does not use gradient "
                "checkpointing"
            )
        if action_head is not None and action_head.enable:
            raise ValueError(
                "heatmap-only training must set model.action_head.enable=false"
            )
        if self.data.dataset_type != "sliding_window":
            raise ValueError(
                "internnav_single_view four-camera supervision requires "
                "data.dataset_type=sliding_window"
            )
        sliding = self.data.sliding_window
        if sliding is None or not sliding.single_view_rgb_input:
            raise ValueError(
                "internnav_single_view requires "
                "data.sliding_window.single_view_rgb_input=true"
            )
        if sliding.defer_heatmap_to_gpu:
            raise ValueError(
                "internnav_single_view requires defer_heatmap_to_gpu=false"
            )
        for stage in self.training.stages:
            if stage.trainable_modules != ["heatmap_vln"]:
                raise ValueError(
                    "internnav_single_view must train exactly ['heatmap_vln']"
                )
            if not stage.strict_trainable_modules:
                raise ValueError(
                    "internnav_single_view requires strict_trainable_modules=true"
                )
            if stage.train_action:
                raise ValueError("internnav_single_view must not train System1")
            if bool(
                stage.train_lm
                if stage.train_lm is not None
                else stage.train_system2_sft
            ):
                raise ValueError("internnav_single_view must not train System2")
            contract = stage.heatmap_warmstart_contract
            if (
                contract is None
                or contract.policy != "internnav_single_view_head_v2"
            ):
                raise ValueError(
                    "internnav_single_view requires the provenance-locked "
                    "internnav_single_view_head_v2 warm-start policy"
                )
        return self


# --- Public API --------------------------------------------------------------

_PATHS_ALLOWED = frozenset(
    {
        "dataset_root",
        "val_root",
        "log_out_dir",
        "tensorboard_dir",
        "llm_model_path",
        "internnav_model_path",
    }
)


def _expand_env_strings(obj: Any) -> Any:
    """Apply ``expanduser`` / ``expandvars`` to every string in nested dict/list."""
    if isinstance(obj, str):
        return os.path.expandvars(os.path.expanduser(obj))
    if isinstance(obj, dict):
        return {k: _expand_env_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_strings(v) for v in obj]
    return obj


def _merge_paths_block(cfg: dict[str, Any]) -> None:
    """Pop optional top-level ``paths`` and merge into ``data`` / ``log`` / ``model``.

    Supported keys:

    * ``dataset_root`` → ``data.root`` (non-empty string only)
    * ``val_root`` → ``data.val_root`` (if key is present; value may be null)
    * ``log_out_dir`` → ``log.out_dir`` (non-empty string only)
    * ``tensorboard_dir`` → ``log.tensorboard_dir`` (if key is present)
    * ``internnav_model_path`` → ``model.llm.model_path`` and
      ``model.action_head.nextdit.internnav_model_path`` (non-empty string only;
      use this for unified InternNav HF checkpoints)
    * ``llm_model_path`` → ``model.llm.model_path`` (non-empty string only;
      e.g. ``$INTERNNAV_BACKBONE`` expanded from the environment; kept for
      split-backbone compatibility)

    Values in ``paths`` should be host-specific; use ``$VAR`` / ``${VAR}`` and
    set environment variables on the machine or scheduler.
    """
    raw_paths = cfg.pop("paths", None)
    if raw_paths is None:
        return
    if not isinstance(raw_paths, dict):
        raise ValueError(f"paths must be a mapping, got {type(raw_paths).__name__}")
    extra = set(raw_paths) - _PATHS_ALLOWED
    if extra:
        raise ValueError(f"Unknown paths keys: {sorted(extra)}. Allowed: {sorted(_PATHS_ALLOWED)}")

    dr = raw_paths.get("dataset_root")
    if isinstance(dr, str) and dr.strip():
        data = cfg.setdefault("data", {})
        if not isinstance(data, dict):
            raise ValueError("data must be a mapping when paths.dataset_root is set")
        data["root"] = dr

    if "val_root" in raw_paths:
        data = cfg.setdefault("data", {})
        if not isinstance(data, dict):
            raise ValueError("data must be a mapping when paths.val_root is set")
        data["val_root"] = raw_paths["val_root"]

    lod = raw_paths.get("log_out_dir")
    if isinstance(lod, str) and lod.strip():
        log = cfg.setdefault("log", {})
        if not isinstance(log, dict):
            raise ValueError("log must be a mapping when paths.log_out_dir is set")
        log["out_dir"] = lod

    if "tensorboard_dir" in raw_paths:
        log = cfg.setdefault("log", {})
        if not isinstance(log, dict):
            raise ValueError("log must be a mapping when paths.tensorboard_dir is set")
        log["tensorboard_dir"] = raw_paths["tensorboard_dir"]

    internnav_mp = raw_paths.get("internnav_model_path")
    if isinstance(internnav_mp, str) and internnav_mp.strip():
        _set_unified_internnav_model_path(cfg, internnav_mp)

    llm_mp = raw_paths.get("llm_model_path")
    if isinstance(llm_mp, str) and llm_mp.strip():
        model = cfg.setdefault("model", {})
        if not isinstance(model, dict):
            raise ValueError("model must be a mapping when paths.llm_model_path is set")
        llm = model.setdefault("llm", {})
        if not isinstance(llm, dict):
            raise ValueError("model.llm must be a mapping when paths.llm_model_path is set")
        llm["model_path"] = llm_mp


def prepare_config_for_use(cfg: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy, expand env vars in strings, merge ``paths`` into data/log/model."""
    out = copy.deepcopy(cfg)
    out = _expand_env_strings(out)
    _merge_paths_block(out)
    _apply_model_path_env_overrides(out)
    data = out.get("data")
    if isinstance(data, dict):
        for section_name in ("sliding_window", "trajectory"):
            if section_name in data:
                data[section_name] = migrate_single_view_history_config(
                    data[section_name],
                    section_path=f"data.{section_name}",
                )
    return out


def _set_unified_internnav_model_path(cfg: dict[str, Any], path: str) -> None:
    """Set the unified InternNav model path for both backbone and System1."""
    model = cfg.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("model must be a mapping when InternNav model path is set")

    llm = model.setdefault("llm", {})
    if not isinstance(llm, dict):
        raise ValueError("model.llm must be a mapping when InternNav model path is set")
    llm["model_path"] = str(path).strip()

    action_head = model.setdefault("action_head", {})
    if not isinstance(action_head, dict):
        raise ValueError("model.action_head must be a mapping when InternNav model path is set")
    nextdit = action_head.setdefault("nextdit", {})
    if not isinstance(nextdit, dict):
        raise ValueError("model.action_head.nextdit must be a mapping when InternNav model path is set")
    nextdit["internnav_model_path"] = str(path).strip()


def _apply_model_path_env_overrides(cfg: dict[str, Any]) -> None:
    raw_unified = os.environ.get("INTERNNAV_MODEL_PATH") or os.environ.get("HEATMAPVLN_INTERNNAV_MODEL_PATH")
    if raw_unified and str(raw_unified).strip():
        _set_unified_internnav_model_path(cfg, str(raw_unified).strip())
        return

    _apply_llm_model_path_env_override(cfg)


def _apply_llm_model_path_env_override(cfg: dict[str, Any]) -> None:
    """If set, ``INTERNNAV_BACKBONE`` or ``HEATMAPVLN_LLM_MODEL_PATH`` overrides ``model.llm.model_path``.

    Applied after ``paths`` merge so a single export can point all stages at the
    host VLM directory without editing YAML per machine.
    """
    raw = os.environ.get("INTERNNAV_BACKBONE") or os.environ.get("HEATMAPVLN_LLM_MODEL_PATH")
    if not raw or not str(raw).strip():
        return
    model = cfg.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("model must be a mapping when VLM env override is set")
    llm = model.setdefault("llm", {})
    if not isinstance(llm, dict):
        raise ValueError("model.llm must be a mapping when VLM env override is set")
    llm["model_path"] = str(raw).strip()


def validate_config(cfg: dict[str, Any]) -> TrainConfig:
    """Validate a raw config dict against the schema.

    Raises ``pydantic.ValidationError`` with a clear message listing all
    violations if the config is invalid.
    """
    prepared = prepare_config_for_use(cfg)
    return TrainConfig.model_validate(prepared)


def normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate a config dict and materialize schema defaults.

    Returns a plain ``dict`` so the rest of the codebase can continue to
    use dictionary access while still benefiting from Pydantic defaults.
    ``None`` sections are excluded to preserve the previous ``dict.get``
    behavior for optional nested configs.
    """
    validated = validate_config(cfg)
    return validated.model_dump(mode="python", exclude_none=True)


def load_and_validate_config(config_path: Union[str, Path]) -> dict[str, Any]:
    """Load a YAML config file and validate it.

    Returns a normalized ``dict`` with schema defaults materialized so
    downstream code that expects ``dict`` keeps working unchanged while
    observing the same defaults as the Pydantic model.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping at top level, got {type(cfg).__name__}")
    return normalize_config(cfg)

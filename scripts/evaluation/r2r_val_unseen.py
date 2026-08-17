"""
Habitat closed-loop evaluation for VLNPipeline on VLN-CE R2R val_unseen.

Uses the VLNPipeline (Qwen2.5-VL + HeatmapVLN + NextDiT System 1) to
navigate in the Habitat simulator and collect standard VLN-CE metrics
(SR, SPL, OS, NE).

Inference flow per high-level step:
    1. Capture 4 panoramic views (front/right/back/left) via sim state
    2. Prepare Qwen2.5-VL input from panoramic views + instruction
    3. Auto-regressive text generation → pixel-goal coordinates or STOP
    4. If coordinates found:
       a. Capture lookdown view (2 × LOOKDOWN, then restore)
       b. generate_latents → traj_hidden_states
       c. (optional) pano-to-InternNav latent adapter projects traj_hidden_states
          into the InternNav-compatible latent manifold expected by frozen NextDiT
       d. NextDiT get_trajectory → continuous trajectory (dx, dy, dyaw)
       e. Convert trajectory to discrete Habitat actions
       f. Execute up to MAX_LOCAL_STEPS actions
    5. If STOP or no coordinates → end episode

Adapted for habitat-lab 0.1.7 (YACS config).
"""

from __future__ import annotations

import faulthandler
import os
import sys
from pathlib import Path
from typing import Any

faulthandler.enable()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ═══════════════════════════════════════════════════════════════════════
# Section 1: Runtime patches (must be before any heavy imports)
# ═══════════════════════════════════════════════════════════════════════

# Keep Transformers on the PyTorch path only.  Importing TensorFlow after a
# Habitat-Sim GL context has been created can segfault in this environment.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# The installed habitat_sim is GLX-based, not EGL/headless.  On this node,
# forcing NVIDIA GLX through Xvfb crashes with X11 BadWindow; Mesa llvmpipe is
# slower but stable enough for correctness evaluation.  Operators can opt back
# into NVIDIA GLX if their display stack supports it.
if (
    os.environ.get("__GLX_VENDOR_LIBRARY_NAME") == "nvidia"
    and os.environ.get("HEATMAPVLN_ALLOW_NVIDIA_GLX", "0") != "1"
):
    os.environ.pop("__GLX_VENDOR_LIBRARY_NAME", None)
    print(
        "WARNING: disabled __GLX_VENDOR_LIBRARY_NAME=nvidia for Habitat GLX "
        "stability. Set HEATMAPVLN_ALLOW_NVIDIA_GLX=1 to keep it.",
        flush=True,
    )

# Block flash_attn import (GLIBC_2.32 not available on this system)
import importlib as _importlib
import importlib.machinery as _importlib_machinery  # noqa: F401  # needed for _importlib.machinery
import types as _types


def _noop(*a, **kw):
    raise RuntimeError("flash_attn stub called – should use SDPA attention instead")


def _make_stub(name, attrs=None):
    m = _types.ModuleType(name)
    m.__spec__ = _importlib.machinery.ModuleSpec(name, None)
    m.__version__ = "2.7.4"
    m.__heatmapvln_stub__ = True
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    sys.modules[name] = m
    return m


class _FlashAttnKernelStub:
    def fwd(self, *_args, **_kwargs):
        return _noop(*_args, **_kwargs)

    def varlen_fwd(self, *_args, **_kwargs):
        return _noop(*_args, **_kwargs)

    def bwd(self, *_args, **_kwargs):
        return _noop(*_args, **_kwargs)

    def varlen_bwd(self, *_args, **_kwargs):
        return _noop(*_args, **_kwargs)


_flash_kernel_stub = _FlashAttnKernelStub()

_fa = _make_stub(
    "flash_attn",
    {
        "flash_attn_func": _noop,
        "flash_attn_varlen_func": _noop,
    },
)
_make_stub("flash_attn_2_cuda")
_fa_iface = _make_stub(
    "flash_attn.flash_attn_interface",
    {
        "flash_attn_func": _noop,
        "flash_attn_varlen_func": _noop,
        "flash_attn_gpu": _flash_kernel_stub,
        "flash_attn_cuda": _flash_kernel_stub,
    },
)
_fa_bert = _make_stub(
    "flash_attn.bert_padding",
    {
        "index_first_axis": _noop,
        "pad_input": _noop,
        "unpad_input": _noop,
    },
)
_fa_rotary = _make_stub("flash_attn.layers", {})
_fa_rotary_mod = _make_stub(
    "flash_attn.layers.rotary",
    {
        "apply_rotary_emb": _noop,
    },
)
_fa.flash_attn_interface = _fa_iface
_fa.bert_padding = _fa_bert
_fa.layers = _fa_rotary
_fa_rotary.rotary = _fa_rotary_mod

import numpy as np

if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "int"):
    np.int = np.int64
if not hasattr(np, "bool"):
    np.bool = np.bool_

# Import torch before habitat_sim (habitat_sim pulls torch during its __init__).
import torch as _torch_preload  # noqa: F401

from scripts.evaluation.episode_cohort import (
    load_episode_cohort,
    restrict_habitat_env_to_episode_keys,
)


LOCAL_FJL_ROOT = Path(os.environ.get("HEATMAPVLN_FJL_ROOT", "/mnt/afs/lixiaoou/intern/fjl"))
LOCAL_VLNCE_DATA_ROOT = Path(
    os.environ.get("HEATMAPVLN_VLNCE_DATA_ROOT", str(LOCAL_FJL_ROOT / "habitat" / "VLN-CE" / "data"))
)
LOCAL_MP3D_ROOT = Path(os.environ.get("HEATMAPVLN_MP3D_ROOT", str(LOCAL_VLNCE_DATA_ROOT / "scene_datasets" / "mp3d")))
LOCAL_R2R_DATASETS_ROOT = Path(os.environ.get("HEATMAPVLN_R2R_DATASETS_ROOT", str(LOCAL_VLNCE_DATA_ROOT / "datasets")))
LOCAL_INTERNNAV_MODEL_PATH = Path(
    os.environ.get("HEATMAPVLN_INTERNNAV_MODEL_PATH", str(LOCAL_FJL_ROOT / "InternNav-Model"))
)


def _find_preinit_scene() -> str | None:
    candidates = [
        os.environ.get("HEATMAPVLN_PREINIT_SCENE"),
        str(LOCAL_MP3D_ROOT / "17DRP5sb8fy" / "17DRP5sb8fy.glb"),
        str(LOCAL_MP3D_ROOT / "zsNo4HB9uLZ" / "zsNo4HB9uLZ.glb"),
        "/root/autodl-tmp/data/scene_datasets/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb",
        "/dataset/mp3d/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb",
        "/workspace/InternNav/data/scene_data/mp3d_ce/zsNo4HB9uLZ/zsNo4HB9uLZ.glb",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate).resolve())

    for root in (
        LOCAL_MP3D_ROOT,
        Path("/root/autodl-tmp/data/scene_datasets/mp3d"),
        Path("/dataset/mp3d/mp3d"),
        Path("/dataset/mp3d"),
    ):
        if not root.exists():
            continue
        try:
            scene = next(sorted(root.glob("*/*.glb")))
            return str(scene.resolve())
        except StopIteration:
            continue
    return None


def _make_camera_sensor_spec(_hsim: Any, uuid: str = "rgb", resolution: tuple[int, int] = (64, 64)) -> Any:
    spec_cls = getattr(_hsim, "CameraSensorSpec", None) or _hsim.SensorSpec
    sensor = spec_cls()
    sensor.uuid = uuid
    sensor.sensor_type = _hsim.SensorType.COLOR
    if hasattr(_hsim, "SensorSubType"):
        sensor.sensor_subtype = _hsim.SensorSubType.PINHOLE
    sensor.resolution = list(resolution)
    return sensor


# Create a real-scene GL context before importing habitat-lab, which imports
# numba/LLVM.  A no-scene dummy simulator is not safe on this node.
if os.environ.get("HEATMAPVLN_PREINIT_GL", "1") != "0" and os.environ.get("DISPLAY"):
    _preinit_scene = _find_preinit_scene()
    if _preinit_scene:
        import habitat_sim as _hsim

        _dummy_cfg = _hsim.SimulatorConfiguration()
        _dummy_cfg.scene_id = _preinit_scene
        _dummy_cfg.gpu_device_id = int(os.environ.get("HABITAT_GL_GPU_ID", "0"))
        _dummy_agent = _hsim.agent.AgentConfiguration()
        _dummy_sensor = _make_camera_sensor_spec(_hsim)
        _dummy_agent.sensor_specifications = [_dummy_sensor]
        _dummy_sim = _hsim.Simulator(_hsim.Configuration(_dummy_cfg, [_dummy_agent]))
        _dummy_sim.get_sensor_observations()
        _dummy_sim.close()
        del _dummy_sim, _dummy_cfg, _dummy_agent, _dummy_sensor
        print(f"GL context pre-initialized with scene: {_preinit_scene}", flush=True)
    else:
        print("WARNING: could not find an MP3D scene for GL pre-initialization", flush=True)

if os.environ.get("HEATMAPVLN_PREINIT_EMPTY_GL", "0") == "1":
    import habitat_sim as _hsim

    _preinit_scene = _find_preinit_scene()
    _dummy_cfg = _hsim.SimulatorConfiguration()
    if _preinit_scene:
        _dummy_cfg.scene_id = _preinit_scene
    else:
        print("WARNING: skipping empty GL pre-initialization; no real MP3D scene found", flush=True)
        _dummy_cfg = None
    if _dummy_cfg is not None:
        # For GLX builds, pre-initialisation must use the local visible GPU index.
        _dummy_cfg.gpu_device_id = int(os.environ.get("HABITAT_GL_GPU_ID", "0"))
        _dummy_agent = _hsim.agent.AgentConfiguration()
        _dummy_agent.sensor_specifications = [_make_camera_sensor_spec(_hsim)]
        _dummy_sim = _hsim.Simulator(_hsim.Configuration(_dummy_cfg, [_dummy_agent]))
        _dummy_sim.get_sensor_observations()
        _dummy_sim.close()
        del _dummy_sim, _dummy_cfg, _dummy_agent
        print(f"GL context pre-initialized with scene: {_preinit_scene}", flush=True)

# Patch gym.spaces.Discrete to allow n=0 (habitat-lab 0.1.7 compatibility)
import gym.spaces

_OrigDiscrete = gym.spaces.Discrete


class _PatchedDiscrete(_OrigDiscrete):
    def __init__(self, n, *args, **kwargs):
        if n == 0:
            n = 1
        super().__init__(n, *args, **kwargs)


gym.spaces.Discrete = _PatchedDiscrete

# ═══════════════════════════════════════════════════════════════════════
# Section 2: Imports
# ═══════════════════════════════════════════════════════════════════════

import argparse
import copy
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import random
import re
from collections import OrderedDict
from enum import IntEnum

import habitat
import quaternion
import torch
import tqdm
from habitat.config.default import Config as CN
from habitat.config.default import get_config as get_habitat_default_config
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.tasks.nav.nav import DistanceToGoal
from PIL import Image

if not hasattr(argparse, "BooleanOptionalAction"):

    class _BooleanOptionalAction(argparse.Action):
        def __init__(
            self,
            option_strings,
            dest,
            default=None,
            type=None,
            choices=None,
            required=False,
            help=None,
            metavar=None,
        ):
            expanded = []
            for option_string in option_strings:
                expanded.append(option_string)
                if option_string.startswith("--"):
                    expanded.append("--no-" + option_string[2:])
            super().__init__(
                option_strings=expanded,
                dest=dest,
                nargs=0,
                default=default,
                type=type,
                choices=choices,
                required=required,
                help=help,
                metavar=metavar,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not str(option_string).startswith("--no-"))

        def format_usage(self):
            return " | ".join(self.option_strings)

    argparse.BooleanOptionalAction = _BooleanOptionalAction

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

from scripts.evaluation.closed_loop_guard import (
    STOP_ACCEPT,
    STOP_PROBE,
    ClosedLoopGuard,
    ClosedLoopGuardConfig,
    should_trust_temporal_terminal,
)
from scripts.evaluation.navigation_metrics import aggregate_navigation_metrics
from scripts.evaluation.rpc_protocol import (
    HEATMAPVLN_RPC_DEFAULT_PROTOCOL_SEED,
    HEATMAPVLN_RPC_PROTOCOL_VERSION,
    HEATMAPVLN_RPC_SAMPLING_FIELD,
    HEATMAPVLN_RPC_SAMPLING_PROTOCOL,
    build_rpc_progress_sampling_contract,
    build_rpc_sampling_metadata,
    validate_rpc_progress_sampling_contract,
    validate_rpc_sampling_metadata,
)
from scripts.evaluation.stop_dagger import (
    BoundaryProbeSweepState,
    HistoricalFalseStopTrigger,
    OracleRecoveryState,
    parse_historical_false_stop_trigger,
    prune_stop_collection_jsonl_for_resume,
    should_finish_oracle_recovery_collection,
    should_force_continue_negative,
    should_record_stop_multimodal_example,
    validate_boundary_probe_collection,
    validate_historical_false_stop_source,
    validate_oracle_path_collection,
    validate_oracle_recovery_actions_per_call,
    validate_oracle_recovery_collection,
)

_INPUT_CONSTRUCTOR_PATH = _PROJECT_ROOT / "src" / "models" / "heatmap" / "input_constructor.py"
_INPUT_CONSTRUCTOR_SPEC = importlib.util.spec_from_file_location(
    "_heatmapvln_input_constructor",
    _INPUT_CONSTRUCTOR_PATH,
)
if _INPUT_CONSTRUCTOR_SPEC is None or _INPUT_CONSTRUCTOR_SPEC.loader is None:
    raise ImportError(f"Could not load input constructor from {_INPUT_CONSTRUCTOR_PATH}")
_input_constructor = importlib.util.module_from_spec(_INPUT_CONSTRUCTOR_SPEC)
sys.modules[_INPUT_CONSTRUCTOR_SPEC.name] = _input_constructor
_INPUT_CONSTRUCTOR_SPEC.loader.exec_module(_input_constructor)
construct_input = _input_constructor.construct_input
parse_structured_pano_output = _input_constructor.parse_structured_pano_output
structured_condition_text = _input_constructor.structured_condition_text
vlm_output_requests_stop = _input_constructor.vlm_output_requests_stop
vlm_output_requests_turn = _input_constructor.vlm_output_requests_turn

_TRAINING_UTILS_PATH = _PROJECT_ROOT / "scripts" / "training" / "utils.py"
_TRAINING_UTILS_SPEC = importlib.util.spec_from_file_location(
    "_heatmapvln_training_utils",
    _TRAINING_UTILS_PATH,
)
if _TRAINING_UTILS_SPEC is None or _TRAINING_UTILS_SPEC.loader is None:
    raise ImportError(f"Could not load training utils from {_TRAINING_UTILS_PATH}")
try:
    _training_utils = importlib.util.module_from_spec(_TRAINING_UTILS_SPEC)
    sys.modules[_TRAINING_UTILS_SPEC.name] = _training_utils
    _TRAINING_UTILS_SPEC.loader.exec_module(_training_utils)
    _normalize_state_key = _training_utils._normalize_state_key
    load_config = _training_utils.load_config
except ModuleNotFoundError as exc:
    if exc.name != "torch.distributed":
        raise

    def _normalize_state_key(name: str) -> str:
        if name.startswith("module."):
            name = name[len("module.") :]
        name = name.replace(".module.", ".")
        prefix_aliases = {
            "qwen3_5.": "qwen2_5_vl.",
            "qwen3_5_vl.": "qwen2_5_vl.",
        }
        for old_prefix, new_prefix in prefix_aliases.items():
            if name.startswith(old_prefix):
                return new_prefix + name[len(old_prefix) :]
        return name

    def load_config(config_path: str, validate: bool = True) -> dict:
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        try:
            if validate:
                from src.config_schema import normalize_config

                return normalize_config(cfg)
            from src.config_schema import prepare_config_for_use

            return prepare_config_for_use(cfg)
        except Exception:
            return cfg


def _load_pano_latent_adapter(
    checkpoint_path: str,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Lazy loader for the pano→InternNav latent adapter.

    Imports are kept inside the function so vanilla Habitat eval runs without
    the adapter never need to touch the training-side modules.

    Returns the adapter in ``eval()`` mode on ``device``. Reuses the inference
    helper from ``eval_pano_latent_adapter.py`` so checkpoint introspection
    (n_layers / pre_norm / output_affine) stays in one place.
    """
    from scripts.evaluation.eval_pano_latent_adapter import (
        _load_adapter_from_checkpoint,
    )

    fallback = argparse.Namespace(
        adapter_hidden_dim=2048,
        adapter_dropout=0.0,
        residual=False,
        pre_norm=False,
    )
    adapter, _saved_args = _load_adapter_from_checkpoint(
        Path(checkpoint_path).expanduser(),
        dim=hidden_dim,
        fallback_args=fallback,
        device=device,
        dtype=dtype,
    )
    return adapter


def _maybe_apply_pano_latent_adapter(
    traj_hs: torch.Tensor,
    adapter,
    *,
    view_id: str | None = None,
    pixel_goal: list[int] | None = None,
    image_size: tuple[int, int] | None = None,
    cond_projector: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Project ``traj_hs`` through the optional adapter, preserving dtype.

    - ``GeometryAwarePanoToNextDiTAdapter``: geometry-aware → 768, bypasses
      cond_projector (legacy).
    - ``PanoLatentSpaceAdapter``: simple MLP → 3584, then routed through
      ``cond_projector`` to 768.
    """
    if adapter is None:
        return traj_hs
    orig_dtype = traj_hs.dtype
    adapter_param = next(adapter.parameters(), None)
    adapter_dtype = adapter_param.dtype if adapter_param is not None else orig_dtype

    if hasattr(adapter, "geometry_token"):
        # Legacy geometry-aware adapter — bypasses cond_projector.
        from src.models.adapters import view_ids_to_indices

        if pixel_goal is None:
            raise RuntimeError("Geometry-aware pano adapter requires pixel_goal")
        goal_view = (view_id or "front").lower()
        if goal_view not in {"front", "right", "back", "left"}:
            goal_view = "front"
        if image_size is None:
            image_size = (384, 384)
        width, height = int(image_size[0]), int(image_size[1])
        view_indices = view_ids_to_indices([goal_view], device=traj_hs.device)
        pixel_xy = torch.tensor(
            [[int(pixel_goal[0]), int(pixel_goal[1])]],
            device=traj_hs.device,
            dtype=adapter_dtype,
        )
        image_hw = torch.tensor(
            [[height, width]],
            device=traj_hs.device,
            dtype=adapter_dtype,
        )
        out = adapter(traj_hs.to(dtype=adapter_dtype), view_indices, pixel_xy, image_hw)
        return out.to(dtype=orig_dtype)

    if hasattr(adapter, "mlp"):
        # PanoLatentSpaceAdapter: MLP → 3584 → cond_projector → 768.
        adapted = adapter(traj_hs.to(dtype=adapter_dtype))  # (B, Q, 3584)
        if cond_projector is not None:
            proj_dtype = next(cond_projector.parameters()).dtype
            adapted = cond_projector(adapted.to(dtype=proj_dtype))
        return adapted.to(dtype=orig_dtype)

    out = adapter(traj_hs.to(dtype=adapter_dtype))
    return out.to(dtype=orig_dtype)


def _parse_pano_view_id(llm_output: str) -> str | None:
    parsed = parse_structured_pano_output(llm_output, image_size=None)
    if parsed.kind == "pixel" and parsed.view_id is not None:
        return parsed.view_id
    if parsed.kind in {"legacy_coord", "stop", "turn", "invalid"}:
        return parsed.view_id
    return None


def _parse_pixel_goal(
    llm_output: str,
    image_size: tuple[int, int],
    *,
    allow_legacy_coord: bool = True,
) -> list[int] | None:
    """Parse structured ``view/pixel`` or legacy ``u v`` pixel goals.

    Clamping of out-of-bounds pixel coordinates is handled (with a
    one-shot warning) inside ``parse_structured_pano_output``.
    """
    parsed = parse_structured_pano_output(
        llm_output,
        image_size=image_size,
        allow_legacy_coord=allow_legacy_coord,
    )
    if parsed.kind in {"pixel", "legacy_coord"} and parsed.pixel_goal is not None:
        return list(parsed.pixel_goal)
    if not re.search(r"\d", llm_output or ""):
        return None
    return None


def _vlm_requests_turn(llm_output: str) -> str | None:
    """Return "left" / "right" for directional turn, or None if not a turn."""
    return vlm_output_requests_turn(llm_output)


def _trajectory_from_condition(
    action_head,
    traj_condition: torch.Tensor,
    *,
    traj_images: torch.Tensor | None,
) -> torch.Tensor:
    if traj_condition.shape[-1] == int(action_head.config.latent_emb_size):
        return action_head.get_trajectory_from_projected(
            traj_condition,
            traj_images=traj_images,
        )
    return action_head.get_trajectory(
        traj_condition,
        traj_images=traj_images,
    )


def _load_force_teacher_internnav(args, device: torch.device):
    """Load the InternNav teacher VLM in-process for ``--force_teacher_coord``.

    Reuses ``scripts.evaluation.collect_internnav_teacher_sidecar._load_teacher``
    by synthesising a SimpleNamespace with only the fields it consumes. Lazy
    import keeps default eval runs free of InternNav-side dependencies.

    NOTE: H7 sanity check helper. Loads ~7 GB extra GPU memory; if VRAM is
    tight, point ``--force_teacher_coord_gpu_id`` at a separate device.
    """
    import types as _types

    teacher_device = device
    teacher_gpu_id = int(getattr(args, "force_teacher_coord_gpu_id", -1))
    if teacher_gpu_id >= 0:
        teacher_device = torch.device(f"cuda:{teacher_gpu_id}")

    from scripts.evaluation.collect_internnav_teacher_sidecar import _load_teacher

    sub = _types.SimpleNamespace(
        internnav_repo=str(args.force_teacher_internnav_repo),
        model_path=str(args.force_teacher_internnav_model_path),
        flash_attn_stub=bool(getattr(args, "force_teacher_flash_attn_stub", True)),
        torch_dtype=str(getattr(args, "force_teacher_torch_dtype", "bf16")),
        attn_implementation=str(getattr(args, "force_teacher_attn_impl", "sdpa")),
        require_nextdit=False,
    )
    model, processor, _traj_to_actions = _load_teacher(sub, teacher_device)
    return model, processor, teacher_device


def _predict_force_teacher_coord(
    teacher_model,
    teacher_processor,
    teacher_device: torch.device,
    current_front_pil,
    lookdown_pil,
    instruction: str,
    *,
    vlm_image_size: tuple[int, int],
    history_front_pils: list | None = None,
    max_new_tokens: int = 64,
) -> tuple[list[int] | None, dict]:
    """Run the InternNav teacher's two-turn protocol to get a teacher coord.

    Returns ``(coord, info)`` where ``coord`` is ``[u, v]`` or ``None`` if
    either turn failed to produce a parseable coordinate. ``info`` carries
    both turns' raw text and which turn produced the final coord, so callers
    can log failure reasons for diagnostics.
    """
    from scripts.evaluation.collect_internnav_teacher_sidecar import (
        DEFAULT_IMAGE_TOKEN,
        INTERNNAV_CONJUNCTIONS,
        PROMPT_TEMPLATE,
        _content_from_text_with_images,
        _parse_coord,
        _strip_instruction_final_period,
    )

    if current_front_pil.size != vlm_image_size:
        current_front_pil = current_front_pil.resize(vlm_image_size)
    if lookdown_pil.size != vlm_image_size:
        lookdown_pil = lookdown_pil.resize(vlm_image_size)
    history_front_pils = history_front_pils or []
    history_front_pils = [
        (img if img.size == vlm_image_size else img.resize(vlm_image_size)) for img in history_front_pils
    ]

    cleaned_instruction = _strip_instruction_final_period(instruction or "")
    prompt_text = PROMPT_TEMPLATE.replace("<instruction>.", cleaned_instruction)
    if history_front_pils:
        prompt_text += (
            f" These are your historical observations: {(DEFAULT_IMAGE_TOKEN + chr(10)) * len(history_front_pils)}."
        )
    prompt_text += f" {INTERNNAV_CONJUNCTIONS[0]}{DEFAULT_IMAGE_TOKEN}."

    first_images = history_front_pils + [current_front_pil]
    first_messages = [
        {
            "role": "user",
            "content": _content_from_text_with_images(prompt_text, first_images),
        }
    ]

    def _run_once(messages, images):
        text = teacher_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = teacher_processor(text=[text], images=images, return_tensors="pt").to(teacher_device)
        with torch.inference_mode():
            out_ids = teacher_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                past_key_values=None,
                return_dict_in_generate=True,
            ).sequences
        prompt_len = int(inputs.input_ids.shape[1])
        return teacher_processor.tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=True).strip()

    turn1 = _run_once(first_messages, first_images)
    info: dict = {
        "turn1_text": turn1,
        "turn2_text": None,
        "used_turn": 1,
        "n_history": len(history_front_pils),
    }
    coord_uv, _ = _parse_coord(turn1)
    if coord_uv is not None:
        return coord_uv, info

    if "↓" not in turn1:
        return None, info

    second_text = f"{INTERNNAV_CONJUNCTIONS[0]}{DEFAULT_IMAGE_TOKEN}."
    second_messages = list(first_messages)
    second_messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": turn1}],
        }
    )
    second_messages.append(
        {
            "role": "user",
            "content": _content_from_text_with_images(second_text, [lookdown_pil]),
        }
    )
    second_images = first_images + [lookdown_pil]

    turn2 = _run_once(second_messages, second_images)
    info["turn2_text"] = turn2
    info["used_turn"] = 2
    coord_uv, _ = _parse_coord(turn2)
    return coord_uv, info


MAX_STEPS = 8
MAX_LOCAL_STEPS = 4
DEFAULT_SCENES_DIR = "data/scene_data/mp3d_ce"
DEFAULT_DATA_PATH = "data/vln_ce/raw_data/r2r/{split}/{split}.json.gz"
DEFAULT_IMAGE_TOKEN = "<image>"
LEGACY_PROMPT_TEMPLATE = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. "
    "Where should you go next to stay on track? Please output the next waypoint's "
    "coordinates in the image. Please output STOP when you have successfully completed the task."
)
LEGACY_CONJUNCTIONS = [
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
]
LEGACY_ACTIONS2IDX = OrderedDict(
    {
        "STOP": [0],
        "↑": [1],
        "←": [2],
        "→": [3],
        "↓": [5],
    }
)


class ActionCode(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


def _closed_loop_guard_config(args) -> ClosedLoopGuardConfig:
    action_chunk_size = int(args.rpc_action_chunk_size)
    if action_chunk_size > MAX_LOCAL_STEPS:
        raise ValueError(
            f"rpc_action_chunk_size must be <= {MAX_LOCAL_STEPS}, got {action_chunk_size}"
        )
    if int(args.closed_loop_recovery_history_keep) < 0:
        raise ValueError("closed_loop_recovery_history_keep must be >= 0")
    recovery_actions = (
        int(args.closed_loop_recovery_turns)
        + int(args.closed_loop_recovery_forward_steps)
    )
    if recovery_actions > MAX_STEPS:
        raise ValueError(
            f"closed-loop recovery actions must be <= {MAX_STEPS}, got {recovery_actions}"
        )
    return ClosedLoopGuardConfig(
        action_chunk_size=action_chunk_size,
        stop_confirmations=int(args.system2_stop_confirmations),
        stop_confirmation_max_gap_calls=int(
            args.system2_stop_confirmation_max_gap_calls
        ),
        stop_confirmation_view_sweep=bool(
            args.system2_stop_confirmation_view_sweep
        ),
        stop_high_confidence_threshold=args.system2_stop_high_confidence_threshold,
        stop_probe_turn=str(args.system2_stop_probe_turn),
        loop_guard_enabled=bool(args.closed_loop_guard),
        collision_epsilon_m=float(args.closed_loop_collision_epsilon_m),
        collision_forward_limit=int(args.closed_loop_collision_forward_limit),
        motion_window_steps=int(args.closed_loop_motion_window_steps),
        motion_min_path_m=float(args.closed_loop_motion_min_path_m),
        motion_max_net_m=float(args.closed_loop_motion_max_net_m),
        plan_window_calls=int(args.closed_loop_plan_window_calls),
        plan_view_dominance=float(args.closed_loop_plan_view_dominance),
        plan_min_path_m=float(args.closed_loop_plan_min_path_m),
        plan_max_net_m=float(args.closed_loop_plan_max_net_m),
        recovery_turns=int(args.closed_loop_recovery_turns),
        recovery_forward_steps=int(args.closed_loop_recovery_forward_steps),
        recovery_follow_last_turn=bool(args.closed_loop_recovery_follow_last_turn),
        recovery_cooldown_steps=int(args.closed_loop_recovery_cooldown_steps),
    )


def _agent_position(env) -> tuple[float, float, float]:
    state = env._sim.get_agent(0).get_state()
    position = np.asarray(state.position, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise RuntimeError(f"Invalid agent position from Habitat: {position!r}")
    return float(position[0]), float(position[1]), float(position[2])


def _trim_recovery_history(
    history: list[dict[str, Image.Image]],
    keep: int,
) -> list[dict[str, Image.Image]]:
    if keep <= 0:
        return []
    return history[-keep:]


def _normalize_instruction(instruction: str) -> str:
    """Match InternNav eval: drop trailing sentence punctuation in prompts."""
    if instruction.endswith((".", "!", "?")):
        return instruction[:-1]
    return instruction


def _finalize_local_actions(action_list: list[int]) -> list[int]:
    """Pad to MAX_STEPS then cap to MAX_LOCAL_STEPS (InternNav dual_system)."""
    if len(action_list) < MAX_STEPS:
        action_list = list(action_list) + [ActionCode.STOP] * (MAX_STEPS - len(action_list))
    if len(action_list) >= MAX_LOCAL_STEPS:
        action_list = action_list[:MAX_LOCAL_STEPS]
    return action_list


def _actions_for_log(actions: list[int]) -> list[int]:
    return [int(action) for action in actions]


_RGB_SENSOR_KEYS = (
    "rgb",
    "RGB_SENSOR",
    "color_sensor",
    "rgba_camera",
    "rgb_camera",
    "color",
)
_panoramic_sensor_warned = False


def _extract_rgb_array(observations: dict) -> np.ndarray | None:
    """Resolve RGB from Habitat task or sim sensor observations."""
    for key in _RGB_SENSOR_KEYS:
        value = observations.get(key)
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            return value

    for key, value in observations.items():
        if not isinstance(value, np.ndarray) or value.ndim < 2:
            continue
        key_lower = str(key).lower()
        if "depth" in key_lower:
            continue
        if value.ndim == 3 and value.shape[-1] in (3, 4):
            return value
        if value.ndim == 2:
            return value
    return None


def _rgb_array_to_pil(rgb: np.ndarray, image_size: tuple | None = None) -> Image.Image:
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    if image_size is not None:
        img = img.resize(image_size)
    return img


def split_and_clean(text: str) -> list[str]:
    """Split by <image> while preserving the token and removing blank text chunks."""
    parts = re.split(r"(<image>)", text)
    results: list[str] = []
    for part in parts:
        if part == DEFAULT_IMAGE_TOKEN:
            results.append(part)
        else:
            clean = part.replace("\n", "").strip()
            if clean:
                results.append(clean)
    return results


def parse_actions(output: str, actions2idx: OrderedDict) -> list[int]:
    action_patterns = "|".join(re.escape(action) for action in actions2idx)
    regex = re.compile(action_patterns)
    matches = regex.findall(output)
    actions = [actions2idx[match] for match in matches]
    return list(itertools.chain.from_iterable(actions))


def ensure_vln_measures_registered() -> None:
    """Register custom VLN measures required by Habitat-Lab 0.1.7 evaluation."""

    if registry.get_measure("OracleNavigationError") is None:

        @registry.register_measure
        class OracleNavigationError(Measure):
            cls_uuid: str = "oracle_navigation_error"

            def _get_uuid(self, *args, **kwargs) -> str:
                return self.cls_uuid

            def reset_metric(self, *args, task: EmbodiedTask, **kwargs) -> None:
                task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
                self._metric = float("inf")
                self.update_metric(task=task)

            def update_metric(self, *args, task: EmbodiedTask, **kwargs) -> None:
                distance_to_target = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
                self._metric = min(self._metric, distance_to_target)

    if registry.get_measure("OracleSuccess") is None:

        @registry.register_measure
        class OracleSuccess(Measure):
            cls_uuid: str = "oracle_success"

            def __init__(self, *args, config=None, **kwargs):
                self._config = config
                super().__init__()

            def _get_uuid(self, *args, **kwargs) -> str:
                return self.cls_uuid

            def reset_metric(self, *args, task: EmbodiedTask, **kwargs) -> None:
                task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
                self._metric = 0.0
                self.update_metric(task=task)

            def update_metric(self, *args, task: EmbodiedTask, **kwargs) -> None:
                distance_to_target = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
                self._metric = float(self._metric or distance_to_target < 3.0)


def _expand_path_template(path_str: str, split: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(path_str))
    try:
        return expanded.format(split=split)
    except (KeyError, IndexError, ValueError):
        return expanded


def _first_dataset_scene_id(data_path: str) -> str | None:
    """Read one scene id so an invalid Habitat scene root fails before native init."""
    path = Path(data_path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes", []) if isinstance(payload, dict) else []
    if not episodes:
        return None
    scene_id = episodes[0].get("scene_id")
    return str(scene_id) if scene_id else None


def _scene_asset_path(scenes_dir: Path, scene_id: str | None) -> Path | None:
    if not scene_id:
        return None
    return scenes_dir / scene_id


def _scene_root_is_compatible(scenes_dir: Path, scene_id: str | None) -> bool:
    if not scenes_dir.is_dir():
        return False
    scene_asset = _scene_asset_path(scenes_dir, scene_id)
    return scene_asset is None or scene_asset.is_file()


def _resolve_eval_paths(args, split: str = "val_unseen") -> None:
    """Resolve dataset/scenes paths for the shared Habitat evaluation environment."""

    requested_data_path = _expand_path_template(args.data_path, split)
    data_path = Path(requested_data_path)
    if data_path.exists():
        args.data_path = str(data_path.resolve())
    elif args.data_path == DEFAULT_DATA_PATH:
        data_candidates = [
            LOCAL_R2R_DATASETS_ROOT / "R2R_VLNCE_v1-3_preprocessed" / split / f"{split}.json.gz",
            LOCAL_R2R_DATASETS_ROOT / "R2R_VLNCE_v1-3" / split / f"{split}.json.gz",
            LOCAL_VLNCE_DATA_ROOT / "vln_ce" / "raw_data" / "r2r" / split / f"{split}.json.gz",
            Path(f"/home/intern/zhr/fjl/InternNav/data/vln_ce/raw_data/r2r/{split}/{split}.json.gz"),
            Path.home() / f"zhr/fjl/InternNav/data/vln_ce/raw_data/r2r/{split}/{split}.json.gz",
            Path.home() / f"InternNav/data/vln_ce/raw_data/r2r/{split}/{split}.json.gz",
            Path(f"/workspace/InternNav/data/vln_ce/raw_data/r2r/{split}/{split}.json.gz"),
            Path(f"/workspace/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz"),
            Path(f"/workspace/R2R_VLNCE_v1-3/{split}/{split}.json.gz"),
        ]
        resolved = next((p for p in data_candidates if p.exists()), None)
        if resolved is None:
            attempted = "\n  - ".join([requested_data_path, *map(str, data_candidates)])
            raise FileNotFoundError(f"Could not find the VLN-CE dataset file. Tried:\n  - {attempted}")
        args.data_path = str(resolved.resolve())
    else:
        raise FileNotFoundError(f"Configured --data_path does not exist: {requested_data_path}")

    first_scene_id = _first_dataset_scene_id(args.data_path)

    requested_scenes_dir = _expand_path_template(args.scenes_dir, split)
    scenes_dir = Path(requested_scenes_dir)
    if _scene_root_is_compatible(scenes_dir, first_scene_id):
        args.scenes_dir = str(scenes_dir.resolve())
    elif args.scenes_dir == DEFAULT_SCENES_DIR:
        scenes_candidates = [
            LOCAL_VLNCE_DATA_ROOT / "scene_datasets",
            LOCAL_MP3D_ROOT,
            LOCAL_VLNCE_DATA_ROOT / "scene_datasets" / "mp3d",
            Path("/home/intern/zhr/fjl/InternNav/data/scene_data"),
            Path("/home/intern/zhr/fjl/InternNav/data/scene_data/mp3d_ce"),
            Path.home() / "zhr/fjl/InternNav/data/scene_data",
            Path.home() / "zhr/fjl/InternNav/data/scene_data/mp3d_ce",
            Path.home() / "InternNav/data/scene_data",
            Path.home() / "InternNav/data/scene_data/mp3d_ce",
            Path("/dataset"),
            Path("/dataset/mp3d"),
            Path("/workspace/InternNav/data/scene_data"),
            Path("/workspace/InternNav/data/scene_data/mp3d_ce"),
        ]
        resolved = next(
            (p for p in scenes_candidates if _scene_root_is_compatible(p, first_scene_id)),
            None,
        )
        if resolved is None:
            attempted = "\n  - ".join([requested_scenes_dir, *map(str, scenes_candidates)])
            raise FileNotFoundError(
                "Could not find a scene root compatible with the dataset's "
                f"first scene_id={first_scene_id!r}. Tried:\n"
                f"  - {attempted}"
            )
        args.scenes_dir = str(resolved.resolve())
    else:
        expected_asset = _scene_asset_path(scenes_dir, first_scene_id)
        raise FileNotFoundError(
            "Configured --scenes_dir is incompatible with the dataset: "
            f"root={requested_scenes_dir!r}, first_scene_id={first_scene_id!r}, "
            f"expected_asset={str(expected_asset)!r}"
        )

    print(f"Using scenes_dir: {args.scenes_dir}")
    print(f"Using data_path:  {args.data_path}")
    if first_scene_id:
        print(f"Verified first scene asset: {Path(args.scenes_dir) / first_scene_id}")


def _extract_checkpoint_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Read a checkpoint and return the tensor state dict it contains."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")

    for key in ("model_state_dict", "trainable_state_dict", "state_dict"):
        state_dict = ckpt.get(key)
        if isinstance(state_dict, dict):
            return state_dict

    if all(torch.is_tensor(value) for value in ckpt.values()):
        return ckpt

    raise KeyError(f"Checkpoint does not contain model_state_dict/trainable_state_dict/state_dict: {checkpoint_path}")


def _extract_checkpoint_config(checkpoint_path: str | None) -> dict:
    if not checkpoint_path:
        return {}
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        return {}
    cfg = ckpt.get("config", {})
    return cfg if isinstance(cfg, dict) else {}


def _state_has_prefix(state_dict: dict[str, torch.Tensor] | None, prefix: str) -> bool:
    if not state_dict:
        return False
    return any(_normalize_state_key(key).startswith(prefix) for key in state_dict)


def _looks_action_only(state_dict: dict[str, torch.Tensor]) -> bool:
    if not state_dict:
        return False
    prefixes = {_normalize_state_key(key).split(".", 1)[0] for key in state_dict}
    return prefixes.issubset({"latent_queries", "nextdit_action_head"})


def _checkpoint_has_base_weights(state_dict: dict[str, torch.Tensor] | None) -> bool:
    return (
        _state_has_prefix(state_dict, "qwen2_5_vl.")
        or _state_has_prefix(state_dict, "qwen3_5.")
        or _state_has_prefix(state_dict, "qwen3_5_vl.")
        or _state_has_prefix(state_dict, "heatmap_vln.")
    )


def _requires_base_checkpoint(cfg: dict, checkpoint_cfg: dict | None = None) -> bool:
    for source in (checkpoint_cfg, cfg):
        if not isinstance(source, dict):
            continue
        stages = source.get("training", {}).get("stages", [])
        if not stages:
            continue
        stage_cfg = stages[0]
        if stage_cfg.get("requires_base_checkpoint") or stage_cfg.get("bridge_only"):
            return True
    return False


def _system2_sft_protocol(cfg: dict) -> str:
    return (cfg.get("data", {}).get("trajectory", {}).get("system2_sft_protocol", "direct")).lower()


def _preflight_checkpoint_args(args) -> None:
    cfg = load_config(args.config)
    checkpoint_cfg = _extract_checkpoint_config(args.checkpoint)

    if not args.base_checkpoint and checkpoint_cfg:
        recorded_base = checkpoint_cfg.get("runtime", {}).get("base_checkpoint")
        if recorded_base and Path(recorded_base).exists():
            args.base_checkpoint = str(Path(recorded_base).resolve())
            print(f"Auto-loading base checkpoint from Stage 2 metadata: {args.base_checkpoint}")
        elif recorded_base:
            print(f"WARNING: Stage 2 metadata records missing base checkpoint: {recorded_base}")

    if not _requires_base_checkpoint(cfg, checkpoint_cfg):
        return

    if args.base_checkpoint:
        return

    checkpoint_state_dict = _extract_checkpoint_state_dict(args.checkpoint) if args.checkpoint else None
    if not _checkpoint_has_base_weights(checkpoint_state_dict):
        raise ValueError(
            "This bridge-only config/checkpoint requires the Stage1-S2 panoramic System2 "
            "base checkpoint. Pass it with --base_checkpoint, or evaluate a checkpoint "
            "whose metadata records runtime.base_checkpoint."
        )


def _load_compatible_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    checkpoint_path: str,
    label: str,
) -> int:
    """Load matching tensors while handling DDP and legacy backbone prefixes."""
    current_state = model.state_dict()
    normalized_to_actual = {_normalize_state_key(name): name for name in current_state}

    remapped: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    skipped_missing: list[str] = []
    for name, value in state_dict.items():
        normalized_name = _normalize_state_key(name)
        actual_name = normalized_to_actual.get(normalized_name)
        if actual_name is None:
            skipped_missing.append(name)
            continue
        if current_state[actual_name].shape != value.shape:
            skipped_shape.append(
                f"{actual_name}: ckpt {tuple(value.shape)} vs model {tuple(current_state[actual_name].shape)}"
            )
            continue
        remapped[actual_name] = value

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(
        f"{label} loaded: {checkpoint_path} "
        f"(loaded={len(remapped)}/{len(state_dict)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if skipped_missing:
        sample = ", ".join(skipped_missing[:5])
        print(f"  skipped unmatched keys: {len(skipped_missing)}; examples: {sample}")
    if skipped_shape:
        print(f"  skipped shape-mismatched keys: {len(skipped_shape)}; examples: {'; '.join(skipped_shape[:3])}")
    return len(remapped)


def _prepare_progress_file(args, output_path: str) -> str:
    if args.resume and args.overwrite_output:
        raise ValueError("--resume and --overwrite_output cannot be used together")

    os.makedirs(output_path, exist_ok=True)
    progress_file = os.path.join(output_path, "progress.json")
    result_file = os.path.join(output_path, "result.json")

    if args.overwrite_output:
        for path in (progress_file, result_file):
            if os.path.exists(path):
                os.remove(path)
        return progress_file

    if os.path.exists(progress_file) and not args.resume:
        raise FileExistsError(
            f"Found existing progress file: {progress_file}. "
            "Pass --resume to continue it, --overwrite_output to start fresh, "
            "or choose a new --output_path."
        )

    return progress_file


def _load_progress(
    progress_file: str,
    *,
    expected_rpc_sampling_contract: dict[str, Any] | None = None,
) -> tuple[list[float], list[float], list[float], list[float], set]:
    sucs, spls, oss, nes = [], [], [], []
    done_set: set = set()
    if not os.path.exists(progress_file):
        return sucs, spls, oss, nes, done_set

    results_by_episode: OrderedDict[tuple[str, int], dict] = OrderedDict()
    loose_results: list[dict] = []
    with open(progress_file) as f:
        for line in f:
            if not line.strip():
                continue
            res = json.loads(line)
            if expected_rpc_sampling_contract is not None:
                validate_rpc_progress_sampling_contract(
                    res,
                    expected=expected_rpc_sampling_contract,
                )
            scene_id = res.get("scene_id")
            episode_id = res.get("episode_id")
            if expected_rpc_sampling_contract is not None and (scene_id is None or episode_id is None):
                raise ValueError("Deterministic RPC progress rows require scene_id and episode_id")
            if scene_id is None or episode_id is None:
                loose_results.append(res)
                continue
            key = (scene_id, int(episode_id))
            results_by_episode[key] = res

    for key, res in results_by_episode.items():
        done_set.add(key)
        sucs.append(res["success"])
        spls.append(res["spl"])
        oss.append(res["os"])
        nes.append(res["ne"])
    for res in loose_results:
        sucs.append(res["success"])
        spls.append(res["spl"])
        oss.append(res["os"])
        nes.append(res["ne"])

    return sucs, spls, oss, nes, done_set


def _load_closed_loop_progress_totals(progress_file: str) -> tuple[int, int]:
    if not os.path.exists(progress_file):
        return 0, 0
    rows: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()
    with open(progress_file) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "scene_id" not in row or "episode_id" not in row:
                continue
            rows[(str(row["scene_id"]), int(row["episode_id"]))] = row
    stop_probes = sum(int(row.get("closed_loop_stop_probes", 0) or 0) for row in rows.values())
    recoveries = 0
    for row in rows.values():
        value = row.get("closed_loop_recoveries", [])
        recoveries += len(value) if isinstance(value, list) else int(value or 0)
    return stop_probes, recoveries


def _load_episode_list(path: str) -> tuple[list[tuple[str, int]], set[tuple[str, int]]]:
    """Load fixed (scene_id, episode_id) pairs for apples-to-apples comparison."""
    keys, _ = load_episode_cohort(path)
    return keys, set(keys)


def _episode_list_from_args(args) -> tuple[list[tuple[str, int]] | None, set[tuple[str, int]] | None]:
    path = getattr(args, "episode_list", None)
    if not path:
        return None, None
    return _load_episode_list(path)


def _episode_metadata_from_args(
    args,
) -> dict[tuple[str, int], dict[str, Any]]:
    path = getattr(args, "episode_list", None)
    if not path:
        return {}
    _, metadata = load_episode_cohort(path)
    return metadata


def _eval_limit(
    args, remaining: int, target_list: list[tuple[str, int]] | None = None, done_set: set | None = None
) -> int:
    if target_list is not None:
        done = done_set or set()
        pending = sum(1 for key in target_list if key not in done)
        if args.max_episodes is None:
            return pending
        return min(pending, max(args.max_episodes, 0))
    if args.max_episodes is None:
        return remaining
    return min(remaining, max(args.max_episodes, 0))


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Habitat config
# ═══════════════════════════════════════════════════════════════════════


def build_habitat_config(args):
    cfg = get_habitat_default_config()
    cfg.defrost()

    cfg.DATASET.TYPE = "R2RVLN-v1"
    cfg.DATASET.SPLIT = "val_unseen"
    cfg.DATASET.SCENES_DIR = args.scenes_dir
    cfg.DATASET.DATA_PATH = args.data_path

    cfg.ENVIRONMENT.MAX_EPISODE_STEPS = 5000
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = 50000

    cfg.SIMULATOR.TYPE = "Sim-v0"
    cfg.SIMULATOR.ACTION_SPACE_CONFIG = "v1"
    cfg.SIMULATOR.FORWARD_STEP_SIZE = 0.25
    cfg.SIMULATOR.TURN_ANGLE = 15
    cfg.SIMULATOR.TILT_ANGLE = 15
    cfg.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.sim_gpu_id
    cfg.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = True

    cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    cfg.SIMULATOR.RGB_SENSOR.WIDTH = 640
    cfg.SIMULATOR.RGB_SENSOR.HEIGHT = 480
    cfg.SIMULATOR.RGB_SENSOR.HFOV = 79
    cfg.SIMULATOR.RGB_SENSOR.POSITION = [0, 1.25, 0]

    cfg.SIMULATOR.DEPTH_SENSOR.WIDTH = 640
    cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT = 480
    cfg.SIMULATOR.DEPTH_SENSOR.HFOV = 79
    cfg.SIMULATOR.DEPTH_SENSOR.POSITION = [0, 1.25, 0]
    cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.0
    cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 10.0
    cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = True

    cfg.TASK.TYPE = "VLN-v0"
    cfg.TASK.SUCCESS_DISTANCE = 3.0
    cfg.TASK.SENSORS = ["INSTRUCTION_SENSOR", "GPS_SENSOR", "COMPASS_SENSOR"]
    cfg.TASK.POSSIBLE_ACTIONS = [
        "STOP",
        "MOVE_FORWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
        "LOOK_UP",
        "LOOK_DOWN",
    ]
    cfg.TASK.MEASUREMENTS = [
        "DISTANCE_TO_GOAL",
        "SUCCESS",
        "SPL",
        "ORACLE_SUCCESS",
        "ORACLE_NAVIGATION_ERROR",
    ]

    cfg.TASK.DISTANCE_TO_GOAL.TYPE = "DistanceToGoal"
    cfg.TASK.DISTANCE_TO_GOAL.DISTANCE_TO = "POINT"
    cfg.TASK.SUCCESS.TYPE = "Success"
    cfg.TASK.SUCCESS.SUCCESS_DISTANCE = 3.0
    cfg.TASK.SPL.TYPE = "SPL"

    cfg.TASK.ORACLE_SUCCESS = CN()
    cfg.TASK.ORACLE_SUCCESS.TYPE = "OracleSuccess"
    cfg.TASK.ORACLE_NAVIGATION_ERROR = CN()
    cfg.TASK.ORACLE_NAVIGATION_ERROR.TYPE = "OracleNavigationError"

    cfg.TASK.ACTIONS.STOP.TYPE = "StopAction"
    cfg.TASK.ACTIONS.MOVE_FORWARD.TYPE = "MoveForwardAction"
    cfg.TASK.ACTIONS.TURN_LEFT.TYPE = "TurnLeftAction"
    cfg.TASK.ACTIONS.TURN_RIGHT.TYPE = "TurnRightAction"
    cfg.TASK.ACTIONS.LOOK_UP.TYPE = "LookUpAction"
    cfg.TASK.ACTIONS.LOOK_DOWN.TYPE = "LookDownAction"

    cfg.freeze()
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Agent pose tracking
# ═══════════════════════════════════════════════════════════════════════


def get_agent_cam2world(env) -> np.ndarray:
    """Extract a 4x4 camera-to-world matrix from the Habitat agent state.

    Habitat convention: Y-up, agent faces -Z.
    The returned matrix matches the format used by the training data
    (``compute_history_rel_poses`` / ``get_trajectory_relative_to_frame``).
    """
    sim = env._sim
    agent = sim.get_agent(0)
    state = agent.get_state()
    rot_mat = quaternion.as_rotation_matrix(state.rotation)  # 3×3
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot_mat
    T[:3, 3] = state.position
    return T


# ═══════════════════════════════════════════════════════════════════════
# Section 5: Panoramic view capture
# ═══════════════════════════════════════════════════════════════════════


def _yaw_quaternion(angle_rad: float):
    """Create a quaternion for rotation around Y-axis (up in Habitat)."""
    half = angle_rad / 2.0
    return np.quaternion(np.cos(half), 0.0, np.sin(half), 0.0)


def _quat_to_heading_deg(rot_xyzw: np.ndarray) -> float:
    """Convert Habitat quaternion [x,y,z,w] to compass heading in degrees.

    Habitat convention: Y-up, agent faces -Z in its local frame.
    Heading is the angle of the agent's forward vector (-Z) projected
    onto the XZ ground plane, measured from +Z (north) clockwise:
    heading = arctan2(forward_x, forward_z).
    """
    q = np.quaternion(float(rot_xyzw[3]), float(rot_xyzw[0]), float(rot_xyzw[1]), float(rot_xyzw[2]))
    rot_mat = quaternion.as_rotation_matrix(q)
    forward = rot_mat @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    heading_rad = np.arctan2(float(forward[0]), float(forward[2]))
    return float(np.degrees(heading_rad) % 360)


_ACTION_NAMES: dict[int, str] = {
    0: "STOP",
    1: "FORWARD",
    2: "LEFT",
    3: "RIGHT",
    4: "LOOKUP",
    5: "LOOKDOWN",
}


def _action_name(action: int) -> str:
    return _ACTION_NAMES.get(int(action), str(action))


def capture_panoramic_views(
    env,
    image_size: tuple = (256, 256),
) -> dict[str, Image.Image]:
    """Capture 4 directional views by manipulating agent state directly.

    This avoids env.step() calls so the episode step counter and metrics
    are not affected.  Only the agent's rotation is temporarily changed;
    position remains the same.
    """
    sim = env._sim
    agent = sim.get_agent(0)
    orig_state = agent.get_state()

    view_names = ["front", "right", "back", "left"]
    yaw_offsets = [0.0, -np.pi / 2, -np.pi, -3 * np.pi / 2]

    views: dict[str, Image.Image] = {}
    for name, yaw in zip(view_names, yaw_offsets):
        state = agent.get_state()
        if yaw != 0.0:
            state.rotation = orig_state.rotation * _yaw_quaternion(yaw)
            agent.set_state(state, reset_sensors=True)

        obs = sim.get_sensor_observations()
        rgb = _extract_rgb_array(obs)
        if rgb is not None:
            views[name] = _rgb_array_to_pil(rgb, image_size)
        else:
            global _panoramic_sensor_warned
            if not _panoramic_sensor_warned:
                available = sorted(obs.keys())
                print(
                    "WARNING: could not find RGB in sim sensor observations; "
                    f"using black placeholders. Available keys: {available}",
                    flush=True,
                )
                _panoramic_sensor_warned = True
            views[name] = Image.fromarray(np.zeros((*image_size[::-1], 3), dtype=np.uint8))

    agent.set_state(orig_state, reset_sensors=True)
    return views


def capture_lookdown_view(env, image_size: tuple = (224, 224)) -> Image.Image:
    """Look down 30° (2 × LOOKDOWN), capture RGB, then restore orientation.

    Unlike panoramic capture, this uses env.step() because the model was
    trained with exactly this procedure and the tilt sensor state must be
    consistent.
    """
    env.step(ActionCode.LOOKDOWN)
    obs = env.step(ActionCode.LOOKDOWN)

    rgb = _extract_rgb_array(obs)
    if rgb is None:
        rgb = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    lookdown_img = _rgb_array_to_pil(rgb, image_size)

    env.step(ActionCode.LOOKUP)
    env.step(ActionCode.LOOKUP)
    return lookdown_img


def _eval_image_sizes(train_cfg: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    """VLM/lookdown size and System1 traj image size from training config."""
    vlm_size = tuple(train_cfg["data"]["image_size"])
    traj_size = tuple(train_cfg["data"]["trajectory"].get("traj_image_size", [224, 224]))
    return vlm_size, traj_size


def _sample_history_panoramas(
    history_panoramas: list[dict[str, Image.Image]],
    num_history: int,
) -> list[dict[str, Image.Image]]:
    """Sample prompt history without mutating the full executed trajectory."""
    indices = _sample_history_indices(len(history_panoramas), num_history)
    return [history_panoramas[i] for i in indices]


def _sample_history_indices(available: int, num_history: int) -> list[int]:
    """Return the exact RPC history indices used by the System2 prompt."""
    if available <= 0 or num_history <= 0:
        return []
    if available <= num_history:
        return list(range(available))
    return np.unique(
        np.linspace(0, available - 1, num_history, dtype=np.int32)
    ).tolist()


def _condition_output_ids_for_pixel_goal(
    output_ids: torch.Tensor,
    prompt_len: int,
    tokenizer,
    pixel_goal: list[int],
    llm_output: str,
    coord_order: str = "generated",
    view_id: str | None = None,
    structured_output: bool = False,
) -> torch.Tensor:
    """Use System1-compatible coordinate text in latent conditioning."""
    parsed = parse_structured_pano_output(llm_output, image_size=None)
    use_structured = structured_output or parsed.kind == "pixel"
    if use_structured:
        resolved_view = (view_id or parsed.view_id or "front").lower()
        desired_text = structured_condition_text(resolved_view, pixel_goal)
        generated_text = (llm_output or "").strip()
        if generated_text == desired_text:
            return output_ids
        print(
            f"  [debug] System1 structured coordinate text: {desired_text!r}",
            flush=True,
        )
        replacement = tokenizer.encode(desired_text, add_special_tokens=False)
    else:
        coord = [int(c) for c in re.findall(r"\d+", llm_output or "")]
        if coord_order == "generated":
            desired = [int(pixel_goal[0]), int(pixel_goal[1])]
        elif coord_order == "internnav_yx":
            desired = [int(pixel_goal[1]), int(pixel_goal[0])]
        else:
            raise ValueError(f"Unsupported coord_order: {coord_order}")

        if len(coord) >= 2 and [coord[0], coord[1]] == desired:
            return output_ids

        coord_text = f"{desired[0]} {desired[1]}"
        print(
            f"  [debug] System1 coordinate text ({coord_order}): {coord_text}",
            flush=True,
        )
        replacement = tokenizer.encode(coord_text, add_special_tokens=False)

    if not replacement:
        return output_ids

    replacement_ids = torch.tensor(
        [replacement],
        device=output_ids.device,
        dtype=output_ids.dtype,
    )

    generated_suffix = output_ids[:, prompt_len:]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if (
        eos_token_id is not None
        and generated_suffix.numel() > 0
        and int(generated_suffix[0, -1].item()) == int(eos_token_id)
    ):
        eos = torch.tensor([[eos_token_id]], device=output_ids.device, dtype=output_ids.dtype)
        replacement_ids = torch.cat([replacement_ids, eos], dim=1)

    return torch.cat([output_ids[:, :prompt_len], replacement_ids], dim=1)


def _vlm_requests_stop(llm_output: str) -> bool:
    if vlm_output_requests_stop(llm_output):
        return True
    return ActionCode.STOP in parse_actions(llm_output, LEGACY_ACTIONS2IDX)


def _apply_habitat_action(env, action: int):
    """Execute one Habitat action; LOOKDOWN is issued twice like InternNav."""
    if action == ActionCode.LOOKDOWN:
        env.step(action)
        observations = env.step(action)
    else:
        observations = env.step(action)
    return observations, env.episode_over


def _lookdown_to_traj_tensor(
    lookdown_img: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    return torch.from_numpy(np.array(lookdown_img)).to(torch.bfloat16) / 255.0


def _metric_distance_to_goal(env) -> float | None:
    try:
        metrics = env.get_metrics()
    except Exception:
        return None
    value = metrics.get("distance_to_goal")
    if value is None:
        value = metrics.get("distance_to_target")
    if value is None:
        return None
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return None
    return distance if np.isfinite(distance) else None


def _system2_stop_rollout_label(
    distance_to_goal_m: float,
    *,
    positive_radius_m: float,
    negative_radius_m: float,
) -> int | None:
    """Map geodesic distance to a STOP label with an ignored margin."""
    distance = float(distance_to_goal_m)
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError(f"Invalid distance_to_goal for STOP collection: {distance}")
    if not 0.0 < positive_radius_m < negative_radius_m:
        raise ValueError(
            "STOP collection radii must satisfy 0 < positive < negative, got "
            f"{positive_radius_m}, {negative_radius_m}"
        )
    if distance <= positive_radius_m:
        return 1
    if distance >= negative_radius_m:
        return 0
    return None


def _debug_input_trace_enabled(args) -> bool:
    return bool(getattr(args, "debug_input_trace", True))


def _image_trace_summary(image: Image.Image) -> str:
    arr = np.asarray(image)
    if arr.size == 0:
        return "empty"
    digest = hashlib.sha1(arr.tobytes()).hexdigest()[:10]
    height, width = arr.shape[:2]
    return f"{width}x{height}:{digest}:mean={float(arr.mean()):.1f}:std={float(arr.std()):.1f}"


def _views_trace_summary(views: dict[str, Image.Image]) -> str:
    parts = []
    for name in ("front", "right", "back", "left"):
        image = views.get(name)
        if image is not None:
            parts.append(f"{name}={_image_trace_summary(image)}")
    return " ".join(parts)


def _agent_pose_summary(env) -> str:
    try:
        state = env._sim.get_agent(0).get_state()
        pos = np.asarray(state.position, dtype=np.float64)
        rot = quaternion.as_float_array(state.rotation)
    except Exception as exc:
        return f"pose=unavailable:{type(exc).__name__}"
    return f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) rot=({rot[0]:.4f},{rot[1]:.4f},{rot[2]:.4f},{rot[3]:.4f})"


def _env_trace_summary(env) -> str:
    distance = _metric_distance_to_goal(env)
    dist_text = "dist=NA" if distance is None else f"dist={distance:.3f}"
    return f"{_agent_pose_summary(env)} {dist_text}"


def _tensor_trace_summary(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "none"
    t = tensor.detach()
    if t.numel() == 0:
        return f"shape={tuple(t.shape)} empty"
    tf = t.float()
    return f"shape={tuple(t.shape)} mean={float(tf.mean().item()):.4f} std={float(tf.std(unbiased=False).item()):.4f}"


def _maybe_save_debug_images(
    args,
    scene_id: str,
    episode_id: int,
    call_idx: int,
    phase: str,
    images: dict[str, Image.Image],
) -> None:
    limit = int(getattr(args, "debug_save_input_images", 0) or 0)
    if limit <= 0 or call_idx > limit:
        return

    debug_dir = Path(args.output_path) / "debug_inputs" / f"{scene_id}_{episode_id:04d}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        image.save(debug_dir / f"{call_idx:04d}_{phase}_{name}.jpg")


# ── Trajectory Step Recorder (for offline HTML visualisation) ──────────


STOP_MULTIMODAL_EXAMPLE_SCHEMA = "heatmapvln-system2-stop-multimodal-example-v1"


class System2StopMultimodalRecorder:
    """Persist exact train-split System2 prompt images for offline LoRA training."""

    _VIEWS = ("front", "right", "back", "left")

    def __init__(
        self,
        output_dir: Path,
        *,
        dataset_split: str,
        jpeg_quality: int,
        regular_min_stop_log_odds: float | None = None,
    ) -> None:
        if dataset_split != "train":
            raise ValueError(
                "Multimodal STOP examples may only be collected from the train split"
            )
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError(f"Invalid multimodal STOP JPEG quality: {jpeg_quality}")
        self.output_dir = Path(output_dir)
        self.examples_dir = self.output_dir / "system2_stop_multimodal_examples"
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.labels_path = self.output_dir / "system2_stop_multimodal_examples.jsonl"
        self.dataset_split = dataset_split
        self.jpeg_quality = int(jpeg_quality)
        if regular_min_stop_log_odds is not None and not math.isfinite(
            float(regular_min_stop_log_odds)
        ):
            raise ValueError("Multimodal regular-negative threshold must be finite")
        self.regular_min_stop_log_odds = (
            float(regular_min_stop_log_odds)
            if regular_min_stop_log_odds is not None
            else None
        )
        self.considered = 0
        self.recorded = 0
        self.skipped = 0
        self.provenance_fallbacks = 0
        self._recorded_episode_keys: set[tuple[str, int]] = set()
        self.collection_namespace = hashlib.sha256(
            str(self.output_dir.expanduser().resolve()).encode("utf-8")
        ).hexdigest()[:12]

    def _key(
        self,
        scene_id: str,
        episode_id: int,
        call_index: int,
        protocol_seed: int,
    ) -> str:
        return (
            f"src{self.collection_namespace}_{scene_id}_ep{int(episode_id):06d}_"
            f"call{int(call_index):05d}_seed{int(protocol_seed)}"
        )

    def _view_paths(
        self,
        scene_id: str,
        episode_id: int,
        call_index: int,
        protocol_seed: int,
    ) -> dict[str, str]:
        key = self._key(scene_id, episode_id, call_index, protocol_seed)
        return {
            view: str(Path("system2_stop_multimodal_examples") / key / f"{view}.jpg")
            for view in self._VIEWS
        }

    def _history_view_paths(self, key: str, history_index: int) -> dict[str, str]:
        return {
            view: str(
                Path("system2_stop_multimodal_examples")
                / key
                / f"history_{int(history_index):02d}_{view}.jpg"
            )
            for view in self._VIEWS
        }

    def _write_views(
        self,
        relative_paths: dict[str, str],
        images: dict[str, Image.Image],
        *,
        key: str,
    ) -> None:
        from vla_rpc.core.image import encode_rgb_to_jpeg

        for view in self._VIEWS:
            image = images.get(view)
            if not isinstance(image, Image.Image):
                raise ValueError(f"Missing {view!r} image for STOP example {key}")
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            jpeg = encode_rgb_to_jpeg(array, quality=self.jpeg_quality)
            destination = self.output_dir / relative_paths[view]
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as handle:
                handle.write(jpeg)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)

    def record(
        self,
        *,
        scene_id: str,
        episode_id: int,
        system2_call_index: int,
        protocol_seed: int,
        instruction: str,
        current_views: dict[str, Image.Image],
        history_views: list[dict[str, Image.Image]],
        history_source_indices: list[int],
        distance_to_goal_m: float,
        stop_target: int | None,
        response: dict[str, Any],
        image_size: tuple[int, int],
        oracle_recovery_active: bool,
    ) -> dict[str, Any] | None:
        self.considered += 1
        stop_policy = response.get("system2_stop_head")
        original_output = (
            stop_policy.get("original_output")
            if isinstance(stop_policy, dict) and stop_policy.get("original_output")
            else response.get("llm_output", "")
        )
        original_terminal = bool(_vlm_requests_stop(str(original_output)))
        decision_scores = response.get("system2_decision_scores")
        stop_log_odds = (
            decision_scores.get("stop_log_odds")
            if isinstance(decision_scores, dict)
            else None
        )
        episode_key = (str(scene_id), int(episode_id))
        episode_has_record = episode_key in self._recorded_episode_keys
        if not should_record_stop_multimodal_example(
            rollout_label=stop_target,
            original_terminal=original_terminal,
            stop_log_odds=stop_log_odds,
            regular_min_stop_log_odds=self.regular_min_stop_log_odds,
            episode_has_record=episode_has_record,
        ):
            self.skipped += 1
            return None
        episode_provenance_fallback = bool(
            self.regular_min_stop_log_odds is not None
            and not episode_has_record
            and stop_target == 0
            and not original_terminal
            and float(stop_log_odds) <= self.regular_min_stop_log_odds
        )

        key = self._key(scene_id, episode_id, system2_call_index, protocol_seed)
        relative_views = self._view_paths(
            scene_id, episode_id, system2_call_index, protocol_seed
        )
        self._write_views(relative_views, current_views, key=key)
        relative_history_views = []
        for history_index, images in enumerate(history_views):
            paths = self._history_view_paths(key, history_index)
            self._write_views(paths, images, key=f"{key}/history_{history_index:02d}")
            relative_history_views.append(paths)
        row = {
            "schema": STOP_MULTIMODAL_EXAMPLE_SCHEMA,
            "key": key,
            "dataset_split": self.dataset_split,
            "scene_id": str(scene_id),
            "episode_id": int(episode_id),
            "system2_call_index": int(system2_call_index),
            "protocol_seed": int(protocol_seed),
            "collection_namespace": self.collection_namespace,
            "instruction": str(instruction),
            "distance_to_goal_m": float(distance_to_goal_m),
            "stop_target": int(stop_target) if stop_target in (0, 1) else None,
            "original_terminal": original_terminal,
            "original_output": str(original_output),
            "effective_output": str(response.get("llm_output", "")),
            "system2_decision_scores": response.get("system2_decision_scores"),
            "system2_stop_policy": stop_policy,
            "deterministic_sampling": response.get(HEATMAPVLN_RPC_SAMPLING_FIELD),
            "current_views": relative_views,
            "history_source_buffer_indices": [
                int(index) for index in history_source_indices
            ],
            "history_views": relative_history_views,
            "image_size": [int(image_size[0]), int(image_size[1])],
            "jpeg_quality": self.jpeg_quality,
            "oracle_recovery_active": bool(oracle_recovery_active),
            "privileged_offline_label": True,
            "regular_min_stop_log_odds": self.regular_min_stop_log_odds,
            "episode_provenance_fallback": episode_provenance_fallback,
        }
        with self.labels_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._recorded_episode_keys.add(episode_key)
        self.recorded += 1
        self.provenance_fallbacks += int(episode_provenance_fallback)
        return row


class TrajectoryStepRecorder:
    """Save per-step agent state, VLM outputs, and panorama images to disk.

    Produces a ``trajectory_steps.json`` alongside per-step JPEG images in a
    subdirectory named ``<scene_id>_<episode_id:04d>``.  The companion script
    ``scripts/visualization/generate_trajectory_html.py`` reads this output
    and renders a self-contained HTML inspection page.
    """

    def __init__(self, output_dir: Path, scene_id: str, episode_id: int) -> None:
        self._out = Path(output_dir) / f"{scene_id}_{int(episode_id):04d}"
        self._out.mkdir(parents=True, exist_ok=True)
        self._meta: dict[str, Any] = {}
        self._steps: list[dict[str, Any]] = []
        self._prev_dist: float | None = None

    def set_metadata(
        self,
        *,
        instruction: str,
        start_pos: list[float],
        start_rot: list[float],
        goal_pos: list[float],
        gt_path: list[list[float]],
    ) -> None:
        self._meta = {
            "scene_id": "",
            "episode_id": -1,
            "instruction": instruction,
            "start_position": list(start_pos),
            "start_heading_deg": _quat_to_heading_deg(np.array(start_rot)),
            "goal_position": list(goal_pos),
            "gt_reference_path": [[float(v) for v in p] for p in gt_path],
        }

    def record_step(self, data: dict[str, Any]) -> None:
        step: dict[str, Any] = {
            "step_id": int(data.get("step_id", len(self._steps))),
            "phase": str(data.get("phase", "unknown")),
            "position": [float(v) for v in data["position"]],
            "heading_deg": float(data.get("heading_deg", 0.0)),
            "rotation": [float(v) for v in data.get("rotation", [0, 0, 0, 1])],
            "distance_to_goal": float(data["distance_to_goal"]) if data.get("distance_to_goal") is not None else None,
        }

        cur_dist = step["distance_to_goal"]
        delta: float | None = None
        if self._prev_dist is not None and cur_dist is not None:
            delta = cur_dist - self._prev_dist  # positive = moving away
        self._prev_dist = cur_dist
        step["delta_dist"] = delta

        # Text / prediction fields (only for VLM steps).
        for key in ("vlm_output", "pano_goal_view"):
            val = data.get(key)
            step[key] = str(val) if val is not None else None
        pg = data.get("pixel_goal")
        step["pixel_goal"] = [int(pg[0]), int(pg[1])] if pg and len(pg) >= 2 else None
        oracle_system2 = data.get("oracle_system2")
        if isinstance(oracle_system2, dict):
            step["oracle_system2"] = {
                key: oracle_system2.get(key)
                for key in (
                    "text",
                    "view",
                    "pixel_goal",
                    "target_position",
                    "heading_delta_deg",
                    "bearing_deg",
                    "offpath_m",
                    "path_progress_m",
                    "target_progress_m",
                    "lookahead_m",
                )
                if key in oracle_system2
            }
        else:
            step["oracle_system2"] = None
        for num_key in ("traj_hs_total_norm",):
            val = data.get(num_key)
            step[num_key] = float(val) if val is not None else None
        per_q = data.get("traj_hs_per_query")
        step["traj_hs_per_query"] = [float(v) for v in per_q] if per_q is not None else None

        # Action fields.
        step["executed_action"] = int(data["executed_action"]) if data.get("executed_action") is not None else None
        step["executed_action_name"] = (
            _action_name(data["executed_action"]) if data.get("executed_action") is not None else None
        )

        # Save panorama images as JPEG files.
        panorama: dict[str, str] = {}
        current_views = data.get("current_views")
        if isinstance(current_views, dict):
            for view_name in ("front", "right", "back", "left"):
                img = current_views.get(view_name)
                if isinstance(img, Image.Image):
                    fname = f"step_{len(self._steps):04d}_{view_name}.jpg"
                    # Downsize for storage efficiency.
                    thumb = img.copy()
                    thumb.thumbnail((256, 256))
                    thumb.save(self._out / fname, "JPEG", quality=65)
                    panorama[view_name] = fname
        step["panorama"] = panorama

        self._steps.append(step)

    def finalize(
        self,
        *,
        scene_id: str,
        episode_id: int,
        success: float,
        spl: float,
        total_steps: int,
        vlm_calls: int = 0,
        traj_calls: int = 0,
    ) -> None:
        self._meta["scene_id"] = str(scene_id)
        self._meta["episode_id"] = int(episode_id)
        self._meta["success"] = bool(success)
        self._meta["spl"] = float(spl)
        self._meta["total_steps"] = int(total_steps)
        self._meta["vlm_calls"] = int(vlm_calls)
        self._meta["trajectory_calls"] = int(traj_calls)
        payload: dict[str, Any] = {"metadata": self._meta, "steps": self._steps}
        with (self._out / "trajectory_steps.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def _record_post_action_step(
    step_recorder: TrajectoryStepRecorder | None,
    env,
    *,
    step_id: int,
    phase: str,
    action: int,
    image_size: tuple[int, int],
    vlm_output: str | None = None,
) -> None:
    """Record the state after a Habitat action has been executed."""
    if step_recorder is None:
        return

    state = env._sim.get_agent(0).get_state()
    pos = np.array(state.position, dtype=float)
    rot = quaternion.as_float_array(state.rotation)
    step_data: dict[str, Any] = {
        "step_id": int(step_id),
        "phase": phase,
        "position": pos,
        "heading_deg": _quat_to_heading_deg(rot),
        "rotation": rot,
        "distance_to_goal": _metric_distance_to_goal(env),
        "executed_action": int(action),
        "current_views": capture_panoramic_views(env, image_size=image_size),
    }
    if vlm_output is not None:
        step_data["vlm_output"] = vlm_output
    step_recorder.record_step(step_data)


def _maybe_stop_at_success(env, args, step_id: int):
    stop_distance = float(getattr(args, "auto_stop_distance", 0.0) or 0.0)
    if stop_distance <= 0.0:
        return None

    distance = _metric_distance_to_goal(env)
    if distance is None or distance > stop_distance:
        return None

    print(
        f"  [debug] auto STOP: distance_to_goal={distance:.3f} <= {stop_distance:.3f}",
        flush=True,
    )
    observations, done = _apply_habitat_action(env, ActionCode.STOP)
    return observations, done, step_id + 1


def _trajectory_debug_summary(
    trajectory: torch.Tensor,
    num_sample_trajs: int,
    action_scale: float,
) -> str:
    if trajectory is None or trajectory.numel() == 0:
        return "trajectory=empty"

    trajs = trajectory[:num_sample_trajs].float().detach().cpu().numpy().copy()
    if trajs.ndim != 3 or trajs.shape[-1] < 2:
        return f"trajectory_shape={tuple(trajectory.shape)}"

    trajs[:, :, :2] /= float(action_scale)
    cumsum_xy = np.cumsum(trajs[:, :, :2], axis=1)
    xy = np.concatenate(
        [np.zeros((trajs.shape[0], 1, 2), dtype=cumsum_xy.dtype), cumsum_xy],
        axis=1,
    )
    mean_xy = xy.mean(axis=0)
    goal_xy = mean_xy[-1]
    direct = float(np.linalg.norm(goal_xy))
    path_len = float(np.linalg.norm(np.diff(mean_xy, axis=0), axis=1).sum())
    return f"traj_goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f}), direct={direct:.2f}, path_len={path_len:.2f}"


def _system1_coord_order(args, *, panoramic_internnav_protocol: bool) -> str:
    requested = getattr(args, "system1_coord_order", "auto")
    if requested != "auto":
        return requested
    # Stage2 bridge-only training tokenizes HeatmapVLN pixel_goal as [u v]
    # before appending TRAJ latent-query tokens.  Auto must therefore match
    # the generated text, even when System2 uses the InternNav two-turn
    # protocol.  Use --system1_coord_order internnav_yx only for raw
    # InternNav hidden-state compatibility experiments.
    return "generated"


# ═══════════════════════════════════════════════════════════════════════
# Section 6: Trajectory → discrete actions conversion
# ═══════════════════════════════════════════════════════════════════════

TRAJECTORY_SELECTION_CHOICES = (
    "mean",
    "endpoint_medoid",
    "path_medoid",
    "median_endpoint_nearest",
    "forward_or_medoid",
    "longest_forward",
)
TRAJECTORY_HEADING_ALIGNMENT_CHOICES = ("none", "pano_pixel")


def reconstruct_xy_from_delta(delta_xyt: np.ndarray) -> np.ndarray:
    start_xy = np.zeros((len(delta_xyt), 2))
    delta_xy = delta_xyt[:, :, :2]
    cumsum_xy = np.cumsum(delta_xy, axis=1)

    batch_size = delta_xyt.shape[0]
    steps = delta_xyt.shape[1]
    xy = np.zeros((batch_size, steps + 1, 2))
    xy[:, 0] = start_xy
    xy[:, 1:] = start_xy[:, None, :] + cumsum_xy
    return xy


def trajectory_xy_path_len(trajectory: np.ndarray) -> float:
    if trajectory.ndim != 2 or trajectory.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1).sum())


def _trajectory_to_discrete_actions_close_to_goal(
    trajectory: np.ndarray,
    step_size: float = 0.25,
    turn_angle_deg: float = 15,
    lookahead: int = 4,
) -> list[int]:
    actions: list[int] = []
    yaw = 0.0
    pos = trajectory[0]
    turn_angle_rad = np.deg2rad(turn_angle_deg)
    goal = trajectory[-1]

    def normalize_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    while np.linalg.norm(pos - goal) > 0.2:
        dists = np.linalg.norm(trajectory - pos, axis=1)
        nearest_idx = np.argmin(dists)
        target_idx = min(nearest_idx + lookahead, len(trajectory) - 1)
        target = trajectory[target_idx]
        target_dir = target - pos
        if np.linalg.norm(target_dir) < 1e-6:
            break

        target_yaw = np.arctan2(target_dir[1], target_dir[0])
        delta_yaw = normalize_angle(target_yaw - yaw)
        n_turns = round(delta_yaw / turn_angle_rad)
        if n_turns > 0:
            actions += [ActionCode.LEFT] * n_turns
        elif n_turns < 0:
            actions += [ActionCode.RIGHT] * (-n_turns)
        yaw = normalize_angle(yaw + n_turns * turn_angle_rad)

        next_pos = pos + step_size * np.array([np.cos(yaw), np.sin(yaw)])
        if np.linalg.norm(next_pos - goal) > np.linalg.norm(pos - goal):
            break

        actions.append(ActionCode.FORWARD)
        pos = next_pos

    return actions


def _endpoint_medoid_index(all_trajectory: np.ndarray) -> int:
    endpoints = all_trajectory[:, -1, :2]
    dists = np.linalg.norm(endpoints[:, None, :] - endpoints[None, :, :], axis=-1)
    return int(np.argmin(dists.sum(axis=1)))


def _path_medoid_index(all_trajectory: np.ndarray) -> int:
    flat = all_trajectory[:, :, :2].reshape(all_trajectory.shape[0], -1)
    dists = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    return int(np.argmin(dists.sum(axis=1)))


def _median_endpoint_nearest_index(all_trajectory: np.ndarray) -> int:
    endpoints = all_trajectory[:, -1, :2]
    median_endpoint = np.median(endpoints, axis=0)
    return int(np.argmin(np.linalg.norm(endpoints - median_endpoint[None, :], axis=-1)))


def _forward_candidate_stats(all_trajectory: np.ndarray) -> list[tuple[int, int, float, list[int]]]:
    candidates: list[tuple[int, int, float, list[int]]] = []
    for idx, trajectory in enumerate(all_trajectory):
        actions = _trajectory_to_discrete_actions_close_to_goal(trajectory)
        forward_count = sum(1 for action in actions if action == ActionCode.FORWARD)
        if forward_count <= 0:
            continue
        candidates.append((idx, forward_count, trajectory_xy_path_len(trajectory), actions))
    return candidates


def select_trajectory_xy(
    all_trajectory: np.ndarray,
    selection: str = "mean",
) -> tuple[np.ndarray, int | None]:
    """Select one XY trajectory from parallel diffusion samples.

    Returns ``(trajectory, selected_index)``.  ``selected_index`` is ``None``
    for the mean trajectory because it is not an original diffusion sample.
    """
    if all_trajectory.ndim != 3 or all_trajectory.shape[0] == 0:
        raise ValueError(f"Expected all_trajectory shape (B,T,2), got {all_trajectory.shape}")

    if selection == "mean":
        return np.mean(all_trajectory, axis=0), None
    if selection == "endpoint_medoid":
        idx = _endpoint_medoid_index(all_trajectory)
        return all_trajectory[idx], idx
    if selection == "path_medoid":
        idx = _path_medoid_index(all_trajectory)
        return all_trajectory[idx], idx
    if selection == "median_endpoint_nearest":
        idx = _median_endpoint_nearest_index(all_trajectory)
        return all_trajectory[idx], idx

    forward_candidates = _forward_candidate_stats(all_trajectory)
    if selection == "forward_or_medoid":
        if forward_candidates:
            medoid_idx = _endpoint_medoid_index(all_trajectory)
            medoid_endpoint = all_trajectory[medoid_idx, -1, :2]
            median_path_len = float(np.median([trajectory_xy_path_len(traj) for traj in all_trajectory]))

            def score(item: tuple[int, int, float, list[int]]) -> tuple[float, int, float]:
                idx, forward_count, path_len, _actions = item
                endpoint_dist = float(np.linalg.norm(all_trajectory[idx, -1, :2] - medoid_endpoint))
                return (endpoint_dist, -forward_count, abs(path_len - median_path_len))

            idx = min(forward_candidates, key=score)[0]
            return all_trajectory[idx], idx
        idx = _endpoint_medoid_index(all_trajectory)
        return all_trajectory[idx], idx

    if selection == "longest_forward":
        if forward_candidates:
            idx = max(forward_candidates, key=lambda item: (item[2], item[1]))[0]
            return all_trajectory[idx], idx
        idx = _endpoint_medoid_index(all_trajectory)
        return all_trajectory[idx], idx

    raise ValueError(f"Unsupported trajectory selection: {selection}; expected one of {TRAJECTORY_SELECTION_CHOICES}")


def traj_to_actions(
    dp_actions: torch.Tensor,
    num_sample_trajs: int = 32,
    action_scale: float = 4.0,
    trajectory_selection: str = "mean",
    trajectory_x_sign: float = 1.0,
) -> list[int]:
    """Convert InternNav trajectory predictions to discrete Habitat actions."""
    if trajectory_x_sign not in (-1.0, 1.0):
        raise ValueError(f"trajectory_x_sign must be -1 or 1, got {trajectory_x_sign}")
    trajs = dp_actions[:num_sample_trajs].float().detach().cpu().numpy().copy()
    trajs[:, :, :2] /= action_scale
    trajs[:, :, 0] *= trajectory_x_sign
    all_trajectory = reconstruct_xy_from_delta(trajs)
    trajectory, _selected_idx = select_trajectory_xy(all_trajectory, trajectory_selection)
    actions = _trajectory_to_discrete_actions_close_to_goal(trajectory)
    return actions if actions else [ActionCode.STOP]


# ═══════════════════════════════════════════════════════════════════════
# Section 7: VLM input preparation
# ═══════════════════════════════════════════════════════════════════════


def _normalize_multimodal_inputs(inputs: dict[str, torch.Tensor]):
    """Replicate HeatmapVLN._normalize_multimodal_inputs."""
    if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
        vgt = inputs["video_grid_thw"]
        if vgt.shape[0] > 0 and vgt[:, 0].max() > 1:
            inputs["video_grid_thw"] = torch.repeat_interleave(
                vgt,
                vgt[:, 0],
                dim=0,
            )
            inputs["video_grid_thw"][:, 0] = 1


def prepare_vlm_inputs(
    processor,
    current_views: dict[str, Image.Image],
    history_panoramas: list[dict[str, Image.Image]],
    instruction: str,
    device: torch.device,
    internnav_protocol: bool = False,
) -> dict[str, torch.Tensor]:
    """Build tokenised Qwen2.5-VL inputs from panoramic observations.

    Uses ``construct_input`` with a dummy ``pixel_goal`` to include the
    waypoint-coordinate prompt, then strips the teacher-forcing assistant
    response so the model generates the coordinates autoregressively.
    """
    messages = construct_input(
        current_views=current_views,
        history_panoramas=history_panoramas,
        instruction=instruction,
        pixel_goal=[0, 0],
        internnav_protocol=internnav_protocol,
    )
    messages = [m for m in messages if m["role"] != "assistant"]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    _normalize_multimodal_inputs(inputs)
    return inputs


# ═══════════════════════════════════════════════════════════════════════
# Section 8: Model building
# ═══════════════════════════════════════════════════════════════════════


def _resolve_internnav_model_path(cfg: dict) -> str:
    raw = (
        os.environ.get("INTERNNAV_MODEL_PATH")
        or os.environ.get("INTERNNAV_BACKBONE")
        or cfg.get("model", {}).get("action_head", {}).get("nextdit", {}).get("internnav_model_path", "")
        or cfg.get("model", {}).get("llm", {}).get("model_path", "")
    )
    resolved = os.path.expandvars(os.path.expanduser(str(raw or "").strip()))
    if (not resolved or resolved.startswith("$")) and LOCAL_INTERNNAV_MODEL_PATH.exists():
        resolved = str(LOCAL_INTERNNAV_MODEL_PATH.resolve())
    return resolved


def _verify_internnav_system1_loaded(model: torch.nn.Module, internnav_path: str) -> None:
    """Fail fast when NextDiT System1 was not loaded from InternNav safetensors."""
    head = getattr(model, "nextdit_action_head", None)
    if head is None:
        return

    unresolved = internnav_path.startswith("$") or not internnav_path
    if unresolved:
        raise RuntimeError(
            "INTERNNAV_MODEL_PATH is not set (config still has an unresolved placeholder). "
            "Stage2 eval requires InternNav System1 weights for traj_dit/rgb_model; "
            "export INTERNNAV_MODEL_PATH=/path/to/InternNav_Model before running."
        )

    model_dir = Path(internnav_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"INTERNNAV_MODEL_PATH does not exist: {internnav_path}")

    from safetensors import safe_open

    checks = (
        (
            "model.traj_dit.model.layers.0.attn1.to_q.weight",
            lambda: head.traj_dit.state_dict()["model.layers.0.attn1.to_q.weight"],
        ),
        (
            "model.rgb_model.patch_embed.proj.weight",
            lambda: head.rgb_model.patch_embed.proj.weight,
        ),
    )

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        import json as _json

        weight_map = _json.loads(index_path.read_text()).get("weight_map", {})
    else:
        single = model_dir / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError(f"No InternNav safetensors found under {internnav_path}")
        weight_map = {k: single.name for k in safe_open(str(single)).keys()}  # noqa: SIM118

    shards: dict[str, object] = {}

    def _tensor_from_internnav(ref_key: str) -> torch.Tensor:
        shard_name = weight_map.get(ref_key)
        if shard_name is None:
            raise RuntimeError(f"{ref_key} missing from {index_path or model_dir}")
        shard_path = model_dir / shard_name
        if shard_name not in shards:
            shards[shard_name] = safe_open(str(shard_path), framework="pt", device="cpu")
        handle = shards[shard_name]
        if ref_key not in handle.keys():  # noqa: SIM118
            raise RuntimeError(f"{ref_key} missing from {shard_path}")
        return handle.get_tensor(ref_key).float()

    rgb_key_count = sum(1 for key in weight_map if key.startswith("model.rgb_model."))
    if rgb_key_count == 0:
        raise RuntimeError(f"No model.rgb_model.* tensors in InternNav weights at {internnav_path}")

    for ref_key, current_fn in checks:
        reference = _tensor_from_internnav(ref_key)
        current = current_fn().detach().float().cpu()
        if current.shape != reference.shape or not torch.allclose(current, reference, atol=1e-4, rtol=1e-3):
            raise RuntimeError(
                "InternNav System1 weights were not loaded into NextDiT "
                f"(mismatch on {ref_key}). Check INTERNNAV_MODEL_PATH."
            )

    print(
        f"Verified InternNav System1 from {internnav_path}: "
        f"traj_dit + DepthAnythingV2 rgb_model ({rgb_key_count} rgb tensors in checkpoint)",
        flush=True,
    )


def load_model(args, device: torch.device):
    """Build VLNPipeline, initialise lazy modules, then load checkpoints."""
    from scripts.training.model_builder import build_model

    cfg = load_config(args.config)
    internnav_path = _resolve_internnav_model_path(cfg)
    if internnav_path:
        print(f"InternNav model path: {internnav_path}", flush=True)

    model = build_model(cfg, device=str(device), verbose=False)

    model = model.to(device)
    _verify_internnav_system1_loaded(model, internnav_path)

    checkpoint_cfg = _extract_checkpoint_config(args.checkpoint)
    if not args.base_checkpoint and checkpoint_cfg:
        recorded_base = checkpoint_cfg.get("runtime", {}).get("base_checkpoint")
        if recorded_base and Path(recorded_base).exists():
            args.base_checkpoint = str(Path(recorded_base).resolve())
            print(f"Auto-loading base checkpoint from Stage 2 metadata: {args.base_checkpoint}")
        elif recorded_base:
            print(f"WARNING: Stage 2 metadata records missing base checkpoint: {recorded_base}")

    base_state_dict = None
    if args.base_checkpoint:
        base_state_dict = _extract_checkpoint_state_dict(args.base_checkpoint)
    checkpoint_state_dict = _extract_checkpoint_state_dict(args.checkpoint) if args.checkpoint else None

    if (
        _requires_base_checkpoint(cfg, checkpoint_cfg)
        and not args.base_checkpoint
        and not _checkpoint_has_base_weights(checkpoint_state_dict)
    ):
        raise ValueError(
            "This bridge-only config/checkpoint requires the Stage1-S2 panoramic System2 "
            "base checkpoint. Pass it with --base_checkpoint, or evaluate a checkpoint "
            "whose metadata records runtime.base_checkpoint."
        )
    if checkpoint_state_dict and _looks_action_only(checkpoint_state_dict) and not args.base_checkpoint:
        print("WARNING: the main checkpoint contains only action-head weights and no base checkpoint was loaded.")
    if not args.base_checkpoint and checkpoint_state_dict is None:
        print(
            "WARNING: no checkpoint was supplied; evaluating the model initialized from config/pretrained weights only."
        )

    # Qwen/LoRA is lazy.  It must exist before loading Stage 1 LoRA weights;
    # otherwise qwen*.model.* keys are silently treated as unexpected.
    model.qwen2_5_vl._load_model()

    if _state_has_prefix(base_state_dict, "heatmap_vln.") or _state_has_prefix(checkpoint_state_dict, "heatmap_vln."):
        model._ensure_heatmap_vln()

    if base_state_dict:
        _load_compatible_state_dict(
            model,
            base_state_dict,
            args.base_checkpoint,
            label="Base checkpoint",
        )
    if checkpoint_state_dict:
        _load_compatible_state_dict(
            model,
            checkpoint_state_dict,
            args.checkpoint,
            label="Main checkpoint",
        )

    del checkpoint_state_dict
    del base_state_dict
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model.eval()
    return model, cfg


def _run_eval_panoramic_vlm(
    args,
    model,
    train_cfg: dict,
    processor,
    device: torch.device,
    action_scale: float,
    num_sample_trajs: int,
    has_nextdit: bool,
    pano_latent_adapter=None,
    force_teacher_model=None,
    force_teacher_processor=None,
    force_teacher_device: torch.device | None = None,
) -> None:
    """Closed-loop Habitat evaluation for checkpoints trained with panoramic VLM input."""
    if pano_latent_adapter is not None:
        print(
            "[panoramic-eval] pano-to-InternNav latent adapter is ACTIVE; "
            "generate_latents output will be projected before NextDiT."
        )
    if force_teacher_model is not None:
        print(
            "[panoramic-eval] --force_teacher_coord is ACTIVE; student VLM coord "
            "will be REPLACED with InternNav teacher's coord at each System2 call."
        )
    hab_cfg = build_habitat_config(args)
    print("Creating Habitat environment ...")
    env = habitat.Env(config=hab_cfg)
    num_episodes = len(list(env.episodes))
    print(f"Total episodes: {num_episodes}")

    vlm_image_size, traj_image_size = _eval_image_sizes(train_cfg)
    image_size = vlm_image_size
    num_history = args.num_history
    max_steps_per_episode = args.max_steps_per_episode
    internnav_protocol = _system2_sft_protocol(train_cfg) == "internnav"
    structured_pano_output = bool(train_cfg.get("data", {}).get("trajectory", {}).get("structured_pano_output", True))
    system1_coord_order = _system1_coord_order(
        args,
        panoramic_internnav_protocol=internnav_protocol,
    )
    print(f"System2 SFT protocol: {'internnav' if internnav_protocol else 'direct'}")
    print(f"structured_pano_output={structured_pano_output}")
    print(f"vlm_image_size={vlm_image_size}, traj_image_size={traj_image_size}")
    print(f"System1 coordinate text order: {system1_coord_order}")

    output_path = args.output_path
    progress_file = _prepare_progress_file(args, output_path)
    sucs, spls, oss, nes, done_set = _load_progress(progress_file)

    target_list, target_set = _episode_list_from_args(args)
    if target_list is not None:
        selected = restrict_habitat_env_to_episode_keys(env, target_list)
        print(f"Fixed episode list ({len(target_list)}): {args.episode_list}")
        print(f"Restricted Habitat iterator to {len(selected)} requested episodes")
    remaining = num_episodes - len(done_set)
    eval_limit = _eval_limit(args, remaining, target_list, done_set)
    print(f"Episodes already done: {len(done_set)}, remaining: {remaining}, this run: {eval_limit}")

    process_bar = tqdm.tqdm(total=eval_limit, desc="Evaluating", ncols=120)
    seen_episodes: set = set()
    eval_count = 0

    while True:
        process_bar.set_postfix(
            SR=f"{float(np.mean(sucs)):.3f}" if sucs else "?",
            SPL=f"{float(np.mean(spls)):.3f}" if spls else "?",
        )
        if eval_count >= eval_limit:
            break

        observations = env.reset()
        episode = env.current_episode
        scene_id = episode.scene_id.split("/")[-2]
        episode_id = int(episode.episode_id)
        ep_key = (scene_id, episode_id)

        if ep_key in seen_episodes:
            break
        seen_episodes.add(ep_key)

        if target_set is not None and ep_key not in target_set:
            continue

        if ep_key in done_set:
            continue

        instruction = _normalize_instruction(episode.instruction.instruction_text)
        eval_count += 1
        print(f"\n[{eval_count}/{eval_limit}] Episode {scene_id}_{episode_id:04d}: {instruction[:80]}...")

        executed_history_panoramas: list[dict[str, Image.Image]] = []
        action_seq: list[int] = []
        local_actions: list[int] = []
        pix_goal_image: torch.Tensor | None = None
        _last_traj_hs: torch.Tensor | None = None
        base_messages: list[dict] | None = None
        awaiting_lookdown = False
        last_llm_output = ""
        forward_action_count = 0
        system2_calls = 0
        trajectory_calls = 0
        step_id = 0
        done = False

        step_recorder: TrajectoryStepRecorder | None = None
        if args.save_trajectory_steps:
            step_recorder = TrajectoryStepRecorder(
                Path(output_path),
                scene_id,
                episode_id,
            )
            init_state = env._sim.get_agent(0).get_state()
            init_pos = np.array(init_state.position, dtype=float)
            init_rot = quaternion.as_float_array(init_state.rotation)
            goal_pos = np.array(episode.goals[0].position, dtype=float)
            gt_ref = getattr(episode, "reference_path", None)
            gt_path = [list(p) for p in gt_ref] if gt_ref is not None else [list(goal_pos)]
            step_recorder.set_metadata(
                instruction=instruction,
                start_pos=init_pos.tolist(),
                start_rot=init_rot.tolist(),
                goal_pos=goal_pos.tolist(),
                gt_path=gt_path,
            )

        while (not done) and (step_id < max_steps_per_episode):
            sys.stdout.flush()
            turn_lookdown_img: Image.Image | None = None

            stop_result = _maybe_stop_at_success(env, args, step_id)
            if stop_result is not None:
                observations, done, step_id = stop_result
                continue

            if local_actions:
                current_views = capture_panoramic_views(env, image_size=image_size)
                executed_history_panoramas.append(current_views)
                action = local_actions.pop(0)
                forward_action_count += 1

                if forward_action_count > MAX_STEPS:
                    pix_goal_image = None
                    _last_traj_hs = None
                    local_actions = []
                    forward_action_count = 0
                    base_messages = None
                    awaiting_lookdown = False
                    continue

                if action == ActionCode.STOP:
                    # This STOP comes from the local System1 action queue
                    # (often padding after a short local trajectory).  InternNav
                    # treats it as "local waypoint finished, ask System2 again",
                    # not as the final VLN episode STOP.
                    print("  [debug] local trajectory STOP -> replan", flush=True)
                    pix_goal_image = None
                    _last_traj_hs = None
                    local_actions = []
                    forward_action_count = 0
                    base_messages = None
                    awaiting_lookdown = False
                    continue

                before = _env_trace_summary(env) if _debug_input_trace_enabled(args) else None
                observations, done = _apply_habitat_action(env, action)
                _record_post_action_step(
                    step_recorder,
                    env,
                    step_id=step_id + 1,
                    phase="local_action",
                    action=int(action),
                    image_size=image_size,
                )
                if before is not None:
                    print(
                        f"  [debug] executed local action={int(action)} {before} -> {_env_trace_summary(env)}",
                        flush=True,
                    )
                step_id += 1
                continue

            if pix_goal_image is not None and _last_traj_hs is not None:
                # Training pins traj_images to direction="front_down" (see
                # src/data/trajectory_dataset.py:577). Feeding s1 the level
                # forward RGB is a domain mismatch; capture a fresh lookdown
                # (LOOKDOWN×2 → RGB → LOOKUP×2 restores pitch).
                current_lookdown_img = capture_lookdown_view(env, image_size=traj_image_size)
                if _debug_input_trace_enabled(args):
                    print(
                        "  [debug] System1 refresh lookdown: "
                        f"{_env_trace_summary(env)} "
                        f"lookdown={_image_trace_summary(current_lookdown_img)}",
                        flush=True,
                    )
                current_traj_t = _lookdown_to_traj_tensor(current_lookdown_img, device)
                traj_images = torch.stack([pix_goal_image, current_traj_t]).unsqueeze(0).to(device)

                print("  [debug] re-calling get_trajectory ...", flush=True)
                trajectory_calls += 1
                with torch.no_grad():
                    trajectory = _trajectory_from_condition(
                        model.nextdit_action_head,
                        _last_traj_hs,
                        traj_images=traj_images,
                    )

                local_actions = _finalize_local_actions(
                    traj_to_actions(
                        trajectory,
                        num_sample_trajs=num_sample_trajs,
                        action_scale=action_scale,
                        trajectory_selection=args.trajectory_selection,
                    )
                )
                print(
                    "  [debug] trajectory "
                    f"{_trajectory_debug_summary(trajectory, num_sample_trajs, action_scale)}, "
                    f"actions={_actions_for_log(local_actions)}",
                    flush=True,
                )
                continue

            print(
                f"  [step_id={step_id}] Capturing panoramic views + VLM inference ...",
                flush=True,
            )

            if awaiting_lookdown and base_messages is not None:
                turn_lookdown_img = capture_lookdown_view(env, image_size=vlm_image_size)
                if _debug_input_trace_enabled(args):
                    print(
                        "  [debug] System2 lookdown input: "
                        f"{_env_trace_summary(env)} "
                        f"lookdown={_image_trace_summary(turn_lookdown_img)}",
                        flush=True,
                    )
                _maybe_save_debug_images(
                    args,
                    scene_id,
                    episode_id,
                    system2_calls + 1,
                    "lookdown",
                    {"lookdown": turn_lookdown_img},
                )
                messages = copy.deepcopy(base_messages)
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": last_llm_output}],
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": random.choice(LEGACY_CONJUNCTIONS)},
                            {"type": "image", "image": turn_lookdown_img},
                        ],
                    }
                )
                awaiting_lookdown = False
            else:
                current_views = capture_panoramic_views(env, image_size=image_size)
                prompt_history = _sample_history_panoramas(
                    executed_history_panoramas,
                    num_history,
                )
                if _debug_input_trace_enabled(args):
                    print(
                        "  [debug] System2 panoramic input: "
                        f"{_env_trace_summary(env)} "
                        f"history={len(prompt_history)} "
                        f"{_views_trace_summary(current_views)}",
                        flush=True,
                    )
                _maybe_save_debug_images(
                    args,
                    scene_id,
                    episode_id,
                    system2_calls + 1,
                    "pano",
                    current_views,
                )
                messages = construct_input(
                    current_views=current_views,
                    history_panoramas=prompt_history,
                    instruction=instruction,
                    pixel_goal=[0, 0],
                    internnav_protocol=internnav_protocol,
                    structured_pano_output=structured_pano_output,
                )
                messages = [m for m in messages if m["role"] != "assistant"]
                base_messages = copy.deepcopy(messages)
                executed_history_panoramas.append(current_views)

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _normalize_multimodal_inputs(inputs)

            print(
                f"  [debug] input_ids shape={inputs['input_ids'].shape}, calling model.generate ...",
                flush=True,
            )
            if _debug_input_trace_enabled(args):
                print(
                    f"  [debug] processor pixel_values {_tensor_trace_summary(inputs.get('pixel_values'))}",
                    flush=True,
                )
            system2_calls += 1
            max_system2_calls = int(getattr(args, "max_system2_calls_per_episode", 0) or 0)
            if max_system2_calls > 0 and system2_calls > max_system2_calls:
                print(
                    f"  [debug] max System2 calls reached ({system2_calls - 1}); stopping episode",
                    flush=True,
                )
                observations, done = _apply_habitat_action(env, ActionCode.STOP)
                _record_post_action_step(
                    step_recorder,
                    env,
                    step_id=step_id + 1,
                    phase="stop",
                    action=int(ActionCode.STOP),
                    image_size=image_size,
                )
                step_id += 1
                continue
            with torch.no_grad():
                output_ids = model.qwen2_5_vl.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                ).sequences

            llm_output = processor.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            last_llm_output = llm_output
            print(f"  step_id: {step_id}, VLM output: {llm_output}")

            if _vlm_requests_stop(llm_output):
                observations, done = _apply_habitat_action(env, ActionCode.STOP)
                _record_post_action_step(
                    step_recorder,
                    env,
                    step_id=step_id + 1,
                    phase="stop",
                    action=int(ActionCode.STOP),
                    image_size=image_size,
                    vlm_output=llm_output,
                )
                step_id += 1
                continue

            turn_dir = _vlm_requests_turn(llm_output)
            if turn_dir is not None:
                action = ActionCode.LEFT if turn_dir == "left" else ActionCode.RIGHT
                observations, done = _apply_habitat_action(env, action)
                _record_post_action_step(
                    step_recorder,
                    env,
                    step_id=step_id + 1,
                    phase="turn",
                    action=int(action),
                    image_size=image_size,
                    vlm_output=llm_output,
                )
                step_id += 1
                continue

            pixel_goal = _parse_pixel_goal(
                llm_output,
                vlm_image_size,
                # Structured output is preferred, but closed-loop eval should
                # still salvage a pure legacy "u v" coordinate.  Malformed
                # structured lines such as "view: front|right|..." stay invalid
                # inside parse_structured_pano_output and do not fall through.
                allow_legacy_coord=True,
            )
            pano_goal_view = _parse_pano_view_id(llm_output) or "front"

            if step_recorder is not None:
                state = env._sim.get_agent(0).get_state()
                pos = np.array(state.position, dtype=float)
                rot = quaternion.as_float_array(state.rotation)
                views_for_record = None
                if executed_history_panoramas:
                    views_for_record = executed_history_panoramas[-1]
                step_recorder.record_step(
                    {
                        "step_id": step_id,
                        "phase": "vlm",
                        "position": pos,
                        "heading_deg": _quat_to_heading_deg(rot),
                        "rotation": rot,
                        "distance_to_goal": _metric_distance_to_goal(env),
                        "vlm_output": llm_output,
                        "pixel_goal": pixel_goal,
                        "pano_goal_view": pano_goal_view,
                        "current_views": views_for_record,
                    }
                )

            # === Force-teacher full-drive (--force_teacher_coord) ============
            #
            # Replace student's whole System2 decision with InternNav teacher's
            # at every call. Teacher's two-turn protocol can yield three modes:
            #   coord  -> use teacher coord, fall through to student System1
            #             (with teacher coord overriding student coord at
            #              _condition_output_ids_for_pixel_goal time)
            #   action -> push teacher's action sequence onto local_actions,
            #             skip System1 entirely, drain via the outer loop
            #   none   -> fall back to student behavior (rare degenerate case)
            # This isolates VLM as the sole controlled variable when measuring
            # the SR ceiling for the adapter+NextDiT stack.
            if force_teacher_model is not None:
                from scripts.evaluation.collect_internnav_teacher_sidecar import (
                    _parse_text_actions,
                )

                if turn_lookdown_img is not None:
                    ld_for_teacher = (
                        turn_lookdown_img
                        if turn_lookdown_img.size == vlm_image_size
                        else turn_lookdown_img.resize(vlm_image_size)
                    )
                else:
                    ld_for_teacher = capture_lookdown_view(
                        env,
                        image_size=vlm_image_size,
                    )
                    turn_lookdown_img = ld_for_teacher

                try:
                    front_for_teacher = current_views["front"]
                except (KeyError, TypeError):
                    front_for_teacher = None

                teacher_coord: list[int] | None = None
                teacher_info: dict = {}
                teacher_actions: list[int] = []
                if front_for_teacher is not None:
                    history_front_pils: list = []
                    if len(executed_history_panoramas) > 1:
                        hist_pano = executed_history_panoramas[-1 - num_history : -1]
                        history_front_pils = [h["front"] for h in hist_pano if "front" in h]
                    teacher_coord, teacher_info = _predict_force_teacher_coord(
                        force_teacher_model,
                        force_teacher_processor,
                        force_teacher_device or device,
                        current_front_pil=front_for_teacher,
                        lookdown_pil=ld_for_teacher,
                        instruction=instruction,
                        vlm_image_size=vlm_image_size,
                        history_front_pils=history_front_pils,
                    )
                    teacher_actions = _parse_text_actions(teacher_info.get("turn1_text") or "")
                    # If teacher's turn-1 contained "↓" (lookdown trigger) we
                    # already executed turn-2 internally inside
                    # _predict_force_teacher_coord. In that case the LOOKDOWN
                    # tokens in turn-1 were a *protocol signal* ("show me the
                    # lookdown frame"), NOT an action sequence. If turn-2 still
                    # didn't yield a coord, treat that as "teacher refused" and
                    # let us fall back to student -- otherwise we re-trigger
                    # awaiting_lookdown on every iteration and burn cycles in
                    # an infinite "LOOKDOWN -> ↓↓ -> LOOKDOWN" loop with no
                    # step_id progress.
                    if teacher_info.get("turn2_text") is not None:
                        teacher_actions = [a for a in teacher_actions if a != int(ActionCode.LOOKDOWN)]

                student_pg_repr = list(pixel_goal) if pixel_goal is not None else None

                if teacher_coord is not None:
                    if pixel_goal is not None and has_nextdit:
                        print(
                            "  [force-teacher] coord override (turn "
                            f"{teacher_info.get('used_turn')}, hist="
                            f"{teacher_info.get('n_history')}): "
                            f"student={student_pg_repr} -> "
                            f"teacher={teacher_coord}",
                            flush=True,
                        )
                        pixel_goal = teacher_coord
                        pano_goal_view = "front"
                    else:
                        print(
                            "  [force-teacher] teacher coord="
                            f"{teacher_coord} but student gave actions "
                            f"(llm_output={llm_output[:40]!r}); "
                            "cannot reuse student latent for teacher coord; "
                            "falling back to student",
                            flush=True,
                        )
                elif teacher_actions:
                    action_name_map = {
                        0: "STOP",
                        1: "FORWARD",
                        2: "LEFT",
                        3: "RIGHT",
                        5: "LOOKDOWN",
                    }
                    pretty_actions = [action_name_map.get(int(a), str(a)) for a in teacher_actions]
                    print(
                        "  [force-teacher] action override (hist="
                        f"{teacher_info.get('n_history')}): "
                        f"student pixel_goal={student_pg_repr} "
                        f"-> teacher actions={pretty_actions}",
                        flush=True,
                    )

                    first_action = teacher_actions[0]
                    if first_action == ActionCode.LOOKDOWN:
                        awaiting_lookdown = True
                        last_llm_output = "\u2193"
                        continue
                    if first_action == ActionCode.STOP:
                        observations, done = _apply_habitat_action(
                            env,
                            ActionCode.STOP,
                        )
                        _record_post_action_step(
                            step_recorder,
                            env,
                            step_id=step_id + 1,
                            phase="stop",
                            action=int(ActionCode.STOP),
                            image_size=image_size,
                            vlm_output=llm_output,
                        )
                        step_id += 1
                        continue

                    local_actions = _finalize_local_actions(teacher_actions)
                    pix_goal_image = None
                    _last_traj_hs = None
                    forward_action_count = 0
                    first_action = local_actions.pop(0)
                    if first_action == ActionCode.STOP:
                        observations, done = _apply_habitat_action(
                            env,
                            ActionCode.STOP,
                        )
                        _record_post_action_step(
                            step_recorder,
                            env,
                            step_id=step_id + 1,
                            phase="stop",
                            action=int(ActionCode.STOP),
                            image_size=image_size,
                            vlm_output=llm_output,
                        )
                        step_id += 1
                        continue
                    observations, done = _apply_habitat_action(
                        env,
                        first_action,
                    )
                    _record_post_action_step(
                        step_recorder,
                        env,
                        step_id=step_id + 1,
                        phase="local_action",
                        action=int(first_action),
                        image_size=image_size,
                        vlm_output=llm_output,
                    )
                    step_id += 1
                    forward_action_count += 1
                    continue
                else:
                    print(
                        "  [force-teacher] teacher unparseable (turn1="
                        f"{(teacher_info.get('turn1_text') or '')!r}); "
                        "falling back to student",
                        flush=True,
                    )

            if has_nextdit and pixel_goal is not None:
                print(f"  predicted pixel_goal {pixel_goal}")

                # Capture lookdown view (matches training direction="front_down"
                # in src/data/trajectory_dataset.py:577). Both slots of
                # traj_images are the same frame at goal-freeze time.
                if turn_lookdown_img is None:
                    current_lookdown_img = capture_lookdown_view(env, image_size=traj_image_size)
                else:
                    current_lookdown_img = (
                        turn_lookdown_img
                        if turn_lookdown_img.size == traj_image_size
                        else turn_lookdown_img.resize(traj_image_size)
                    )
                if _debug_input_trace_enabled(args):
                    print(
                        "  [debug] System1 goal-freeze lookdown: "
                        f"{_env_trace_summary(env)} "
                        f"lookdown={_image_trace_summary(current_lookdown_img)}",
                        flush=True,
                    )
                current_traj_t = _lookdown_to_traj_tensor(current_lookdown_img, device)
                pix_goal_image = current_traj_t.clone()
                traj_images = torch.stack([pix_goal_image, current_traj_t]).unsqueeze(0).to(device)

                print("  [debug] calling generate_latents ...", flush=True)
                lq = model.latent_queries.expand(1, -1, -1).to(
                    device=device,
                    dtype=model.config.dtype,
                )
                condition_output_ids = _condition_output_ids_for_pixel_goal(
                    output_ids=output_ids,
                    prompt_len=inputs["input_ids"].shape[1],
                    tokenizer=processor.tokenizer,
                    pixel_goal=pixel_goal,
                    llm_output=llm_output,
                    coord_order=system1_coord_order,
                    view_id=pano_goal_view,
                    structured_output=structured_pano_output,
                )
                with torch.no_grad():
                    _last_traj_hs = model.qwen2_5_vl.generate_latents(
                        output_ids=condition_output_ids,
                        pixel_values=inputs.get("pixel_values"),
                        image_grid_thw=inputs.get("image_grid_thw"),
                        latent_queries=lq,
                        attention_mask=inputs.get("attention_mask"),
                        mm_token_type_ids=inputs.get("mm_token_type_ids"),
                    )
                    if _debug_input_trace_enabled(args):
                        _per_q = [
                            float(_last_traj_hs[0, i].float().norm().item()) for i in range(_last_traj_hs.shape[1])
                        ]
                        print(
                            "  [debug] traj_hs total_norm="
                            f"{float(_last_traj_hs.float().norm().item()):.3f} "
                            f"per_query={_per_q}",
                            flush=True,
                        )
                    if step_recorder is not None and _last_traj_hs is not None:
                        try:
                            ths = _last_traj_hs.detach()
                            step_recorder._steps[-1]["traj_hs_total_norm"] = float(ths.float().norm().item())
                            step_recorder._steps[-1]["traj_hs_per_query"] = [
                                float(ths[0, i].float().norm().item()) for i in range(ths.shape[1])
                            ]
                        except Exception:
                            pass
                    if pano_latent_adapter is not None:
                        _last_traj_hs = _maybe_apply_pano_latent_adapter(
                            _last_traj_hs,
                            pano_latent_adapter,
                            cond_projector=model.nextdit_action_head.cond_projector
                            if model.nextdit_action_head is not None
                            else None,
                        )

                print("  [debug] calling get_trajectory ...", flush=True)
                trajectory_calls += 1
                with torch.no_grad():
                    trajectory = _trajectory_from_condition(
                        model.nextdit_action_head,
                        _last_traj_hs,
                        traj_images=traj_images,
                    )

                local_actions = _finalize_local_actions(
                    traj_to_actions(
                        trajectory,
                        num_sample_trajs=num_sample_trajs,
                        action_scale=action_scale,
                        trajectory_selection=args.trajectory_selection,
                    )
                )
                print(
                    "  [debug] trajectory "
                    f"{_trajectory_debug_summary(trajectory, num_sample_trajs, action_scale)}, "
                    f"actions={_actions_for_log(local_actions)}",
                    flush=True,
                )
                forward_action_count = 0

                if local_actions:
                    first_action = local_actions.pop(0)
                    if first_action == ActionCode.STOP:
                        # Mirror InternNav: if s1 predicts STOP on the very
                        # first action after s2, force a LEFT turn so the next
                        # s2 call sees a different panorama and can replan
                        # (otherwise s2→s1→STOP loops with identical views).
                        print("  [debug] first local action STOP -> LEFT anti-deadlock", flush=True)
                        pix_goal_image = None
                        _last_traj_hs = None
                        local_actions = []
                        base_messages = None
                        awaiting_lookdown = False
                        forward_action_count = 0
                        before = _env_trace_summary(env) if _debug_input_trace_enabled(args) else None
                        observations, done = _apply_habitat_action(env, ActionCode.LEFT)
                        _record_post_action_step(
                            step_recorder,
                            env,
                            step_id=step_id + 1,
                            phase="local_action",
                            action=int(ActionCode.LEFT),
                            image_size=image_size,
                            vlm_output=llm_output,
                        )
                        if before is not None:
                            print(
                                "  [debug] executed anti-deadlock action="
                                f"{int(ActionCode.LEFT)} {before} -> "
                                f"{_env_trace_summary(env)}",
                                flush=True,
                            )
                        step_id += 1
                        continue

                    before = _env_trace_summary(env) if _debug_input_trace_enabled(args) else None
                    observations, done = _apply_habitat_action(env, first_action)
                    _record_post_action_step(
                        step_recorder,
                        env,
                        step_id=step_id + 1,
                        phase="local_action",
                        action=int(first_action),
                        image_size=image_size,
                        vlm_output=llm_output,
                    )
                    if before is not None:
                        print(
                            f"  [debug] executed first local action={int(first_action)} "
                            f"{before} -> {_env_trace_summary(env)}",
                            flush=True,
                        )
                    step_id += 1
                    forward_action_count += 1
                    continue

                observations, done = _apply_habitat_action(env, ActionCode.STOP)
                _record_post_action_step(
                    step_recorder,
                    env,
                    step_id=step_id + 1,
                    phase="stop",
                    action=int(ActionCode.STOP),
                    image_size=image_size,
                    vlm_output=llm_output,
                )
                step_id += 1
                continue

            action_seq = parse_actions(llm_output, LEGACY_ACTIONS2IDX)
            if action_seq:
                action = action_seq.pop(0)
                if action == ActionCode.LOOKDOWN:
                    awaiting_lookdown = True
                    continue
                observations, done = _apply_habitat_action(env, action)
                _record_post_action_step(
                    step_recorder,
                    env,
                    step_id=step_id + 1,
                    phase="vlm_action",
                    action=int(action),
                    image_size=image_size,
                    vlm_output=llm_output,
                )
                step_id += 1
            elif not (llm_output or "").strip():
                # VLM generated blank/empty output — fall back to a turn
                # instead of immediately stopping the episode.
                print(
                    "  [warn] VLM output empty; falling back to LEFT turn",
                    flush=True,
                )
                observations, done = _apply_habitat_action(env, ActionCode.LEFT)
                _record_post_action_step(
                    step_recorder,
                    env,
                    step_id=step_id + 1,
                    phase="fallback_action",
                    action=int(ActionCode.LEFT),
                    image_size=image_size,
                    vlm_output=llm_output,
                )
                step_id += 1
            else:
                observations, done = _apply_habitat_action(env, ActionCode.STOP)
                _record_post_action_step(
                    step_recorder,
                    env,
                    step_id=step_id + 1,
                    phase="stop",
                    action=int(ActionCode.STOP),
                    image_size=image_size,
                    vlm_output=llm_output,
                )
                step_id += 1

        metrics = env.get_metrics()
        sucs.append(metrics["success"])
        spls.append(metrics["spl"])
        oss.append(metrics["oracle_success"])
        nes.append(metrics["distance_to_goal"])

        if step_recorder is not None:
            step_recorder.finalize(
                scene_id=scene_id,
                episode_id=episode_id,
                success=metrics["success"],
                spl=metrics["spl"],
                total_steps=step_id,
                vlm_calls=system2_calls,
                traj_calls=trajectory_calls,
            )

        print(
            f"  => success: {metrics['success']}, spl: {metrics['spl']:.4f}, "
            f"os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']:.4f}, "
            f"vlm_calls: {system2_calls}, trajectory_calls: {trajectory_calls}"
        )

        result = {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "success": metrics["success"],
            "spl": metrics["spl"],
            "os": metrics["oracle_success"],
            "ne": metrics["distance_to_goal"],
            "steps": step_id,
            "episode_instruction": instruction,
            "vlm_calls": system2_calls,
            "trajectory_calls": trajectory_calls,
            "system1_coord_order": system1_coord_order,
        }
        with open(progress_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        done_set.add(ep_key)
        process_bar.update(1)

        if eval_count % 50 == 0:
            torch.cuda.empty_cache()

    env.close()

    final_result = aggregate_navigation_metrics(sucs, spls, oss, nes)
    final_result["oracle_system2"] = bool(getattr(args, "oracle_system2", False))
    if bool(getattr(args, "oracle_system2", False)):
        final_result["oracle_system2_lookahead_m"] = float(args.oracle_system2_lookahead_m)
        final_result["oracle_system2_strategy"] = str(args.oracle_system2_strategy)
        final_result["oracle_system2_max_side_dist_m"] = float(args.oracle_system2_max_side_dist_m)

    print("\n" + "=" * 60)
    print("Final Results:")
    print(f"  NE  (Navigation Error):  {final_result['NE']:.4f}")
    print(f"  OS  (Oracle Success):    {final_result['OS']:.4f}")
    print(f"  SR  (Success Rate):      {final_result['SR']:.4f}")
    print(f"  SPL (Success w/ Path):   {final_result['SPL']:.4f}")
    print(f"  Total episodes:          {final_result['total_episodes']}")
    print("=" * 60)

    with open(os.path.join(output_path, "result.json"), "w") as f:
        json.dump(final_result, f, indent=2)
    print(f"Results saved to {os.path.join(output_path, 'result.json')}")


def _rpc_blob_from_pil(name: str, image: Image.Image, quality: int) -> dict:
    from vla_rpc.core.image import encode_rgb_to_jpeg

    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return {
        "name": name,
        "data": encode_rgb_to_jpeg(arr, quality=quality),
        "mime_type": "image/jpeg",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
    }


_ORACLE_PANO_VIEW_CENTERS_DEG: dict[str, float] = {
    "front": 0.0,
    "right": 90.0,
    "back": 180.0,
    "left": -90.0,
}


def _normalize_angle_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def _oracle_view_from_delta(delta_deg: float) -> str:
    delta = _normalize_angle_deg(delta_deg)
    if -45.0 <= delta <= 45.0:
        return "front"
    if 45.0 < delta <= 135.0:
        return "right"
    if -135.0 <= delta < -45.0:
        return "left"
    return "back"


def _path_xz(points: list[list[float]]) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[2])) for p in points if len(p) >= 3]


def _path_cumulative(points: list[tuple[float, float]]) -> list[float]:
    distances = [0.0]
    for (x0, z0), (x1, z1) in zip(points, points[1:]):
        distances.append(distances[-1] + float(math.hypot(x1 - x0, z1 - z0)))
    return distances


def _closest_progress_on_path(
    points: list[tuple[float, float]],
    cumulative: list[float],
    pos: np.ndarray,
) -> tuple[float, float]:
    px, pz = float(pos[0]), float(pos[2])
    if len(points) == 1:
        return float(math.hypot(px - points[0][0], pz - points[0][1])), 0.0

    best_dist = float("inf")
    best_progress = 0.0
    for idx, ((x0, z0), (x1, z1)) in enumerate(zip(points, points[1:])):
        dx, dz = x1 - x0, z1 - z0
        seg_len_sq = dx * dx + dz * dz
        if seg_len_sq <= 1e-9:
            continue
        t = ((px - x0) * dx + (pz - z0) * dz) / seg_len_sq
        t = max(0.0, min(1.0, t))
        qx, qz = x0 + t * dx, z0 + t * dz
        dist = float(math.hypot(px - qx, pz - qz))
        if dist < best_dist:
            best_dist = dist
            best_progress = cumulative[idx] + t * math.sqrt(seg_len_sq)
    return best_dist, best_progress


def _point_at_path_progress(
    points: list[tuple[float, float]],
    cumulative: list[float],
    progress: float,
) -> tuple[float, float]:
    if progress <= 0.0:
        return points[0]
    if progress >= cumulative[-1]:
        return points[-1]
    for idx in range(len(points) - 1):
        if cumulative[idx] <= progress <= cumulative[idx + 1]:
            seg_len = max(cumulative[idx + 1] - cumulative[idx], 1e-9)
            t = (progress - cumulative[idx]) / seg_len
            x = points[idx][0] + t * (points[idx + 1][0] - points[idx][0])
            z = points[idx][1] + t * (points[idx + 1][1] - points[idx][1])
            return x, z
    return points[-1]


def _episode_reference_path(episode) -> list[list[float]]:
    gt_ref = getattr(episode, "reference_path", None)
    if gt_ref is not None:
        path = [[float(v) for v in p] for p in gt_ref]
        if path:
            return path
    goal_pos = getattr(episode.goals[0], "position", None)
    return [[float(v) for v in goal_pos]] if goal_pos is not None else []


def _build_oracle_system2_from_reference_path(
    env,
    episode,
    *,
    image_size: tuple[int, int],
    strategy: str,
    lookahead_m: float,
    min_ahead_m: float,
    max_side_dist_m: float,
) -> dict[str, Any] | None:
    """Reference-path oracle for panoramic System2.

    This uses the closed-loop agent pose and the episode reference path.  It is
    intentionally depth-free: it is a direction/pixel oracle for isolating
    System1+adapter behavior, not a replica of the offline pano labeler.
    """
    gt_path = _episode_reference_path(episode)
    points = _path_xz(gt_path)
    if not points:
        return None

    state = env._sim.get_agent(0).get_state()
    pos = np.asarray(state.position, dtype=np.float64)
    rot = quaternion.as_float_array(state.rotation)
    heading_deg = _quat_to_heading_deg(rot)

    cumulative = _path_cumulative(points)
    offpath_m, progress_m = _closest_progress_on_path(points, cumulative, pos)
    min_progress_m = min(cumulative[-1], progress_m + float(min_ahead_m))

    if strategy == "lookahead":
        ahead_m = max(float(lookahead_m), float(min_ahead_m))
        target_progress_m = min(cumulative[-1], progress_m + ahead_m)
        target_x, target_z = _point_at_path_progress(points, cumulative, target_progress_m)
    else:
        target_progress_m = min_progress_m
        target_x, target_z = _point_at_path_progress(points, cumulative, target_progress_m)
        span = max(cumulative[-1] - min_progress_m, 0.0)
        num_samples = max(1, int(math.ceil(span / 0.25)))
        for sample_idx in range(num_samples + 1):
            cand_progress = cumulative[-1] - (span * sample_idx / num_samples)
            cand_x, cand_z = _point_at_path_progress(points, cumulative, cand_progress)
            cand_dx = cand_x - float(pos[0])
            cand_dz = cand_z - float(pos[2])
            cand_dist = float(math.hypot(cand_dx, cand_dz))
            if cand_dist <= 1e-6:
                continue
            cand_bearing = math.degrees(math.atan2(cand_dx, cand_dz)) % 360.0
            cand_delta = _normalize_angle_deg(cand_bearing - heading_deg)
            cand_view = _oracle_view_from_delta(cand_delta)
            if cand_view == "front" or cand_dist <= float(max_side_dist_m):
                target_progress_m = cand_progress
                target_x, target_z = cand_x, cand_z
                break

    dx = target_x - float(pos[0])
    dz = target_z - float(pos[2])
    if math.hypot(dx, dz) <= 1e-6:
        target_x, target_z = points[-1]
        dx = target_x - float(pos[0])
        dz = target_z - float(pos[2])
    if math.hypot(dx, dz) <= 1e-6:
        return {"text": "view: stop", "view": "stop", "pixel_goal": None}

    bearing_deg = math.degrees(math.atan2(dx, dz)) % 360.0
    delta_deg = _normalize_angle_deg(bearing_deg - heading_deg)
    view_id = _oracle_view_from_delta(delta_deg)
    local_deg = _normalize_angle_deg(delta_deg - _ORACLE_PANO_VIEW_CENTERS_DEG[view_id])

    width, height = int(image_size[0]), int(image_size[1])
    u_float = (width / 2.0) + (width / 2.0) * math.tan(math.radians(local_deg))
    u = max(0, min(width - 1, int(round(u_float))))
    v = max(0, min(height - 1, height // 2))
    pixel_goal = [u, v]
    return {
        "text": structured_condition_text(view_id, pixel_goal),
        "view": view_id,
        "pixel_goal": pixel_goal,
        "target_position": [float(target_x), float(pos[1]), float(target_z)],
        "heading_delta_deg": float(delta_deg),
        "bearing_deg": float(bearing_deg),
        "offpath_m": float(offpath_m),
        "path_progress_m": float(progress_m),
        "target_progress_m": float(target_progress_m),
        "strategy": str(strategy),
        "lookahead_m": float(lookahead_m),
        "max_side_dist_m": float(max_side_dist_m),
    }


def _next_shortest_path_recovery_action(
    follower: Any,
    goal_position: Any,
) -> tuple[int, bool]:
    """Return one privileged recovery action, probing in place at the goal."""
    action = follower.get_next_action(np.asarray(goal_position, dtype=np.float32))
    if action is None:
        raise RuntimeError("Habitat shortest-path follower returned no recovery action")
    action = int(action)
    if action == int(ActionCode.STOP):
        return int(ActionCode.LEFT), True
    valid_actions = {
        int(ActionCode.FORWARD),
        int(ActionCode.LEFT),
        int(ActionCode.RIGHT),
    }
    if action not in valid_actions:
        raise RuntimeError(f"Invalid Habitat shortest-path recovery action: {action}")
    return action, False


def _rpc_plan_panoramic(
    client,
    *,
    instruction: str,
    current_views: dict[str, Image.Image],
    history_panoramas: list[dict[str, Image.Image]],
    lookdown_img: Image.Image,
    vlm_image_size: tuple[int, int],
    traj_image_size: tuple[int, int],
    system1_coord_order: str,
    trajectory_selection: str,
    trajectory_x_sign: float,
    trajectory_heading_alignment: str,
    jpeg_quality: int,
    scene_id: str,
    episode_id: int,
    system2_call_index: int,
    protocol_seed: int,
    require_deterministic_sampling: bool,
    oracle_system2: dict[str, Any] | None = None,
    force_non_stop: bool = False,
) -> dict:
    blobs = []
    for view in ("front", "right", "back", "left"):
        blobs.append(_rpc_blob_from_pil(f"current/{view}", current_views[view], jpeg_quality))
    for idx, hist in enumerate(history_panoramas):
        for view in ("front", "right", "back", "left"):
            blobs.append(_rpc_blob_from_pil(f"history/{idx}/{view}", hist[view], jpeg_quality))
    blobs.append(_rpc_blob_from_pil("lookdown", lookdown_img, jpeg_quality))

    sampling_metadata = build_rpc_sampling_metadata(
        protocol_seed=protocol_seed,
        scene_id=scene_id,
        episode_id=episode_id,
        system2_call_index=system2_call_index,
    )
    payload = {
        "instruction": instruction,
        "num_history": len(history_panoramas),
        "vlm_image_size": list(vlm_image_size),
        "traj_image_size": list(traj_image_size),
        "system1_coord_order": system1_coord_order,
        "trajectory_selection": trajectory_selection,
        "trajectory_x_sign": trajectory_x_sign,
        "trajectory_heading_alignment": trajectory_heading_alignment,
        "require_deterministic_sampling": bool(require_deterministic_sampling),
        HEATMAPVLN_RPC_SAMPLING_FIELD: sampling_metadata,
    }
    if oracle_system2 is not None:
        payload["oracle_system2"] = oracle_system2
    if force_non_stop:
        payload["system2_force_non_stop"] = True
    result = client.infer_json("plan_panoramic", payload, blobs)
    if result is None:
        raise RuntimeError("RPC model server returned no response")
    response, _response_blobs = result
    if not response.get("ok", False):
        raise RuntimeError(f"RPC model server error: {response}")
    if response.get("proto_v") != HEATMAPVLN_RPC_PROTOCOL_VERSION:
        raise RuntimeError(
            "RPC response protocol mismatch: "
            f"server={response.get('proto_v')!r} "
            f"expected={HEATMAPVLN_RPC_PROTOCOL_VERSION!r}"
        )
    response_sampling = validate_rpc_sampling_metadata(
        response.get(HEATMAPVLN_RPC_SAMPLING_FIELD),
        require_deterministic=True,
    )
    if response_sampling != sampling_metadata:
        raise RuntimeError(
            "RPC server did not echo the exact deterministic sampling record: "
            f"request={sampling_metadata!r} response={response_sampling!r}"
        )
    return response


def run_eval_rpc_panoramic(args):
    """Run Habitat in this process and send model inference to RPC server."""
    import yaml
    from vla_rpc.client import VLAClient

    ensure_vln_measures_registered()
    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)
    panoramic_vlm_input = bool(train_cfg.get("data", {}).get("trajectory", {}).get("panoramic_vlm_input", False))
    if not panoramic_vlm_input:
        raise RuntimeError("--rpc_server currently supports panoramic_vlm_input configs only")

    vlm_image_size, traj_image_size = _eval_image_sizes(train_cfg)
    image_size = vlm_image_size
    num_history = args.num_history
    max_steps_per_episode = args.max_steps_per_episode
    system1_coord_order = _system1_coord_order(args, panoramic_internnav_protocol=False)

    print(f"Using RPC model server: {args.rpc_server}")
    print(f"vlm_image_size={vlm_image_size}, traj_image_size={traj_image_size}")
    print(f"trajectory_selection={args.trajectory_selection}")
    print(f"trajectory_x_sign={args.trajectory_x_sign:g}")
    print(f"trajectory_heading_alignment={args.trajectory_heading_alignment}")
    guard_config = _closed_loop_guard_config(args)
    print(
        "closed_loop_policy="
        f"action_chunk={guard_config.action_chunk_size} "
        f"stop_confirmations={guard_config.stop_confirmations} "
        "stop_confirmation_max_gap_calls="
        f"{guard_config.stop_confirmation_max_gap_calls} "
        "stop_confirmation_view_sweep="
        f"{guard_config.stop_confirmation_view_sweep} "
        "stop_high_confidence_threshold="
        f"{guard_config.stop_high_confidence_threshold} "
        f"stop_probe_turn={guard_config.stop_probe_turn} "
        f"loop_guard={guard_config.loop_guard_enabled} "
        f"recovery_turns={guard_config.recovery_turns} "
        f"recovery_forward_steps={guard_config.recovery_forward_steps} "
        f"recovery_follow_last_turn={guard_config.recovery_follow_last_turn}"
    )
    print(
        "rpc_sampling="
        f"{HEATMAPVLN_RPC_SAMPLING_PROTOCOL} "
        f"protocol_seed={args.rpc_protocol_seed} "
        f"required={bool(args.rpc_require_deterministic_sampling)}"
    )
    if bool(getattr(args, "oracle_system2", False)):
        print(
            "[rpc-eval] --oracle_system2 is ACTIVE; System2 text will be "
            "replaced by a reference-path pano view/pixel oracle before RPC "
            "System1 inference."
        )

    client = VLAClient(
        server_addr=args.rpc_server,
        timeout_ms=args.rpc_timeout_ms,
        jpeg_quality=args.rpc_jpeg_quality,
    )
    client.connect()
    if not client.health_check():
        raise RuntimeError(f"RPC model server is not healthy: {args.rpc_server}")
    info = client.get_server_info()
    if info is not None:
        print(f"RPC server: version={info.version}, model={info.model_version}")
        if info.version != HEATMAPVLN_RPC_PROTOCOL_VERSION:
            raise RuntimeError(
                f"RPC server protocol mismatch: server={info.version!r} expected={HEATMAPVLN_RPC_PROTOCOL_VERSION!r}"
            )

    hab_cfg = build_habitat_config(args)
    print("Creating Habitat environment ...")
    env = habitat.Env(config=hab_cfg)
    num_episodes = len(list(env.episodes))
    print(f"Total episodes: {num_episodes}")

    output_path = args.output_path
    progress_file = _prepare_progress_file(args, output_path)
    collect_stop_features = bool(args.collect_system2_stop_features)
    collect_stop_multimodal = bool(args.collect_system2_stop_multimodal_examples)
    dataset_split = Path(args.data_path).parent.name
    if collect_stop_multimodal and not collect_stop_features:
        raise ValueError(
            "--collect_system2_stop_multimodal_examples requires "
            "--collect_system2_stop_features"
        )
    if collect_stop_multimodal and dataset_split != "train":
        raise ValueError(
            "Multimodal STOP collection is train-split only; "
            f"got dataset split {dataset_split!r}"
        )
    force_continue_stop_negatives = bool(
        args.system2_stop_collect_force_continue_negatives
    )
    oracle_recovery_after_negative = validate_oracle_recovery_collection(
        collection_enabled=collect_stop_features,
        force_continue_negatives=force_continue_stop_negatives,
        oracle_recovery_after_negative=bool(
            args.system2_stop_collect_oracle_recovery_after_negative
        ),
    )
    oracle_path_from_start = validate_oracle_path_collection(
        collection_enabled=collect_stop_features,
        force_continue_negatives=force_continue_stop_negatives,
        oracle_path_from_start=bool(
            args.system2_stop_collect_oracle_path_from_start
        ),
    )
    if oracle_path_from_start and oracle_recovery_after_negative:
        raise ValueError(
            "Oracle path-from-start and recovery-after-negative modes are mutually exclusive"
        )
    boundary_probe_sweep = validate_boundary_probe_collection(
        collection_enabled=collect_stop_features,
        force_continue_negatives=force_continue_stop_negatives,
        oracle_path_from_start=oracle_path_from_start,
        boundary_probe_sweep=bool(args.system2_stop_collect_boundary_probe_sweep),
        min_distance_m=float(args.system2_stop_boundary_probe_min_distance_m),
        max_distance_m=float(args.system2_stop_boundary_probe_max_distance_m),
        probes=int(args.system2_stop_boundary_probe_views),
    )
    oracle_recovery_from_cohort_triggers = bool(
        args.system2_stop_oracle_recovery_from_cohort_triggers
    )
    if oracle_recovery_from_cohort_triggers and not oracle_recovery_after_negative:
        raise ValueError(
            "Cohort-triggered oracle recovery requires the complete privileged "
            "STOP recovery collection mode"
        )
    should_force_continue_negative(
        collection_enabled=collect_stop_features,
        force_continue_negatives=force_continue_stop_negatives,
        terminal=False,
        rollout_label=None,
    )
    should_finish_oracle_recovery_collection(
        goal_probe_count=0,
        max_goal_probes=int(args.system2_stop_oracle_recovery_goal_probes),
    )
    oracle_recovery_actions_per_call = validate_oracle_recovery_actions_per_call(
        args.system2_stop_oracle_recovery_actions_per_call
    )
    stop_feature_labels_path = Path(output_path) / "system2_stop_rollout_labels.jsonl"
    stop_multimodal_labels_path = (
        Path(output_path) / "system2_stop_multimodal_examples.jsonl"
    )
    stop_multimodal_recorder = None
    if collect_stop_features:
        if not bool(args.rpc_require_deterministic_sampling):
            raise ValueError(
                "--collect_system2_stop_features requires "
                "--rpc_require_deterministic_sampling"
            )
        if not (
            0.0
            < float(args.system2_stop_positive_radius_m)
            < float(args.system2_stop_negative_radius_m)
        ):
            raise ValueError(
                "System2 STOP collection requires 0 < positive radius < negative radius"
            )
        if args.overwrite_output and stop_feature_labels_path.exists():
            stop_feature_labels_path.unlink()
        elif stop_feature_labels_path.exists() and not args.resume:
            raise FileExistsError(
                f"Found existing STOP rollout labels: {stop_feature_labels_path}"
            )
        if collect_stop_multimodal:
            if args.overwrite_output and stop_multimodal_labels_path.exists():
                stop_multimodal_labels_path.unlink()
            elif stop_multimodal_labels_path.exists() and not args.resume:
                raise FileExistsError(
                    "Found existing multimodal STOP rollout labels: "
                    f"{stop_multimodal_labels_path}"
                )
            stop_multimodal_recorder = System2StopMultimodalRecorder(
                Path(output_path),
                dataset_split=dataset_split,
                jpeg_quality=int(args.rpc_jpeg_quality),
                regular_min_stop_log_odds=(
                    float(args.system2_stop_multimodal_regular_min_stop_log_odds)
                    if args.system2_stop_multimodal_regular_min_stop_log_odds
                    is not None
                    else None
                ),
            )
        print(
            "WARNING: privileged System2 STOP feature collection is ACTIVE; "
            "distance-to-goal is written only as an offline training label and "
            "must not be used by the navigation policy. "
            f"force_continue_negatives={force_continue_stop_negatives} "
            f"oracle_recovery_after_negative={oracle_recovery_after_negative} "
            f"oracle_path_from_start={oracle_path_from_start} "
            f"boundary_probe_sweep={boundary_probe_sweep} "
            f"multimodal_examples={collect_stop_multimodal} "
            "multimodal_regular_min_stop_log_odds="
            f"{args.system2_stop_multimodal_regular_min_stop_log_odds} "
            f"oracle_actions_per_call={oracle_recovery_actions_per_call}",
            flush=True,
        )
    target_list, target_set = _episode_list_from_args(args)
    target_metadata = _episode_metadata_from_args(args)
    historical_recovery_triggers: dict[
        tuple[str, int], HistoricalFalseStopTrigger
    ] = {}
    if oracle_recovery_from_cohort_triggers:
        if target_list is None:
            raise ValueError(
                "Cohort-triggered oracle recovery requires --episode_list"
            )
        for key in target_list:
            trigger = parse_historical_false_stop_trigger(
                target_metadata[key],
                expected_protocol_seed=int(args.rpc_protocol_seed),
                negative_radius_m=float(args.system2_stop_negative_radius_m),
            )
            validate_historical_false_stop_source(
                trigger,
                scene_id=key[0],
                episode_id=key[1],
            )
            historical_recovery_triggers[key] = trigger
        print(
            "WARNING: historical false-STOP call triggers are ACTIVE for "
            f"{len(historical_recovery_triggers)} audited episodes; this is "
            "privileged offline collection only.",
            flush=True,
        )
    sucs, spls, oss, nes, done_set = _load_progress(
        progress_file,
        expected_rpc_sampling_contract=build_rpc_progress_sampling_contract(
            protocol_seed=int(args.rpc_protocol_seed),
            require_deterministic_sampling=bool(args.rpc_require_deterministic_sampling),
        ),
    )
    if args.resume and collect_stop_features:
        resume_label_paths = [stop_feature_labels_path]
        if collect_stop_multimodal:
            resume_label_paths.append(stop_multimodal_labels_path)
        for labels_path in resume_label_paths:
            kept_rows, dropped_rows = prune_stop_collection_jsonl_for_resume(
                labels_path,
                done_set,
            )
            print(
                "STOP collection resume cleanup: "
                f"path={labels_path} kept={kept_rows} "
                f"dropped_incomplete={dropped_rows}",
                flush=True,
            )
    if target_set is not None and not done_set.issubset(target_set):
        unexpected = sorted(done_set - target_set)
        raise ValueError(f"RPC progress contains episodes outside the requested fixed cohort: {unexpected[:10]}")
    if target_list is not None:
        selected = restrict_habitat_env_to_episode_keys(env, target_list)
        print(f"Fixed episode list ({len(target_list)}): {args.episode_list}")
        print(f"Restricted Habitat iterator to {len(selected)} requested episodes")
    remaining = num_episodes - len(done_set)
    eval_limit = _eval_limit(args, remaining, target_list, done_set)
    print(f"Episodes already done: {len(done_set)}, remaining: {remaining}, this run: {eval_limit}")

    process_bar = tqdm.tqdm(total=eval_limit, desc="Evaluating", ncols=120)
    seen_episodes: set = set()
    eval_count = 0
    total_stop_probes, total_recoveries = _load_closed_loop_progress_totals(progress_file)

    while True:
        process_bar.set_postfix(
            SR=f"{float(np.mean(sucs)):.3f}" if sucs else "?",
            SPL=f"{float(np.mean(spls)):.3f}" if spls else "?",
        )
        if eval_count >= eval_limit:
            break

        observations = env.reset()
        episode = env.current_episode
        scene_id = episode.scene_id.split("/")[-2]
        episode_id = int(episode.episode_id)
        ep_key = (scene_id, episode_id)
        if ep_key in seen_episodes:
            break
        seen_episodes.add(ep_key)
        if target_set is not None and ep_key not in target_set:
            continue
        if ep_key in done_set:
            continue

        instruction = _normalize_instruction(episode.instruction.instruction_text)
        eval_count += 1
        print(f"\n[{eval_count}/{eval_limit}] Episode {scene_id}_{episode_id:04d}: {instruction[:80]}...")

        executed_history_panoramas: list[dict[str, Image.Image]] = []
        local_actions: list[int] = []
        forward_action_count = 0
        system2_calls = 0
        trajectory_calls = 0
        step_id = 0
        done = False
        stop_probes = 0
        stop_score_records: list[dict[str, Any]] = []
        stop_head_records: list[dict[str, Any]] = []
        collected_stop_features = 0
        forced_continue_calls = 0
        oracle_recovery_calls = 0
        oracle_recovery_primitive_actions = 0
        oracle_recovery_goal_probes = 0
        boundary_probe_rows = 0
        boundary_probe_turns = 0
        oracle_recovery_state = OracleRecoveryState()
        if oracle_path_from_start:
            oracle_recovery_state.activate_from_start()
        boundary_probe_state = BoundaryProbeSweepState(
            enabled=boundary_probe_sweep,
            min_distance_m=float(args.system2_stop_boundary_probe_min_distance_m),
            max_distance_m=float(args.system2_stop_boundary_probe_max_distance_m),
            max_probes=int(args.system2_stop_boundary_probe_views),
        )
        historical_recovery_trigger = historical_recovery_triggers.get(ep_key)
        historical_recovery_trigger_reached = False
        oracle_recovery_follower = None
        oracle_recovery_goal_radius_m = 0.0
        if oracle_recovery_after_negative or oracle_path_from_start:
            from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

            oracle_recovery_goal_radius_m = max(
                0.25,
                float(args.system2_stop_positive_radius_m) - 0.5,
            )
            oracle_recovery_follower = ShortestPathFollower(
                env._sim,
                goal_radius=oracle_recovery_goal_radius_m,
                return_one_hot=False,
                stop_on_error=False,
            )
        recovery_reasons: list[str] = []
        closed_loop_guard = ClosedLoopGuard(
            guard_config,
            forward_action=int(ActionCode.FORWARD),
            left_action=int(ActionCode.LEFT),
            right_action=int(ActionCode.RIGHT),
        )
        closed_loop_guard.reset_episode(_agent_position(env))

        step_recorder: TrajectoryStepRecorder | None = None
        if args.save_trajectory_steps:
            step_recorder = TrajectoryStepRecorder(
                Path(output_path),
                scene_id,
                episode_id,
            )
            init_state = env._sim.get_agent(0).get_state()
            init_pos = np.array(init_state.position, dtype=float)
            init_rot = quaternion.as_float_array(init_state.rotation)
            goal_pos = np.array(episode.goals[0].position, dtype=float)
            gt_path = _episode_reference_path(episode) or [list(goal_pos)]
            step_recorder.set_metadata(
                instruction=instruction,
                start_pos=init_pos.tolist(),
                start_rot=init_rot.tolist(),
                goal_pos=goal_pos.tolist(),
                gt_path=gt_path,
            )

        while (not done) and (step_id < max_steps_per_episode):
            sys.stdout.flush()
            stop_result = _maybe_stop_at_success(env, args, step_id)
            if stop_result is not None:
                observations, done, new_step_id = stop_result
                _record_post_action_step(
                    step_recorder,
                    env,
                    step_id=new_step_id,
                    phase="auto_stop",
                    action=int(ActionCode.STOP),
                    image_size=image_size,
                )
                step_id = new_step_id
                continue

            if local_actions:
                current_views = capture_panoramic_views(env, image_size=image_size)
                executed_history_panoramas.append(current_views)
                action = int(local_actions.pop(0))
                forward_action_count += 1
                if forward_action_count > MAX_STEPS:
                    local_actions = []
                    forward_action_count = 0
                    continue
                if action == ActionCode.STOP:
                    print("  [debug] local trajectory STOP -> replan", flush=True)
                    local_actions = []
                    forward_action_count = 0
                    continue
                before_position = _agent_position(env)
                before = _env_trace_summary(env) if _debug_input_trace_enabled(args) else None
                observations, done = _apply_habitat_action(env, action)
                after_position = _agent_position(env)
                if before is not None:
                    print(
                        f"  [debug] executed local action={int(action)} {before} -> {_env_trace_summary(env)}",
                        flush=True,
                    )
                step_id += 1
                _record_post_action_step(
                    step_recorder,
                    env,
                    step_id=step_id,
                    phase="local_action",
                    action=int(action),
                    image_size=image_size,
                )
                recovery = closed_loop_guard.observe_action(
                    action,
                    before_position,
                    after_position,
                )
                if recovery is not None and not done:
                    recovery_reasons.append(recovery.reason)
                    local_actions = list(recovery.actions)
                    forward_action_count = 0
                    executed_history_panoramas = _trim_recovery_history(
                        executed_history_panoramas,
                        int(args.closed_loop_recovery_history_keep),
                    )
                    print(
                        "  [guard] recovery: "
                        f"reason={recovery.reason} actions={list(recovery.actions)} "
                        f"history={len(executed_history_panoramas)}",
                        flush=True,
                    )
                continue

            max_system2_calls = int(getattr(args, "max_system2_calls_per_episode", 0) or 0)
            if max_system2_calls > 0 and system2_calls >= max_system2_calls:
                print(
                    f"  [debug] max System2 calls reached ({system2_calls}); stopping episode",
                    flush=True,
                )
                observations, done = _apply_habitat_action(env, ActionCode.STOP)
                step_id += 1
                _record_post_action_step(
                    step_recorder,
                    env,
                    step_id=step_id,
                    phase="max_system2_stop",
                    action=int(ActionCode.STOP),
                    image_size=image_size,
                )
                continue

            current_views = capture_panoramic_views(env, image_size=image_size)
            prompt_history_indices = _sample_history_indices(
                len(executed_history_panoramas), num_history
            )
            prompt_history = [
                executed_history_panoramas[index]
                for index in prompt_history_indices
            ]
            if _debug_input_trace_enabled(args):
                print(
                    "  [debug] RPC System2 input: "
                    f"{_env_trace_summary(env)} "
                    f"history={len(prompt_history)} "
                    f"{_views_trace_summary(current_views)}",
                    flush=True,
                )
            lookdown_img = capture_lookdown_view(env, image_size=traj_image_size)
            executed_history_panoramas.append(current_views)
            system2_calls += 1
            oracle_system2 = None
            if bool(getattr(args, "oracle_system2", False)):
                oracle_system2 = _build_oracle_system2_from_reference_path(
                    env,
                    episode,
                    image_size=vlm_image_size,
                    strategy=str(getattr(args, "oracle_system2_strategy", "farthest_visible")),
                    lookahead_m=float(getattr(args, "oracle_system2_lookahead_m", 2.0)),
                    min_ahead_m=float(getattr(args, "oracle_system2_min_ahead_m", 0.5)),
                    max_side_dist_m=float(getattr(args, "oracle_system2_max_side_dist_m", 6.0)),
                )
                if oracle_system2 is None:
                    raise RuntimeError(f"Could not build oracle System2 for {scene_id}_{episode_id:04d}")
                print(
                    "  [oracle-system2] "
                    f"{oracle_system2['text'].replace(chr(10), ' | ')} "
                    f"delta={oracle_system2.get('heading_delta_deg', 0.0):.1f} "
                    f"offpath={oracle_system2.get('offpath_m', 0.0):.2f}",
                    flush=True,
                )

            response = _rpc_plan_panoramic(
                client,
                instruction=instruction,
                current_views=current_views,
                history_panoramas=prompt_history,
                lookdown_img=lookdown_img,
                vlm_image_size=vlm_image_size,
                traj_image_size=traj_image_size,
                system1_coord_order=system1_coord_order,
                trajectory_selection=args.trajectory_selection,
                trajectory_x_sign=args.trajectory_x_sign,
                trajectory_heading_alignment=args.trajectory_heading_alignment,
                jpeg_quality=args.rpc_jpeg_quality,
                scene_id=scene_id,
                episode_id=episode_id,
                # Explicitly zero-based and independent of any calls made by
                # the other experimental arm.
                system2_call_index=system2_calls - 1,
                protocol_seed=args.rpc_protocol_seed,
                require_deterministic_sampling=args.rpc_require_deterministic_sampling,
                oracle_system2=oracle_system2,
            )
            llm_output = response.get("llm_output", "")
            raw_actions = [int(action) for action in response.get("actions", [])]
            actions = closed_loop_guard.limit_actions(raw_actions)
            print(
                f"  step_id: {step_id}, RPC kind={response.get('kind')}, VLM output: {llm_output}",
                flush=True,
            )
            if response.get("trajectory_summary"):
                trajectory_calls += 1
                print(
                    f"  [debug] trajectory {response['trajectory_summary']}, actions={raw_actions}",
                    flush=True,
                )
                if actions != raw_actions:
                    print(f"  [guard] action chunk: {raw_actions} -> {actions}", flush=True)
            elif raw_actions:
                print(f"  [debug] actions={raw_actions}", flush=True)

            if step_recorder is not None:
                state = env._sim.get_agent(0).get_state()
                pos = np.array(state.position, dtype=float)
                rot = quaternion.as_float_array(state.rotation)
                step_recorder.record_step(
                    {
                        "step_id": step_id,
                        "phase": "rpc_vlm",
                        "position": pos,
                        "heading_deg": _quat_to_heading_deg(rot),
                        "rotation": rot,
                        "distance_to_goal": _metric_distance_to_goal(env),
                        "vlm_output": llm_output,
                        "pixel_goal": response.get("pixel_goal"),
                        "pano_goal_view": response.get("pano_goal_view"),
                        "oracle_system2": response.get("oracle_system2"),
                        "system2_stop_head": response.get("system2_stop_head"),
                        HEATMAPVLN_RPC_SAMPLING_FIELD: response.get(HEATMAPVLN_RPC_SAMPLING_FIELD),
                        "current_views": current_views,
                    }
                )

            terminal = bool(response.get("terminal", False))
            if stop_multimodal_recorder is not None:
                distance_to_goal = _metric_distance_to_goal(env)
                if distance_to_goal is None:
                    raise RuntimeError(
                        "Habitat distance_to_goal metric is unavailable for "
                        "multimodal STOP collection"
                    )
                rollout_label = _system2_stop_rollout_label(
                    distance_to_goal,
                    positive_radius_m=float(args.system2_stop_positive_radius_m),
                    negative_radius_m=float(args.system2_stop_negative_radius_m),
                )
                stop_multimodal_recorder.record(
                    scene_id=scene_id,
                    episode_id=episode_id,
                    system2_call_index=system2_calls - 1,
                    protocol_seed=int(args.rpc_protocol_seed),
                    instruction=instruction,
                    current_views=current_views,
                    history_views=prompt_history,
                    history_source_indices=prompt_history_indices,
                    distance_to_goal_m=float(distance_to_goal),
                    stop_target=rollout_label,
                    response=response,
                    image_size=vlm_image_size,
                    oracle_recovery_active=oracle_recovery_state.active,
                )
            if collect_stop_features:
                feature_record = response.get("system2_stop_feature")
                if not isinstance(feature_record, dict):
                    raise RuntimeError(
                        "RPC server did not return system2_stop_feature while "
                        "--collect_system2_stop_features is active"
                    )
                distance_to_goal = _metric_distance_to_goal(env)
                if distance_to_goal is None:
                    raise RuntimeError(
                        "Habitat distance_to_goal metric is unavailable for STOP collection"
                    )
                rollout_label = _system2_stop_rollout_label(
                    distance_to_goal,
                    positive_radius_m=float(args.system2_stop_positive_radius_m),
                    negative_radius_m=float(args.system2_stop_negative_radius_m),
                )
                stop_feature_row = {
                    **feature_record,
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "data_path": str(args.data_path),
                    "dataset_split": Path(args.data_path).parent.name,
                    "system2_call_index": system2_calls - 1,
                    "step": int(step_id),
                    "distance_to_goal_m": float(distance_to_goal),
                    "stop_target": rollout_label,
                    "positive_radius_m": float(args.system2_stop_positive_radius_m),
                    "negative_radius_m": float(args.system2_stop_negative_radius_m),
                    "original_terminal": terminal,
                    "llm_output": str(response.get("llm_output", "")),
                    "system2_decision_scores": response.get("system2_decision_scores"),
                    "trajectory_metrics": response.get("trajectory_metrics"),
                }
                boundary_probe_index = boundary_probe_state.observe(
                    distance_m=float(distance_to_goal),
                    rollout_label=rollout_label,
                )
                stop_feature_row["boundary_probe_sweep"] = (
                    boundary_probe_index is not None
                )
                stop_feature_row["boundary_probe_index"] = boundary_probe_index
                stop_feature_row["boundary_probe_views"] = int(
                    args.system2_stop_boundary_probe_views
                )
                stop_feature_row["boundary_probe_activation_distance_m"] = (
                    boundary_probe_state.activation_distance_m
                )
                stop_feature_row["boundary_probe_sweep_id"] = (
                    f"{scene_id}:{episode_id}:{int(args.rpc_protocol_seed)}:boundary"
                    if boundary_probe_index is not None
                    else None
                )
                goal_probe_index = (
                    oracle_recovery_goal_probes
                    if oracle_recovery_state.active
                    and rollout_label == 1
                    and float(distance_to_goal) <= oracle_recovery_goal_radius_m
                    else None
                )
                stop_feature_row["goal_probe_sweep"] = goal_probe_index is not None
                stop_feature_row["goal_probe_index"] = goal_probe_index
                stop_feature_row["goal_probe_sweep_id"] = (
                    f"{scene_id}:{episode_id}:{int(args.rpc_protocol_seed)}:goal"
                    if goal_probe_index is not None
                    else None
                )
                force_continue = should_force_continue_negative(
                    collection_enabled=collect_stop_features,
                    force_continue_negatives=force_continue_stop_negatives,
                    terminal=terminal,
                    rollout_label=rollout_label,
                )
                oracle_recovery_override = oracle_recovery_state.active
                historical_trigger_due = bool(
                    historical_recovery_trigger is not None
                    and system2_calls - 1
                    == historical_recovery_trigger.system2_call_index
                )
                if historical_trigger_due:
                    historical_recovery_trigger_reached = True
                if oracle_recovery_after_negative:
                    oracle_recovery_override = oracle_recovery_state.observe(
                        terminal=terminal,
                        rollout_label=rollout_label,
                    )
                if (
                    historical_recovery_trigger is not None
                    and not oracle_recovery_override
                    and terminal
                    and rollout_label == 1
                ):
                    oracle_recovery_override = (
                        oracle_recovery_state.activate_from_cohort(
                            rollout_label=rollout_label,
                            reason="current_positive_stop",
                        )
                    )
                elif historical_trigger_due and not oracle_recovery_override:
                    oracle_recovery_override = (
                        oracle_recovery_state.activate_from_cohort(
                            rollout_label=rollout_label,
                            reason="historical_false_stop_call",
                        )
                    )
                if (
                    historical_recovery_trigger is not None
                    and system2_calls - 1
                    > historical_recovery_trigger.system2_call_index
                    and oracle_recovery_state.activations == 0
                ):
                    raise RuntimeError(
                        "Missed historical false-STOP recovery trigger for "
                        f"{scene_id}_{episode_id}: "
                        f"call={historical_recovery_trigger.system2_call_index}"
                    )
                stop_feature_row["oracle_forced_continue"] = force_continue
                stop_feature_row["oracle_recovery_active"] = (
                    oracle_recovery_override
                )
                stop_feature_row["oracle_recovery_activation_reason"] = (
                    oracle_recovery_state.activation_reason
                    if oracle_recovery_override
                    else None
                )
                stop_feature_row["oracle_recovery_actions_per_call"] = (
                    oracle_recovery_actions_per_call
                )
                stop_feature_row["historical_false_stop_trigger_due"] = (
                    historical_trigger_due
                )
                stop_feature_row["historical_false_stop_trigger"] = (
                    {
                        "system2_call_index": (
                            historical_recovery_trigger.system2_call_index
                        ),
                        "step": historical_recovery_trigger.step,
                        "distance_m": historical_recovery_trigger.distance_m,
                        "protocol_seed": historical_recovery_trigger.protocol_seed,
                        "source_labels": historical_recovery_trigger.source_labels,
                    }
                    if historical_recovery_trigger is not None
                    else None
                )
                stop_feature_row["intervention_policy"] = (
                    "habitat_shortest_path_recovery_from_audited_cohort"
                    if oracle_recovery_override
                    and oracle_recovery_state.activation_reason
                    in {"historical_false_stop_call", "current_positive_stop"}
                    else "habitat_shortest_path_recovery"
                    if oracle_recovery_override
                    else "stop_constrained_system2"
                    if force_continue
                    else None
                )
                with stop_feature_labels_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(stop_feature_row, sort_keys=True) + "\n")
                collected_stop_features += 1
                if force_continue:
                    original_decision_scores = response.get("system2_decision_scores") or {}
                    original_class_probabilities = (
                        original_decision_scores.get("class_probabilities") or {}
                    )
                    stop_score_records.append(
                        {
                            "step": int(step_id),
                            "decision": "oracle_force_continue",
                            "stop_head_decision": None,
                            "stop_head_probability": None,
                            "stop_probability": float(
                                original_class_probabilities.get("stop", 0.0)
                            )
                            if original_decision_scores
                            else None,
                            "stop_log_odds": float(
                                original_decision_scores.get("stop_log_odds", 0.0)
                            )
                            if original_decision_scores
                            else None,
                        }
                    )

                if oracle_recovery_override:
                    if oracle_recovery_follower is None:
                        raise RuntimeError(
                            "Oracle recovery became active without a shortest-path follower"
                        )
                    oracle_recovery_calls += 1
                    if force_continue:
                        forced_continue_calls += 1
                    if boundary_probe_index is not None:
                        boundary_probe_rows += 1
                        sweep_complete = boundary_probe_state.finish_current_probe()
                        if not sweep_complete:
                            recovery_action = int(ActionCode.LEFT)
                            before_position = _agent_position(env)
                            observations, done = _apply_habitat_action(
                                env,
                                recovery_action,
                            )
                            after_position = _agent_position(env)
                            step_id += 1
                            oracle_recovery_primitive_actions += 1
                            boundary_probe_turns += 1
                            _record_post_action_step(
                                step_recorder,
                                env,
                                step_id=step_id,
                                phase="rpc_dagger_boundary_probe",
                                action=recovery_action,
                                image_size=image_size,
                                vlm_output=llm_output,
                            )
                            print(
                                "  [dagger] boundary view sweep: "
                                f"index={boundary_probe_index + 1}/"
                                f"{args.system2_stop_boundary_probe_views} "
                                f"distance={distance_to_goal:.3f} "
                                f"moved={float(np.linalg.norm(np.asarray(after_position) - np.asarray(before_position))):.6f}",
                                flush=True,
                            )
                            continue
                    recovery_actions: list[int] = []
                    recovery_moved_m = 0.0
                    executed_goal_probe = False
                    deferred_goal_probe = False
                    collection_stop = False
                    primitive_budget = min(
                        oracle_recovery_actions_per_call,
                        max_steps_per_episode - step_id,
                    )
                    for _ in range(primitive_budget):
                        recovery_action, goal_probe = (
                            _next_shortest_path_recovery_action(
                                oracle_recovery_follower,
                                episode.goals[0].position,
                            )
                        )
                        if goal_probe and recovery_actions:
                            # The current System2 feature was captured before this
                            # chunk reached the goal. Query a fresh goal view before
                            # counting or executing an in-place probe.
                            deferred_goal_probe = True
                            break
                        if goal_probe:
                            executed_goal_probe = True
                            oracle_recovery_goal_probes += 1
                            collection_stop = (
                                should_finish_oracle_recovery_collection(
                                    goal_probe_count=oracle_recovery_goal_probes,
                                    max_goal_probes=int(
                                        args.system2_stop_oracle_recovery_goal_probes
                                    ),
                                )
                            )
                            if collection_stop:
                                recovery_action = int(ActionCode.STOP)
                                oracle_recovery_state.complete()
                        before_position = _agent_position(env)
                        observations, done = _apply_habitat_action(
                            env,
                            recovery_action,
                        )
                        after_position = _agent_position(env)
                        recovery_moved_m += float(
                            np.linalg.norm(
                                np.asarray(after_position)
                                - np.asarray(before_position)
                            )
                        )
                        step_id += 1
                        oracle_recovery_primitive_actions += 1
                        recovery_actions.append(recovery_action)
                        _record_post_action_step(
                            step_recorder,
                            env,
                            step_id=step_id,
                            phase=(
                                "rpc_dagger_collection_stop"
                                if collection_stop
                                else "rpc_dagger_goal_probe"
                                if executed_goal_probe
                                else "rpc_dagger_shortest_path_recovery"
                            ),
                            action=recovery_action,
                            image_size=image_size,
                            vlm_output=llm_output,
                        )
                        if executed_goal_probe or done:
                            break
                    recovery_mode = (
                        "collection_stop"
                        if collection_stop
                        else "goal_probe"
                        if executed_goal_probe
                        else "navigate_to_goal"
                        if deferred_goal_probe
                        else "navigate"
                    )
                    print(
                        "  [dagger] persistent shortest-path recovery: "
                        f"original={'STOP' if terminal else 'non-STOP'} "
                        f"label={rollout_label} distance={distance_to_goal:.3f} "
                        f"activation={oracle_recovery_state.activation_reason} "
                        f"actions={recovery_actions} "
                        f"mode={recovery_mode} "
                        f"goal_probes={oracle_recovery_goal_probes}/"
                        f"{args.system2_stop_oracle_recovery_goal_probes} "
                        f"moved={recovery_moved_m:.3f}",
                        flush=True,
                    )
                    continue

                if force_continue:
                    response = _rpc_plan_panoramic(
                        client,
                        instruction=instruction,
                        current_views=current_views,
                        history_panoramas=prompt_history,
                        lookdown_img=lookdown_img,
                        vlm_image_size=vlm_image_size,
                        traj_image_size=traj_image_size,
                        system1_coord_order=system1_coord_order,
                        trajectory_selection=args.trajectory_selection,
                        trajectory_x_sign=args.trajectory_x_sign,
                        trajectory_heading_alignment=args.trajectory_heading_alignment,
                        jpeg_quality=args.rpc_jpeg_quality,
                        scene_id=scene_id,
                        episode_id=episode_id,
                        system2_call_index=system2_calls - 1,
                        protocol_seed=args.rpc_protocol_seed,
                        require_deterministic_sampling=(
                            args.rpc_require_deterministic_sampling
                        ),
                        oracle_system2=None,
                        force_non_stop=True,
                    )
                    if response.get("system2_force_non_stop") is not True:
                        raise RuntimeError(
                            "RPC server did not acknowledge forced DAgger continuation"
                        )
                    terminal = bool(response.get("terminal", False))
                    if terminal:
                        raise RuntimeError(
                            "Forced DAgger continuation returned a terminal response"
                        )
                    forced_continue_calls += 1
                    llm_output = str(response.get("llm_output", ""))
                    raw_actions = [
                        int(action) for action in response.get("actions", [])
                    ]
                    actions = closed_loop_guard.limit_actions(raw_actions)
                    print(
                        "  [dagger] oracle-labelled false STOP -> "
                        f"constrained non-STOP output: {llm_output}",
                        flush=True,
                    )
                    if response.get("trajectory_summary"):
                        trajectory_calls += 1
                        print(
                            "  [dagger] constrained trajectory "
                            f"{response['trajectory_summary']}, actions={raw_actions}",
                            flush=True,
                        )
                    elif raw_actions:
                        print(
                            f"  [dagger] constrained actions={raw_actions}",
                            flush=True,
                        )
                    if step_recorder is not None:
                        state = env._sim.get_agent(0).get_state()
                        position = np.array(state.position, dtype=float)
                        rotation = quaternion.as_float_array(state.rotation)
                        step_recorder.record_step(
                            {
                                "step_id": step_id,
                                "phase": "rpc_dagger_force_continue",
                                "position": position,
                                "heading_deg": _quat_to_heading_deg(rotation),
                                "rotation": rotation,
                                "distance_to_goal": _metric_distance_to_goal(env),
                                "vlm_output": llm_output,
                                "pixel_goal": response.get("pixel_goal"),
                                "pano_goal_view": response.get("pano_goal_view"),
                                "oracle_system2": response.get("oracle_system2"),
                                HEATMAPVLN_RPC_SAMPLING_FIELD: response.get(
                                    HEATMAPVLN_RPC_SAMPLING_FIELD
                                ),
                                "current_views": current_views,
                            }
                        )
            stop_head = response.get("system2_stop_head") or {}
            head_decision = str(stop_head.get("decision", ""))
            trusted_terminal = should_trust_temporal_terminal(
                enabled=bool(args.system2_stop_accept_temporal_confirmed),
                decision=head_decision,
                observed_margin=stop_head.get("temporal_min_margin"),
                min_margin=float(args.system2_stop_temporal_trust_min_margin),
            )
            trusted_terminal_source = (
                "temporal_confirms_original_stop" if trusted_terminal else None
            )
            if head_decision:
                stop_head_records.append(
                    {
                        "step": int(step_id),
                        "decision": head_decision,
                        "mode": stop_head.get("mode"),
                        "stop_probability": stop_head.get("stop_probability"),
                        "threshold": stop_head.get("threshold"),
                        "add_stop_threshold": stop_head.get("add_stop_threshold"),
                        "veto_stop_threshold": stop_head.get("veto_stop_threshold"),
                        "qwen_stop_probability": stop_head.get(
                            "qwen_stop_probability"
                        ),
                        "stop_decision_class_probabilities": stop_head.get(
                            "class_probabilities"
                        ),
                        "stop_decision_stop_log_odds": stop_head.get(
                            "stop_log_odds"
                        ),
                        "add_min_qwen_stop_probability": stop_head.get(
                            "add_min_qwen_stop_probability"
                        ),
                        "policy_kind": stop_head.get("policy_kind"),
                        "temporal_accepted": stop_head.get("temporal_accepted"),
                        "temporal_decision": stop_head.get("temporal_decision"),
                        "static_add_decision": stop_head.get("static_add_decision"),
                        "static_add_stop_probability": stop_head.get(
                            "static_add_stop_probability"
                        ),
                        "temporal_min_margin": stop_head.get("temporal_min_margin"),
                        "temporal_trust_min_margin": float(
                            args.system2_stop_temporal_trust_min_margin
                        ),
                        "member_probabilities": stop_head.get(
                            "member_probabilities"
                        ),
                        "member_thresholds": stop_head.get("member_thresholds"),
                        "member_margins": stop_head.get("member_margins"),
                        "original_output": stop_head.get("original_output"),
                        "constrained_output": stop_head.get("constrained_output"),
                        "constrained_generation_output": stop_head.get(
                            "constrained_generation_output"
                        ),
                        "constrained_generation_fallback": stop_head.get(
                            "constrained_generation_fallback"
                        ),
                        "trusted_terminal": trusted_terminal,
                        "trusted_terminal_source": trusted_terminal_source,
                    }
                )
                member_probabilities = stop_head.get("member_probabilities") or []
                member_thresholds = stop_head.get("member_thresholds") or []
                member_summary = ""
                if member_probabilities:
                    member_summary = (
                        f" members={[round(float(value), 4) for value in member_probabilities]}"
                        f" member_thresholds={[round(float(value), 4) for value in member_thresholds]}"
                    )
                print(
                    "  [debug] System2 STOP head: "
                    f"decision={head_decision} "
                    f"p={float(stop_head.get('stop_probability', 0.0)):.4f} "
                    f"effective_threshold={float(stop_head.get('threshold', 0.5)):.4f} "
                    f"add={float(stop_head.get('add_stop_threshold', 0.5)):.4f} "
                    f"veto={float(stop_head.get('veto_stop_threshold', 0.5)):.4f}"
                    f"{member_summary}",
                    flush=True,
                )
            decision_scores = response.get("system2_decision_scores") or {}
            class_probabilities = decision_scores.get("class_probabilities") or {}
            stop_probability = float(class_probabilities.get("stop", 0.0))
            stop_vote_probability = stop_probability
            if stop_head.get("mode") == "stop_decision_adapter":
                stop_vote_probability = float(
                    stop_head.get("stop_probability", stop_probability)
                )
            if decision_scores and (terminal or stop_probability >= 0.01):
                print(
                    "  [debug] System2 decision scores: "
                    f"selected={decision_scores.get('selected')} "
                    f"stop_p={stop_probability:.4f} "
                    f"stop_log_odds={float(decision_scores.get('stop_log_odds', 0.0)):.3f}",
                    flush=True,
                )
            terminal_vote_source = None
            if terminal:
                terminal_vote_source = head_decision or "system2_original_stop"
            stop_decision = closed_loop_guard.observe_system2_terminal(
                terminal,
                stop_probability=(
                    stop_vote_probability
                    if decision_scores or stop_head.get("mode") == "stop_decision_adapter"
                    else None
                ),
                trusted_terminal=trusted_terminal,
                terminal_source=terminal_vote_source,
            )
            if terminal:
                stop_score_records.append(
                    {
                        "step": int(step_id),
                        "decision": str(stop_decision),
                        "stop_head_decision": head_decision or None,
                        "stop_head_probability": stop_head.get("stop_probability"),
                        "stop_head_policy_kind": stop_head.get("policy_kind"),
                        "stop_head_member_probabilities": stop_head.get(
                            "member_probabilities"
                        ),
                        "stop_head_member_thresholds": stop_head.get(
                            "member_thresholds"
                        ),
                        "trusted_terminal": trusted_terminal,
                        "trusted_terminal_source": trusted_terminal_source,
                        "terminal_vote_source": terminal_vote_source,
                        "stop_probability": float(
                            class_probabilities.get("stop", 0.0)
                        )
                        if decision_scores
                        else None,
                        "stop_log_odds": float(
                            decision_scores.get("stop_log_odds", 0.0)
                        )
                        if decision_scores
                        else None,
                    }
                )
            if stop_decision == STOP_PROBE:
                action = closed_loop_guard.next_stop_probe_action()
                before_position = _agent_position(env)
                observations, done = _apply_habitat_action(env, action)
                after_position = _agent_position(env)
                closed_loop_guard.observe_action(action, before_position, after_position)
                step_id += 1
                stop_probes += 1
                print(
                    "  [guard] System2 STOP verification sweep; "
                    f"terminal={terminal} "
                    f"trusted_terminal={trusted_terminal} "
                    f"trusted_terminal_source={trusted_terminal_source} "
                    f"vote_source={closed_loop_guard.stop_vote_source} "
                    f"probe_action={int(action)} "
                    f"vote={closed_loop_guard.stop_votes}/"
                    f"{guard_config.stop_confirmations} "
                    f"gap={closed_loop_guard.stop_gap_calls}/"
                    f"{guard_config.stop_confirmation_max_gap_calls}",
                    flush=True,
                )
                _record_post_action_step(
                    step_recorder,
                    env,
                    step_id=step_id,
                    phase="rpc_stop_probe",
                    action=int(action),
                    image_size=image_size,
                    vlm_output=llm_output,
                )
                continue

            if terminal:
                if stop_decision != STOP_ACCEPT:
                    raise RuntimeError(f"Unexpected STOP guard decision: {stop_decision!r}")
                action = actions[0] if actions else ActionCode.STOP
                observations, done = _apply_habitat_action(env, action)
                step_id += 1
                _record_post_action_step(
                    step_recorder,
                    env,
                    step_id=step_id,
                    phase="rpc_terminal",
                    action=int(action),
                    image_size=image_size,
                    vlm_output=llm_output,
                )
                continue

            plan_recovery = closed_loop_guard.observe_plan(
                response.get("pano_goal_view"),
                _agent_position(env),
            )
            if plan_recovery is not None:
                recovery_reasons.append(plan_recovery.reason)
                local_actions = list(plan_recovery.actions)
                forward_action_count = 0
                executed_history_panoramas = _trim_recovery_history(
                    executed_history_panoramas,
                    int(args.closed_loop_recovery_history_keep),
                )
                print(
                    "  [guard] recovery before plan execution: "
                    f"reason={plan_recovery.reason} actions={list(plan_recovery.actions)} "
                    f"history={len(executed_history_panoramas)}",
                    flush=True,
                )
                continue

            if not actions:
                observations, done = _apply_habitat_action(env, ActionCode.STOP)
                step_id += 1
                _record_post_action_step(
                    step_recorder,
                    env,
                    step_id=step_id,
                    phase="rpc_empty_actions",
                    action=int(ActionCode.STOP),
                    image_size=image_size,
                    vlm_output=llm_output,
                )
                continue

            first_action = int(actions.pop(0))
            local_actions = actions
            forward_action_count = 0
            if first_action == ActionCode.STOP:
                local_actions = []
                continue
            before_position = _agent_position(env)
            before = _env_trace_summary(env) if _debug_input_trace_enabled(args) else None
            observations, done = _apply_habitat_action(env, first_action)
            after_position = _agent_position(env)
            if before is not None:
                print(
                    f"  [debug] executed first RPC action={int(first_action)} {before} -> {_env_trace_summary(env)}",
                    flush=True,
                )
            step_id += 1
            forward_action_count += 1
            _record_post_action_step(
                step_recorder,
                env,
                step_id=step_id,
                phase="rpc_first_action",
                action=int(first_action),
                image_size=image_size,
                vlm_output=llm_output,
            )
            recovery = closed_loop_guard.observe_action(
                first_action,
                before_position,
                after_position,
            )
            if recovery is not None and not done:
                recovery_reasons.append(recovery.reason)
                local_actions = list(recovery.actions)
                forward_action_count = 0
                executed_history_panoramas = _trim_recovery_history(
                    executed_history_panoramas,
                    int(args.closed_loop_recovery_history_keep),
                )
                print(
                    "  [guard] recovery: "
                    f"reason={recovery.reason} actions={list(recovery.actions)} "
                    f"history={len(executed_history_panoramas)}",
                    flush=True,
                )

        metrics = env.get_metrics()
        sucs.append(metrics["success"])
        spls.append(metrics["spl"])
        oss.append(metrics["oracle_success"])
        nes.append(metrics["distance_to_goal"])
        total_stop_probes += stop_probes
        total_recoveries += len(recovery_reasons)
        if step_recorder is not None:
            step_recorder.finalize(
                scene_id=scene_id,
                episode_id=episode_id,
                success=metrics["success"],
                spl=metrics["spl"],
                total_steps=step_id,
                vlm_calls=system2_calls,
                traj_calls=trajectory_calls,
            )
        print(
            f"  => success: {metrics['success']}, spl: {metrics['spl']:.4f}, "
            f"os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']:.4f}, "
            f"vlm_calls: {system2_calls}, trajectory_calls: {trajectory_calls}, "
            f"stop_probes: {stop_probes}, recoveries: {len(recovery_reasons)}"
        )
        result = {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "success": metrics["success"],
            "spl": metrics["spl"],
            "os": metrics["oracle_success"],
            "ne": metrics["distance_to_goal"],
            "steps": step_id,
            "episode_instruction": instruction,
            "vlm_calls": system2_calls,
            "trajectory_calls": trajectory_calls,
            "rpc_server": args.rpc_server,
            "rpc_protocol": HEATMAPVLN_RPC_PROTOCOL_VERSION,
            "rpc_sampling_protocol": HEATMAPVLN_RPC_SAMPLING_PROTOCOL,
            "rpc_deterministic_sampling_enabled": True,
            "rpc_protocol_seed": int(args.rpc_protocol_seed),
            "rpc_require_deterministic_sampling": bool(args.rpc_require_deterministic_sampling),
            "auto_stop_distance": float(args.auto_stop_distance),
            "trajectory_selection": str(args.trajectory_selection),
            "trajectory_x_sign": float(args.trajectory_x_sign),
            "trajectory_heading_alignment": str(args.trajectory_heading_alignment),
            "system1_coord_order": str(system1_coord_order),
            "oracle_system2": bool(getattr(args, "oracle_system2", False)),
            "rpc_action_chunk_size": guard_config.action_chunk_size,
            "system2_stop_confirmations": guard_config.stop_confirmations,
            "system2_stop_confirmation_max_gap_calls": (
                guard_config.stop_confirmation_max_gap_calls
            ),
            "system2_stop_confirmation_view_sweep": (
                guard_config.stop_confirmation_view_sweep
            ),
            "system2_stop_accept_temporal_confirmed": bool(
                args.system2_stop_accept_temporal_confirmed
            ),
            "system2_stop_temporal_trust_min_margin": float(
                args.system2_stop_temporal_trust_min_margin
            ),
            "system2_stop_high_confidence_threshold": (
                guard_config.stop_high_confidence_threshold
            ),
            "system2_stop_probe_turn": guard_config.stop_probe_turn,
            "closed_loop_guard": guard_config.loop_guard_enabled,
            "closed_loop_stop_probes": stop_probes,
            "system2_stop_score_records": stop_score_records,
            "system2_stop_head_records": stop_head_records,
            "system2_stop_feature_collection": collect_stop_features,
            "system2_stop_collect_oracle_recovery_after_negative": (
                oracle_recovery_after_negative
            ),
            "system2_stop_collect_oracle_path_from_start": oracle_path_from_start,
            "system2_stop_features_collected": collected_stop_features,
            "system2_stop_forced_continue_calls": forced_continue_calls,
            "system2_stop_oracle_recovery_calls": oracle_recovery_calls,
            "system2_stop_oracle_recovery_primitive_actions": (
                oracle_recovery_primitive_actions
            ),
            "system2_stop_oracle_recovery_actions_per_call": (
                oracle_recovery_actions_per_call
            ),
            "system2_stop_oracle_recovery_goal_probes": (
                oracle_recovery_goal_probes
            ),
            "system2_stop_boundary_probe_rows": boundary_probe_rows,
            "system2_stop_boundary_probe_turns": boundary_probe_turns,
            "system2_stop_boundary_probe_completed": boundary_probe_state.completed,
            "system2_stop_oracle_recovery_goal_probe_limit": int(
                args.system2_stop_oracle_recovery_goal_probes
            ),
            "system2_stop_oracle_recovery_activations": (
                oracle_recovery_state.activations
            ),
            "system2_stop_oracle_recovery_activation_reason": (
                oracle_recovery_state.activation_reason
            ),
            "system2_stop_oracle_recovery_from_cohort_triggers": (
                oracle_recovery_from_cohort_triggers
            ),
            "system2_stop_historical_trigger_reached": (
                historical_recovery_trigger_reached
            ),
            "system2_stop_historical_trigger_call_index": (
                historical_recovery_trigger.system2_call_index
                if historical_recovery_trigger is not None
                else None
            ),
            "system2_stop_historical_trigger_step": (
                historical_recovery_trigger.step
                if historical_recovery_trigger is not None
                else None
            ),
            "system2_stop_historical_trigger_distance_m": (
                historical_recovery_trigger.distance_m
                if historical_recovery_trigger is not None
                else None
            ),
            "system2_stop_historical_trigger_source_labels": (
                historical_recovery_trigger.source_labels
                if historical_recovery_trigger is not None
                else None
            ),
            "system2_stop_oracle_recovery_goal_radius_m": (
                oracle_recovery_goal_radius_m
            ),
            "closed_loop_recoveries": recovery_reasons,
            "closed_loop_recovery_forward_steps": guard_config.recovery_forward_steps,
            "closed_loop_recovery_follow_last_turn": guard_config.recovery_follow_last_turn,
        }
        if bool(getattr(args, "oracle_system2", False)):
            result["oracle_system2_lookahead_m"] = float(args.oracle_system2_lookahead_m)
            result["oracle_system2_strategy"] = str(args.oracle_system2_strategy)
            result["oracle_system2_max_side_dist_m"] = float(args.oracle_system2_max_side_dist_m)
        with open(progress_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        done_set.add(ep_key)
        process_bar.update(1)

    env.close()
    client.close()

    final_result = aggregate_navigation_metrics(sucs, spls, oss, nes)
    final_result.update(
        {
            "rpc_protocol": HEATMAPVLN_RPC_PROTOCOL_VERSION,
            "rpc_sampling_protocol": HEATMAPVLN_RPC_SAMPLING_PROTOCOL,
            "rpc_deterministic_sampling_enabled": True,
            "rpc_protocol_seed": int(args.rpc_protocol_seed),
            "rpc_require_deterministic_sampling": bool(args.rpc_require_deterministic_sampling),
            "auto_stop_distance": float(args.auto_stop_distance),
            "trajectory_selection": str(args.trajectory_selection),
            "trajectory_x_sign": float(args.trajectory_x_sign),
            "trajectory_heading_alignment": str(args.trajectory_heading_alignment),
            "system1_coord_order": str(system1_coord_order),
            "oracle_system2": bool(getattr(args, "oracle_system2", False)),
            "rpc_action_chunk_size": guard_config.action_chunk_size,
            "system2_stop_confirmations": guard_config.stop_confirmations,
            "system2_stop_confirmation_max_gap_calls": (
                guard_config.stop_confirmation_max_gap_calls
            ),
            "system2_stop_confirmation_view_sweep": (
                guard_config.stop_confirmation_view_sweep
            ),
            "system2_stop_accept_temporal_confirmed": bool(
                args.system2_stop_accept_temporal_confirmed
            ),
            "system2_stop_temporal_trust_min_margin": float(
                args.system2_stop_temporal_trust_min_margin
            ),
            "system2_stop_high_confidence_threshold": (
                guard_config.stop_high_confidence_threshold
            ),
            "system2_stop_probe_turn": guard_config.stop_probe_turn,
            "system2_stop_feature_collection": collect_stop_features,
            "system2_stop_collect_oracle_recovery_after_negative": (
                oracle_recovery_after_negative
            ),
            "system2_stop_collect_oracle_path_from_start": oracle_path_from_start,
            "system2_stop_oracle_recovery_from_cohort_triggers": (
                oracle_recovery_from_cohort_triggers
            ),
            "system2_stop_oracle_recovery_goal_probe_limit": int(
                args.system2_stop_oracle_recovery_goal_probes
            ),
            "system2_stop_oracle_recovery_actions_per_call": (
                oracle_recovery_actions_per_call
            ),
            "system2_stop_feature_labels": (
                str(stop_feature_labels_path) if collect_stop_features else None
            ),
            "system2_stop_multimodal_regular_min_stop_log_odds": (
                args.system2_stop_multimodal_regular_min_stop_log_odds
            ),
            "system2_stop_multimodal_examples_considered_this_process": (
                stop_multimodal_recorder.considered
                if stop_multimodal_recorder is not None
                else 0
            ),
            "system2_stop_multimodal_examples_recorded_this_process": (
                stop_multimodal_recorder.recorded
                if stop_multimodal_recorder is not None
                else 0
            ),
            "system2_stop_multimodal_examples_skipped_this_process": (
                stop_multimodal_recorder.skipped
                if stop_multimodal_recorder is not None
                else 0
            ),
            "system2_stop_multimodal_episode_provenance_fallbacks_this_process": (
                stop_multimodal_recorder.provenance_fallbacks
                if stop_multimodal_recorder is not None
                else 0
            ),
            "closed_loop_guard": guard_config.loop_guard_enabled,
            "closed_loop_stop_probes": total_stop_probes,
            "closed_loop_recoveries": total_recoveries,
            "closed_loop_recovery_forward_steps": guard_config.recovery_forward_steps,
            "closed_loop_recovery_follow_last_turn": guard_config.recovery_follow_last_turn,
        }
    )

    print("\n" + "=" * 60)
    print("Final Results:")
    print(f"  NE  (Navigation Error):  {final_result['NE']:.4f}")
    print(f"  OS  (Oracle Success):    {final_result['OS']:.4f}")
    print(f"  SR  (Success Rate):      {final_result['SR']:.4f}")
    print(f"  SPL (Success w/ Path):   {final_result['SPL']:.4f}")
    print(f"  Total episodes:          {final_result['total_episodes']}")
    print("=" * 60)

    with open(os.path.join(output_path, "result.json"), "w") as f:
        json.dump(final_result, f, indent=2)
    print(f"Results saved to {os.path.join(output_path, 'result.json')}")


# ═══════════════════════════════════════════════════════════════════════
# Section 9: Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════


def run_eval(args):
    if getattr(args, "rpc_server", ""):
        return run_eval_rpc_panoramic(args)

    device = torch.device(f"cuda:{args.gpu_id}")
    ensure_vln_measures_registered()

    print(f"Loading model from config={args.config}, checkpoint={args.checkpoint or '<none>'}")
    if args.base_checkpoint:
        print(f"Loading base checkpoint first: {args.base_checkpoint}")
    model, train_cfg = load_model(args, device)
    processor = model.qwen2_5_vl.processor
    processor.tokenizer.padding_side = "left"

    action_scale = train_cfg.get("data", {}).get("trajectory", {}).get("action_scale", 4.0)
    num_sample_trajs = train_cfg.get("model", {}).get("action_head", {}).get("nextdit", {}).get("num_sample_trajs", 32)
    has_nextdit = model.nextdit_action_head is not None and model.latent_queries is not None
    print(f"NextDiT action head available: {has_nextdit}")
    print(f"  action_scale={action_scale}, num_sample_trajs={num_sample_trajs}")
    print(f"  trajectory_selection={args.trajectory_selection}")

    panoramic_vlm_input = bool(train_cfg.get("data", {}).get("trajectory", {}).get("panoramic_vlm_input", False))
    print(f"Panoramic VLM input: {panoramic_vlm_input}")

    pano_latent_adapter = None
    if getattr(args, "pano_latent_adapter_checkpoint", None):
        hidden_dim = int(train_cfg.get("model", {}).get("llm", {}).get("hidden_dim", 3584))
        print(f"Loading pano-latent adapter from {args.pano_latent_adapter_checkpoint} (hidden_dim={hidden_dim})")
        pano_latent_adapter = _load_pano_latent_adapter(
            args.pano_latent_adapter_checkpoint,
            hidden_dim=hidden_dim,
            device=device,
            dtype=model.config.dtype,
        )
    elif getattr(model, "pano_latent_adapter", None) is not None:
        pano_latent_adapter = model.pano_latent_adapter
        pano_latent_adapter.eval()
        print("Using model-attached pano-latent adapter from config/checkpoint")

    force_teacher_model = None
    force_teacher_processor = None
    force_teacher_device = None
    if bool(getattr(args, "force_teacher_coord", False)):
        if not getattr(args, "force_teacher_internnav_model_path", ""):
            raise RuntimeError("--force_teacher_coord requires --force_teacher_internnav_model_path")
        if not getattr(args, "force_teacher_internnav_repo", ""):
            raise RuntimeError("--force_teacher_coord requires --force_teacher_internnav_repo")
        print(f"Loading InternNav teacher VLM for --force_teacher_coord from {args.force_teacher_internnav_model_path}")
        force_teacher_model, force_teacher_processor, force_teacher_device = _load_force_teacher_internnav(args, device)

    if panoramic_vlm_input:
        return _run_eval_panoramic_vlm(
            args=args,
            model=model,
            train_cfg=train_cfg,
            processor=processor,
            device=device,
            action_scale=action_scale,
            num_sample_trajs=num_sample_trajs,
            has_nextdit=has_nextdit,
            pano_latent_adapter=pano_latent_adapter,
            force_teacher_model=force_teacher_model,
            force_teacher_processor=force_teacher_processor,
            force_teacher_device=force_teacher_device,
        )

    hab_cfg = build_habitat_config(args)
    print("Creating Habitat environment ...")
    env = habitat.Env(config=hab_cfg)
    num_episodes = len(list(env.episodes))
    print(f"Total episodes: {num_episodes}")

    num_history = args.num_history
    max_steps_per_episode = args.max_steps_per_episode
    vlm_image_size, traj_image_size = _eval_image_sizes(train_cfg)
    print(f"vlm_image_size={vlm_image_size}, traj_image_size={traj_image_size}")

    # ── Resume / output management ──
    output_path = args.output_path
    progress_file = _prepare_progress_file(args, output_path)
    sucs, spls, oss, nes, done_set = _load_progress(progress_file)

    target_list, target_set = _episode_list_from_args(args)
    if target_list is not None:
        selected = restrict_habitat_env_to_episode_keys(env, target_list)
        print(f"Fixed episode list ({len(target_list)}): {args.episode_list}")
        print(f"Restricted Habitat iterator to {len(selected)} requested episodes")
    remaining = num_episodes - len(done_set)
    eval_limit = _eval_limit(args, remaining, target_list, done_set)
    print(f"Episodes already done: {len(done_set)}, remaining: {remaining}, this run: {eval_limit}")

    process_bar = tqdm.tqdm(total=eval_limit, desc="Evaluating", ncols=120)
    seen_episodes: set = set()
    eval_count = 0

    # ── Episode loop (iterator-driven, see ReadBeforeEvaluatingHabitat.md §16) ──
    while True:
        process_bar.set_postfix(
            SR=f"{float(np.mean(sucs)):.3f}" if sucs else "?",
            SPL=f"{float(np.mean(spls)):.3f}" if spls else "?",
        )
        if eval_count >= eval_limit:
            break

        observations = env.reset()
        episode = env.current_episode
        scene_id = episode.scene_id.split("/")[-2]
        episode_id = int(episode.episode_id)
        ep_key = (scene_id, episode_id)

        if ep_key in seen_episodes:
            break
        seen_episodes.add(ep_key)

        if target_set is not None and ep_key not in target_set:
            continue

        if ep_key in done_set:
            continue

        instruction = _normalize_instruction(episode.instruction.instruction_text)
        eval_count += 1
        print(f"\n[{eval_count}/{eval_limit}] Episode {scene_id}_{episode_id:04d}: {instruction[:80]}...")

        # ── Per-episode state (InternNav dual-system logic) ──
        rgb_history: list[Image.Image] = []
        action_seq: list[int] = []
        input_images: list[Image.Image] = []
        llm_output = ""
        action: int | None = None
        messages: list[dict] = []
        pix_goal_image: torch.Tensor | None = None
        _last_traj_hs: torch.Tensor | None = None
        local_actions: list[int] = []
        forward_action_count = 0
        step_id = 0
        done = False

        while (not done) and (step_id < max_steps_per_episode):
            sys.stdout.flush()
            stop_result = _maybe_stop_at_success(env, args, step_id)
            if stop_result is not None:
                observations, done, step_id = stop_result
                continue

            print(
                f"  [step_id={step_id}] Capturing observations + VLM inference ...",
                flush=True,
            )

            rgb_arr = _extract_rgb_array(observations)
            if rgb_arr is None:
                rgb_arr = np.zeros((480, 640, 3), dtype=np.uint8)
            image = _rgb_array_to_pil(rgb_arr)

            if action == ActionCode.LOOKDOWN:
                lookdown_img = image.resize(vlm_image_size)
            else:
                rgb_history.append(image.resize((args.resize_w, args.resize_h)))

                down_observations = env.step(ActionCode.LOOKDOWN)
                down_observations = env.step(ActionCode.LOOKDOWN)
                down_rgb = _extract_rgb_array(down_observations)
                if down_rgb is None:
                    down_rgb = np.zeros((vlm_image_size[1], vlm_image_size[0], 3), dtype=np.uint8)
                lookdown_img = _rgb_array_to_pil(down_rgb, vlm_image_size)
                env.step(ActionCode.LOOKUP)
                env.step(ActionCode.LOOKUP)

            if len(action_seq) == 0 and pix_goal_image is None:
                if action == ActionCode.LOOKDOWN:
                    sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                    input_images += [lookdown_img]
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": llm_output}]})
                    input_img_id = -1
                else:
                    sources = [
                        {"from": "human", "value": LEGACY_PROMPT_TEMPLATE},
                        {"from": "gpt", "value": ""},
                    ]
                    sources[0]["value"] = sources[0]["value"].replace("<instruction>.", instruction)

                    cur_images = rgb_history[-1:]
                    if step_id == 0:
                        history_id = []
                    else:
                        history_id = np.unique(np.linspace(0, step_id - 1, num_history, dtype=np.int32)).tolist()
                        placeholder = (DEFAULT_IMAGE_TOKEN + "\n") * len(history_id)
                        sources[0]["value"] += f" These are your historical observations: {placeholder}."

                    history_id = sorted(history_id)
                    input_images = [rgb_history[i] for i in history_id] + cur_images
                    input_img_id = 0

                prompt = random.choice(LEGACY_CONJUNCTIONS) + DEFAULT_IMAGE_TOKEN
                sources[0]["value"] += f" {prompt}."
                prompt_instruction = copy.deepcopy(sources[0]["value"])
                parts = split_and_clean(prompt_instruction)

                content = []
                for part in parts:
                    if part == DEFAULT_IMAGE_TOKEN:
                        content.append({"type": "image", "image": input_images[input_img_id]})
                        input_img_id += 1
                    else:
                        content.append({"type": "text", "text": part})

                messages.append({"role": "user", "content": content})

                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = processor(
                    text=[text],
                    images=input_images,
                    return_tensors="pt",
                ).to(device)

                print(
                    f"  [debug] input_ids shape={inputs.input_ids.shape}, calling model.generate ...",
                    flush=True,
                )
                with torch.no_grad():
                    output_ids = model.qwen2_5_vl.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        use_cache=True,
                        return_dict_in_generate=True,
                    ).sequences

                llm_output = processor.tokenizer.decode(
                    output_ids[0][inputs.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )
                print(f"  step_id: {step_id}, VLM output: {llm_output}")

                if action == ActionCode.LOOKDOWN:
                    env.step(ActionCode.LOOKUP)
                    observations = env.step(ActionCode.LOOKUP)
                    done = env.episode_over

                if _vlm_requests_stop(llm_output):
                    observations, done = _apply_habitat_action(env, ActionCode.STOP)
                    step_id += 1
                    messages = []
                    continue

                pixel_goal = _parse_pixel_goal(
                    llm_output,
                    vlm_image_size,
                    # Same compatibility policy as the in-process panoramic
                    # path: accept bare "u v" as front-view fallback, while
                    # keeping malformed structured view lines invalid.
                    allow_legacy_coord=True,
                )
                if pixel_goal is not None:
                    print(f"  predicted pixel_goal {pixel_goal}")

                    if not has_nextdit:
                        observations, done = _apply_habitat_action(env, ActionCode.STOP)
                        step_id += 1
                        messages = []
                        continue

                    lookdown_traj_img = (
                        lookdown_img if lookdown_img.size == traj_image_size else lookdown_img.resize(traj_image_size)
                    )
                    lookdown_t = _lookdown_to_traj_tensor(lookdown_traj_img, device)
                    pix_goal_image = lookdown_t.clone()
                    traj_images = torch.stack([pix_goal_image, lookdown_t]).unsqueeze(0).to(device)

                    print("  [debug] calling generate_latents ...", flush=True)
                    lq = model.latent_queries.expand(1, -1, -1).to(
                        device=device,
                        dtype=model.config.dtype,
                    )
                    condition_output_ids = _condition_output_ids_for_pixel_goal(
                        output_ids=output_ids,
                        prompt_len=inputs.input_ids.shape[1],
                        tokenizer=processor.tokenizer,
                        pixel_goal=pixel_goal,
                        llm_output=llm_output,
                        coord_order="generated",
                    )
                    with torch.no_grad():
                        _last_traj_hs = model.qwen2_5_vl.generate_latents(
                            output_ids=condition_output_ids,
                            pixel_values=inputs.get("pixel_values"),
                            image_grid_thw=inputs.get("image_grid_thw"),
                            latent_queries=lq,
                            attention_mask=inputs.get("attention_mask"),
                            mm_token_type_ids=inputs.get("mm_token_type_ids"),
                        )
                        if _debug_input_trace_enabled(args):
                            _per_q = [
                                float(_last_traj_hs[0, i].float().norm().item()) for i in range(_last_traj_hs.shape[1])
                            ]
                            print(
                                "  [debug] traj_hs total_norm="
                                f"{float(_last_traj_hs.float().norm().item()):.3f} "
                                f"per_query={_per_q}",
                                flush=True,
                            )
                    if pano_latent_adapter is not None:
                        _last_traj_hs = _maybe_apply_pano_latent_adapter(
                            _last_traj_hs,
                            pano_latent_adapter,
                            view_id=_parse_pano_view_id(llm_output) or "front",
                            pixel_goal=pixel_goal,
                            image_size=vlm_image_size,
                        )

                    print("  [debug] calling get_trajectory ...", flush=True)
                    with torch.no_grad():
                        trajectory = _trajectory_from_condition(
                            model.nextdit_action_head,
                            _last_traj_hs,
                            traj_images=traj_images,
                        )

                    local_actions = _finalize_local_actions(
                        traj_to_actions(
                            trajectory,
                            num_sample_trajs=num_sample_trajs,
                            action_scale=action_scale,
                            trajectory_selection=args.trajectory_selection,
                        )
                    )

                    forward_action_count = 0
                    action = local_actions[0] if local_actions else ActionCode.STOP
                    if action == ActionCode.STOP:
                        pix_goal_image = None
                        _last_traj_hs = None
                        local_actions = []
                        action = ActionCode.LEFT
                        observations, done = _apply_habitat_action(env, action)
                        step_id += 1
                        messages = []
                        continue
                else:
                    action_seq = parse_actions(llm_output, LEGACY_ACTIONS2IDX)
                    print(f"  actions {action_seq}")

            if len(action_seq) != 0:
                action = action_seq.pop(0)
            elif pix_goal_image is not None:
                if len(local_actions) == 0:
                    lookdown_traj_img = (
                        lookdown_img if lookdown_img.size == traj_image_size else lookdown_img.resize(traj_image_size)
                    )
                    lookdown_t = _lookdown_to_traj_tensor(lookdown_traj_img, device)
                    traj_images = torch.stack([pix_goal_image, lookdown_t]).unsqueeze(0).to(device)

                    with torch.no_grad():
                        trajectory = _trajectory_from_condition(
                            model.nextdit_action_head,
                            _last_traj_hs,
                            traj_images=traj_images,
                        )

                    local_actions = _finalize_local_actions(
                        traj_to_actions(
                            trajectory,
                            num_sample_trajs=num_sample_trajs,
                            action_scale=action_scale,
                            trajectory_selection=args.trajectory_selection,
                        )
                    )
                    action = local_actions.pop(0) if local_actions else ActionCode.STOP
                else:
                    action = local_actions.pop(0)

                forward_action_count += 1
                if forward_action_count > MAX_STEPS:
                    pix_goal_image = None
                    _last_traj_hs = None
                    messages = []
                    forward_action_count = 0
                    local_actions = []
                    continue

                if action == ActionCode.STOP:
                    # Local System1 STOP means replan from System2.  Only an
                    # explicit high-level VLM STOP should finish the episode.
                    pix_goal_image = None
                    _last_traj_hs = None
                    messages = []
                    forward_action_count = 0
                    local_actions = []
                    continue
            else:
                action = ActionCode.STOP

            observations, done = _apply_habitat_action(env, action)
            step_id += 1
            messages = []

        # ── Collect metrics ──
        metrics = env.get_metrics()
        sucs.append(metrics["success"])
        spls.append(metrics["spl"])
        oss.append(metrics["oracle_success"])
        nes.append(metrics["distance_to_goal"])

        print(
            f"  => success: {metrics['success']}, spl: {metrics['spl']:.4f}, "
            f"os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']:.4f}"
        )

        result = {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "success": metrics["success"],
            "spl": metrics["spl"],
            "os": metrics["oracle_success"],
            "ne": metrics["distance_to_goal"],
            "steps": step_id,
            "episode_instruction": instruction,
        }
        with open(progress_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        done_set.add(ep_key)
        process_bar.update(1)

        if eval_count % 50 == 0:
            torch.cuda.empty_cache()

    env.close()

    # ── Aggregate results ──
    final_result = aggregate_navigation_metrics(sucs, spls, oss, nes)

    print("\n" + "=" * 60)
    print("Final Results:")
    print(f"  NE  (Navigation Error):  {final_result['NE']:.4f}")
    print(f"  OS  (Oracle Success):    {final_result['OS']:.4f}")
    print(f"  SR  (Success Rate):      {final_result['SR']:.4f}")
    print(f"  SPL (Success w/ Path):   {final_result['SPL']:.4f}")
    print(f"  Total episodes:          {final_result['total_episodes']}")
    print("=" * 60)

    with open(os.path.join(output_path, "result.json"), "w") as f:
        json.dump(final_result, f, indent=2)
    print(f"Results saved to {os.path.join(output_path, 'result.json')}")


# ═══════════════════════════════════════════════════════════════════════
# Section 10: CLI
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLNPipeline on VLN-CE R2R val_unseen (Habitat closed-loop)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config used for training (e.g. configs/train_config_internnav.yaml)",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional main/Stage 2 checkpoint path (.pth)")
    parser.add_argument(
        "--base_checkpoint", type=str, default=None, help="Optional Stage 1/base checkpoint loaded before --checkpoint"
    )
    parser.add_argument(
        "--pano_latent_adapter_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional pano latent adapter checkpoint. Accepts Stage2 adapter-only "
            "checkpoints with adapter_state_dict or Stage3 full checkpoints with "
            "pano_latent_adapter.* in trainable_state_dict. When set, the "
            "panoramic generate_latents output is projected through the adapter "
            "before NextDiT."
        ),
    )
    parser.add_argument(
        "--force_teacher_coord",
        action="store_true",
        default=False,
        help=(
            "H7 sanity check: at each System2 call, run the InternNav teacher VLM "
            "alongside the student to predict the pixel goal, and override the "
            "student's coord with the teacher's before conditioning the student "
            "latent. Used to test whether closing the coord-distribution gap is "
            "enough to close the closed-loop SR gap."
        ),
    )
    parser.add_argument(
        "--force_teacher_internnav_model_path",
        type=str,
        default=os.environ.get("INTERNNAV_MODEL_PATH", ""),
        help="Path to InternNav teacher VLM (required when --force_teacher_coord).",
    )
    parser.add_argument(
        "--force_teacher_internnav_repo",
        type=str,
        default=os.environ.get("INTERNNAV_REPO", ""),
        help="Path to InternNav source repo (required when --force_teacher_coord).",
    )
    parser.add_argument(
        "--force_teacher_torch_dtype",
        type=str,
        default="bf16",
        help="Teacher dtype: bf16 | fp16 | fp32.",
    )
    parser.add_argument(
        "--force_teacher_attn_impl",
        type=str,
        default="sdpa",
        help="Teacher attn impl: sdpa | flash_attention_2 | eager.",
    )
    parser.add_argument(
        "--force_teacher_flash_attn_stub",
        action="store_true",
        default=True,
        help=(
            "Install a flash_attn stub when teacher uses SDPA (mirrors "
            "collect_internnav_teacher_sidecar default). Disable with "
            "--no_force_teacher_flash_attn_stub if you actually have flash_attn."
        ),
    )
    parser.add_argument(
        "--no_force_teacher_flash_attn_stub",
        dest="force_teacher_flash_attn_stub",
        action="store_false",
    )
    parser.add_argument(
        "--force_teacher_coord_gpu_id",
        type=int,
        default=-1,
        help="Optional separate GPU id for the teacher VLM; -1 = use --gpu_id.",
    )
    parser.add_argument("--scenes_dir", type=str, default=DEFAULT_SCENES_DIR)
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_path", type=str, default="./logs/eval_r2r_val_unseen")
    parser.add_argument("--gpu_id", type=int, default=0, help="Torch CUDA device id for model inference")
    parser.add_argument(
        "--sim_gpu_id", type=int, default=0, help="Habitat-Sim GL device id; keep 0 for GLX/Xvfb builds"
    )
    parser.add_argument(
        "--rpc_server",
        type=str,
        default="",
        help=(
            "Optional model RPC server address (host:port). When set, this "
            "process runs Habitat only and sends panoramic observations to the "
            "model server."
        ),
    )
    parser.add_argument(
        "--rpc_timeout_ms",
        type=int,
        default=600000,
        help="RPC timeout for one model planning call in milliseconds.",
    )
    parser.add_argument(
        "--rpc_jpeg_quality",
        type=int,
        default=90,
        help="JPEG quality for RGB observations sent to the model server.",
    )
    parser.add_argument(
        "--rpc_protocol_seed",
        type=int,
        default=HEATMAPVLN_RPC_DEFAULT_PROTOCOL_SEED,
        help=(
            "Fixed cross-arm protocol seed used with scene/episode/System2 "
            "call index to derive each NextDiT SHA256 seed."
        ),
    )
    parser.add_argument(
        "--rpc_require_deterministic_sampling",
        action="store_true",
        default=False,
        help=("Ask the server to fail closed unless the deterministic NextDiT sampling record is complete and valid."),
    )
    parser.add_argument(
        "--oracle_system2",
        action="store_true",
        default=False,
        help=(
            "RPC-only ablation: replace generated panoramic System2 text with "
            "a reference-path oracle 'view: <front|right|back|left>\\n"
            "pixel: <u> <v>' before System1 latent/action inference."
        ),
    )
    parser.add_argument(
        "--oracle_system2_strategy",
        choices=("farthest_visible", "lookahead"),
        default="farthest_visible",
        help=(
            "How to select the reference-path oracle target. farthest_visible "
            "scans backward from the goal and uses side/back targets only "
            "within --oracle_system2_max_side_dist_m; lookahead uses a fixed "
            "--oracle_system2_lookahead_m target."
        ),
    )
    parser.add_argument(
        "--oracle_system2_lookahead_m",
        type=float,
        default=2.0,
        help="Reference-path lookahead distance in meters for --oracle_system2_strategy lookahead.",
    )
    parser.add_argument(
        "--oracle_system2_min_ahead_m",
        type=float,
        default=0.5,
        help="Minimum forward path progress in meters for --oracle_system2.",
    )
    parser.add_argument(
        "--oracle_system2_max_side_dist_m",
        type=float,
        default=6.0,
        help="Maximum side/back target distance for --oracle_system2_strategy farthest_visible.",
    )
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--max_steps_per_episode", type=int, default=500)
    parser.add_argument(
        "--auto_stop_distance",
        type=float,
        default=3.0,
        help="Execute Habitat STOP once distance_to_goal is within this threshold; set <=0 to disable.",
    )
    parser.add_argument(
        "--max_system2_calls_per_episode",
        type=int,
        default=0,
        help="Optional debug safety cap for VLM calls per episode; 0 disables the cap.",
    )
    parser.add_argument(
        "--rpc_action_chunk_size",
        type=int,
        default=MAX_LOCAL_STEPS,
        help=(
            "Maximum low-level actions executed from one RPC trajectory before "
            "System2 replans. Use 2 for tighter endpoint/obstacle feedback; 4 "
            "reproduces the original InternNav execution cadence."
        ),
    )
    parser.add_argument(
        "--system2_stop_confirmations",
        type=int,
        default=1,
        help=(
            "Consecutive System2 STOP votes required at the same position. "
            "Unconfirmed votes trigger an in-place probe turn; 1 disables verification."
        ),
    )
    parser.add_argument(
        "--system2_stop_confirmation_max_gap_calls",
        type=int,
        default=0,
        help=(
            "Maximum non-terminal System2 replans allowed between STOP votes. "
            "Zero preserves strict consecutive confirmation."
        ),
    )
    parser.add_argument(
        "--system2_stop_confirmation_view_sweep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "While a STOP vote is pending, consume allowed non-terminal gap calls "
            "as same-direction in-place view probes instead of executing their "
            "trajectory. Requires max gap calls >= 1."
        ),
    )
    parser.add_argument(
        "--system2_stop_accept_temporal_confirmed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Let a temporal verifier's unanimous confirmation of an original "
            "System2 STOP bypass generic multi-view confirmation. Static-head "
            "added STOP decisions still require normal confirmation."
        ),
    )
    parser.add_argument(
        "--system2_stop_temporal_trust_min_margin",
        type=float,
        default=0.005,
        help=(
            "Minimum unanimous temporal-verifier margin required before an "
            "original STOP may bypass generic confirmation. Borderline "
            "temporal confirmations fall back to normal multi-view voting."
        ),
    )
    parser.add_argument(
        "--system2_stop_high_confidence_threshold",
        type=float,
        default=None,
        help=(
            "When set, System2 STOP predictions at or above this structured-class "
            "probability bypass confirmation; lower-confidence STOP predictions "
            "still require --system2_stop_confirmations. Requires confirmations >= 2."
        ),
    )
    parser.add_argument(
        "--collect_system2_stop_features",
        action="store_true",
        help=(
            "Collect frozen-Qwen System2 decision features and privileged geodesic "
            "STOP labels for offline training. Never enable for official metrics."
        ),
    )
    parser.add_argument(
        "--collect_system2_stop_multimodal_examples",
        action="store_true",
        help=(
            "Alongside privileged STOP features, save the exact train-split "
            "System2 panorama/history prompt inputs for isolated LoRA training. "
            "The loader rejects val/val_unseen examples."
        ),
    )
    parser.add_argument(
        "--system2_stop_multimodal_regular_min_stop_log_odds",
        type=float,
        default=None,
        help=(
            "Optional collection-only filter: always persist labelled STOP "
            "positives and original false STOPs, but persist regular non-STOP "
            "rows only when their frozen-System2 STOP log-odds exceed this value."
        ),
    )
    parser.add_argument(
        "--system2_stop_collect_force_continue_negatives",
        action="store_true",
        help=(
            "DAgger-only oracle intervention: retain an original far-away STOP "
            "as a negative feature, then request a STOP-constrained continuation. "
            "Requires --collect_system2_stop_features and is never valid for metrics."
        ),
    )
    parser.add_argument(
        "--system2_stop_collect_oracle_recovery_after_negative",
        action="store_true",
        help=(
            "Privileged DAgger-only recovery: after recording a false original "
            "STOP, keep querying and recording the real System2 while Habitat's "
            "shortest-path follower supplies recovery actions. Inside the success "
            "radius, collect a bounded set of real-policy views before a privileged "
            "collection-only STOP. Requires forced negative continuation."
        ),
    )
    parser.add_argument(
        "--system2_stop_collect_oracle_path_from_start",
        action="store_true",
        help=(
            "Privileged offline collection only: query the real System2 while "
            "Habitat's shortest-path follower supplies every environment action, "
            "then collect bounded in-place goal views. Requires STOP feature "
            "collection and forced negative continuation."
        ),
    )
    parser.add_argument(
        "--system2_stop_collect_boundary_probe_sweep",
        action="store_true",
        help=(
            "Privileged oracle-path collection only: pause once in the configured "
            "negative distance band and collect a fixed-position multi-view sweep."
        ),
    )
    parser.add_argument(
        "--system2_stop_boundary_probe_min_distance_m",
        type=float,
        default=3.01,
    )
    parser.add_argument(
        "--system2_stop_boundary_probe_max_distance_m",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--system2_stop_boundary_probe_views",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--system2_stop_oracle_recovery_goal_probes",
        type=int,
        default=8,
        help=(
            "Number of real System2 views to collect after privileged recovery "
            "reaches the goal radius before ending the collection episode."
        ),
    )
    parser.add_argument(
        "--system2_stop_oracle_recovery_actions_per_call",
        type=int,
        default=1,
        help=(
            "Maximum privileged shortest-path primitive actions executed after "
            "one real System2 query. Goal probes always use a fresh query and "
            "one primitive action."
        ),
    )
    parser.add_argument(
        "--system2_stop_oracle_recovery_from_cohort_triggers",
        action="store_true",
        help=(
            "Privileged offline collection only: use each episode-list entry's "
            "audited historical false-STOP call as a deterministic fallback "
            "recovery trigger. A real false STOP or positive STOP may trigger "
            "collection earlier. Requires the full oracle-recovery mode."
        ),
    )
    parser.add_argument("--system2_stop_positive_radius_m", type=float, default=3.0)
    parser.add_argument("--system2_stop_negative_radius_m", type=float, default=4.0)
    parser.add_argument(
        "--system2_stop_probe_turn",
        choices=("left", "right"),
        default="left",
        help="Direction of the first in-place STOP verification turn; later probes alternate.",
    )
    parser.add_argument(
        "--closed_loop_guard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable non-privileged collision and local-loop recovery using only "
            "issued actions, agent self-motion, and predicted waypoint views."
        ),
    )
    parser.add_argument("--closed_loop_collision_epsilon_m", type=float, default=0.03)
    parser.add_argument("--closed_loop_collision_forward_limit", type=int, default=3)
    parser.add_argument("--closed_loop_motion_window_steps", type=int, default=32)
    parser.add_argument("--closed_loop_motion_min_path_m", type=float, default=2.0)
    parser.add_argument("--closed_loop_motion_max_net_m", type=float, default=0.75)
    parser.add_argument("--closed_loop_plan_window_calls", type=int, default=20)
    parser.add_argument("--closed_loop_plan_view_dominance", type=float, default=0.9)
    parser.add_argument("--closed_loop_plan_min_path_m", type=float, default=3.0)
    parser.add_argument("--closed_loop_plan_max_net_m", type=float, default=1.5)
    parser.add_argument("--closed_loop_recovery_turns", type=int, default=3)
    parser.add_argument("--closed_loop_recovery_forward_steps", type=int, default=0)
    parser.add_argument(
        "--closed_loop_recovery_follow_last_turn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Continue the latest System1 turn direction during collision recovery "
            "instead of alternating blindly."
        ),
    )
    parser.add_argument("--closed_loop_recovery_cooldown_steps", type=int, default=12)
    parser.add_argument(
        "--closed_loop_recovery_history_keep",
        type=int,
        default=2,
        help="Number of most recent panoramas retained after loop recovery.",
    )
    parser.add_argument(
        "--trajectory_selection",
        choices=TRAJECTORY_SELECTION_CHOICES,
        default="mean",
        help=(
            "How to select one local trajectory from parallel diffusion samples "
            "before decoding Habitat actions. mean preserves the original "
            "InternNav-compatible behavior."
        ),
    )
    parser.add_argument(
        "--trajectory_x_sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help=(
            "Multiply predicted trajectory x by this sign before InternNav "
            "discretization. Stage3 pose-derived targets use -x as forward."
        ),
    )
    parser.add_argument(
        "--trajectory_heading_alignment",
        choices=TRAJECTORY_HEADING_ALIGNMENT_CHOICES,
        default="none",
        help=(
            "RPC-only optional calibration. pano_pixel rotates the selected "
            "local trajectory so its endpoint bearing matches the panoramic "
            "view/pixel ray while preserving path length and curvature."
        ),
    )
    parser.add_argument(
        "--debug_input_trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Print compact pose, distance, image hash, and processor tensor stats for System2/System1 inputs."),
    )
    parser.add_argument(
        "--debug_save_input_images",
        type=int,
        default=0,
        help=(
            "Save raw System2 input images for the first N VLM calls in "
            "output_path/debug_inputs; 0 disables image dumps."
        ),
    )
    parser.add_argument(
        "--system1_coord_order",
        choices=("auto", "generated", "internnav_yx"),
        default="auto",
        help=(
            "Coordinate text used for System1 latent conditioning. auto keeps "
            "HeatmapVLN Stage2's generated [u v] order; use internnav_yx only "
            "for raw InternNav compatibility checks."
        ),
    )
    parser.add_argument("--max_episodes", type=int, default=None, help="Evaluate at most this many new episodes")
    parser.add_argument(
        "--episode_list", type=str, default=None, help="JSON file with fixed episodes [{scene_id, episode_id}, ...]"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from output_path/progress.json")
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="Delete output_path/progress.json and result.json before evaluating",
    )
    parser.add_argument(
        "--save_trajectory_steps",
        action="store_true",
        default=False,
        help=(
            "Record per-step agent state, VLM outputs, panorama images into "
            "output_path/<scene>_<ep>/trajectory_steps.json for offline HTML "
            "visualization via scripts/visualization/generate_trajectory_html.py."
        ),
    )
    args = parser.parse_args()
    if args.oracle_system2 and not args.rpc_server:
        raise RuntimeError("--oracle_system2 currently requires --rpc_server")
    if not args.rpc_server:
        _preflight_checkpoint_args(args)
    _resolve_eval_paths(args)
    run_eval(args)


if __name__ == "__main__":
    main()

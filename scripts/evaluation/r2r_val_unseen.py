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
       c. NextDiT get_trajectory → continuous trajectory (dx, dy, dyaw)
       d. Convert trajectory to discrete Habitat actions
       e. Execute up to MAX_LOCAL_STEPS actions
    5. If STOP or no coordinates → end episode

Adapted for habitat-lab 0.1.7 (YACS config).
"""

import faulthandler
import os
import sys
from pathlib import Path

faulthandler.enable()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

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
import types as _types


def _noop(*a, **kw):
    raise RuntimeError("flash_attn stub called – should use SDPA attention instead")

def _make_stub(name, attrs=None):
    m = _types.ModuleType(name)
    m.__spec__ = _importlib.machinery.ModuleSpec(name, None)
    m.__version__ = '2.7.4'
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

_fa = _make_stub('flash_attn', {
    'flash_attn_func': _noop,
    'flash_attn_varlen_func': _noop,
})
_make_stub('flash_attn_2_cuda')
_fa_iface = _make_stub('flash_attn.flash_attn_interface', {
    'flash_attn_func': _noop,
    'flash_attn_varlen_func': _noop,
    'flash_attn_gpu': _flash_kernel_stub,
    'flash_attn_cuda': _flash_kernel_stub,
})
_fa_bert = _make_stub('flash_attn.bert_padding', {
    'index_first_axis': _noop,
    'pad_input': _noop,
    'unpad_input': _noop,
})
_fa_rotary = _make_stub('flash_attn.layers', {})
_fa_rotary_mod = _make_stub('flash_attn.layers.rotary', {
    'apply_rotary_emb': _noop,
})
_fa.flash_attn_interface = _fa_iface
_fa.bert_padding = _fa_bert
_fa.layers = _fa_rotary
_fa_rotary.rotary = _fa_rotary_mod

import numpy as np

if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int64
if not hasattr(np, 'bool'):
    np.bool = np.bool_

# Import torch before habitat_sim (habitat_sim pulls torch during its __init__).
import torch as _torch_preload  # noqa: F401

def _find_preinit_scene() -> str | None:
    candidates = [
        os.environ.get("HEATMAPVLN_PREINIT_SCENE"),
        "/dataset/mp3d/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb",
        "/workspace/InternNav/data/scene_data/mp3d_ce/zsNo4HB9uLZ/zsNo4HB9uLZ.glb",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate

    for root in (Path("/dataset/mp3d/mp3d"), Path("/dataset/mp3d")):
        if not root.exists():
            continue
        try:
            scene = next(root.glob("*/*.glb"))
            return str(scene)
        except StopIteration:
            continue
    return None


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
        _dummy_sensor = _hsim.CameraSensorSpec()
        _dummy_sensor.uuid = "rgb"
        _dummy_sensor.sensor_type = _hsim.SensorType.COLOR
        _dummy_sensor.resolution = [64, 64]
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

    _dummy_cfg = _hsim.SimulatorConfiguration()
    # For GLX builds, pre-initialisation must use the local visible GPU index.
    _dummy_cfg.gpu_device_id = int(os.environ.get("HABITAT_GL_GPU_ID", "0"))
    _dummy_agent = _hsim.agent.AgentConfiguration()
    _dummy_agent.sensor_specifications = [_hsim.CameraSensorSpec()]
    _dummy_sim = _hsim.Simulator(_hsim.Configuration(_dummy_cfg, [_dummy_agent]))
    _dummy_sim.close()
    del _dummy_sim, _dummy_cfg, _dummy_agent
    print("GL context pre-initialized (NVIDIA GPU)", flush=True)

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
import hashlib
import itertools
import json
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
from scripts.training.model_builder import build_model
from scripts.training.utils import _normalize_state_key, load_config

from src.models.heatmap.input_constructor import construct_input

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


def _resolve_eval_paths(args, split: str = "val_unseen") -> None:
    """Resolve dataset/scenes paths for the shared Habitat evaluation environment."""

    requested_data_path = _expand_path_template(args.data_path, split)
    data_path = Path(requested_data_path)
    if data_path.exists():
        args.data_path = str(data_path.resolve())
    elif args.data_path == DEFAULT_DATA_PATH:
        data_candidates = [
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
            raise FileNotFoundError(
                "Could not find the VLN-CE dataset file. Tried:\n"
                f"  - {attempted}"
            )
        args.data_path = str(resolved.resolve())
    else:
        raise FileNotFoundError(
            f"Configured --data_path does not exist: {requested_data_path}"
        )

    requested_scenes_dir = _expand_path_template(args.scenes_dir, split)
    scenes_dir = Path(requested_scenes_dir)
    if scenes_dir.exists():
        args.scenes_dir = str(scenes_dir.resolve())
    elif args.scenes_dir == DEFAULT_SCENES_DIR:
        scenes_candidates = [
            Path("/home/intern/zhr/fjl/InternNav/data/scene_data/mp3d_ce"),
            Path.home() / "zhr/fjl/InternNav/data/scene_data/mp3d_ce",
            Path.home() / "InternNav/data/scene_data/mp3d_ce",
            Path("/dataset/mp3d"),
            Path("/dataset"),
            Path("/workspace/InternNav/data/scene_data/mp3d_ce"),
        ]
        resolved = next((p for p in scenes_candidates if p.exists()), None)
        if resolved is None:
            attempted = "\n  - ".join([requested_scenes_dir, *map(str, scenes_candidates)])
            raise FileNotFoundError(
                "Could not find the MP3D scene directory. Tried:\n"
                f"  - {attempted}"
            )
        args.scenes_dir = str(resolved.resolve())
    else:
        raise FileNotFoundError(
            f"Configured --scenes_dir does not exist: {requested_scenes_dir}"
        )

    print(f"Using scenes_dir: {args.scenes_dir}")
    print(f"Using data_path:  {args.data_path}")


def _extract_checkpoint_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Read a checkpoint and return the tensor state dict it contains."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")

    for key in ("model_state_dict", "trainable_state_dict", "state_dict"):
        state_dict = ckpt.get(key)
        if isinstance(state_dict, dict):
            return state_dict

    if all(torch.is_tensor(value) for value in ckpt.values()):
        return ckpt

    raise KeyError(
        f"Checkpoint does not contain model_state_dict/trainable_state_dict/state_dict: "
        f"{checkpoint_path}"
    )


def _extract_checkpoint_config(checkpoint_path: str | None) -> dict:
    if not checkpoint_path:
        return {}
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    return (
        cfg.get("data", {})
        .get("trajectory", {})
        .get("system2_sft_protocol", "direct")
    ).lower()


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

    checkpoint_state_dict = (
        _extract_checkpoint_state_dict(args.checkpoint)
        if args.checkpoint else None
    )
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
    normalized_to_actual = {
        _normalize_state_key(name): name
        for name in current_state
    }

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
                f"{actual_name}: ckpt {tuple(value.shape)} vs "
                f"model {tuple(current_state[actual_name].shape)}"
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
        print(
            "  skipped shape-mismatched keys: "
            f"{len(skipped_shape)}; examples: {'; '.join(skipped_shape[:3])}"
        )
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


def _load_progress(progress_file: str) -> tuple[list[float], list[float], list[float], list[float], set]:
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
            scene_id = res.get("scene_id")
            episode_id = res.get("episode_id")
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


def _load_episode_list(path: str) -> tuple[list[tuple[str, int]], set[tuple[str, int]]]:
    """Load fixed (scene_id, episode_id) pairs for apples-to-apples comparison."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    episodes = data.get("episodes", data)
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"episode_list must contain a non-empty 'episodes' array: {path}")
    keys: list[tuple[str, int]] = []
    for item in episodes:
        keys.append((str(item["scene_id"]), int(item["episode_id"])))
    return keys, set(keys)


def _episode_list_from_args(args) -> tuple[list[tuple[str, int]] | None, set[tuple[str, int]] | None]:
    path = getattr(args, "episode_list", None)
    if not path:
        return None, None
    return _load_episode_list(path)


def _eval_limit(args, remaining: int, target_list: list[tuple[str, int]] | None = None,
                done_set: set | None = None) -> int:
    if target_list is not None:
        done = done_set or set()
        return sum(1 for key in target_list if key not in done)
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
        "STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN",
    ]
    cfg.TASK.MEASUREMENTS = [
        "DISTANCE_TO_GOAL", "SUCCESS", "SPL", "ORACLE_SUCCESS", "ORACLE_NAVIGATION_ERROR",
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


def capture_panoramic_views(
    env, image_size: tuple = (256, 256),
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
            views[name] = Image.fromarray(
                np.zeros((*image_size[::-1], 3), dtype=np.uint8)
            )

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
    if num_history <= 0:
        return []
    if len(history_panoramas) <= num_history:
        return list(history_panoramas)
    indices = np.unique(
        np.linspace(0, len(history_panoramas) - 1, num_history, dtype=np.int32)
    ).tolist()
    return [history_panoramas[i] for i in indices]


_pixel_goal_clamp_warned = False


def _parse_pixel_goal(
    llm_output: str,
    image_size: tuple[int, int],
) -> list[int] | None:
    """Parse ``u v`` pixel goal in training order; clamp to image bounds."""
    global _pixel_goal_clamp_warned
    if not re.search(r"\d", llm_output):
        return None
    coord = [int(c) for c in re.findall(r"\d+", llm_output)]
    if len(coord) < 2:
        return None
    w, h = int(image_size[0]), int(image_size[1])
    u, v = int(coord[0]), int(coord[1])
    if u > w - 1 or v > h - 1 or u < 0 or v < 0:
        raw_u, raw_v = u, v
        u = max(0, min(w - 1, u))
        v = max(0, min(h - 1, v))
        if not _pixel_goal_clamp_warned:
            print(
                f"WARNING: clamped pixel_goal from [{raw_u}, {raw_v}] to [{u}, {v}] "
                f"for image_size={image_size}",
                flush=True,
            )
            _pixel_goal_clamp_warned = True
    return [u, v]


def _condition_output_ids_for_pixel_goal(
    output_ids: torch.Tensor,
    prompt_len: int,
    tokenizer,
    pixel_goal: list[int],
    llm_output: str,
    coord_order: str = "generated",
) -> torch.Tensor:
    """Use the System1-compatible coordinate text in latent conditioning."""
    coord = [int(c) for c in re.findall(r"\d+", llm_output)]
    if coord_order == "generated":
        desired = [int(pixel_goal[0]), int(pixel_goal[1])]
    elif coord_order == "internnav_yx":
        # InternNav System1 was trained from text like "301 225" while the
        # eval-side pixel goal state is [u=225, v=301].  Keep this mode for
        # raw InternNav compatibility checks; HeatmapVLN Stage2 bridge
        # checkpoints are trained on the generated [u v] text.
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
    if re.search(r"\bSTOP\b", llm_output, flags=re.I):
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


def _debug_input_trace_enabled(args) -> bool:
    return bool(getattr(args, "debug_input_trace", True))


def _image_trace_summary(image: Image.Image) -> str:
    arr = np.asarray(image)
    if arr.size == 0:
        return "empty"
    digest = hashlib.sha1(arr.tobytes()).hexdigest()[:10]
    height, width = arr.shape[:2]
    return (
        f"{width}x{height}:{digest}:"
        f"mean={float(arr.mean()):.1f}:std={float(arr.std()):.1f}"
    )


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
    return (
        f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
        f"rot=({rot[0]:.4f},{rot[1]:.4f},{rot[2]:.4f},{rot[3]:.4f})"
    )


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
    return (
        f"shape={tuple(t.shape)} "
        f"mean={float(tf.mean().item()):.4f} "
        f"std={float(tf.std(unbiased=False).item()):.4f}"
    )


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

    debug_dir = (
        Path(args.output_path)
        / "debug_inputs"
        / f"{scene_id}_{episode_id:04d}"
    )
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        image.save(debug_dir / f"{call_idx:04d}_{phase}_{name}.jpg")


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
    return (
        f"traj_goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f}), "
        f"direct={direct:.2f}, path_len={path_len:.2f}"
    )


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

def traj_to_actions(
    dp_actions: torch.Tensor,
    num_sample_trajs: int = 32,
    action_scale: float = 4.0,
) -> list[int]:
    """Convert InternNav trajectory predictions to discrete Habitat actions.

    This mirrors the verified logic from
    ``internnav.model.utils.vln_utils.traj_to_actions`` instead of the
    simplified threshold-based decoder.
    """

    def reconstruct_xy_from_delta(delta_xyt: np.ndarray) -> np.ndarray:
        start_xy = np.zeros((len(delta_xyt), 2))
        delta_xy = delta_xyt[:, :, :2]
        cumsum_xy = np.cumsum(delta_xy, axis=1)

        B = delta_xyt.shape[0]
        T = delta_xyt.shape[1]
        xy = np.zeros((B, T + 1, 2))
        xy[:, 0] = start_xy
        xy[:, 1:] = start_xy[:, None, :] + cumsum_xy
        return xy

    def trajectory_to_discrete_actions_close_to_goal(
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

    trajs = dp_actions[:num_sample_trajs].float().cpu().numpy()
    trajs[:, :, :2] /= action_scale
    all_trajectory = reconstruct_xy_from_delta(trajs)
    trajectory = np.mean(all_trajectory, axis=0)
    actions = trajectory_to_discrete_actions_close_to_goal(trajectory)
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
                vgt, vgt[:, 0], dim=0,
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
        or cfg.get("model", {})
        .get("action_head", {})
        .get("nextdit", {})
        .get("internnav_model_path", "")
        or cfg.get("model", {}).get("llm", {}).get("model_path", "")
    )
    return os.path.expandvars(os.path.expanduser(str(raw or "").strip()))


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
            raise FileNotFoundError(
                f"No InternNav safetensors found under {internnav_path}"
            )
        weight_map = {k: single.name for k in safe_open(str(single)).keys()}

    shards: dict[str, object] = {}

    def _tensor_from_internnav(ref_key: str) -> torch.Tensor:
        shard_name = weight_map.get(ref_key)
        if shard_name is None:
            raise RuntimeError(f"{ref_key} missing from {index_path or model_dir}")
        shard_path = model_dir / shard_name
        if shard_name not in shards:
            shards[shard_name] = safe_open(str(shard_path), framework="pt", device="cpu")
        handle = shards[shard_name]
        if ref_key not in handle.keys():
            raise RuntimeError(f"{ref_key} missing from {shard_path}")
        return handle.get_tensor(ref_key).float()

    rgb_key_count = sum(1 for key in weight_map if key.startswith("model.rgb_model."))
    if rgb_key_count == 0:
        raise RuntimeError(
            f"No model.rgb_model.* tensors in InternNav weights at {internnav_path}"
        )

    for ref_key, current_fn in checks:
        reference = _tensor_from_internnav(ref_key)
        current = current_fn().detach().float().cpu()
        if current.shape != reference.shape or not torch.allclose(
            current, reference, atol=1e-4, rtol=1e-3
        ):
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
    checkpoint_state_dict = (
        _extract_checkpoint_state_dict(args.checkpoint)
        if args.checkpoint else None
    )

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
        print(
            "WARNING: the main checkpoint contains only action-head weights and no "
            "base checkpoint was loaded."
        )
    if not args.base_checkpoint and checkpoint_state_dict is None:
        print(
            "WARNING: no checkpoint was supplied; evaluating the model initialized "
            "from config/pretrained weights only."
        )

    # Qwen/LoRA is lazy.  It must exist before loading Stage 1 LoRA weights;
    # otherwise qwen*.model.* keys are silently treated as unexpected.
    model.qwen2_5_vl._load_model()

    if (
        _state_has_prefix(base_state_dict, "heatmap_vln.")
        or _state_has_prefix(checkpoint_state_dict, "heatmap_vln.")
    ):
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
) -> None:
    """Closed-loop Habitat evaluation for checkpoints trained with panoramic VLM input."""
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
    system1_coord_order = _system1_coord_order(
        args,
        panoramic_internnav_protocol=internnav_protocol,
    )
    print(f"System2 SFT protocol: {'internnav' if internnav_protocol else 'direct'}")
    print(f"vlm_image_size={vlm_image_size}, traj_image_size={traj_image_size}")
    print(f"System1 coordinate text order: {system1_coord_order}")

    output_path = args.output_path
    progress_file = _prepare_progress_file(args, output_path)
    sucs, spls, oss, nes, done_set = _load_progress(progress_file)

    target_list, target_set = _episode_list_from_args(args)
    if target_list is not None:
        print(f"Fixed episode list ({len(target_list)}): {args.episode_list}")
    remaining = num_episodes - len(done_set)
    eval_limit = _eval_limit(args, remaining, target_list, done_set)
    print(
        f"Episodes already done: {len(done_set)}, remaining: {remaining}, "
        f"this run: {eval_limit}"
    )

    process_bar = tqdm.tqdm(total=eval_limit, desc="Evaluating")
    seen_episodes: set = set()
    eval_count = 0

    while True:
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
        print(
            f"\n[{eval_count}/{eval_limit}] Episode {scene_id}_{episode_id:04d}: "
            f"{instruction[:80]}..."
        )

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

        while (not done) and (step_id <= max_steps_per_episode):
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
                if before is not None:
                    print(
                        f"  [debug] executed local action={int(action)} "
                        f"{before} -> {_env_trace_summary(env)}",
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
                    trajectory = model.nextdit_action_head.get_trajectory(
                        _last_traj_hs,
                        traj_images=traj_images,
                    )

                local_actions = _finalize_local_actions(
                    traj_to_actions(
                        trajectory,
                        num_sample_trajs=num_sample_trajs,
                        action_scale=action_scale,
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
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": last_llm_output}],
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": random.choice(LEGACY_CONJUNCTIONS)},
                        {"type": "image", "image": turn_lookdown_img},
                    ],
                })
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
                    "  [debug] processor pixel_values "
                    f"{_tensor_trace_summary(inputs.get('pixel_values'))}",
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
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            last_llm_output = llm_output
            print(f"  step_id: {step_id}, VLM output: {llm_output}")

            if _vlm_requests_stop(llm_output):
                observations, done = _apply_habitat_action(env, ActionCode.STOP)
                step_id += 1
                continue

            pixel_goal = _parse_pixel_goal(llm_output, vlm_image_size)
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
                    device=device, dtype=model.config.dtype,
                )
                condition_output_ids = _condition_output_ids_for_pixel_goal(
                    output_ids=output_ids,
                    prompt_len=inputs["input_ids"].shape[1],
                    tokenizer=processor.tokenizer,
                    pixel_goal=pixel_goal,
                    llm_output=llm_output,
                    coord_order=system1_coord_order,
                )
                with torch.no_grad():
                    _last_traj_hs = model.qwen2_5_vl.generate_latents(
                        output_ids=condition_output_ids,
                        pixel_values=inputs.get("pixel_values"),
                        image_grid_thw=inputs.get("image_grid_thw"),
                        latent_queries=lq,
                    )

                print("  [debug] calling get_trajectory ...", flush=True)
                trajectory_calls += 1
                with torch.no_grad():
                    trajectory = model.nextdit_action_head.get_trajectory(
                        _last_traj_hs,
                        traj_images=traj_images,
                    )

                local_actions = _finalize_local_actions(
                    traj_to_actions(
                        trajectory,
                        num_sample_trajs=num_sample_trajs,
                        action_scale=action_scale,
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
                        before = (
                            _env_trace_summary(env)
                            if _debug_input_trace_enabled(args)
                            else None
                        )
                        observations, done = _apply_habitat_action(env, ActionCode.LEFT)
                        if before is not None:
                            print(
                                "  [debug] executed anti-deadlock action="
                                f"{int(ActionCode.LEFT)} {before} -> "
                                f"{_env_trace_summary(env)}",
                                flush=True,
                            )
                        step_id += 1
                        continue

                    before = (
                        _env_trace_summary(env)
                        if _debug_input_trace_enabled(args)
                        else None
                    )
                    observations, done = _apply_habitat_action(env, first_action)
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
                step_id += 1
                continue

            action_seq = parse_actions(llm_output, LEGACY_ACTIONS2IDX)
            if action_seq:
                action = action_seq.pop(0)
                if action == ActionCode.LOOKDOWN:
                    awaiting_lookdown = True
                    continue
                observations, done = _apply_habitat_action(env, action)
                step_id += 1
            else:
                observations, done = _apply_habitat_action(env, ActionCode.STOP)
                step_id += 1

        metrics = env.get_metrics()
        sucs.append(metrics["success"])
        spls.append(metrics["spl"])
        oss.append(metrics["oracle_success"])
        nes.append(metrics["distance_to_goal"])

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

    if len(sucs) > 0:
        sucs_t = torch.tensor(sucs)
        spls_t = torch.tensor(spls)
        oss_t = torch.tensor(oss)
        nes_t = torch.tensor(nes)
        torch.nan_to_num(spls_t, nan=0.0, posinf=0.0, neginf=0.0, out=spls_t)
        nes_finite = nes_t[torch.isfinite(nes_t)]

        final_result = {
            "SR": float(sucs_t.mean().item()),
            "SPL": float(spls_t.mean().item()),
            "OS": float(oss_t.mean().item()),
            "NE": float(nes_finite.mean().item()) if len(nes_finite) > 0 else 0.0,
            "total_episodes": len(sucs),
        }
    else:
        final_result = {"SR": 0, "SPL": 0, "OS": 0, "NE": 0, "total_episodes": 0}

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
    device = torch.device(f"cuda:{args.gpu_id}")
    ensure_vln_measures_registered()

    print(f"Loading model from config={args.config}, checkpoint={args.checkpoint or '<none>'}")
    if args.base_checkpoint:
        print(f"Loading base checkpoint first: {args.base_checkpoint}")
    model, train_cfg = load_model(args, device)
    processor = model.qwen2_5_vl.processor
    processor.tokenizer.padding_side = "left"

    action_scale = (
        train_cfg.get("data", {}).get("trajectory", {}).get("action_scale", 4.0)
    )
    num_sample_trajs = (
        train_cfg.get("model", {})
        .get("action_head", {})
        .get("nextdit", {})
        .get("num_sample_trajs", 32)
    )
    has_nextdit = model.nextdit_action_head is not None and model.latent_queries is not None
    print(f"NextDiT action head available: {has_nextdit}")
    print(f"  action_scale={action_scale}, num_sample_trajs={num_sample_trajs}")

    panoramic_vlm_input = bool(
        train_cfg.get("data", {}).get("trajectory", {}).get("panoramic_vlm_input", False)
    )
    print(f"Panoramic VLM input: {panoramic_vlm_input}")
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
        print(f"Fixed episode list ({len(target_list)}): {args.episode_list}")
    remaining = num_episodes - len(done_set)
    eval_limit = _eval_limit(args, remaining, target_list, done_set)
    print(
        f"Episodes already done: {len(done_set)}, remaining: {remaining}, "
        f"this run: {eval_limit}"
    )

    process_bar = tqdm.tqdm(total=eval_limit, desc="Evaluating")
    seen_episodes: set = set()
    eval_count = 0

    # ── Episode loop (iterator-driven, see ReadBeforeEvaluatingHabitat.md §16) ──
    while True:
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
        print(
            f"\n[{eval_count}/{eval_limit}] Episode {scene_id}_{episode_id:04d}: "
            f"{instruction[:80]}..."
        )

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

        while (not done) and (step_id <= max_steps_per_episode):
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
                    messages.append(
                        {"role": "assistant", "content": [{"type": "text", "text": llm_output}]}
                    )
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
                        history_id = np.unique(
                            np.linspace(0, step_id - 1, num_history, dtype=np.int32)
                        ).tolist()
                        placeholder = (DEFAULT_IMAGE_TOKEN + "\n") * len(history_id)
                        sources[0]["value"] += (
                            f" These are your historical observations: {placeholder}."
                        )

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
                    output_ids[0][inputs.input_ids.shape[1]:],
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

                pixel_goal = _parse_pixel_goal(llm_output, vlm_image_size)
                if pixel_goal is not None:
                    print(f"  predicted pixel_goal {pixel_goal}")

                    if not has_nextdit:
                        observations, done = _apply_habitat_action(env, ActionCode.STOP)
                        step_id += 1
                        messages = []
                        continue

                    lookdown_traj_img = (
                        lookdown_img
                        if lookdown_img.size == traj_image_size
                        else lookdown_img.resize(traj_image_size)
                    )
                    lookdown_t = _lookdown_to_traj_tensor(lookdown_traj_img, device)
                    pix_goal_image = lookdown_t.clone()
                    traj_images = torch.stack([pix_goal_image, lookdown_t]).unsqueeze(0).to(device)

                    print("  [debug] calling generate_latents ...", flush=True)
                    lq = model.latent_queries.expand(1, -1, -1).to(
                        device=device, dtype=model.config.dtype,
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
                        )

                    print("  [debug] calling get_trajectory ...", flush=True)
                    with torch.no_grad():
                        trajectory = model.nextdit_action_head.get_trajectory(
                            _last_traj_hs,
                            traj_images=traj_images,
                        )

                    local_actions = _finalize_local_actions(
                        traj_to_actions(
                            trajectory,
                            num_sample_trajs=num_sample_trajs,
                            action_scale=action_scale,
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
                        lookdown_img
                        if lookdown_img.size == traj_image_size
                        else lookdown_img.resize(traj_image_size)
                    )
                    lookdown_t = _lookdown_to_traj_tensor(lookdown_traj_img, device)
                    traj_images = torch.stack([pix_goal_image, lookdown_t]).unsqueeze(0).to(device)

                    with torch.no_grad():
                        trajectory = model.nextdit_action_head.get_trajectory(
                            _last_traj_hs,
                            traj_images=traj_images,
                        )

                    local_actions = _finalize_local_actions(
                        traj_to_actions(
                            trajectory,
                            num_sample_trajs=num_sample_trajs,
                            action_scale=action_scale,
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
    if len(sucs) > 0:
        sucs_t = torch.tensor(sucs)
        spls_t = torch.tensor(spls)
        oss_t = torch.tensor(oss)
        nes_t = torch.tensor(nes)
        torch.nan_to_num(spls_t, nan=0.0, posinf=0.0, neginf=0.0, out=spls_t)
        nes_finite = nes_t[torch.isfinite(nes_t)]

        final_result = {
            "SR": float(sucs_t.mean().item()),
            "SPL": float(spls_t.mean().item()),
            "OS": float(oss_t.mean().item()),
            "NE": float(nes_finite.mean().item()) if len(nes_finite) > 0 else 0.0,
            "total_episodes": len(sucs),
        }
    else:
        final_result = {"SR": 0, "SPL": 0, "OS": 0, "NE": 0, "total_episodes": 0}

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
    parser = argparse.ArgumentParser(
        description="Evaluate VLNPipeline on VLN-CE R2R val_unseen (Habitat closed-loop)"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config used for training (e.g. configs/train_config_internnav.yaml)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional main/Stage 2 checkpoint path (.pth)")
    parser.add_argument("--base_checkpoint", type=str, default=None,
                        help="Optional Stage 1/base checkpoint loaded before --checkpoint")
    parser.add_argument("--scenes_dir", type=str, default=DEFAULT_SCENES_DIR)
    parser.add_argument("--data_path", type=str,
                        default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_path", type=str, default="./logs/eval_r2r_val_unseen")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="Torch CUDA device id for model inference")
    parser.add_argument("--sim_gpu_id", type=int, default=0,
                        help="Habitat-Sim GL device id; keep 0 for GLX/Xvfb builds")
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
        "--debug_input_trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Print compact pose, distance, image hash, and processor tensor stats "
            "for System2/System1 inputs."
        ),
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
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Evaluate at most this many new episodes")
    parser.add_argument("--episode_list", type=str, default=None,
                        help="JSON file with fixed episodes [{scene_id, episode_id}, ...]")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from output_path/progress.json")
    parser.add_argument("--overwrite_output", action="store_true",
                        help="Delete output_path/progress.json and result.json before evaluating")
    args = parser.parse_args()
    _preflight_checkpoint_args(args)
    _resolve_eval_paths(args)
    run_eval(args)


if __name__ == "__main__":
    main()

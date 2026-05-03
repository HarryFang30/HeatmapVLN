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

faulthandler.enable()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ═══════════════════════════════════════════════════════════════════════
# Section 1: Runtime patches (must be before any heavy imports)
# ═══════════════════════════════════════════════════════════════════════

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

# Initialize NVIDIA GL context BEFORE numba/LLVM is loaded.
import habitat_sim as _hsim

_dummy_cfg = _hsim.SimulatorConfiguration()
# For GLX builds, pre-initialisation must use the *local visible* GPU index.
# When users launch with e.g. CUDA_VISIBLE_DEVICES=4 and --gpu_id 0, Habitat's
# actual simulator later runs on local GPU 0 (the remapped visible device).
# Parsing CUDA_VISIBLE_DEVICES here would incorrectly request physical GPU 4
# from GLX, which fails on multi-GPU systems.
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
import itertools
import json
import random
import re
from collections import OrderedDict
from enum import IntEnum
from pathlib import Path

import habitat
import quaternion
import torch
import tqdm
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.config.default import Config as CN
from habitat.config.default import get_config as get_habitat_default_config
from habitat.tasks.nav.nav import DistanceToGoal
from PIL import Image
from scripts.training.model_builder import build_model
from scripts.training.utils import load_config

from src.data.vln_sliding_window_dataset import compute_history_rel_poses
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
    cfg.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu_id
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
        rgb = obs.get("rgba_camera", obs.get("color_sensor", obs.get("rgb")))
        if isinstance(rgb, np.ndarray):
            if rgb.ndim == 3 and rgb.shape[-1] == 4:
                rgb = rgb[:, :, :3]
            views[name] = Image.fromarray(rgb).convert("RGB").resize(image_size)
        else:
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

    rgb = obs["rgb"]
    lookdown_img = Image.fromarray(rgb).convert("RGB").resize(image_size)

    env.step(ActionCode.LOOKUP)
    env.step(ActionCode.LOOKUP)
    return lookdown_img


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
            n_turns = int(round(delta_yaw / turn_angle_rad))
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

def load_model(args, device: torch.device):
    """Build VLNPipeline, load checkpoint, and initialise the Qwen backbone."""
    cfg = load_config(args.config)
    model = build_model(cfg, device=str(device), verbose=False)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get(
        "model_state_dict", ckpt.get("trainable_state_dict", ckpt)
    )
    if state_dict and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Checkpoint loaded: {args.checkpoint}  "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )

    model = model.to(device)
    model.eval()
    model.qwen2_5_vl._load_model()
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

    image_size = tuple(train_cfg["data"]["image_size"])
    num_history = args.num_history
    max_steps_per_episode = args.max_steps_per_episode

    sucs, spls, oss, nes = [], [], [], []
    done_set: set = set()
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    progress_file = os.path.join(output_path, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            for line in f:
                res = json.loads(line)
                sucs.append(res["success"])
                spls.append(res["spl"])
                oss.append(res["os"])
                nes.append(res["ne"])
                if "scene_id" in res:
                    done_set.add((res["scene_id"], res["episode_id"]))

    remaining = num_episodes - len(done_set)
    print(f"Episodes already done: {len(done_set)}, remaining: {remaining}")

    process_bar = tqdm.tqdm(total=remaining, desc="Evaluating")
    seen_episodes: set = set()
    eval_count = 0

    while True:
        observations = env.reset()
        episode = env.current_episode
        scene_id = episode.scene_id.split("/")[-2]
        episode_id = int(episode.episode_id)
        ep_key = (scene_id, episode_id)

        if ep_key in seen_episodes:
            break
        seen_episodes.add(ep_key)

        if ep_key in done_set:
            continue

        instruction = episode.instruction.instruction_text
        eval_count += 1
        print(
            f"\n[{eval_count}/{remaining}] Episode {scene_id}_{episode_id:04d}: "
            f"{instruction[:80]}..."
        )

        history_panoramas: list[dict[str, Image.Image]] = []
        action_seq: list[int] = []
        local_actions: list[int] = []
        pix_goal_image: torch.Tensor | None = None
        _last_traj_hs: torch.Tensor | None = None
        base_messages: list[dict] | None = None
        awaiting_lookdown = False
        last_llm_output = ""
        forward_action_count = 0
        step_id = 0
        done = False

        while (not done) and (step_id <= max_steps_per_episode):
            sys.stdout.flush()

            if local_actions:
                action = local_actions.pop(0)
                forward_action_count += 1

                if forward_action_count > MAX_STEPS:
                    pix_goal_image = None
                    local_actions = []
                    forward_action_count = 0
                    step_id += 1
                    continue

                if action == ActionCode.STOP:
                    pix_goal_image = None
                    local_actions = []
                    forward_action_count = 0
                    step_id += 1
                    continue

                observations = env.step(action)
                done = env.episode_over
                step_id += 1
                continue

            if pix_goal_image is not None and _last_traj_hs is not None:
                # Training pins traj_images to direction="front_down" (see
                # src/data/trajectory_dataset.py:577). Feeding s1 the level
                # forward RGB is a domain mismatch; capture a fresh lookdown
                # (LOOKDOWN×2 → RGB → LOOKUP×2 restores pitch).
                current_lookdown_img = capture_lookdown_view(env, image_size=(224, 224))
                current_traj_t = (
                    torch.from_numpy(np.array(current_lookdown_img)).to(torch.bfloat16) / 255.0
                )
                traj_images = torch.stack([pix_goal_image, current_traj_t]).unsqueeze(0).to(device)

                print("  [debug] re-calling get_trajectory ...", flush=True)
                with torch.no_grad():
                    trajectory = model.nextdit_action_head.get_trajectory(
                        _last_traj_hs,
                        traj_images=traj_images,
                    )

                local_actions = traj_to_actions(
                    trajectory,
                    num_sample_trajs=num_sample_trajs,
                    action_scale=action_scale,
                )
                if len(local_actions) < MAX_STEPS:
                    local_actions = list(local_actions) + [ActionCode.STOP] * (
                        MAX_STEPS - len(local_actions)
                    )
                if len(local_actions) >= MAX_LOCAL_STEPS:
                    local_actions = local_actions[:MAX_LOCAL_STEPS]
                continue

            print(
                f"  [step_id={step_id}] Capturing panoramic views + VLM inference ...",
                flush=True,
            )

            if awaiting_lookdown and base_messages is not None:
                lookdown_img = capture_lookdown_view(env, image_size=(640, 480))
                messages = copy.deepcopy(base_messages)
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": last_llm_output}],
                })
                messages.append({
                    "role": "user",
                    "content": [{"type": "image", "image": lookdown_img}],
                })
                awaiting_lookdown = False
            else:
                current_views = capture_panoramic_views(env, image_size=image_size)
                messages = construct_input(
                    current_views=current_views,
                    history_panoramas=history_panoramas,
                    instruction=instruction,
                    pixel_goal=[0, 0],
                )
                messages = [m for m in messages if m["role"] != "assistant"]
                base_messages = copy.deepcopy(messages)
                history_panoramas.append(current_views)
                if len(history_panoramas) > num_history:
                    indices = np.unique(
                        np.linspace(0, len(history_panoramas) - 1, num_history, dtype=np.int32)
                    ).tolist()
                    history_panoramas = [history_panoramas[i] for i in indices]

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

            if has_nextdit and bool(re.search(r"\d", llm_output)):
                coord = [int(c) for c in re.findall(r"\d+", llm_output)]
                if len(coord) >= 2:
                    pixel_goal = [int(coord[1]), int(coord[0])]
                    print(f"  predicted pixel_goal {pixel_goal}")
                else:
                    env.step(ActionCode.STOP)
                    done = True
                    continue

                # Capture lookdown view (matches training direction="front_down"
                # in src/data/trajectory_dataset.py:577). Both slots of
                # traj_images are the same frame at goal-freeze time.
                current_lookdown_img = capture_lookdown_view(env, image_size=(224, 224))
                current_traj_t = (
                    torch.from_numpy(np.array(current_lookdown_img)).to(torch.bfloat16) / 255.0
                )
                pix_goal_image = current_traj_t.clone()
                traj_images = torch.stack([pix_goal_image, current_traj_t]).unsqueeze(0).to(device)

                print("  [debug] calling generate_latents ...", flush=True)
                lq = model.latent_queries.expand(1, -1, -1).to(
                    device=device, dtype=model.config.dtype,
                )
                with torch.no_grad():
                    _last_traj_hs = model.qwen2_5_vl.generate_latents(
                        output_ids=output_ids,
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

                local_actions = traj_to_actions(
                    trajectory,
                    num_sample_trajs=num_sample_trajs,
                    action_scale=action_scale,
                )
                if len(local_actions) < MAX_STEPS:
                    local_actions = list(local_actions) + [ActionCode.STOP] * (
                        MAX_STEPS - len(local_actions)
                    )
                if len(local_actions) >= MAX_LOCAL_STEPS:
                    local_actions = local_actions[:MAX_LOCAL_STEPS]
                forward_action_count = 0

                if local_actions:
                    first_action = local_actions.pop(0)
                    if first_action == ActionCode.STOP:
                        # Mirror InternNav: if s1 predicts STOP on the very
                        # first action after s2, force a LEFT turn so the next
                        # s2 call sees a different panorama and can replan
                        # (otherwise s2→s1→STOP loops with identical views).
                        pix_goal_image = None
                        _last_traj_hs = None
                        local_actions = []
                        base_messages = None
                        awaiting_lookdown = False
                        forward_action_count = 0
                        observations = env.step(ActionCode.LEFT)
                        done = env.episode_over
                        step_id += 1
                        continue

                    observations = env.step(first_action)
                    done = env.episode_over
                    step_id += 1
                    forward_action_count += 1
                    continue

                env.step(ActionCode.STOP)
                done = True
                continue

            action_seq = parse_actions(llm_output, LEGACY_ACTIONS2IDX)
            if action_seq:
                action = action_seq.pop(0)
                if action == ActionCode.LOOKDOWN:
                    awaiting_lookdown = True
                    continue
                else:
                    observations = env.step(action)
                    done = env.episode_over
                    step_id += 1
            else:
                env.step(ActionCode.STOP)
                done = True

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

    print(f"Loading model from config={args.config}, checkpoint={args.checkpoint}")
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

    # ── Resume support ──
    sucs, spls, oss, nes = [], [], [], []
    done_set: set = set()
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    progress_file = os.path.join(output_path, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            for line in f:
                res = json.loads(line)
                sucs.append(res["success"])
                spls.append(res["spl"])
                oss.append(res["os"])
                nes.append(res["ne"])
                if "scene_id" in res:
                    done_set.add((res["scene_id"], res["episode_id"]))

    remaining = num_episodes - len(done_set)
    print(f"Episodes already done: {len(done_set)}, remaining: {remaining}")

    process_bar = tqdm.tqdm(total=remaining, desc="Evaluating")
    seen_episodes: set = set()
    eval_count = 0

    # ── Episode loop (iterator-driven, see ReadBeforeEvaluatingHabitat.md §16) ──
    while True:
        observations = env.reset()
        episode = env.current_episode
        scene_id = episode.scene_id.split("/")[-2]
        episode_id = int(episode.episode_id)
        ep_key = (scene_id, episode_id)

        if ep_key in seen_episodes:
            break
        seen_episodes.add(ep_key)

        if ep_key in done_set:
            continue

        instruction = episode.instruction.instruction_text
        eval_count += 1
        print(
            f"\n[{eval_count}/{remaining}] Episode {scene_id}_{episode_id:04d}: "
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
            print(
                f"  [step_id={step_id}] Capturing observations + VLM inference ...",
                flush=True,
            )

            rgb = observations["rgb"]
            image = Image.fromarray(rgb).convert("RGB")

            if action == ActionCode.LOOKDOWN:
                lookdown_img = image.resize((224, 224))
            else:
                rgb_history.append(image.resize((args.resize_w, args.resize_h)))

                down_observations = env.step(ActionCode.LOOKDOWN)
                down_observations = env.step(ActionCode.LOOKDOWN)
                lookdown_img = Image.fromarray(down_observations["rgb"]).convert("RGB").resize((224, 224))
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
                    prompt_instruction = instruction[:-1] if instruction.endswith((".", "!", "?")) else instruction
                    sources[0]["value"] = sources[0]["value"].replace("<instruction>.", prompt_instruction)

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

                if bool(re.search(r"\d", llm_output)):
                    coord = [int(c) for c in re.findall(r"\d+", llm_output)]
                    if len(coord) >= 2:
                        pixel_goal = [int(coord[1]), int(coord[0])]
                        print(f"  predicted pixel_goal {pixel_goal}")
                    else:
                        action = ActionCode.LEFT
                        observations = env.step(action)
                        step_id += 1
                        done = env.episode_over
                        messages = []
                        continue

                    if not has_nextdit:
                        action = ActionCode.STOP
                        observations = env.step(action)
                        done = True
                        messages = []
                        continue

                    lookdown_t = (
                        torch.from_numpy(np.array(lookdown_img)).to(torch.bfloat16) / 255.0
                    )
                    pix_goal_image = lookdown_t.clone()
                    traj_images = torch.stack([pix_goal_image, lookdown_t]).unsqueeze(0).to(device)

                    print("  [debug] calling generate_latents ...", flush=True)
                    lq = model.latent_queries.expand(1, -1, -1).to(
                        device=device, dtype=model.config.dtype,
                    )
                    with torch.no_grad():
                        _last_traj_hs = model.qwen2_5_vl.generate_latents(
                            output_ids=output_ids,
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

                    local_actions = traj_to_actions(
                        trajectory,
                        num_sample_trajs=num_sample_trajs,
                        action_scale=action_scale,
                    )
                    if len(local_actions) >= MAX_LOCAL_STEPS:
                        local_actions = local_actions[:MAX_LOCAL_STEPS]

                    forward_action_count = 0
                    action = local_actions[0] if local_actions else ActionCode.STOP
                    if action == ActionCode.STOP:
                        pix_goal_image = None
                        local_actions = []
                        action = ActionCode.LEFT
                        observations = env.step(action)
                        step_id += 1
                        done = env.episode_over
                        messages = []
                        continue
                else:
                    action_seq = parse_actions(llm_output, LEGACY_ACTIONS2IDX)
                    print(f"  actions {action_seq}")

            if len(action_seq) != 0:
                action = action_seq.pop(0)
            elif pix_goal_image is not None:
                if len(local_actions) == 0:
                    lookdown_t = (
                        torch.from_numpy(np.array(lookdown_img)).to(torch.bfloat16) / 255.0
                    )
                    traj_images = torch.stack([pix_goal_image, lookdown_t]).unsqueeze(0).to(device)

                    with torch.no_grad():
                        trajectory = model.nextdit_action_head.get_trajectory(
                            _last_traj_hs,
                            traj_images=traj_images,
                        )

                    local_actions = traj_to_actions(
                        trajectory,
                        num_sample_trajs=num_sample_trajs,
                        action_scale=action_scale,
                    )
                    if len(local_actions) >= MAX_LOCAL_STEPS:
                        local_actions = local_actions[:MAX_LOCAL_STEPS]
                    action = local_actions.pop(0) if local_actions else ActionCode.STOP
                else:
                    action = local_actions.pop(0)

                forward_action_count += 1
                if forward_action_count > MAX_STEPS:
                    pix_goal_image = None
                    messages = []
                    step_id += 1
                    forward_action_count = 0
                    local_actions = []
                    continue

                if action == ActionCode.STOP:
                    pix_goal_image = None
                    messages = []
                    step_id += 1
                    forward_action_count = 0
                    local_actions = []
                    continue
            else:
                action = ActionCode.STOP

            if action == ActionCode.LOOKDOWN:
                env.step(action)
                observations = env.step(action)
                done = env.episode_over
            else:
                observations = env.step(action)
                done = env.episode_over
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
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Model checkpoint path (.pth)")
    parser.add_argument("--scenes_dir", type=str, default=DEFAULT_SCENES_DIR)
    parser.add_argument("--data_path", type=str,
                        default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_path", type=str, default="./logs/eval_r2r_val_unseen")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--max_steps_per_episode", type=int, default=500)
    args = parser.parse_args()
    _resolve_eval_paths(args)
    run_eval(args)


if __name__ == "__main__":
    main()

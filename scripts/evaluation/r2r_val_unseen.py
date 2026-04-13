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
    m.__version__ = '2.8.3'
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    sys.modules[name] = m
    return m

_fa = _make_stub('flash_attn', {
    'flash_attn_func': _noop,
    'flash_attn_varlen_func': _noop,
})
_make_stub('flash_attn_2_cuda')
_fa_iface = _make_stub('flash_attn.flash_attn_interface', {
    'flash_attn_func': _noop,
    'flash_attn_varlen_func': _noop,
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
_dummy_cfg.gpu_device_id = (
    int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    if os.environ.get("CUDA_VISIBLE_DEVICES") else 0
)
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
import json
import re
from enum import IntEnum

import habitat
import quaternion
import torch
import tqdm
import yaml
from habitat.config.default import Config as CN
from habitat.config.default import get_config as get_habitat_default_config
from PIL import Image

from src.data.vln_sliding_window_dataset import compute_history_rel_poses
from src.models.heatmap.input_constructor import construct_input
from src.models.lora_utils import resolve_lora_layer_indices
from src.models.pipeline import VLNPipeline, VLNPipelineConfig

MAX_STEPS = 8
MAX_LOCAL_STEPS = 4


class ActionCode(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


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
    """Convert NextDiT continuous trajectory to discrete Habitat actions.

    The trajectory represents relative poses (dx, dy, dyaw) scaled by
    ``action_scale`` during training.  We accumulate these increments and
    emit discrete FORWARD / LEFT / RIGHT actions when the accumulated
    change exceeds the corresponding Habitat action magnitude.

    Args:
        dp_actions: (B * num_sample_trajs, T, 3) trajectory predictions.
        num_sample_trajs: number of parallel trajectory samples per batch.
        action_scale: scaling factor used during training.

    Returns:
        List of discrete action codes (ActionCode values).
    """
    trajs = dp_actions[:num_sample_trajs].float()
    mean_traj = trajs.mean(dim=0).cpu().numpy() / action_scale  # (T, 3)

    forward_step = 0.25   # Habitat FORWARD_STEP_SIZE
    turn_step = np.deg2rad(15)  # Habitat TURN_ANGLE

    actions: list[int] = []
    accum_dx = 0.0
    accum_dyaw = 0.0

    for step in mean_traj:
        dx, _dy, dyaw = step
        accum_dx += dx
        accum_dyaw += dyaw

        if abs(accum_dyaw) >= turn_step * 0.5:
            if accum_dyaw > 0:
                actions.append(ActionCode.LEFT)
                accum_dyaw -= turn_step
            else:
                actions.append(ActionCode.RIGHT)
                accum_dyaw += turn_step
        elif accum_dx >= forward_step * 0.5:
            actions.append(ActionCode.FORWARD)
            accum_dx -= forward_step

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

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_vln_pipeline(cfg: dict, device: str = "cuda:0") -> VLNPipeline:
    """Build VLNPipeline from YAML config (mirrors scripts/evaluation/general.py)."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    llm_cfg = model_cfg.get("llm", {})
    heatmap_cfg = model_cfg.get("heatmap", {})
    action_cfg = model_cfg.get("action_head", {})
    nextdit_cfg = action_cfg.get("nextdit", {})

    import logging
    _logger = logging.getLogger("evaluate")
    resolved_lora_layers = resolve_lora_layer_indices(
        llm_cfg, heatmap_cfg, logger=_logger,
    )

    config = VLNPipelineConfig(
        llm_model_path=llm_cfg.get("model_path", "./models/internnav_backbone"),
        llm_backbone_type=llm_cfg.get("backbone_type", "qwen2_5_vl"),
        llm_hidden_dim=llm_cfg.get("hidden_dim", 3584),
        llm_token_dim=llm_cfg.get("token_dim", 896),
        llm_torch_dtype=llm_cfg.get("torch_dtype", "bfloat16"),
        llm_attn_implementation=llm_cfg.get("attn_implementation", "sdpa"),
        max_video_frames=llm_cfg.get("max_video_frames", -1),
        enable_packing=llm_cfg.get("enable_packing", False),
        max_seq_length=llm_cfg.get("max_seq_length", 8192),
        spatial_merge_size=llm_cfg.get("spatial_merge_size", 2),
        internnav_system1_path=nextdit_cfg.get("internnav_system1_path", ""),
        device=device,
        enable_heatmap=heatmap_cfg.get("enable", True),
        heatmap_c_vit=heatmap_cfg.get("c_vit", 1280),
        heatmap_c_llm=heatmap_cfg.get("c_llm", 3584),
        heatmap_c_fused=heatmap_cfg.get("c_fused", 256),
        heatmap_vit_layer_indices=heatmap_cfg.get("vit_layer_indices", [7, 15, 23, 31]),
        heatmap_llm_layer_indices=heatmap_cfg.get("llm_layer_indices", [6, 13, 20]),
        heatmap_size=tuple(heatmap_cfg.get("heatmap_size", data_cfg["init_hm_size"])),
        image_size=heatmap_cfg.get("image_size", data_cfg["image_size"][0]),
        heatmap_trajectory_config=heatmap_cfg.get("trajectory", None),
        use_lora=llm_cfg.get("use_lora", False),
        lora_rank=llm_cfg.get("lora_rank", 16),
        lora_alpha=llm_cfg.get("lora_alpha", 32),
        lora_num_layers=llm_cfg.get("lora_num_layers", 4),
        lora_layer_indices=resolved_lora_layers,
        lora_dropout=llm_cfg.get("lora_dropout", 0.05),
        lora_target_modules=llm_cfg.get("lora_target_modules", None),
        enable_action_head=action_cfg.get("enable", True),
        nextdit_enabled=nextdit_cfg.get("enabled", False),
        nextdit_vlm_hidden_dim=nextdit_cfg.get("vlm_hidden_dim", 3584),
        nextdit_latent_emb_size=nextdit_cfg.get("latent_emb_size", 768),
        nextdit_n_query=nextdit_cfg.get("n_query", 4),
        nextdit_dit_dim=nextdit_cfg.get("dit_dim", 384),
        nextdit_dit_layers=nextdit_cfg.get("dit_layers", 12),
        nextdit_dit_heads=nextdit_cfg.get("dit_heads", 6),
        nextdit_dit_kv_heads=nextdit_cfg.get("dit_kv_heads", 6),
        nextdit_dit_ffn_dim_multiplier=nextdit_cfg.get("dit_ffn_dim_multiplier", 2 / 3),
        nextdit_predict_steps=nextdit_cfg.get("predict_steps", 32),
        nextdit_action_dim=nextdit_cfg.get("action_dim", 3),
        nextdit_num_inference_steps=nextdit_cfg.get("num_inference_steps", 10),
        nextdit_guidance_scale=nextdit_cfg.get("guidance_scale", 1.0),
        nextdit_num_sample_trajs=nextdit_cfg.get("num_sample_trajs", 32),
        nextdit_dav2_ckpt_path=nextdit_cfg.get("dav2_ckpt_path", ""),
        nextdit_enable_gradient_checkpointing=nextdit_cfg.get(
            "enable_gradient_checkpointing", True,
        ),
        verbose=False,
    )
    return VLNPipeline(config)


def load_model(args, device: torch.device):
    """Build VLNPipeline, load checkpoint, and initialise HeatmapVLN."""
    cfg = load_config(args.config)
    model = build_vln_pipeline(cfg, device=str(device))

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
    model._ensure_heatmap_vln()
    return model, cfg


# ═══════════════════════════════════════════════════════════════════════
# Section 9: Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════

def run_eval(args):
    device = torch.device(f"cuda:{args.gpu_id}")

    print(f"Loading model from config={args.config}, checkpoint={args.checkpoint}")
    model, train_cfg = load_model(args, device)
    processor = model.qwen2_5_vl.processor

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

    hab_cfg = build_habitat_config(args)
    print("Creating Habitat environment ...")
    env = habitat.Env(config=hab_cfg)
    num_episodes = len(list(env.episodes))
    print(f"Total episodes: {num_episodes}")

    image_size = tuple(train_cfg["data"]["image_size"])  # (256, 256)
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
        env.reset()
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

        # ── Per-episode state ──
        history_panoramas: list[dict[str, Image.Image]] = []
        history_poses: list[np.ndarray] = []  # 4×4 cam2world at each panoramic capture
        pix_goal_image: torch.Tensor | None = None
        _last_traj_hs: torch.Tensor | None = None
        local_actions: list[int] = []
        forward_action_count = 0
        step_id = 0
        done = False

        while (not done) and (step_id <= max_steps_per_episode):
            sys.stdout.flush()

            # ── If there are queued local actions, execute them ──
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

                env.step(action)
                done = env.episode_over
                step_id += 1
                continue

            # ── If pixel_goal is active but local_actions exhausted → re-predict ──
            if pix_goal_image is not None:
                lookdown_img = capture_lookdown_view(env, image_size=(224, 224))
                lookdown_t = (
                    torch.from_numpy(np.array(lookdown_img)).to(torch.bfloat16) / 255.0
                )
                traj_images = (
                    torch.stack([pix_goal_image, lookdown_t]).unsqueeze(0).to(device)
                )

                with torch.no_grad():
                    trajectory = model.nextdit_action_head.get_trajectory(
                        _last_traj_hs, traj_images=traj_images,
                    )
                local_actions = traj_to_actions(
                    trajectory, num_sample_trajs=num_sample_trajs,
                    action_scale=action_scale,
                )
                if len(local_actions) >= MAX_LOCAL_STEPS:
                    local_actions = local_actions[:MAX_LOCAL_STEPS]
                continue

            # ── High-level step: VLM inference ──
            print(
                f"  [step_id={step_id}] Capturing panoramic views + VLM inference ...",
                flush=True,
            )

            current_views = capture_panoramic_views(env, image_size=image_size)
            current_pose = get_agent_cam2world(env)

            # Compute relative poses for TrajectoryGuidedAttention
            if history_poses:
                rel_poses_np = compute_history_rel_poses(history_poses, current_pose)
                history_rel_poses = (
                    torch.from_numpy(rel_poses_np).float().unsqueeze(0).to(device)
                )  # (1, N, 4)
            else:
                history_rel_poses = None

            inputs = prepare_vlm_inputs(
                processor, current_views, history_panoramas, instruction, device,
            )

            # ── Optional heatmap forward: feed history_rel_poses to the
            #    TrajectoryGuidedAttention coarse localisation head.
            #    This runs one non-generative VLM pass with heatmap hooks so
            #    that the model's spatial understanding is exercised. ──
            if history_rel_poses is not None and model.heatmap_vln is not None:
                try:
                    hm_inputs, n_hist_list = model.heatmap_vln.prepare_qwen_inputs_batch(
                        current_views=current_views,
                        history_panoramas=[history_panoramas],
                        instruction=[instruction],
                        device=device,
                    )
                    model.heatmap_vln.feat_extractor.clear()
                    from src.models.heatmap.input_constructor import find_text_anchor_positions
                    img_pos_batch = [
                        model.heatmap_vln._find_image_positions_from_ids(
                            hm_inputs["input_ids"][b]
                        )
                        for b in range(hm_inputs["input_ids"].shape[0])
                    ]
                    txt_anc_batch = [
                        find_text_anchor_positions(
                            hm_inputs["input_ids"][b:b + 1],
                            model.heatmap_vln.processor.tokenizer,
                            num_history=n_hist_list[b],
                        )
                        for b in range(hm_inputs["input_ids"].shape[0])
                    ]
                    model.heatmap_vln.feat_extractor.prepare_batch_capture(
                        image_token_positions_batch=img_pos_batch,
                        text_anchor_positions_batch=txt_anc_batch,
                        image_grid_thw=hm_inputs.get("image_grid_thw"),
                    )
                    with torch.inference_mode():
                        model.qwen2_5_vl.model(
                            **hm_inputs,
                            output_hidden_states=False,
                            return_dict=True,
                        )
                    heatmap_output = model.heatmap_vln.decode_from_inputs_batch(
                        hm_inputs, n_hist_list,
                        image_positions_batch=img_pos_batch,
                        text_anchors_batch=txt_anc_batch,
                        history_rel_poses=history_rel_poses,
                    )
                    print(
                        f"  [heatmap] visibility shape="
                        f"{heatmap_output['visibility'].shape}, "
                        f"heatmaps shape={heatmap_output['heatmaps'].shape}",
                        flush=True,
                    )
                except Exception as e:
                    import traceback
                    print(f"  [heatmap] skipped: {e}\n{traceback.format_exc()}", flush=True)

            print(
                f"  [debug] input_ids shape={inputs['input_ids'].shape}, "
                f"calling model.generate ...",
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
            print(f"  step_id: {step_id}, VLM output: {llm_output}")

            # ── Parse output: coordinates or STOP ──
            if has_nextdit and bool(re.search(r"\d", llm_output)):
                coord = [int(c) for c in re.findall(r"\d+", llm_output)]
                if len(coord) >= 2:
                    pixel_goal = [int(coord[1]), int(coord[0])]
                    print(f"  predicted pixel_goal {pixel_goal}")
                else:
                    env.step(ActionCode.LEFT)
                    step_id += 1
                    done = env.episode_over
                    history_panoramas.append(current_views)
                    continue

                lookdown_img = capture_lookdown_view(env, image_size=(224, 224))
                lookdown_t = (
                    torch.from_numpy(np.array(lookdown_img)).to(torch.bfloat16) / 255.0
                )
                pix_goal_image = lookdown_t.clone()
                traj_images = (
                    torch.stack([pix_goal_image, lookdown_t]).unsqueeze(0).to(device)
                )

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
                        _last_traj_hs, traj_images=traj_images,
                    )
                print(
                    f"  [debug] trajectory shape={trajectory.shape}",
                    flush=True,
                )

                local_actions = traj_to_actions(
                    trajectory, num_sample_trajs=num_sample_trajs,
                    action_scale=action_scale,
                )
                if len(local_actions) >= MAX_LOCAL_STEPS:
                    local_actions = local_actions[:MAX_LOCAL_STEPS]
                forward_action_count = 0

                first_action = local_actions.pop(0) if local_actions else ActionCode.STOP
                if first_action == ActionCode.STOP:
                    pix_goal_image = None
                    local_actions = []
                    env.step(ActionCode.LEFT)
                    step_id += 1
                    done = env.episode_over
                else:
                    env.step(first_action)
                    step_id += 1
                    forward_action_count += 1
                    done = env.episode_over
            else:
                env.step(ActionCode.STOP)
                done = True

            # ── Update history (panoramic views + agent poses) ──
            history_panoramas.append(current_views)
            history_poses.append(current_pose)
            if len(history_panoramas) > num_history:
                indices = np.unique(
                    np.linspace(
                        0, len(history_panoramas) - 1, num_history, dtype=np.int32,
                    )
                ).tolist()
                history_panoramas = [history_panoramas[i] for i in indices]
                history_poses = [history_poses[i] for i in indices]

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
    parser.add_argument("--scenes_dir", type=str, default="data/scene_data/mp3d_ce")
    parser.add_argument("--data_path", type=str,
                        default="data/vln_ce/raw_data/r2r/{split}/{split}.json.gz")
    parser.add_argument("--output_path", type=str, default="./logs/eval_r2r_val_unseen")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--max_steps_per_episode", type=int, default=500)
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()

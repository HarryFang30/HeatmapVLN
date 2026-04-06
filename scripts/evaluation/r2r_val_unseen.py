"""
Standalone evaluation script for InternVLA-N1 on VLN-CE R2R val_unseen.
Strictly follows the dual_system evaluation logic from HabitatVLNEvaluator._run_eval_dual_system().
Adapted for habitat-lab 0.1.7 (YACS config).
"""

import sys
import os
import faulthandler
faulthandler.enable()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Block flash_attn import (GLIBC_2.32 not available on this system)
# Provide a complete stub so transformers can import without error,
# but we will use SDPA attention so these functions are never called at runtime.
import types as _types, importlib as _importlib

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
# numba's LLVM runtime conflicts with NVIDIA's GLX shader compiler if
# numba is imported first, causing fatal X11 BadWindow errors.
# Requires: __GLX_VENDOR_LIBRARY_NAME=nvidia in the environment.
import habitat_sim as _hsim
_dummy_cfg = _hsim.SimulatorConfiguration()
_dummy_cfg.gpu_device_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]) if os.environ.get("CUDA_VISIBLE_DEVICES") else 0
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

import argparse
import copy
import importlib
import importlib.util
import itertools
import json
import random
import re
from collections import OrderedDict
from enum import IntEnum

import cv2
import torch
import tqdm
from depth_camera_filtering import filter_depth
from PIL import Image
from transformers import AutoProcessor

import habitat
from habitat.config.default import get_config as get_habitat_default_config
from habitat.config.default import Config as CN

# Directly load modules via file path to avoid triggering __init__.py which imports
# HabitatVLNEvaluator (incompatible with habitat-lab 0.1.7)
_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

def _load_module_from_file(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_load_module_from_file("vln_measures",
    os.path.join(_base_dir, 'internnav', 'habitat_extensions', 'vln', 'measures.py'))
_utils_mod = _load_module_from_file("vln_utils_ext",
    os.path.join(_base_dir, 'internnav', 'habitat_extensions', 'vln', 'utils.py'))
preprocess_depth_image_v2 = _utils_mod.preprocess_depth_image_v2

# Patch build_depthanythingv2 to skip loading separate checkpoint
# (weights will be loaded from safetensors via from_pretrained)
import internnav.model.basemodel.internvla_n1.internvla_n1_arch as _arch_mod
def _patched_build_dav2(config):
    from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import DepthAnythingV2
    model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
    DAv2_model = DepthAnythingV2(**model_configs['vits'])
    return DAv2_model.pretrained
_arch_mod.build_depthanythingv2 = _patched_build_dav2

from internnav.model.basemodel.internvla_n1.internvla_n1 import (
    InternVLAN1ForCausalLM, InternVLAN1ModelConfig,
)
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions

from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("internvla_n1", InternVLAN1ModelConfig)
AutoModelForCausalLM.register(InternVLAN1ModelConfig, InternVLAN1ForCausalLM)

DEFAULT_IMAGE_TOKEN = "<image>"
MAX_STEPS = 8
MAX_LOCAL_STEPS = 4


class action_code(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


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
    cfg.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"]
    cfg.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "ORACLE_SUCCESS", "ORACLE_NAVIGATION_ERROR"]

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


def parse_actions(output, actions2idx):
    action_patterns = '|'.join(re.escape(action) for action in actions2idx)
    regex = re.compile(action_patterns)
    matches = regex.findall(output)
    actions = [actions2idx[match] for match in matches]
    actions = itertools.chain.from_iterable(actions)
    return list(actions)


def run_eval(args):
    device = torch.device(f"cuda:{args.gpu_id}")

    hab_cfg = build_habitat_config(args)

    print(f"Loading model from {args.model_path} ...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = 'left'

    model = InternVLAN1ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": device},
    )
    model.eval()
    print("Model loaded successfully.")

    print("Creating Habitat environment ...")
    env = habitat.Env(config=hab_cfg)
    all_episodes = list(env.episodes)
    num_episodes = len(all_episodes)
    print(f"Total episodes: {num_episodes}")

    min_depth = hab_cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
    max_depth = hab_cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH

    resize_w = args.resize_w
    resize_h = args.resize_h
    num_history = args.num_history
    max_steps_per_episode = args.max_steps_per_episode

    prompt_template = (
        "You are an autonomous navigation assistant. Your task is to <instruction>. "
        "Where should you go next to stay on track? Please output the next waypoint's "
        "coordinates in the image. Please output STOP when you have successfully completed the task."
    )
    conversation = [{"from": "human", "value": prompt_template}, {"from": "gpt", "value": ""}]

    conjunctions = [
        'you can see ',
        'in front of you is ',
        'there is ',
        'you can spot ',
        'you are toward the ',
        'ahead of you is ',
        'in your sight is ',
    ]

    actions2idx = OrderedDict(
        {
            'STOP': [0],
            "↑": [1],
            "←": [2],
            "→": [3],
            "↓": [5],
        }
    )

    sucs, spls, oss, nes = [], [], [], []
    done_set = set()
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    progress_file = os.path.join(output_path, 'progress.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            for line in f:
                res = json.loads(line)
                sucs.append(res['success'])
                spls.append(res['spl'])
                oss.append(res['os'])
                nes.append(res['ne'])
                if "scene_id" in res:
                    done_set.add((res["scene_id"], res["episode_id"]))

    remaining = num_episodes - len(done_set)
    print(f"Episodes already done: {len(done_set)}, remaining: {remaining}")

    process_bar = tqdm.tqdm(total=remaining, desc="Evaluating")
    seen_episodes = set()
    eval_count = 0

    while True:
        observations = env.reset()
        episode = env.current_episode
        scene_id = episode.scene_id.split('/')[-2]
        episode_id = int(episode.episode_id)
        ep_key = (scene_id, episode_id)

        if ep_key in seen_episodes:
            break
        seen_episodes.add(ep_key)

        if ep_key in done_set:
            continue

        episode_instruction = episode.instruction.instruction_text
        eval_count += 1
        print(f"\n[{eval_count}/{remaining}] Episode {scene_id}_{episode_id:04d}: {episode_instruction[:80]}...")

        rgb_list = []
        action_seq = []
        input_images = []
        output_ids = None
        llm_outputs = ""
        action = None
        messages = []
        local_actions = []

        done = False
        pixel_goal = None
        step_id = 0

        while (not done) and (step_id <= max_steps_per_episode):
            sys.stdout.flush()
            print(f'  [loop] step_id={step_id}, action={action}, pixel_goal={pixel_goal}, action_seq={action_seq}', flush=True)
            rgb = observations["rgb"]
            depth = observations["depth"]
            depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
            depth = depth * (max_depth - min_depth) + min_depth
            depth = depth * 1000

            image = Image.fromarray(rgb).convert('RGB')

            if action == action_code.LOOKDOWN:
                look_down_image = image
                look_down_depth, _ = preprocess_depth_image_v2(
                    Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                    do_depth_scale=True, depth_scale=1000,
                    target_height=224, target_width=224,
                )
                look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                look_down_depth[look_down_depth > 5.0] = 5.0
            else:
                image = image.resize((resize_w, resize_h))
                rgb_list.append(image)

                down_observations = env.step(action_code.LOOKDOWN)
                down_observations = env.step(action_code.LOOKDOWN)

                look_down_image = Image.fromarray(down_observations["rgb"]).convert('RGB')
                depth = down_observations["depth"]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (max_depth - min_depth) + min_depth
                depth = depth * 1000
                look_down_depth, _ = preprocess_depth_image_v2(
                    Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                    do_depth_scale=True, depth_scale=1000,
                    target_height=224, target_width=224,
                )
                look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                look_down_depth[look_down_depth > 5.0] = 5.0

                env.step(action_code.LOOKUP)
                env.step(action_code.LOOKUP)

            if len(action_seq) == 0 and pixel_goal is None:
                if action == action_code.LOOKDOWN:
                    sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                    input_images += [look_down_image]
                    messages.append(
                        {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}
                    )
                    input_img_id = -1
                else:
                    sources = copy.deepcopy(conversation)
                    sources[0]["value"] = sources[0]["value"].replace(
                        '<instruction>.', episode.instruction.instruction_text[:-1]
                    )
                    cur_images = rgb_list[-1:]
                    if step_id == 0:
                        history_id = []
                    else:
                        history_id = np.unique(
                            np.linspace(0, step_id - 1, num_history, dtype=np.int32)
                        ).tolist()
                        placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                        sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                    history_id = sorted(history_id)
                    input_images = [rgb_list[i] for i in history_id] + cur_images
                    input_img_id = 0

                prompt = random.choice(conjunctions) + DEFAULT_IMAGE_TOKEN
                sources[0]["value"] += f" {prompt}."
                prompt_instruction = copy.deepcopy(sources[0]["value"])
                parts = split_and_clean(prompt_instruction)

                content = []
                for i in range(len(parts)):
                    if parts[i] == "<image>":
                        content.append({"type": "image", "image": input_images[input_img_id]})
                        input_img_id += 1
                    else:
                        content.append({"type": "text", "text": parts[i]})

                messages.append({'role': 'user', 'content': content})

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                print(f'  [debug] processing inputs, num_images={len(input_images)}', flush=True)
                inputs = processor(text=[text], images=input_images, return_tensors="pt").to(device)
                print(f'  [debug] input_ids shape={inputs.input_ids.shape}, calling model.generate ...', flush=True)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        use_cache=True,
                        past_key_values=None,
                        return_dict_in_generate=True,
                    ).sequences
                print(f'  [debug] model.generate done', flush=True)

                llm_outputs = processor.tokenizer.decode(
                    output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
                )
                print(f'  step_id: {step_id}, output: {llm_outputs}')

                if bool(re.search(r'\d', llm_outputs)):
                    forward_action = 0
                    coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]
                    pixel_goal = [int(coord[1]), int(coord[0])]

                    env.step(action_code.LOOKUP)
                    env.step(action_code.LOOKUP)

                    local_actions = []
                    pixel_values = inputs.pixel_values
                    image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                    print(f'  [debug] calling generate_latents ...', flush=True)
                    with torch.no_grad():
                        traj_latents = model.generate_latents(output_ids, pixel_values, image_grid_thw)
                    print(f'  [debug] generate_latents done', flush=True)

                    image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                    pix_goal_image = copy.copy(image_dp)
                    images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(device)
                    depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                    pix_goal_depth = copy.copy(depth_dp)
                    depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(device)

                    print(f'  [debug] calling generate_traj ...', flush=True)
                    with torch.no_grad():
                        dp_actions = model.generate_traj(traj_latents, images_dp, depths_dp)
                    print(f'  [debug] generate_traj done, dp_actions={dp_actions}', flush=True)

                    action_list = traj_to_actions(dp_actions)
                    if len(action_list) < MAX_STEPS:
                        action_list += [0] * (MAX_STEPS - len(action_list))

                    local_actions = action_list
                    if len(local_actions) >= MAX_LOCAL_STEPS:
                        local_actions = local_actions[:MAX_LOCAL_STEPS]

                    action = local_actions[0]
                    if action == action_code.STOP:
                        pixel_goal = None
                        output_ids = None
                        action = action_code.LEFT
                        observations = env.step(action)
                        step_id += 1
                        done = env.episode_over
                        messages = []
                        continue
                    print(f'  predicted goal {pixel_goal}')

                else:
                    action_seq = parse_actions(llm_outputs, actions2idx)
                    print(f'  actions {action_seq}')

            if len(action_seq) != 0:
                action = action_seq[0]
                action_seq.pop(0)
            elif pixel_goal is not None:
                if len(local_actions) == 0:
                    local_actions = []
                    image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                    images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(device)
                    depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                    depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(device)

                    with torch.no_grad():
                        dp_actions = model.generate_traj(traj_latents, images_dp, depths_dp)

                    action_list = traj_to_actions(dp_actions)
                    if len(action_list) < MAX_STEPS:
                        action_list += [0] * (MAX_STEPS - len(action_list))

                    local_actions = action_list
                    if len(local_actions) >= MAX_LOCAL_STEPS:
                        local_actions = local_actions[:MAX_LOCAL_STEPS]
                    action = local_actions.pop(0)
                else:
                    action = local_actions.pop(0)

                forward_action += 1
                if forward_action > MAX_STEPS:
                    pixel_goal = None
                    output_ids = None
                    messages = []
                    step_id += 1
                    forward_action = 0
                    local_actions = []
                    continue
                if action == action_code.STOP:
                    pixel_goal = None
                    output_ids = None
                    messages = []
                    step_id += 1
                    forward_action = 0
                    local_actions = []
                    continue
            else:
                action = 0

            if action == action_code.LOOKDOWN:
                env.step(action)
                observations = env.step(action)
                done = env.episode_over
            else:
                observations = env.step(action)
                done = env.episode_over
                step_id += 1
                messages = []

        metrics = env.get_metrics()
        sucs.append(metrics['success'])
        spls.append(metrics['spl'])
        oss.append(metrics['oracle_success'])
        nes.append(metrics['distance_to_goal'])

        print(
            f"  => success: {metrics['success']}, spl: {metrics['spl']:.4f}, "
            f"os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']:.4f}"
        )

        result = {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "success": metrics["success"],
            "spl": metrics["spl"],
            "os": metrics['oracle_success'],
            "ne": metrics["distance_to_goal"],
            "steps": step_id,
            "episode_instruction": episode_instruction,
        }
        with open(progress_file, 'a') as f:
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

    with open(os.path.join(output_path, 'result.json'), 'w') as f:
        json.dump(final_result, f, indent=2)
    print(f"Results saved to {os.path.join(output_path, 'result.json')}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate InternVLA-N1 on VLN-CE R2R val_unseen")
    parser.add_argument("--model_path", type=str, default="/workspace/InternNav_Model")
    parser.add_argument("--scenes_dir", type=str, default="data/scene_data/mp3d_ce")
    parser.add_argument("--data_path", type=str,
                        default="data/vln_ce/raw_data/r2r/{split}/{split}.json.gz")
    parser.add_argument("--output_path", type=str, default="./logs/eval_r2r_val_unseen")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--max_steps_per_episode", type=int, default=500)
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    main()

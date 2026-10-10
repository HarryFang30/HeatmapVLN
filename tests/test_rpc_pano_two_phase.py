from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

pytest.importorskip("vla_rpc")

from scripts.evaluation import rpc_model_server as server


class _Tokenizer:
    eos_token_id = 99

    def encode(self, _text, add_special_tokens=False):
        assert not add_special_tokens
        return [41, 42]

    def decode(self, _ids, skip_special_tokens=True):
        assert skip_special_tokens
        return "view: right\npixel: 10 20"


class _Processor:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        has_assistant = any(message.get("role") == "assistant" for message in messages)
        assert has_assistant is (not kwargs["add_generation_prompt"])
        ids = [1, 2, 3, 41, 42, 99] if has_assistant else [1, 2, 3]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "pixel_values": torch.zeros(4, 3, 2, 2),
            "image_grid_thw": torch.ones(4, 3, dtype=torch.long),
        }


class _Qwen:
    def __init__(self):
        self.generate_calls = 0
        self.latent_calls = 0
        self.last_latent_output_ids = None
        self.model = self

    def generate(self, **kwargs):
        self.generate_calls += 1
        suffix = torch.tensor([[7, 8]], dtype=kwargs["input_ids"].dtype)
        return SimpleNamespace(sequences=torch.cat([kwargs["input_ids"], suffix], dim=1))

    def generate_latents(self, **kwargs):
        self.latent_calls += 1
        self.last_latent_output_ids = kwargs["output_ids"].detach().clone()
        return torch.zeros(1, 4, 8)


def _runtime():
    runtime = object.__new__(server.HeatmapVLNRuntime)
    runtime.require_deterministic_sampling = False
    runtime.device = torch.device("cpu")
    runtime.processor = _Processor()
    runtime.train_cfg = {
        "data": {
            "image_size": [256, 256],
            "trajectory": {
                "system2_sft_protocol": "direct",
                "structured_pano_output": True,
            },
        }
    }
    qwen = _Qwen()
    runtime.model = SimpleNamespace(
        qwen2_5_vl=qwen,
        latent_queries=torch.zeros(1, 4, 8),
        config=SimpleNamespace(dtype=torch.float32),
        nextdit_action_head=SimpleNamespace(cond_projector=None),
    )
    runtime.pano_latent_adapter = object()
    runtime.has_nextdit = True
    runtime.num_sample_trajs = 32
    runtime.action_scale = 4.0
    runtime.ppa_stage0_action_arm = "disabled"
    # The deployed shape for the arm this test covers.  Two-phase recenter is by
    # construction the *non*-PPA arm: the client refuses to launch the two
    # together, in two separate guards --
    #   "formal online PPA uses native front-only InternNav and forbids
    #    pano_recenter_before_system1"
    #   "online AMB3R PPA requires --no-pano_recenter_before_system1"
    # -- so ppa_online_amb3r=False is what --pano_recenter_before_system1 really
    # runs with, not a value picked to dodge the PPA branch.  The server
    # advertises the two-phase capability unconditionally, and the PPA path has
    # its own coverage in tests/test_latency_timing.py (MODEL_CASES).
    runtime.ppa_online_amb3r = False
    runtime.system2_cognition_arm = False
    return runtime, qwen


def _payload(phase, **extra):
    return {
        "phase": phase,
        "instruction": "go",
        "num_history": 0,
        "vlm_image_size": [256, 256],
        "traj_image_size": [224, 224],
        **extra,
    }


def test_two_phase_rpc_skips_system1_until_after_real_recenter(monkeypatch):
    monkeypatch.setattr(server, "_blobs_by_name", lambda _blobs: defaultdict(object))
    monkeypatch.setattr(
        server,
        "_pil_from_blob",
        lambda *_args, **_kwargs: Image.new("RGB", (8, 8)),
    )
    monkeypatch.setattr(
        server,
        "_lookdown_to_traj_tensor",
        lambda *_args, **_kwargs: torch.zeros(3, 8, 8),
    )
    adapter_calls = []

    def apply_adapter(traj_hs, _adapter, **kwargs):
        adapter_calls.append(kwargs)
        return traj_hs

    monkeypatch.setattr(server, "_maybe_apply_pano_latent_adapter", apply_adapter)
    monkeypatch.setattr(
        server,
        "_trajectory_from_condition",
        lambda *_args, **_kwargs: torch.zeros(32, 4, 3),
    )
    monkeypatch.setattr(server, "traj_to_actions", lambda *_args, **_kwargs: [1])

    runtime, qwen = _runtime()
    system2 = runtime.plan_panoramic(_payload("system2"), [])

    assert system2["kind"] == "pano_goal"
    assert system2["pano_goal_view"] == "right"
    assert system2["pixel_goal"] == [10, 20]
    assert qwen.generate_calls == 1
    assert qwen.latent_calls == 0
    assert adapter_calls == []

    front = runtime.plan_panoramic(
        _payload("front_system1", pixel_goal=[10, 20]),
        [],
    )

    assert front["kind"] == "trajectory"
    assert front["pano_goal_view"] == "front"
    assert front["trajectory_heading_alignment"] == "none"
    assert qwen.generate_calls == 1
    assert qwen.latent_calls == 1
    # front_system1 uses the same complete teacher-forced assistant turn as
    # training, including the chat-template terminator.
    assert qwen.last_latent_output_ids.tolist() == [[1, 2, 3, 41, 42, 99]]
    phase2_messages, phase2_template_kwargs = runtime.processor.template_calls[-1]
    assert phase2_template_kwargs["add_generation_prompt"] is False
    assert phase2_messages[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "view: front\npixel: 10 20"}],
    }
    assert adapter_calls[0]["view_id"] == "front"
    assert adapter_calls[0]["pixel_goal"] == [10, 20]


def test_internnav_lookdown_helpers_preserve_native_coordinate_order():
    lookdown = Image.new("RGB", (384, 384))
    first_messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]

    assert server._internnav_requests_lookdown("↓")
    assert server._internnav_requests_lookdown("TILT DOWN ↓")
    assert not server._internnav_requests_lookdown("216 308")
    assert not server._internnav_requests_lookdown("←")
    assert server._parse_internnav_pixel_goal("216 308") == [308, 216]

    second_messages = server._append_internnav_lookdown_turn(
        first_messages,
        first_output="↓",
        lookdown_frame=lookdown,
    )
    assert len(first_messages) == 1
    assert [message["role"] for message in second_messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert second_messages[-1]["content"][0] == {
        "type": "text",
        "text": "you can see ",
    }
    assert second_messages[-1]["content"][1]["image"] is lookdown


# test_internnav_rpc_runs_second_lookdown_generation lived here.  Retired: every
# contract it held is now covered against the shape the native protocol is
# actually deployed in, which this test was not.
#
#   - that a "down" first turn triggers a second generation carrying the
#     lookdown frame:  tests/test_latency_timing.py MODEL_CASES["ready_two_turns"]
#     drives it through the servicer with ppa_online_amb3r=True, real pose
#     fields and phase="joint", asserts the system2_turn2_prep/_generate stages,
#     and pins the whole response by sha256.
#   - the two-turn message shape (one image, then assistant + a second user turn
#     with two):  tests/test_native_internnav_exact.py
#     ::test_lookdown_turn_appends_assistant_and_user, in more detail than here,
#     plus ::test_lookdown_request_requires_leading_arrow_and_no_digits.
#   - the "v u" -> [u, v] order:  test_internnav_lookdown_helpers_preserve_
#     native_coordinate_order, just above.
#
# It had also drifted out of reach of the code it claimed to test: it fed a
# 384x384 lookdown, where ed46c76 made the native path fail closed on anything
# but NATIVE_LOOKDOWN_SIZE (640x480), and it drove phase="system2", which the
# native protocol never sees -- native InternNav is the PPA arm, and PPA forbids
# --pano_recenter_before_system1, so the client only ever sends phase="joint".

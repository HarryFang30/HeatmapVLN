"""EXP-37 serving: ``rpc_model_server --system2_base_decides_stop``.

On every System2 call the released weights (the adapter disabled) greedily produce the
first turn-1 token; STOP ends the call there, otherwise the adapter decodes as usual
with both STOP ids suppressed at its first generated position.  Without the flag the
server is unchanged.  Three layers:

- the deployed ``_plan_panoramic_native`` on the CPU fakes of ``test_latency_timing``
  (needs ``vla_rpc``): which calls run, in which adapter state and vision scope, with
  which kwargs, and the response against the pinned no-flag payloads;
- the start-up checks, and transformers' own ``begin_suppress_tokens`` on a tiny random
  Qwen2.5-VL with an image, wrapped by the server's PEFT loader (needs ``peft``);
- the NPU launcher: it passes the flag, checks the server's evidence and records the
  mode in servers.json.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests import test_external_server_guards as guards
from tests import test_latency_timing as tl
from tests.test_latency_timing import model_server  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[1]
NPU_LAUNCHER = REPO / "scripts" / "ascend" / "run_ppa_servers_npu.sh"
SERVE_910B = REPO / "scripts" / "exp28" / "serve_910b.sh"
SHA = "ab" * 32
# InternNav_Model's vocab: "STOP" and " STOP" (docs/experiments/README.md, EXP-13 runbook).
STOP_IDS = [50669, 45537]
NOT_STOP = 2189  # any other first token


class _BaseStopQwen:
    """Stands in for Qwen2_5VLIntegration: answers the base's first token, records every call."""

    def __init__(self, vision_scope, base_token):
        self.enabled = True
        self.base_token = base_token
        self.calls: list[dict] = []
        self.model = self
        self._vision_scope = vision_scope

    def _note(self, name, kwargs):
        scope = self._vision_scope() or {}
        self.calls.append(
            {
                "name": name,
                "enabled": self.enabled,
                "recording": scope.get("recording", 0),
                "serving": scope.get("serving", 0),
                # The decoding kwargs only: the model inputs are tensors.
                "kwargs": {k: v for k, v in kwargs.items() if not isinstance(v, torch.Tensor)},
            }
        )

    @contextlib.contextmanager
    def adapters_disabled(self):
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True

    def generate(self, **kwargs):
        self._note("generate", kwargs)
        if not self.enabled:  # the base's one token
            return SimpleNamespace(sequences=torch.cat([kwargs["input_ids"], torch.tensor([[self.base_token]])], dim=1))
        return SimpleNamespace(sequences=torch.cat([kwargs["input_ids"], torch.tensor([[7, 8]])], dim=1))

    def generate_latents(self, **kwargs):
        self._note("generate_latents", {})
        return torch.zeros(1, 4, tl.HIDDEN_DIM)


def _serve(server, vla_pb2, monkeypatch, case, *, base_stop, base_token=NOT_STOP, answers=None):
    """One InferJSON on the fakes; ``base_stop`` is the runtime's system2_base_stop (or "absent")."""
    from src.models import qwen2_5_vl_vision_count as vision_count

    monkeypatch.delenv(tl.latency.TIMING_ENV, raising=False)
    kwargs, case_answers, _ = tl.MODEL_CASES[case]
    runtime = tl._model_runtime(server, case_answers if answers is None else answers)
    qwen = _BaseStopQwen(vision_count._scope, base_token)
    runtime.model.qwen2_5_vl = qwen
    runtime.system2_lora = {"adapter_sha256": SHA, "z0_source": "adapters_disabled"}
    if base_stop != "absent":
        runtime.system2_base_stop = base_stop
    servicer = server.HeatmapVLNRPCServicer(runtime)
    errors = []
    context = SimpleNamespace(set_details=errors.append, set_code=errors.append)
    response = servicer.InferJSON(tl._model_request(vla_pb2, 0, **kwargs), context)
    assert errors == []
    return json.loads(response.json_payload), qwen, servicer


ON = {"stop_ids": STOP_IDS}
# (case, the Qwen calls the adapter path makes today)
CALLS = {
    "ready_two_turns": ["generate", "generate", "generate_latents"],
    "warmup_one_turn": ["generate", "generate_latents"],
    "native_arrows": ["generate"],
}


def _pinned_rest(payload):
    """The payload minus the LoRA identity: must hash to the pinned no-flag payload."""
    payload = dict(payload)
    assert payload.pop("system2_lora") == {"adapter_sha256": SHA, "z0_source": "adapters_disabled"}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# The deployed call on the CPU fakes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", sorted(CALLS))
def test_base_stop_ends_the_call_without_running_the_adapter(model_server, monkeypatch, caplog, case):
    server, vla_pb2 = model_server
    caplog.set_level(logging.INFO, logger="heatmapvln-rpc-server")
    payload, qwen, _ = _serve(server, vla_pb2, monkeypatch, case, base_stop=ON, base_token=STOP_IDS[0], answers=["STOP"])
    # One call: the base's single token, adapter off, recorded like turn 1.  No adapter
    # decode, no turn 2, no System1.
    assert [call["name"] for call in qwen.calls] == ["generate"]
    (base,) = qwen.calls
    assert not base["enabled"] and base["recording"] == 1 and base["serving"] == 0
    assert base["kwargs"] == {**server.system2_generate_kwargs(), "max_new_tokens": 1}
    assert qwen.enabled  # restored
    assert payload["kind"] == "stop" and payload["terminal"] is True and payload["actions"] == [0]
    assert payload["llm_output"] == "STOP" and payload["native_first_output"] == "STOP"
    assert payload["native_lookdown_turns"] == 0 and payload["native_front_only"] is True
    assert payload["system2_stop_source"] == "base"
    assert payload["system2_lora"] == {"adapter_sha256": SHA, "z0_source": "adapters_disabled"}
    assert "System2 stop source: base (kind stop)" in caplog.text


def test_the_spaced_spelling_stops_too(model_server, monkeypatch):
    server, vla_pb2 = model_server
    payload, qwen, _ = _serve(
        server, vla_pb2, monkeypatch, "native_arrows", base_stop=ON, base_token=STOP_IDS[1], answers=[" STOP"]
    )
    assert len(qwen.calls) == 1
    assert payload["kind"] == "stop" and payload["system2_stop_source"] == "base"


@pytest.mark.parametrize("case", sorted(CALLS))
def test_base_goes_on_and_the_adapter_decodes_with_stop_suppressed(model_server, monkeypatch, caplog, case):
    server, vla_pb2 = model_server
    caplog.set_level(logging.INFO, logger="heatmapvln-rpc-server")
    payload, qwen, _ = _serve(server, vla_pb2, monkeypatch, case, base_stop=ON)
    assert [call["name"] for call in qwen.calls] == ["generate", *CALLS[case]]
    base, turn1, *rest = qwen.calls
    assert not base["enabled"] and base["recording"] == 1
    assert base["kwargs"] == {**server.system2_generate_kwargs(), "max_new_tokens": 1}
    # The adapter's turn 1: today's kwargs plus the first-position mask, nothing else.
    assert turn1["enabled"] and turn1["recording"] == 1 and turn1["serving"] == 0
    assert turn1["kwargs"] == {**server.system2_generate_kwargs(), "begin_suppress_tokens": STOP_IDS}
    for call in rest:
        if call["name"] == "generate":  # turn 2 decodes exactly as today
            assert call["enabled"] and call["recording"] == 1
            assert call["kwargs"] == server.system2_generate_kwargs()
        else:  # Z0: adapter off, served from the record
            assert not call["enabled"] and call["serving"] == 1 and call["recording"] == 0
    assert payload.pop("system2_stop_source") == "none"
    # Everything else is the pinned no-flag payload: same keys, same order, same values.
    assert _pinned_rest(payload) == tl.HEAD_PAYLOAD_SHA256[case]
    assert f"System2 stop source: none (kind {'native_actions' if case == 'native_arrows' else 'trajectory'})" in caplog.text


@pytest.mark.parametrize(("answer", "kind"), [("STOP", "stop"), ("I am not sure", "fallback_stop")])
def test_a_stop_the_adapter_still_makes_is_counted_as_the_adapters(model_server, monkeypatch, caplog, answer, kind):
    server, vla_pb2 = model_server
    caplog.set_level(logging.INFO, logger="heatmapvln-rpc-server")
    payload, qwen, _ = _serve(server, vla_pb2, monkeypatch, "native_arrows", base_stop=ON, answers=[answer])
    assert [call["enabled"] for call in qwen.calls] == [False, True]
    assert payload["kind"] == kind and payload["terminal"] is True
    assert payload["system2_stop_source"] == "adapter"
    assert f"System2 stop source: adapter (kind {kind})" in caplog.text


@pytest.mark.parametrize("base_stop", ["absent", None], ids=["attribute_absent", "flag_off"])
@pytest.mark.parametrize("case", sorted(CALLS))
def test_without_the_flag_nothing_changes(model_server, monkeypatch, caplog, case, base_stop):
    server, vla_pb2 = model_server
    caplog.set_level(logging.INFO, logger="heatmapvln-rpc-server")
    payload, qwen, servicer = _serve(server, vla_pb2, monkeypatch, case, base_stop=base_stop)
    assert [call["name"] for call in qwen.calls] == CALLS[case]  # no extra generate
    for call in qwen.calls:
        if call["name"] == "generate":
            assert call["enabled"] and call["kwargs"] == server.system2_generate_kwargs()
    assert "system2_stop_source" not in payload
    assert _pinned_rest(payload) == tl.HEAD_PAYLOAD_SHA256[case]
    assert "System2 stop source" not in caplog.text
    assert "system2-base-stop-v1" not in list(servicer.GetServerInfo(None, None).supported_formats)


def test_server_info_names_the_mode_after_the_adapter(model_server):
    server, _ = model_server
    runtime = tl._model_runtime(server, [])
    runtime.system2_lora = {"adapter_sha256": SHA, "z0_source": "adapters_disabled"}
    runtime.system2_base_stop = ON
    servicer = server.HeatmapVLNRPCServicer(runtime, server_instance="run/slot0")
    formats = list(servicer.GetServerInfo(None, None).supported_formats)
    assert formats[: formats.index("heatmapvln-instance:run/slot0")][-3:] == [
        "system2-lora-v1",
        f"system2-lora-sha256:{SHA}",
        "system2-base-stop-v1",
    ]
    # The client's adapter pin reads only the sha entries; the mode adds none.
    assert [item for item in formats if item.startswith("system2-lora-sha256:")] == [f"system2-lora-sha256:{SHA}"]


def test_model_version_names_the_mode_after_the_adapter_sha():
    source = (REPO / "scripts" / "evaluation" / "rpc_model_server.py").read_text(encoding="utf-8")
    init = source.split("class HeatmapVLNRuntime:", 1)[1].split("def _preflight_system2_base_stop", 1)[0]
    lora = init.index('+system2-lora:{self.system2_lora')
    mode = init.index("SYSTEM2_BASE_STOP_VERSION_SUFFIX}")
    assert lora < mode
    assert init.index("self.system2_base_stop = self._preflight_system2_base_stop()") < lora


# --------------------------------------------------------------------------- #
# Start-up checks
# --------------------------------------------------------------------------- #
class _Vocab:
    def __init__(self, table):
        self.table = table

    def encode(self, text, add_special_tokens=False):
        return list(self.table[text])


VOCAB = {"STOP": [STOP_IDS[0]], " STOP": [STOP_IDS[1]]}


def _preflight_runtime(**overrides):
    runtime = SimpleNamespace(
        args=SimpleNamespace(system2_base_decides_stop=True),
        system2_cognition_arm=False,
        system2_lora={"adapter_sha256": SHA},
        processor=SimpleNamespace(tokenizer=_Vocab(VOCAB)),
    )
    for name, value in overrides.items():
        setattr(runtime, name, value)
    return runtime


REFUSED = [
    ({"system2_cognition_arm": True}, {}, "--system2_cognition_arm"),
    ({"system2_lora": None}, {}, "needs --system2_lora_adapter"),
    ({}, {"PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE": "0.7"}, "greedy decoding only"),
    ({"processor": SimpleNamespace(tokenizer=_Vocab({**VOCAB, "STOP": [3, 4]}))}, {}, "'STOP' is not a single token"),
    ({"processor": SimpleNamespace(tokenizer=_Vocab({**VOCAB, " STOP": []}))}, {}, "' STOP' is not a single token"),
    ({"processor": SimpleNamespace(tokenizer=_Vocab({"STOP": [9], " STOP": [9]}))}, {}, "map to the same id"),
]


@pytest.mark.parametrize(("overrides", "env", "phrase"), REFUSED, ids=[p for *_, p in REFUSED])
def test_the_mode_refuses_what_it_cannot_serve(model_server, monkeypatch, overrides, env, phrase):
    server, _ = model_server
    monkeypatch.delenv("PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=re.escape(phrase)):
        server.HeatmapVLNRuntime._preflight_system2_base_stop(_preflight_runtime(**overrides))


def test_the_mode_resolves_both_stop_ids_and_says_so(model_server, monkeypatch, caplog):
    server, _ = model_server
    monkeypatch.delenv("PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE", raising=False)
    caplog.set_level(logging.INFO, logger="heatmapvln-rpc-server")
    assert server.HeatmapVLNRuntime._preflight_system2_base_stop(_preflight_runtime()) == {"stop_ids": STOP_IDS}
    assert f"System2 base decides STOP enabled (stop ids {STOP_IDS})" in caplog.text
    # Off: nothing is checked, nothing is logged.
    off = _preflight_runtime(args=SimpleNamespace(system2_base_decides_stop=False), system2_lora=None)
    assert server.HeatmapVLNRuntime._preflight_system2_base_stop(off) is None
    absent = _preflight_runtime(args=SimpleNamespace(), system2_lora=None)
    assert server.HeatmapVLNRuntime._preflight_system2_base_stop(absent) is None


def test_the_cli_flag_defaults_off(model_server, monkeypatch):
    server, _ = model_server
    base = ["rpc_model_server.py", "--config", "c.yaml"]
    monkeypatch.setattr(sys, "argv", base)
    assert server.parse_args().system2_base_decides_stop is False
    monkeypatch.setattr(sys, "argv", [*base, "--system2_lora_adapter", "/a", "--system2_base_decides_stop"])
    assert server.parse_args().system2_base_decides_stop is True


# --------------------------------------------------------------------------- #
# transformers 4.51's begin_suppress_tokens on a real multimodal generate
# --------------------------------------------------------------------------- #
def _mm_inputs():
    """One 28x28 image (4 patches, 1 merged token) between vision start/end, then text."""
    generator = torch.Generator().manual_seed(3)
    ids = torch.tensor([[0, 5, 62, 60, 63, 9, 17, 23]])
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "pixel_values": torch.randn(4, 3 * 2 * 14 * 14, generator=generator),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }


KWARGS = {
    "max_new_tokens": 5,
    "do_sample": False,
    "use_cache": True,
    "return_dict_in_generate": True,
    "repetition_penalty": 1.05,  # InternNav_Model's generation_config
    "output_scores": True,
}


def test_begin_suppress_tokens_masks_only_the_first_generated_position(model_server, tmp_path):
    from tests.test_system2_lora_serving import _save_adapter, _tiny_qwen

    server, _ = model_server
    adapter = _save_adapter(tmp_path / "trained", zero=False)
    wrapped, _ = server._wrap_system2_lora(_tiny_qwen(), str(adapter))
    inputs = _mm_inputs()
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        free = wrapped.generate(**inputs, **KWARGS)
        first = int(free.sequences[0, prompt_len])
        masked = wrapped.generate(**inputs, **KWARGS, begin_suppress_tokens=[first])
    assert masked.scores[0][0, first] == -float("inf")
    assert int(masked.sequences[0, prompt_len]) != first
    # The mask is the only change at that position (the penalty and the logits are as without it).
    keep = torch.ones_like(free.scores[0][0], dtype=torch.bool)
    keep[first] = False
    assert torch.equal(masked.scores[0][0][keep], free.scores[0][0][keep])
    assert all(torch.isfinite(step[0, first]) for step in masked.scores[1:])  # later positions untouched


def test_the_base_first_token_is_the_first_token_of_a_server_without_the_adapter(model_server, tmp_path):
    from tests.test_system2_lora_serving import _save_adapter, _tiny_qwen

    server, _ = model_server
    adapter = _save_adapter(tmp_path / "trained", zero=False)
    wrapped, _ = server._wrap_system2_lora(_tiny_qwen(), str(adapter))
    plain = _tiny_qwen()
    inputs = _mm_inputs()
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        released = plain.generate(**inputs, **KWARGS)
        with wrapped.disable_adapter():
            one = wrapped.generate(**inputs, **{**KWARGS, "max_new_tokens": 1})
        adapted = wrapped.generate(**inputs, **KWARGS)
    assert one.sequences.shape[1] == prompt_len + 1
    assert torch.equal(one.sequences[0], released.sequences[0, : prompt_len + 1])
    assert torch.equal(one.scores[0], released.scores[0])
    assert not torch.equal(adapted.scores[0], released.scores[0])  # the adapter really is on otherwise


# --------------------------------------------------------------------------- #
# The launchers
# --------------------------------------------------------------------------- #
def test_npu_launcher_passes_the_flag_and_checks_the_evidence():
    text = NPU_LAUNCHER.read_text(encoding="utf-8")
    assert 'SYSTEM2_BASE_STOP="${PPA_EVAL_SYSTEM2_BASE_STOP:-0}"' in text
    assert '[[ "$SYSTEM2_BASE_STOP" -eq 1 ]] && MODEL_EXTRA+=(--system2_base_decides_stop)' in text
    assert 'grep -F "System2 base decides STOP enabled (stop ids "' in text
    assert "PPA_EVAL_SYSTEM2_BASE_STOP=1 needs PPA_EVAL_SYSTEM2_LORA" in text
    server = (REPO / "scripts" / "evaluation" / "rpc_model_server.py").read_text(encoding="utf-8")
    assert '"System2 base decides STOP enabled (stop ids %s)"' in server
    assert '"--system2_base_decides_stop"' in server
    serve = SERVE_910B.read_text(encoding="utf-8")
    assert 'lora_env=(PPA_EVAL_SYSTEM2_BASE_STOP="$BASE_STOP")' in serve
    assert "\"system2_base_stop\": true" in serve


def _servers_json(path: Path, *extra: str) -> dict:
    argv = [str(path), "2", "52400", "52500", "0,1", "2,3", guards.COMMIT, str(guards.VO_SEED), "0", "0",
            guards.INSTANCE, guards.MODEL_VERSION, "0", "", "", *extra]
    result = subprocess.run([sys.executable, "-", *argv], input=guards._servers_json_writer_source(),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("extra", "expected"),
    [((SHA, "1"), True), ((SHA, "0"), False), ((SHA,), False), ((), False)],
    ids=["on", "off", "record_before_the_field", "record_before_lora"],
)
def test_servers_json_records_the_mode(tmp_path, extra, expected):
    record = _servers_json(tmp_path / "servers.json", *extra)
    assert record["system2_base_stop"] is expected
    assert record["system2_lora_sha256"] == (extra[0] if extra else "")

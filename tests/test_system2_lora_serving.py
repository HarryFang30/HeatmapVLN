"""EXP-23 serving: ``rpc_model_server --system2_lora_adapter``.

System2 decides with a LoRA (both native turns), System1's Z0 is computed with the
adapter disabled, and without the flag nothing changes.  Three layers:

- the deployed ``_plan_panoramic_native`` on the CPU fakes of ``test_latency_timing``
  (needs ``vla_rpc``): which adapter state each Qwen call runs in, the vision-pass
  scopes, the response and GetServerInfo identity, and the pinned no-flag payloads;
- the real PEFT wrapping on a tiny random Qwen2.5-VL (needs ``peft``/``transformers``):
  a B = 0 adapter serves byte-identically, ``adapters_disabled()`` restores the base,
  and adapters outside the decoder blocks, incomplete or non-LoRA ones are refused;
- the three launchers: they pass the flag, check the server's evidence, and the
  client launcher refuses a server set serving another adapter (or one it did not ask for).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests import test_external_server_guards as guards
from tests import test_latency_timing as tl
from tests.test_external_server_guards import tree  # noqa: F401  (fixture)
from tests.test_latency_timing import model_server  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[1]
NPU_LAUNCHER = REPO / "scripts" / "ascend" / "run_ppa_servers_npu.sh"
CUDA_LAUNCHER = REPO / "scripts" / "run_ppa_r2r_val_unseen_cuda.sh"
NAV_LAUNCHER = REPO / "scripts" / "deploy" / "start_nav_servers_cuda.sh"
SHA = "ab" * 32
# The tiny model's language-model blocks, the only place an EXP-23 adapter may sit
# (transformers 4.51 names: model.layers.N.*; the tower is visual.blocks.N.*).
LLM_ONLY = r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"
# A plain list matches by suffix, and the vision tower's MLP has gate/up/down_proj too.
SUFFIXES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# --------------------------------------------------------------------------- #
# The deployed call on the CPU fakes
# --------------------------------------------------------------------------- #
class _AdapterQwen:
    """Stands in for Qwen2_5VLIntegration: records the adapter state of every call."""

    def __init__(self, vision_scope):
        self.enabled = True
        self.disabled_blocks = 0
        self.calls: list[tuple[str, bool, int, int]] = []
        self.model = self
        self._vision_scope = vision_scope

    def _note(self, name):
        scope = self._vision_scope() or {}
        self.calls.append((name, self.enabled, scope.get("recording", 0), scope.get("serving", 0)))

    @contextlib.contextmanager
    def adapters_disabled(self):
        self.disabled_blocks += 1
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True

    def generate(self, **kwargs):
        self._note("generate")
        return SimpleNamespace(sequences=torch.cat([kwargs["input_ids"], torch.tensor([[7, 8]])], dim=1))

    def generate_latents(self, **_kwargs):
        self._note("generate_latents")
        return torch.zeros(1, 4, tl.HIDDEN_DIM)


def _lora_identity(sha=SHA):
    return {"adapter_sha256": sha, "z0_source": "adapters_disabled", "capability": "system2-lora-v1"}


def _serve(server, vla_pb2, monkeypatch, case, *, lora):
    """One InferJSON on the fakes; ``lora`` is the runtime's system2_lora (or "absent")."""
    from src.models import qwen2_5_vl_vision_count as vision_count

    monkeypatch.delenv(tl.latency.TIMING_ENV, raising=False)
    kwargs, answers, _ = tl.MODEL_CASES[case]
    runtime = tl._model_runtime(server, answers)
    qwen = _AdapterQwen(vision_count._scope)
    runtime.model.qwen2_5_vl = qwen
    if lora != "absent":
        runtime.system2_lora = lora
    servicer = server.HeatmapVLNRPCServicer(runtime)
    errors = []
    context = SimpleNamespace(set_details=errors.append, set_code=errors.append)
    response = servicer.InferJSON(tl._model_request(vla_pb2, 0, **kwargs), context)
    assert errors == []
    return response, qwen, servicer


# (case, the Qwen calls it makes)
CALLS = {
    "ready_two_turns": ["generate", "generate", "generate_latents"],
    "warmup_one_turn": ["generate", "generate_latents"],
    "native_arrows": ["generate"],
}


@pytest.mark.parametrize("case", sorted(CALLS))
def test_the_adapter_decides_and_z0_is_taken_with_it_disabled(model_server, monkeypatch, case):
    server, vla_pb2 = model_server
    response, qwen, _ = _serve(server, vla_pb2, monkeypatch, case, lora=_lora_identity())
    assert [name for name, *_ in qwen.calls] == CALLS[case]
    for name, enabled, recording, serving in qwen.calls:
        if name == "generate":
            # Both native turns: adapter on, and recorded for the latent pass to reuse.
            assert enabled and recording == 1 and serving == 0
        else:
            # Z0: adapter off, inside the one scope that may be served from the record.
            assert not enabled and serving == 1 and recording == 0
    assert qwen.enabled  # restored after the request
    assert qwen.disabled_blocks == CALLS[case].count("generate_latents")

    payload = json.loads(response.json_payload)
    assert payload.pop("system2_lora") == {"adapter_sha256": SHA, "z0_source": "adapters_disabled"}
    # Everything else is the pinned no-flag payload: same keys, same order, same values.
    rest = json.dumps(payload, ensure_ascii=False)
    assert hashlib.sha256(rest.encode("utf-8")).hexdigest() == tl.HEAD_PAYLOAD_SHA256[case]


@pytest.mark.parametrize("lora", ["absent", None], ids=["attribute_absent", "flag_off"])
@pytest.mark.parametrize("case", sorted(CALLS))
def test_without_the_flag_the_server_is_unchanged(model_server, monkeypatch, case, lora):
    server, vla_pb2 = model_server
    response, qwen, servicer = _serve(server, vla_pb2, monkeypatch, case, lora=lora)
    assert hashlib.sha256(response.json_payload.encode("utf-8")).hexdigest() == tl.HEAD_PAYLOAD_SHA256[case]
    assert qwen.disabled_blocks == 0 and all(enabled for _, enabled, *_ in qwen.calls)
    assert response.model_v == "ppa-stage2-online-amb3r:test"
    formats = list(servicer.GetServerInfo(None, None).supported_formats)
    assert not [item for item in formats if item.startswith("system2-lora")]


def test_server_info_names_the_adapter_before_the_instance_entries(model_server):
    server, _ = model_server
    runtime = tl._model_runtime(server, [])
    runtime.system2_lora = _lora_identity()
    servicer = server.HeatmapVLNRPCServicer(runtime, server_instance="run/slot0")
    formats = list(servicer.GetServerInfo(None, None).supported_formats)
    assert formats[: formats.index("heatmapvln-instance:run/slot0")][-2:] == [
        "system2-lora-v1",
        f"system2-lora-sha256:{SHA}",
    ]
    assert "ppa-online-amb3r-v1" in formats


# --------------------------------------------------------------------------- #
# Incompatible modes are refused before anything is loaded
# --------------------------------------------------------------------------- #
def _preflight_runtime(server, adapter="/nonexistent/system2_lora", **overrides):
    runtime = object.__new__(server.HeatmapVLNRuntime)
    runtime.args = SimpleNamespace(system2_lora_adapter=adapter)
    runtime.system2_cognition_arm = False
    runtime.ppa_stage0_action_arm = "disabled"
    runtime.pano_latent_adapter = None
    runtime.train_cfg = {
        "data": {"trajectory": {"system2_sft_protocol": "internnav", "structured_pano_output": False}}
    }
    runtime.model = SimpleNamespace(qwen2_5_vl=SimpleNamespace(config=SimpleNamespace(use_lora=False), model=None))
    for name, value in overrides.items():
        setattr(runtime, name, value)
    return runtime


REFUSED_MODES = [
    ({"system2_cognition_arm": True}, "--system2_cognition_arm"),
    ({"ppa_stage0_action_arm": "baseline"}, "Stage-0"),
    ({"pano_latent_adapter": object()}, "panoramic latent adapter"),
    ({"args": SimpleNamespace(system2_lora_adapter="/nonexistent/system2_lora", workers=4)}, "--workers 1"),
    (
        {"train_cfg": {"data": {"trajectory": {"system2_sft_protocol": "internnav", "structured_pano_output": True}}}},
        "two-turn InternNav protocol",
    ),
    ({"train_cfg": {"data": {"trajectory": {"system2_sft_protocol": "direct"}}}}, "two-turn InternNav protocol"),
    (
        {"model": SimpleNamespace(qwen2_5_vl=SimpleNamespace(config=SimpleNamespace(use_lora=True), model=None))},
        "use_lora=false",
    ),
]


@pytest.mark.parametrize(("overrides", "phrase"), REFUSED_MODES, ids=[p for _, p in REFUSED_MODES])
def test_modes_with_their_own_lora_or_z0_transform_are_refused(model_server, overrides, phrase):
    server, _ = model_server
    with pytest.raises(RuntimeError, match=re.escape(phrase)):
        server.HeatmapVLNRuntime._preflight_system2_lora(_preflight_runtime(server, **overrides))


def test_the_compatible_path_reaches_the_adapter_and_no_flag_loads_nothing(model_server):
    server, _ = model_server
    with pytest.raises(RuntimeError, match="has no adapter_config.json"):
        server.HeatmapVLNRuntime._preflight_system2_lora(_preflight_runtime(server))
    assert server.HeatmapVLNRuntime._preflight_system2_lora(_preflight_runtime(server, adapter=None)) is None
    assert server.HeatmapVLNRuntime._preflight_system2_lora(_preflight_runtime(server, adapter="")) is None


# --------------------------------------------------------------------------- #
# The real PEFT wrapping on a tiny random Qwen2.5-VL
# --------------------------------------------------------------------------- #
def _tiny_qwen(seed=0):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(seed)
    config = transformers.Qwen2_5_VLConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "fullatt_block_indexes": [1],
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    )
    return transformers.Qwen2_5_VLForConditionalGeneration(config).eval()


def _save_adapter(directory: Path, *, target_modules=LLM_ONLY, zero=True, seed=0) -> Path:
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(
        _tiny_qwen(seed),
        peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, target_modules=target_modules, task_type="CAUSAL_LM"),
    )
    if not zero:
        generator = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.copy_(torch.randn(parameter.shape, generator=generator))
    model.save_pretrained(str(directory))
    return directory


IDS = torch.tensor([[0, 5, 9, 17, 23, 31, 4, 40]])


def _logits(model):
    with torch.no_grad():
        return model(input_ids=IDS).logits


def _greedy(model):
    with torch.no_grad():
        return model.generate(input_ids=IDS, max_new_tokens=6, do_sample=False)


def test_a_zero_adapter_serves_byte_identically(model_server, tmp_path):
    server, _ = model_server
    adapter = _save_adapter(tmp_path / "zero")
    base = _tiny_qwen()
    wrapped, contract = server._wrap_system2_lora(_tiny_qwen(), str(adapter))
    assert contract["lora_b_all_zero"] is True
    assert contract["lora_layers"] == 2 * 7 and contract["lora_tensors"] == 2 * 2 * 7
    assert contract["target_modules"] == sorted(name for name in SUFFIXES)
    weights = adapter / "adapter_model.safetensors"
    assert contract["adapter_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert contract["z0_source"] == "adapters_disabled"
    assert torch.equal(_logits(wrapped), _logits(base))
    assert torch.equal(_greedy(wrapped), _greedy(base))
    assert not any(parameter.requires_grad for parameter in wrapped.parameters())


def test_adapters_disabled_restores_the_released_weights(model_server, tmp_path):
    """The integration's own context manager (the one generate_latents runs in) on a real PeftModel."""
    server, _ = model_server
    from src.models.qwen2_5_vl.integration import Qwen2_5VLConfig, Qwen2_5VLIntegration

    adapter = _save_adapter(tmp_path / "trained", zero=False)
    base = _logits(_tiny_qwen())
    backbone = Qwen2_5VLIntegration(Qwen2_5VLConfig(device="cpu"))
    backbone.model, backbone._model_loaded = _tiny_qwen(), True
    with backbone.adapters_disabled():  # no PEFT wrapper yet: nothing to disable, runs unchanged
        assert torch.equal(_logits(backbone.model), base)

    runtime = _preflight_runtime(server, adapter=str(adapter), model=SimpleNamespace(qwen2_5_vl=backbone))
    contract = server.HeatmapVLNRuntime._preflight_system2_lora(runtime)
    runtime.system2_lora = contract
    assert contract["lora_b_all_zero"] is False
    adapted = _logits(backbone.model)
    assert not torch.equal(adapted, base)  # the adapter really decides
    with runtime._system1_condition_scope():
        assert torch.equal(_logits(backbone.model), base)  # Z0's weights are the released ones
    assert torch.equal(_logits(backbone.model), adapted)  # and the adapter is back on afterwards


def _rewrite_config(adapter: Path, **changes) -> None:
    path = adapter / "adapter_config.json"
    config = json.loads(path.read_text())
    config.update(changes)
    path.write_text(json.dumps(config))


def _rewrite_weights(adapter: Path, edit) -> None:
    from safetensors.torch import load_file, save_file

    path = adapter / "adapter_model.safetensors"
    save_file(edit(dict(load_file(str(path)))), str(path))


def _break_vision(adapter):
    shutil.rmtree(adapter)
    _save_adapter(adapter, target_modules=SUFFIXES)


def _break_merger(adapter):
    shutil.rmtree(adapter)
    _save_adapter(adapter, target_modules=r"^model\.layers\.0\.self_attn\.q_proj$|^visual\.merger\.mlp\.0$")


def _first_b(tensors):
    return next(key for key in sorted(tensors) if key.endswith("lora_B.weight"))


def _nan(tensors):
    key = _first_b(tensors)
    tensors[key] = tensors[key].clone()
    tensors[key][0, 0] = float("nan")
    return tensors


BAD_ADAPTERS = [
    ("vision_tower", _break_vision, "outside the language model's transformer blocks"),
    ("one_in_the_merger", _break_merger, "outside the language model's transformer blocks"),
    ("no_weights", lambda a: (a / "adapter_model.safetensors").unlink(), "has no adapter_model.safetensors"),
    ("no_config", lambda a: (a / "adapter_config.json").unlink(), "has no adapter_config.json"),
    ("empty", lambda a: _rewrite_weights(a, lambda t: {}), "holds no tensors"),
    (
        "keys_do_not_land",
        lambda a: _rewrite_weights(a, lambda t: {k.replace(".layers.", ".blocks."): v for k, v in t.items()}),
        "does not match this model key for key",
    ),
    (
        "half_the_keys",
        lambda a: _rewrite_weights(a, lambda t: {k: v for k, v in t.items() if ".layers.0." in k}),
        "does not match this model key for key",
    ),
    ("non_finite", lambda a: _rewrite_weights(a, _nan), "non-finite"),
    (
        "foreign_tensor",
        lambda a: _rewrite_weights(a, lambda t: {**t, "base_model.model.lm_head.weight": torch.zeros(64, 32)}),
        "not LoRA A/B weights",
    ),
    ("not_lora", lambda a: _rewrite_config(a, peft_type="IA3"), "peft_type='IA3'"),
    ("modules_to_save", lambda a: _rewrite_config(a, modules_to_save=["lm_head"]), "modules_to_save"),
    ("dora", lambda a: _rewrite_config(a, use_dora=True), "use_dora"),
    ("bias", lambda a: _rewrite_config(a, bias="all"), "bias='all'"),
]


@pytest.mark.parametrize(("name", "breaks", "phrase"), BAD_ADAPTERS, ids=[case[0] for case in BAD_ADAPTERS])
def test_bad_adapters_are_refused(model_server, tmp_path, name, breaks, phrase):
    server, _ = model_server
    adapter = _save_adapter(tmp_path / name)
    breaks(adapter)
    with pytest.raises(RuntimeError, match=re.escape(phrase)):
        server._wrap_system2_lora(_tiny_qwen(), str(adapter))


def test_an_already_wrapped_backbone_is_not_stacked(model_server, tmp_path):
    server, _ = model_server
    adapter = _save_adapter(tmp_path / "zero")
    wrapped, _ = server._wrap_system2_lora(_tiny_qwen(), str(adapter))
    with pytest.raises(RuntimeError, match="already carries a PEFT adapter"):
        server._wrap_system2_lora(wrapped, str(adapter))


# --------------------------------------------------------------------------- #
# The launchers
# --------------------------------------------------------------------------- #
EXTRA_LINE = '[[ -n "$SYSTEM2_LORA" ]] && MODEL_EXTRA+=(--system2_lora_adapter "$SYSTEM2_LORA")'


def test_all_three_launchers_pass_the_adapter_and_check_the_evidence():
    for launcher in (NPU_LAUNCHER, CUDA_LAUNCHER):
        text = launcher.read_text(encoding="utf-8")
        assert 'SYSTEM2_LORA="${PPA_EVAL_SYSTEM2_LORA:-}"' in text
        assert EXTRA_LINE in text  # the same line on both platforms (test_ascend_npu_port compares them)
        assert 'grep -F "System2 LoRA adapter enabled: "' in text
    nav = NAV_LAUNCHER.read_text(encoding="utf-8")
    assert 'SYSTEM2_LORA="${NAV_SYSTEM2_LORA:-${PPA_EVAL_SYSTEM2_LORA:-}}"' in nav
    assert "MODEL_EXTRA+=(--system2_lora_adapter" in nav and '"${MODEL_EXTRA[@]}"' in nav
    assert 'grep -F "System2 LoRA adapter enabled: "' in nav
    server = (REPO / "scripts" / "evaluation" / "rpc_model_server.py").read_text(encoding="utf-8")
    assert '"--system2_lora_adapter"' in server and '"System2 LoRA adapter enabled: %s"' in server


def _servers_json(path: Path, sha: str) -> dict:
    """servers.json from the NPU launcher's own writer, with the LoRA argument it appends."""
    argv = [str(path), "2", "52400", "52500", "0,1", "2,3", guards.COMMIT, str(guards.VO_SEED), "0", "0",
            guards.INSTANCE, guards.MODEL_VERSION, "0", "", "", sha]
    result = subprocess.run([sys.executable, "-", *argv], input=guards._servers_json_writer_source(),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(path.read_text(encoding="utf-8"))


def _external_check(path: Path, sha: str | None):
    argv = [str(path), "2", "52400", "52500", "0", "0", "", "", guards.COMMIT, "0"]
    if sha is not None:
        argv.append(sha)
    return subprocess.run([sys.executable, "-", *argv], input=guards._external_check_source(),
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize(
    ("recorded", "expected", "ok"),
    [
        ("", "", True),
        (SHA, SHA, True),
        (SHA, "", False),  # a base-arm run must not be served by an adapter
        ("", SHA, False),  # an adapter run must not be served by the released System2
        (SHA, "cd" * 32, False),
    ],
    ids=["base", "same_adapter", "unasked_adapter", "missing_adapter", "other_adapter"],
)
def test_servers_json_names_the_adapter_and_the_client_checks_it(tmp_path, recorded, expected, ok):
    path = tmp_path / "servers.json"
    assert _servers_json(path, recorded)["system2_lora_sha256"] == recorded
    result = _external_check(path, expected)
    assert (result.returncode == 0) is ok, result.stderr
    if not ok:
        assert "system2_lora_sha256=" in result.stderr and "PPA_EVAL_SYSTEM2_LORA" in result.stderr


def test_a_record_from_before_the_field_reads_as_no_adapter(tmp_path):
    path = tmp_path / "servers.json"
    guards._write_servers_json(path)  # the 15-argument call of every existing record
    assert json.loads(path.read_text())["system2_lora_sha256"] == ""
    record = json.loads(path.read_text())
    record.pop("system2_lora_sha256")
    path.write_text(json.dumps(record))
    assert _external_check(path, None).returncode == 0
    assert _external_check(path, "").returncode == 0
    assert _external_check(path, SHA).returncode == 1


# A local model server stub that, like the real one, logs the adapter it was given.
STUB_LORA_MODEL_SERVER = r'''
import hashlib
import json
import os
import sys
import time
from pathlib import Path

print("Formal PPA online AMB3R runtime enabled", flush=True)
if "--ppa_bridge_off" in sys.argv:
    print("PPA bridge off (EXP-20 A1)", flush=True)
if "--system2_lora_adapter" in sys.argv:
    adapter = Path(sys.argv[sys.argv.index("--system2_lora_adapter") + 1])
    sha = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    print("System2 LoRA adapter enabled: " + json.dumps({"adapter_sha256": sha}, sort_keys=True), flush=True)
Path(os.environ["FAKE_READY_DIR"], "model").touch()
time.sleep(300)
'''


def _fake_adapter(directory: Path) -> str:
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    (directory / "adapter_model.safetensors").write_bytes(b"not really weights")
    return hashlib.sha256(b"not really weights").hexdigest()


def _lora_formats(sha):
    return ["ppa-online-amb3r-v1", "system2-lora-v1", f"system2-lora-sha256:{sha}"]


@pytest.mark.parametrize(
    ("served", "ok"),
    [("same", True), ("other", False), ("none", False)],
)
def test_local_servers_get_the_adapter_and_must_name_it(tree, served, ok):  # noqa: F811
    sha = _fake_adapter(tree.tmp / "adapter")
    (tree.root / "HeatmapVLN" / "scripts" / "evaluation" / "rpc_model_server.py").write_text(STUB_LORA_MODEL_SERVER)
    formats = {"same": _lora_formats(sha), "other": _lora_formats("cd" * 32), "none": ["ppa-online-amb3r-v1"]}
    tree.set_servers(model=formats[served], vo=["json+jpeg"], wait_for=[tree.ready / "model", tree.ready / "vo"])
    result = tree.run(PPA_EVAL_SYSTEM2_LORA=tree.tmp / "adapter", **guards.LOCAL_CANARY)
    assert f"system2_lora={tree.tmp / 'adapter'} sha256={sha}" in result.stdout
    if ok:
        assert result.returncode == 0, result.stderr
        assert "[ppa-eval] COMPLETE" in result.stdout
    else:
        assert result.returncode == 2
        assert "are not the ones in this run" in result.stderr and "System2 LoRA" in result.stderr
        assert tree.client_calls() == []


def test_a_base_run_refuses_local_servers_that_serve_an_adapter(tree):  # noqa: F811
    tree.set_servers(model=_lora_formats(SHA), vo=["json+jpeg"], wait_for=[tree.ready / "model", tree.ready / "vo"])
    result = tree.run(**guards.LOCAL_CANARY)
    assert result.returncode == 2
    assert "are not the ones in" in result.stderr
    assert tree.client_calls() == []


def _external_lora(tree, sha):
    server_dir = tree.external()
    path = server_dir / "servers.json"
    record = json.loads(path.read_text())
    record["system2_lora_sha256"] = sha
    path.write_text(json.dumps(record))
    tree.set_servers(model=[*guards._model_formats(), "system2-lora-v1", f"system2-lora-sha256:{sha}"],
                     vo=guards._vo_formats())
    with (server_dir / "logs" / "model_0.log").open("a") as log:
        log.write("System2 LoRA adapter enabled: " + json.dumps({"adapter_sha256": sha}) + "\n")
    return server_dir


def test_external_servers_with_the_named_adapter_complete(tree):  # noqa: F811
    sha = _fake_adapter(tree.tmp / "adapter_copy")
    server_dir = _external_lora(tree, sha)
    by_sha = tree.run(PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir, PPA_EVAL_SYSTEM2_LORA=sha, **guards.EXTERNAL_CANARY)
    assert by_sha.returncode == 0, by_sha.stderr
    assert "[ppa-eval] COMPLETE" in by_sha.stdout
    # A local copy of the adapter directory names it as well: the launcher hashes it.
    by_copy = tree.run(PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir, PPA_EVAL_SYSTEM2_LORA=tree.tmp / "adapter_copy",
                       PPA_EVAL_OUTPUT_ROOT=tree.tmp / "out_copy", **guards.EXTERNAL_CANARY)
    assert by_copy.returncode == 0, by_copy.stderr
    assert f"sha256={sha}" in by_copy.stdout


def test_external_servers_with_an_adapter_the_run_did_not_ask_for_are_refused(tree):  # noqa: F811
    server_dir = _external_lora(tree, SHA)
    result = tree.run(PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir, **guards.EXTERNAL_CANARY)
    assert result.returncode == 2
    assert "external servers.json does not match this run" in result.stderr
    assert "system2_lora_sha256" in result.stderr
    assert tree.client_calls() == []

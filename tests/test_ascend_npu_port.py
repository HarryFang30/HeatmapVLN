"""The Ascend NPU port: the device must be explicit, and never silently CPU.

The whole risk of this port is a change that does not crash.  A device that falls
back to CPU, a synchronisation that is skipped, a dtype that drops to fp16 or a
chunked attention path that is silently not taken all leave a server that answers
every request with different numbers.  These tests pin the places where that could
happen; nothing here needs an NPU.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import shlex
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODEL_SERVER = REPO / "scripts" / "evaluation" / "rpc_model_server.py"
VO_SERVER = REPO / "scripts" / "amb3r_vo" / "rpc_amb3r_vo_server.py"
NPU_LAUNCHER = REPO / "scripts" / "ascend" / "run_ppa_servers_npu.sh"
CUDA_LAUNCHER = REPO / "scripts" / "run_ppa_r2r_val_unseen_cuda.sh"
AMB3R_PATCH = REPO / "scripts" / "ascend" / "amb3r_npu.patch"


def _function_source(path: Path, name: str, *, inside: str | None = None) -> str:
    """Source of a function, or of a method of the class named by ``inside``."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    scopes = [tree]
    if inside is not None:
        scopes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == inside
        ]
        assert scopes, f"class {inside} not found in {path}"
    for scope in scopes:
        for node in ast.walk(scope):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(text, node)
    raise AssertionError(f"{name} not found in {path}")


def _code_only(path: Path) -> str:
    """The file without comment text, so a prose mention cannot stand in for code."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        lines.append(line.split("  # ", 1)[0])
    return "\n".join(lines)


def _load_resolve_device():
    """``_resolve_device`` alone, so the test needs neither torch nor the model stack.

    The npu branch ends in two module-level helpers that need a real NPU.  They are
    recording stubs here, so a test sees that the npu branch reached them and that
    the certified cuda path and the cpu path did not.  Returns the function, the fake
    torch and the list of recorded calls.
    """
    text = MODEL_SERVER.read_text(encoding="utf-8")
    module = ast.parse(text)
    functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
    # A stub stands in for a name, so a rename or a new argument in the server would
    # leave these tests green while the real call fails.  Pin both to the call shape
    # the stubs below accept.
    for name, arity in (("_disable_fused_mha_fastpath", 0), ("_warm_up_npu", 1)):
        helper = functions.get(name)
        assert helper is not None, f"{name} is no longer a module-level function of {MODEL_SERVER}"
        arguments = helper.args
        assert len(arguments.posonlyargs + arguments.args) == arity, name
        assert arguments.vararg is None and not arguments.kwonlyargs, name
    node = functions["_resolve_device"]
    # Padded to the function's own first line, so a traceback names the real line of
    # rpc_model_server.py instead of a line counted from the start of the excerpt.
    source = "\n" * (node.lineno - 1) + ast.get_source_segment(text, node)
    calls = []
    torch = types.SimpleNamespace(
        device=lambda spec: spec,
        cuda=types.SimpleNamespace(is_available=lambda: False),
        npu=types.SimpleNamespace(is_available=lambda: False, set_device=lambda index: None),
        __version__="0.0",
    )
    namespace = {
        "torch": torch,
        "argparse": argparse,
        "LOGGER": types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
        "_disable_fused_mha_fastpath": lambda: calls.append("fastpath"),
        "_warm_up_npu": lambda device: calls.append(("warm_up", device)),
    }
    exec(compile(source, str(MODEL_SERVER), "exec"), namespace)
    return namespace["_resolve_device"], torch, calls


def test_model_server_device_never_falls_back_to_cpu():
    resolve, torch, calls = _load_resolve_device()
    args = argparse.Namespace(device="cuda", gpu_id=0)
    with pytest.raises(RuntimeError, match="refusing to fall back to CPU"):
        resolve(args)

    torch.cuda.is_available = lambda: True
    assert resolve(args) == "cuda:0"
    # The CUDA path is certified: it must not pick up the NPU-only setup.
    assert calls == []


def test_model_server_npu_requires_a_real_npu(monkeypatch):
    resolve, torch, calls = _load_resolve_device()
    args = argparse.Namespace(device="npu", gpu_id=3)

    monkeypatch.setitem(sys.modules, "torch_npu", None)
    with pytest.raises(RuntimeError, match="torch_npu cannot be imported"):
        resolve(args)
    assert calls == []

    monkeypatch.setitem(sys.modules, "torch_npu", types.SimpleNamespace(__version__="2.7.1"))
    with pytest.raises(RuntimeError, match=r"torch\.npu\.is_available\(\) is False"):
        resolve(args)
    assert calls == []

    chosen = []
    torch.npu.is_available = lambda: True
    torch.npu.is_bf16_supported = lambda: True
    torch.npu.set_device = chosen.append
    assert resolve(args) == "npu:3"
    assert chosen == [3]
    # The fake torch.device returns its spec, so the warm-up records the string.
    assert calls == ["fastpath", ("warm_up", "npu:3")]


def test_model_server_cpu_is_only_ever_explicit():
    resolve, _, calls = _load_resolve_device()
    assert resolve(argparse.Namespace(device="cpu", gpu_id=0)) == "cpu"
    with pytest.raises(ValueError, match="unsupported --device"):
        resolve(argparse.Namespace(device="mps", gpu_id=0))
    assert calls == []


def test_model_server_exposes_a_device_flag_defaulting_to_cuda():
    source = MODEL_SERVER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    flags = {
        call.args[0].value: {
            keyword.arg: keyword.value
            for keyword in call.keywords
        }
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_argument"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    }
    assert "--device" in flags, "the accelerator type must be nameable, not inferred"
    default = flags["--device"].get("default")
    # The unchanged CUDA launcher passes no --device, so the default must stay cuda.
    assert isinstance(default, ast.Constant) and default.value == "cuda"
    choices = flags["--device"].get("choices")
    assert choices is not None
    assert {element.value for element in choices.elts} == {"cuda", "npu", "cpu"}


def test_no_unconditional_cuda_calls_remain_in_the_servers():
    """Any ``torch.cuda.X`` left on these paths would raise on a non-CUDA torch."""
    for path in (MODEL_SERVER, VO_SERVER):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "cuda"
                and isinstance(value.value, ast.Name)
                and value.value.id == "torch"
            ):
                offenders.append(f"{path.name}:{node.lineno} torch.cuda.{node.attr}")
        # is_available is the one allowed use: it is how --device cuda is validated.
        assert [item for item in offenders if not item.endswith("is_available")] == []


def test_vo_server_constructs_da3_with_the_requested_device():
    """load_model("da3") takes no device and would place the model on cuda regardless."""
    source = _code_only(VO_SERVER)
    assert 'load_model("da3"' not in source
    assert "from amb3r.model_zoo import DA3" in source
    assert re.search(r"DA3\(\s*device=args\.device", source)


def test_vo_server_brings_up_and_checks_its_device():
    source = _function_source(VO_SERVER, "_prepare_accelerator")
    assert "torch_npu" in source
    assert "is_available()" in source
    assert "device_count()" in source
    assert "set_device" in source
    # The chunked-SDPA setting must be visible in the log, since taking the
    # unchunked path is a ~20 GiB difference and not otherwise observable.
    assert "DA3_SDPA_QUERY_CHUNK_SIZE" in source
    body = VO_SERVER.read_text(encoding="utf-8")
    assert "_prepare_accelerator(args.device)" in body


def test_vo_server_can_seed_the_device_rng_per_episode():
    source = VO_SERVER.read_text(encoding="utf-8")
    assert '"--rng-seed"' in source
    assert "rng_seed=args.rng_seed" in source

    online = (REPO / "src" / "vo" / "online_amb3r.py").read_text(encoding="utf-8")
    assert "def _seed_episode" in online
    # Seeding belongs in the backend's own reset, which runs at every episode.
    reset = _function_source(
        REPO / "src" / "vo" / "online_amb3r.py", "reset", inside="StatefulAMB3RBackend"
    )
    assert "_seed_episode()" in reset
    assert "rng_seed=rng_seed" in online


def test_vo_backend_hands_amb3r_a_config_that_already_names_the_device():
    """AMB3R_VO.__init__ moves the model before any later cfg.device override."""
    online = REPO / "src" / "vo" / "online_amb3r.py"
    source = online.read_text(encoding="utf-8")
    assert "AMB3R_VO(model, cfg_path=self._config_for_device(" in source
    helper = _function_source(online, "_config_for_device")
    assert "OmegaConf.load" in helper and "OmegaConf.save" in helper
    # The released config belongs to the AMB3R checkout and must stay untouched.
    assert "mkdtemp" in helper


def test_config_for_device_copies_only_the_device_key(tmp_path):
    omegaconf = pytest.importorskip("omegaconf")
    online = REPO / "src" / "vo" / "online_amb3r.py"
    source = _function_source(online, "_config_for_device")
    namespace: dict = {"Path": Path}
    exec(compile(f"import tempfile\n{source}", str(online), "exec"), namespace)
    config_for_device = namespace["_config_for_device"]

    original = tmp_path / "slam_config.yaml"
    original.write_text("device: 'cuda:0'\nmap_every: 8\nblend: true\n", encoding="utf-8")
    before = original.read_text(encoding="utf-8")

    same = config_for_device(original, "cuda:0")
    assert Path(same) == original, "an already-matching config is used as it is"

    copy = Path(config_for_device(original, "npu:0"))
    assert copy != original
    assert original.read_text(encoding="utf-8") == before
    written = omegaconf.OmegaConf.load(str(copy))
    assert written.device == "npu:0"
    assert written.map_every == 8 and written.blend is True


def test_amb3r_patch_fixes_both_silent_precision_sites():
    """The AMB3R tree is third-party, so its two NPU fixes ship as a tracked patch."""
    patch = AMB3R_PATCH.read_text(encoding="utf-8")
    assert "slam/pipeline.py" in patch and "thirdparty/depth_anything_3/api.py" in patch
    # Removed: a cuda-named autocast (off CUDA torch only warns and runs fp32) and a
    # cuda-only bf16 probe (off CUDA it is False, so the forward drops to fp16).
    assert "-        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):" in patch
    assert "-        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16" in patch
    assert "+        autocast_dtype = torch.bfloat16" in patch
    assert "refusing to run in fp16 silently" in patch


def test_npu_launcher_runs_servers_only_and_pins_the_reference_path():
    script = NPU_LAUNCHER.read_text(encoding="utf-8")
    code = _code_only(NPU_LAUNCHER)
    # Servers only: no Habitat, no Xvfb, no dataset, no clients, no merge.
    for absent in ("Xvfb", "r2r_val_unseen.py", "merge_shards", "SCENES_DIR", "--resume"):
        assert absent not in code, f"the NPU half must not run {absent}"
    # Devices: Ascend ignores CUDA_VISIBLE_DEVICES, so using it would put every slot
    # on card 0 while appearing to spread them.
    assert "ASCEND_RT_VISIBLE_DEVICES=" in script
    assert "CUDA_VISIBLE_DEVICES=\"${GPUS" not in script
    assert "--device npu --gpu_id 0" in script
    assert "--device npu:0" in script
    # The certified DA3 path and the offline weights.
    assert "DA3_DISABLE_XFORMERS=1" in script
    assert "DA3_SDPA_QUERY_CHUNK_SIZE=256" in script
    assert "HF_HUB_OFFLINE=1" in script
    # A non-interactive shell does not necessarily load CANN.
    assert 'source "$ASCEND_ENV"' in script
    assert "127.0.0.1" in script and "0.0.0.0" not in script, "gRPC has no auth: bind loopback only"
    # Evidence the client side greps for, and the device lines only this platform
    # can get wrong.
    for evidence in (
        "Formal PPA online AMB3R runtime enabled",
        "Model server device: npu:0",
        "VO server device: npu:0",
    ):
        assert f'grep -F "{evidence}"' in script
    # The chunked-attention evidence must come from DA3, not from this script's own
    # export.  Grepping "DA3_SDPA_QUERY_CHUNK_SIZE=256" -- which the VO server echoed
    # back out of the environment the launcher had just set -- could not fail whatever
    # DA3 did with the variable, and the deploy doc quoted it as proof the certified
    # chunked path was taken.  tests/test_ascend_amb3r_patch_check.py covers the
    # replacement in detail.
    assert 'grep -F "DA3_SDPA_QUERY_CHUNK_SIZE=256"' not in script
    assert (
        'grep -F "DA3 attention: query_chunk=$DA3_SDPA_QUERY_CHUNK_SIZE (parsed by DA3), '
        'memory_bounded=True"'
    ) in script


def test_npu_launcher_takes_only_free_cards_and_cleans_up():
    script = NPU_LAUNCHER.read_text(encoding="utf-8")
    assert "npu-smi" in script
    assert "PPA_NPU_MAX_USED_MIB" in script
    assert "already has" in script  # the refusal message for a busy card
    assert "trap cleanup EXIT" in script
    assert "ASCEND_RT_VISIBLE_DEVICES is already set" in script


def _server_flags(path: Path, server: str) -> dict[str, str]:
    """Flags and VALUES of the `python -u "$SERVER"` invocation, up to the `&`.

    Values, not just names: comparing names alone let the two halves drift in any
    argument that both pass -- a different --map-every or --resolution on one
    platform would have gone unnoticed, and the arms would no longer be comparable.
    """
    code = _code_only(path)
    marker = f'-u "{server}"'
    assert marker in code, f"{server} is never invoked in {path.name}"
    block = code.split(marker, 1)[1].split("&\n", 1)[0]
    block = block.replace("\\\n", " ")
    tokens = shlex.split(block, comments=True)
    flags: dict[str, str] = {}
    current = None
    for token in tokens:
        if token.startswith("--"):
            current = token
            flags[current] = ""
        elif re.fullmatch(r"\$\{[A-Z_]+\[@\]\}", token):
            # An array expansion carries its own flags at runtime, so it is a marker in
            # its own right rather than the value of whatever flag precedes it.
            flags[token] = ""
            current = None
        elif current is not None:
            flags[current] = f"{flags[current]} {token}".strip()
    return flags


# Flags whose value is legitimately platform-specific: the device itself, the port
# expression, the VO seed CUDA got for free, and the instance token only the NPU
# launcher issues.  Everything else must match on both sides, value included.
PLATFORM_ONLY_FLAGS = {"--device", "--port", "--rng-seed", "--server_instance", "--server-instance"}
# The NPU profiler is an engineering instrument for latency work, passed only when
# PPA_NPU_PROFILE_DIR is set, and it exists only on this platform.  It is listed here
# rather than silently tolerated so that adding any other NPU-only server flag has to
# come past this line.
NPU_ONLY_MARKERS = {"${MODEL_PROFILE[@]}"}


def test_npu_launcher_and_cuda_launcher_pass_the_same_server_flags():
    """The two halves must differ only in the device, or the arms are not comparable."""
    for server, platform_only in (
        ("$MODEL_SERVER", {"--device", "--port", "--server_instance"}),
        ("$VO_SERVER", {"--device", "--port", "--rng-seed", "--server-instance"}),
    ):
        npu = _server_flags(NPU_LAUNCHER, server)
        cuda = _server_flags(CUDA_LAUNCHER, server)
        only_on_one = (set(npu) ^ set(cuda)) - NPU_ONLY_MARKERS
        assert only_on_one == platform_only - (set(npu) & set(cuda)), server
        # Both halves must still expand the same EXP-20 / EXP-21 switch array.
        assert "${MODEL_EXTRA[@]}" in npu and "${MODEL_EXTRA[@]}" in cuda or server == "$VO_SERVER"
        shared = (set(npu) & set(cuda)) - PLATFORM_ONLY_FLAGS
        differing = {flag: (npu[flag], cuda[flag]) for flag in shared if npu[flag] != cuda[flag]}
        assert differing == {}, f"{server} is passed different values: {differing}"

    # The EXP-20 / EXP-21 switches are built outside the invocation, so compare the
    # lines that build them too: an override that reached only one platform would
    # make the two halves different arms under the same name.
    def model_extra(path: Path) -> list[str]:
        # The lines that BUILD the array, not the invocation that expands it: the two
        # launchers may format the invocation differently, but an override that reached
        # only one platform would make them different arms under the same name.
        return [
            " ".join(line.split())
            for line in _code_only(path).splitlines()
            if ("MODEL_EXTRA=" in line or "MODEL_EXTRA+=" in line) and "declare" not in line
        ]

    assert model_extra(NPU_LAUNCHER) == model_extra(CUDA_LAUNCHER)


def test_npu_launcher_publishes_what_the_client_needs():
    script = NPU_LAUNCHER.read_text(encoding="utf-8")
    assert "servers.json" in script
    # v2 adds what the client needs in order to tell WHICH servers answered: the
    # ports cannot, since 52400+k / 52500+k are also the 4090's own defaults.
    assert "heatmapvln-npu-servers-v2" in script
    assert "heatmapvln-npu-servers-v1" not in script
    for key in (
        "model_ports",
        "vo_ports",
        "repo_commit",
        "repo_dirty",
        "vo_rng_seed",
        "bridge_off",
        "server_instance",
        "model_version",
        "num_sample_trajs",
        "num_inference_steps",
    ):
        assert key in script
    # The commit must be read, not defaulted: "unknown" in servers.json would make
    # the client's commit check vacuous.
    assert "|| echo unknown" not in script
    assert 'rev-parse HEAD' in script


def test_client_external_mode_refuses_to_reuse_a_local_runs_directory():
    """Resuming into a CUDA-server directory would mix two platforms in one result."""
    script = CUDA_LAUNCHER.read_text(encoding="utf-8")
    assert 'EXTERNAL_SERVERS="${PPA_EVAL_EXTERNAL_SERVERS:-0}"' in script
    assert "requires an explicit PPA_EVAL_OUTPUT_ROOT" in script
    assert "requires PPA_EVAL_EXTERNAL_SERVER_DIR" in script
    # External mode starts no servers of its own ...
    assert 'if [[ "$EXTERNAL_SERVERS" -eq 1 ]]; then' in script
    # ... and checks the tunnel instead of a local pid.
    assert "is the SSH tunnel up?" in script
    assert "stopped listening (remote server or tunnel down)" in script
    # The preflight evidence comes from the servers' own archived log.
    assert "external_server_logs" in script
    assert 'require_file "$model_log"' in script


def test_client_external_mode_checks_the_servers_match_the_run():
    script = CUDA_LAUNCHER.read_text(encoding="utf-8")
    assert "external servers.json does not match this run" in script
    assert "heatmapvln-npu-servers-v2" in script
    assert "servers bridge_off=" in script
    # tests/test_external_server_guards.py runs these checks; here we only pin that
    # the launcher still asks for each of them, since ports alone cannot identify a
    # server set that shares the 4090's own default ports.
    for field in ("server_instance", "repo_commit", "repo_dirty", "timing", "device"):
        assert field in script


def test_cuda_launcher_still_defaults_to_local_cuda_servers():
    """The 4090 runs must be unaffected: same command, same local servers."""
    script = CUDA_LAUNCHER.read_text(encoding="utf-8")
    assert "--device cuda:0" in script  # the VO server flag is unchanged
    assert "--gpu_id 0 --host 127.0.0.1" in script  # the model server gets no --device
    assert 'CUDA_VISIBLE_DEVICES="${GPUS[$slot]}"' in script
    assert "PPA_EVAL_EXTERNAL_SERVERS:-0" in script  # off unless asked for


def test_npu_launcher_keeps_tmpdir_short_enough_for_an_af_unix_socket():
    """CANN's kernel bank opens an AF_UNIX socket under TMPDIR, and 108 bytes is the limit."""
    script = NPU_LAUNCHER.read_text(encoding="utf-8")
    # Not under the timestamped runtime directory: that path alone crossed the limit,
    # and the failure surfaced as an unrelated-looking ACL/GEInitialize error.
    assert 'TMPDIR="$runtime/model/tmp"' not in script
    assert 'TMPDIR="$model_tmp"' in script and 'TMPDIR="$vo_tmp"' in script
    assert "AF_UNIX_MAX=108" in script
    assert "too long for an AF_UNIX socket" in script


def test_both_servers_leave_the_fused_mha_fastpath_off_on_npu():
    """The fused encoder-layer kernel has no NPU implementation and runs on the CPU."""
    model = _code_only(MODEL_SERVER)
    vo = _code_only(VO_SERVER)
    assert "set_fastpath_enabled(False)" in model
    assert "set_fastpath_enabled(False)" in vo
    # Logged, because the fused and ordinary paths differ in summation order: this is
    # a platform choice, not a free optimisation.
    assert "Fused MHA fastpath disabled" in model
    assert "Fused MHA fastpath disabled" in vo
    # NPU only; the certified CUDA path keeps whatever torch picks there.
    helper = _function_source(MODEL_SERVER, "_resolve_device")
    assert "_disable_fused_mha_fastpath()" in helper
    assert helper.index("kind == \"npu\"") < helper.index("_disable_fused_mha_fastpath()")


def test_shard_restarts_are_opt_in_and_resume():
    """One failed RPC call raises and takes the whole shard; --resume bounds the loss."""
    script = CUDA_LAUNCHER.read_text(encoding="utf-8")
    assert 'SHARD_RETRIES="${PPA_EVAL_SHARD_RETRIES:-0}"' in script, "off by default: the certified runs used no retries"
    assert "run_shard_once" in script and "run_shard() {" in script
    assert "--resume" in script
    # A restart must append to the client log, not truncate the evidence of the death.
    assert '>>"$RUNTIME_DIR/logs/client_shard_0${shard}.log"' in script
    # In external mode a restart waits for a server that answers a real RPC, not for
    # an open port: an Ascend device error leaves the process alive and LISTENing
    # (observed: a VO server served HealthCheck, GetServerInfo and 19 ingests for an
    # hour after its device died), and the thing that accepts the TCP connection is
    # the local ssh forwarder in any case.  The old wait returned immediately and the
    # restarted client burned every remaining retry on a server that could not compute.
    assert "tcp_open $((MODEL_PORT_BASE + slot)) && tcp_open $((VO_PORT_BASE + slot)) && break" not in script
    assert "until rpc_ready " in script
    # Exit 3 means "answered, but not the recorded server set", which no amount of
    # waiting fixes, so the shard is given up instead of retried.
    assert "probe == 3" in script
    assert "giving up the shard" in script

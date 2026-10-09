"""The AMB3R NPU patch gate, and the DA3 attention evidence the launcher greps.

AMB3R is third party and is not vendored here.  Its two NPU fixes ship as
scripts/ascend/amb3r_npu.patch and are applied by hand on each instance, and both
fail silently when they are missing: a cuda-named autocast runs the mapping forward
in fp32 off CUDA (torch only warns), and torch.cuda.is_bf16_supported() being False
off CUDA drops DA3 to fp16.  scripts/ascend/check_amb3r_patch.sh is the only thing
that tells a patched tree from a clean clone, so these tests run it against trees
built from the patch's own hunks -- the fixtures cannot drift from the patch.

The second half pins the chunked-attention evidence.  The launcher used to grep its
own export echoed back by the VO server, a check that could not fail.  It now greps
a line the VO server builds from DA3's own parser and the attention function the
dinov2 layers bound.  That function needs DA3 and torch, neither of which is here,
so it is executed from its source against fake DA3 modules, and the launcher's
grep block is executed against the logs it produces.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECK = REPO / "scripts" / "ascend" / "check_amb3r_patch.sh"
PATCH = REPO / "scripts" / "ascend" / "amb3r_npu.patch"
NPU_LAUNCHER = REPO / "scripts" / "ascend" / "run_ppa_servers_npu.sh"
VO_SERVER = REPO / "scripts" / "amb3r_vo" / "rpc_amb3r_vo_server.py"

PIPELINE = "slam/pipeline.py"
API = "thirdparty/depth_anything_3/api.py"

BASH = shutil.which("bash")
GIT = shutil.which("git")
if BASH is None:
    pytest.skip("bash is required to run the shell under test", allow_module_level=True)




def _patch_images() -> dict[str, tuple[str, str]]:
    """{path: (pre-image, post-image)} for every file in the patch, from its hunks.

    The pre-image is the context plus the '-' lines and the post-image the context
    plus the '+' lines, so applying the patch to the pre-image gives the post-image.
    Only the hunks are reproduced, not the whole upstream file, which is all the
    check reads.
    """
    images: dict[str, tuple[list[str], list[str]]] = {}
    current: str | None = None
    in_hunk = False
    for line in PATCH.read_text(encoding="utf-8").splitlines():
        if line.startswith("diff --git "):
            current, in_hunk = None, False
        elif line.startswith("+++ "):
            current = line[4:].removeprefix("b/")
            images[current] = ([], [])
        elif line.startswith("@@") and current is not None:
            in_hunk = True
        elif in_hunk and current is not None:
            before, after = images[current]
            if line.startswith("\\"):
                continue
            tag, body = (line[:1], line[1:]) if line else (" ", "")
            if tag in (" ", "-"):
                before.append(body)
            if tag in (" ", "+"):
                after.append(body)
    return {path: ("\n".join(b) + "\n", "\n".join(a) + "\n") for path, (b, a) in images.items()}


IMAGES = _patch_images()


def _write_tree(root: Path, *, patched: dict[str, bool]) -> Path:
    for relative, is_patched in patched.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before, after = IMAGES[relative]
        target.write_text(after if is_patched else before, encoding="utf-8")
    return root


def _git(tree: Path, *args: str) -> str:
    result = subprocess.run(
        [GIT, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
         "-c", f"safe.directory={tree}", "-C", str(tree), *args],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _git_tree(root: Path, *, patched: bool) -> Path:
    """A git checkout whose HEAD is the unpatched base, with the work tree patched or not."""
    if GIT is None:
        pytest.skip("git is required for the checkout cases")
    _write_tree(root, patched={PIPELINE: False, API: False})
    (root / "README.md").write_text("amb3r fixture\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    if patched:
        _write_tree(root, patched={PIPELINE: True, API: True})
    return root


def _run(tree: Path, *extra: str, cwd: Path | None = None, env: dict[str, str] | None = None,
         script: Path = CHECK) -> subprocess.CompletedProcess[str]:
    full_env = dict(os.environ)
    full_env.pop("PPA_NPU_AMB3R_BASE_TREE", None)
    full_env.update(env or {})
    return subprocess.run(
        [BASH, str(script), str(tree), *extra],
        cwd=str(cwd or REPO), env=full_env, capture_output=True, text=True, timeout=60,
    )


def _tree_digest(root: Path) -> dict[str, tuple[str, int, int]]:
    """Every path under root, .git included, with its content hash, mode and mtime."""
    digest = {}
    for path in sorted(root.rglob("*")):
        stat = path.lstat()
        content = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "dir"
        digest[str(path.relative_to(root))] = (content, stat.st_mode, stat.st_mtime_ns)
    return digest


def test_fixture_images_come_from_both_files_of_the_patch():
    # If the parser silently dropped a file or a hunk, every case below would be
    # testing the check against something other than the patch.
    assert set(IMAGES) == {PIPELINE, API}
    for before, after in IMAGES.values():
        assert before != after




def test_patched_plain_directory_passes_with_a_non_git_warning(tmp_path):
    # A plain copy of the tree is still checkable by its two files.  Refusing it
    # would block a correct deployment for want of a .git directory.
    tree = _write_tree(tmp_path / "amb3r", patched={PIPELINE: True, API: True})
    result = _run(tree)
    assert result.returncode == 0, result.stderr
    assert "carries the NPU patch" in result.stdout
    assert "is not a git checkout" in result.stderr
    assert "ERROR" not in result.stderr


def test_patched_git_checkout_passes(tmp_path):
    # The hunks must reverse-apply on a patched checkout; a spurious drift warning
    # here would teach operators to ignore the one that matters.
    tree = _git_tree(tmp_path / "amb3r", patched=True)
    result = _run(tree)
    assert result.returncode == 0, result.stderr
    assert "carries the NPU patch" in result.stdout
    assert "does not reverse-apply" not in result.stderr
    assert "certified base tree" in result.stderr and "is not in this checkout" in result.stderr


def test_clean_clone_is_refused_for_the_fp32_mapping_forward(tmp_path):
    # The case the check exists for: before it, a clean clone passed every gate.
    tree = _write_tree(tmp_path / "amb3r", patched={PIPELINE: False, API: False})
    result = _run(tree)
    assert result.returncode == 2
    assert "not patched" in result.stderr
    assert PIPELINE in result.stderr and "fp32" in result.stderr
    assert "carries the NPU patch" not in result.stdout


def test_only_api_reverted_is_refused_for_fp16(tmp_path):
    # Each fix is checked on its own: a tree with the pipeline fix and a reverted
    # api.py would otherwise run DA3 in fp16 with no trace.
    tree = _write_tree(tmp_path / "amb3r", patched={PIPELINE: True, API: False})
    result = _run(tree)
    assert result.returncode == 2
    assert API in result.stderr and "fp16" in result.stderr
    assert "fp32" not in result.stderr


@pytest.mark.parametrize("present", [(), (PIPELINE,), (API,)])
def test_missing_files_say_incomplete_not_unpatched(tmp_path, present):
    # "Apply the patch" and "the share is not mounted" are different actions.  An
    # operator told to apply a patch to an empty mount point would get a confusing
    # git error instead of being pointed at PPA_EVAL_AMB3R_ROOT.
    tree = tmp_path / "amb3r"
    tree.mkdir()
    _write_tree(tree, patched={path: True for path in present})
    result = _run(tree)
    assert result.returncode == 2
    assert "tree incomplete" in result.stderr and "not readable" in result.stderr
    assert "not patched" not in result.stderr
    assert "apply" not in result.stderr


def test_missing_tree_is_refused_as_not_mounted(tmp_path):
    result = _run(tmp_path / "nowhere")
    assert result.returncode == 2
    assert "no AMB3R tree" in result.stderr
    assert "not patched" not in result.stderr


def test_missing_patch_file_is_refused(tmp_path):
    tree = _write_tree(tmp_path / "amb3r", patched={PIPELINE: True, API: True})
    result = _run(tree, str(tmp_path / "missing.patch"))
    assert result.returncode == 2
    assert "missing patch" in result.stderr


def test_a_half_applied_tree_is_refused_even_though_it_carries_the_fix(tmp_path):
    # The dangerous shape: someone adds the fixed line by hand and leaves the old one
    # live above it.  Both literals are then present, and the cuda-named one is what
    # runs -- so this must be refused, not merely warned about.  Scoped to these two
    # files, where each literal appears exactly once before the patch and not at all
    # after (checked on the 910B tree), so a correct tree cannot trip it.
    tree = _git_tree(tmp_path / "amb3r", patched=True)
    pipeline = tree / PIPELINE
    pipeline.write_text(
        pipeline.read_text(encoding="utf-8")
        + "\n    def other(self):\n"
        "        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):\n"
        "            pass\n",
        encoding="utf-8",
    )
    result = _run(tree)
    assert result.returncode == 2, result.stderr
    assert "half-applied" in result.stderr
    assert "cuda-named autocast" in result.stderr

    # The same for the dtype probe, on its own.
    tree2 = _git_tree(tmp_path / "amb3r2", patched=True)
    api = tree2 / API
    api.write_text(api.read_text(encoding="utf-8") + "\nX = torch.cuda.is_bf16_supported()\n",
                   encoding="utf-8")
    result2 = _run(tree2)
    assert result2.returncode == 2, result2.stderr
    assert "still probes bf16 through torch.cuda" in result2.stderr


def test_drifted_context_warns_but_passes(tmp_path):
    # Both fixed lines are present but the surrounding upstream code moved: the
    # patch needs regenerating, which is not a reason to stop serving.
    tree = _git_tree(tmp_path / "amb3r", patched=True)
    pipeline = tree / PIPELINE
    text = pipeline.read_text(encoding="utf-8")
    assert "@torch.no_grad()" in text
    pipeline.write_text(text.replace("@torch.no_grad()", "@torch.inference_mode()"), encoding="utf-8")
    result = _run(tree)
    assert result.returncode == 0, result.stderr
    assert "does not reverse-apply cleanly" in result.stderr


@pytest.mark.parametrize("as_git", [False, True], ids=["plain", "git"])
def test_refusal_carries_an_apply_command_that_actually_fixes_the_tree(tmp_path, as_git):
    # The repo documents the apply command nowhere else, so the one in the refusal
    # has to work as printed: run it from an unrelated directory, then the same check
    # must pass.  The tree path has a space in it to prove the quoting holds.
    tree = tmp_path / "amb3r tree"
    if as_git:
        _git_tree(tree, patched=False)
    else:
        if GIT is None:
            pytest.skip("git is required to run the printed apply command")
        _write_tree(tree, patched={PIPELINE: False, API: False})
    refused = _run(tree)
    assert refused.returncode == 2
    match = re.search(r"Apply it with: (git .* apply .*)$", refused.stderr, re.MULTILINE)
    assert match, refused.stderr
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    applied = subprocess.run([BASH, "-c", match.group(1)], cwd=elsewhere, capture_output=True, text=True)
    assert applied.returncode == 0, applied.stderr
    for relative in (PIPELINE, API):
        assert (tree / relative).read_text(encoding="utf-8") == IMAGES[relative][1]
    assert _run(tree).returncode == 0


@pytest.mark.parametrize(
    "state",
    ["patched-git", "unpatched-git", "patched-plain", "unpatched-plain"],
)
def test_check_never_writes_to_the_tree(tmp_path, state):
    # Applying the patch is the operator's decision.  The digest covers .git too:
    # git apply --check and git diff must not touch the index either.
    patched = state.startswith("patched")
    if state.endswith("git"):
        tree = _git_tree(tmp_path / "amb3r", patched=patched)
    else:
        tree = _write_tree(tmp_path / "amb3r", patched={PIPELINE: patched, API: patched})
    before = _tree_digest(tree)
    result = _run(tree)
    assert result.returncode == (0 if patched else 2), result.stderr
    assert _tree_digest(tree) == before


def test_repo_root_resolves_from_any_working_directory(tmp_path):
    # The launcher calls the check by absolute path, and the default patch is found
    # relative to the script.  If that resolution used the caller's cwd, the check
    # would refuse every tree for a "missing patch" from anywhere but the repo root.
    tree = _write_tree(tmp_path / "amb3r", patched={PIPELINE: True, API: True})
    cwd = tmp_path / "somewhere" / "else"
    cwd.mkdir(parents=True)
    result = _run(tree, cwd=cwd)
    assert result.returncode == 0, result.stderr
    assert "missing patch" not in result.stderr
    # And the refusal names the repo's patch by its absolute path, so the printed
    # command works from where the operator stands.
    unpatched = _write_tree(tmp_path / "clean", patched={PIPELINE: False, API: False})
    refused = _run(unpatched, cwd=cwd)
    assert refused.returncode == 2
    assert str(PATCH) in refused.stderr


def test_certified_base_mismatch_outside_the_patched_files_is_refused(tmp_path):
    # Everything outside the two patched files must match the base the certified
    # numbers came from.  The fixture's own base commit stands in for the real one.
    tree = _git_tree(tmp_path / "amb3r", patched=True)
    base_tree = _git(tree, "rev-parse", "HEAD^{tree}")
    env = {"PPA_NPU_AMB3R_BASE_TREE": base_tree}
    matching = _run(tree, env=env)
    assert matching.returncode == 0, matching.stderr
    assert "tree matches the certified base" in matching.stdout
    (tree / "README.md").write_text("local edit\n", encoding="utf-8")
    drifted = _run(tree, env=env)
    assert drifted.returncode == 2
    assert "differs from the certified base" in drifted.stderr




def _launcher_chunk_export() -> str:
    exports = re.findall(r"^export DA3_SDPA_QUERY_CHUNK_SIZE=(\S+)$", NPU_LAUNCHER.read_text(encoding="utf-8"),
                         re.MULTILINE)
    assert len(exports) == 1, exports
    return exports[0]


def _launcher_da3_gate() -> str:
    """The launcher's DA3 evidence grep, cut out verbatim so it can be executed.

    One grep over the whole reported line: the chunk size and memory_bounded=True have
    to appear together, so a "memory_bounded=True" elsewhere in the log cannot stand in
    for the chunk size DA3 reported.
    """
    lines = NPU_LAUNCHER.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if 'grep -F "DA3 attention:' in line)
    end = next(i for i in range(start, len(lines)) if "unpatched or unexpected AMB3R tree" in lines[i])
    return "\n".join(lines[start:end + 1])


def _run_gate(tmp_path: Path, log_text: str, *, chunk: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run the launcher's DA3 gate for slot 0 against a VO log holding log_text."""
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "logs" / "vo_0.log").write_text(log_text, encoding="utf-8")
    script = "\n".join([
        "set -Eeuo pipefail",
        "die() { printf 'ERROR: %s\\n' \"$*\" >&2; exit 2; }",
        f"export DA3_SDPA_QUERY_CHUNK_SIZE={chunk if chunk is not None else _launcher_chunk_export()}",
        f"RUNTIME_DIR='{tmp_path}'",
        "slot=0",
        _launcher_da3_gate(),
        "echo GATE_PASSED",
    ])
    return subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=30)


ECHO_LINE = (
    "2026-10-09 INFO heatmapvln-amb3r-vo-rpc-server: VO server device: npu:0 (npu 2.7.1, "
    "bf16=True, DA3_SDPA_QUERY_CHUNK_SIZE=256, DA3_DISABLE_XFORMERS=1)\n"
)


def test_launcher_no_longer_greps_its_own_echoed_export():
    # String check, and the only option for an absence: the old gate grepped
    # "DA3_SDPA_QUERY_CHUNK_SIZE=256", the server's echo of this script's own export.
    # The behavioural counterpart is the next test, which feeds that echo to the gate.
    for line in NPU_LAUNCHER.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        if "grep" in line:
            assert "DA3_SDPA_QUERY_CHUNK_SIZE=" not in line, line


def test_gate_fails_on_the_echo_alone(tmp_path):
    # A log holding only the server's echo of the environment is exactly what an
    # unpatched or unexpected DA3 produces.  The old gate passed it; this one must not.
    result = _run_gate(tmp_path, ECHO_LINE)
    assert result.returncode == 2
    assert "does not report the configured query chunk" in result.stderr
    assert "GATE_PASSED" not in result.stdout


def test_gate_fails_when_da3_parsed_no_chunk(tmp_path):
    # Measured on the 910B: with the variable unset DA3 reports query_chunk=0.
    result = _run_gate(tmp_path, ECHO_LINE + (
        "INFO DA3 attention: query_chunk=0 (parsed by DA3), memory_bounded=True, xformers_disabled=True\n"
    ))
    assert result.returncode == 2
    assert "does not report the configured query chunk" in result.stderr


def test_gate_fails_when_the_layers_are_not_memory_bounded(tmp_path):
    # The chunk size is right and the attention layers still bound something else, so
    # the configured chunk would never be applied.  One grep covers both, because the
    # two facts are only worth anything together.
    result = _run_gate(tmp_path, ECHO_LINE + (
        "INFO DA3 attention: query_chunk=256 (parsed by DA3), memory_bounded=False, xformers_disabled=True\n"
    ))
    assert result.returncode == 2
    assert "memory-bounded attention bound" in result.stderr


def test_gate_passes_on_the_line_measured_on_the_machine(tmp_path):
    result = _run_gate(tmp_path, ECHO_LINE + (
        "INFO DA3 attention: query_chunk=256 (parsed by DA3), memory_bounded=True, xformers_disabled=True\n"
    ))
    assert result.returncode == 0, result.stderr
    assert "GATE_PASSED" in result.stdout


def test_gate_follows_the_exported_variable_not_a_literal(tmp_path):
    # The expected chunk is the variable the launcher exports, so the export and the
    # grep cannot drift apart.  At 128, a DA3 report of 256 is now the wrong one.
    good = "INFO DA3 attention: query_chunk=128 (parsed by DA3), memory_bounded=True, xformers_disabled=True\n"
    stale = good.replace("query_chunk=128", "query_chunk=256")
    assert _run_gate(tmp_path, good, chunk="128").returncode == 0
    assert _run_gate(tmp_path, stale, chunk="128").returncode == 2
    # A longer number sharing the prefix must not satisfy the gate either.
    assert _run_gate(tmp_path, good.replace("=128", "=1280"), chunk="128").returncode == 2


def _report_function_source() -> str:
    tree = ast.parse(VO_SERVER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_report_da3_attention":
            return ast.get_source_segment(VO_SERVER.read_text(encoding="utf-8"), node)
    raise AssertionError("_report_da3_attention is missing from the VO server")


def test_report_is_called_on_the_vo_server_startup_path():
    # Structural (ast) check: the real call path needs torch, DA3 and an NPU.  The
    # report must run unconditionally in _build_real_application, after DA3 is built
    # (so DA3's import order is the certified one) and before the session, and
    # main() must go through _build_real_application.
    text = VO_SERVER.read_text(encoding="utf-8")
    functions = {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}

    def top_level_calls(function: ast.FunctionDef) -> list[str]:
        names = []
        for statement in function.body:
            for node in ast.walk(statement):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    names.append(node.func.id)
        return names

    build = functions["_build_real_application"]
    report_calls = [
        statement for statement in build.body
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name) and statement.value.func.id == "_report_da3_attention"
    ]
    assert len(report_calls) == 1, "the report must be a plain top-level statement, not behind a branch"
    calls = top_level_calls(build)
    assert calls.index("DA3") < calls.index("_report_da3_attention") < calls.index("build_online_amb3r_session")
    assert "_build_real_application" in top_level_calls(functions["main"])


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def report(monkeypatch):
    """_report_da3_attention executed from its source, with a capturing logger."""
    for name in list(sys.modules):
        if name == "depth_anything_3" or name.startswith("depth_anything_3."):
            monkeypatch.delitem(sys.modules, name)
    logger = logging.getLogger("test-ascend-amb3r-patch-check")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    capture = _Capture()
    logger.addHandler(capture)
    namespace: dict[str, object] = {"os": os, "LOGGER": logger, "__name__": "vo_server_extract"}
    exec(compile(_report_function_source(), str(VO_SERVER), "exec"), namespace)
    yield namespace["_report_da3_attention"], capture
    logger.removeHandler(capture)


def _install_fake_da3(monkeypatch, *, parse, bound_is_memory_bounded: bool | None) -> None:
    """Fake DA3 modules; bound_is_memory_bounded=None leaves the layer attribute unset."""

    def memory_bounded_scaled_dot_product_attention(*args, **kwargs):
        raise AssertionError("evidence code must not run attention")

    def plain_sdpa(*args, **kwargs):
        raise AssertionError("evidence code must not run attention")

    names = [
        "depth_anything_3", "depth_anything_3.model", "depth_anything_3.model.dinov2",
        "depth_anything_3.model.dinov2.layers", "depth_anything_3.model.dinov2.layers.attention",
        "depth_anything_3.model.utils", "depth_anything_3.model.utils.memory_bounded_attention",
    ]
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        if name != names[-1] and name != names[4]:
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    modules["depth_anything_3.model.dinov2.layers"].attention = modules[names[4]]
    mba = modules["depth_anything_3.model.utils.memory_bounded_attention"]
    mba._configured_query_chunk_size = parse
    mba.memory_bounded_scaled_dot_product_attention = memory_bounded_scaled_dot_product_attention
    if bound_is_memory_bounded is not None:
        modules[names[4]].memory_bounded_scaled_dot_product_attention = (
            memory_bounded_scaled_dot_product_attention if bound_is_memory_bounded else plain_sdpa
        )


def test_report_logs_exactly_what_the_launcher_greps(tmp_path, report, monkeypatch):
    # End to end without the machine: the line the server logs is fed to the
    # launcher's own gate, so a change of wording on either side fails here.
    function, capture = report
    monkeypatch.setenv("DA3_DISABLE_XFORMERS", "1")
    chunk = _launcher_chunk_export()
    _install_fake_da3(monkeypatch, parse=lambda: int(chunk), bound_is_memory_bounded=True)
    function()
    [record] = capture.records
    assert record.levelno == logging.INFO
    message = record.getMessage()
    assert message == (
        f"DA3 attention: query_chunk={chunk} (parsed by DA3), memory_bounded=True, xformers_disabled=True"
    )
    result = _run_gate(tmp_path, ECHO_LINE + f"INFO heatmapvln-amb3r-vo-rpc-server: {message}\n")
    assert result.returncode == 0, result.stderr


def test_report_says_false_when_the_layers_bound_another_function(tmp_path, report, monkeypatch):
    # The dinov2 layers holding some other attention is precisely the case the
    # evidence must show; it must log False, not crash or claim True.
    function, capture = report
    monkeypatch.delenv("DA3_DISABLE_XFORMERS", raising=False)
    _install_fake_da3(monkeypatch, parse=lambda: 256, bound_is_memory_bounded=False)
    function()
    [record] = capture.records
    assert record.levelno == logging.INFO
    assert "memory_bounded=False" in record.getMessage()
    assert "xformers_disabled=False" in record.getMessage()
    assert _run_gate(tmp_path, record.getMessage() + "\n").returncode == 2


def test_report_says_false_when_the_layers_bound_nothing(report, monkeypatch):
    function, capture = report
    _install_fake_da3(monkeypatch, parse=lambda: 256, bound_is_memory_bounded=None)
    function()
    [record] = capture.records
    assert "memory_bounded=False" in record.getMessage()


def test_report_swallows_a_failing_parser(report, monkeypatch):
    # DA3's parser rejecting the variable must leave a warning, not stop the server:
    # the launcher's gate is what refuses, with a message that says why.
    function, capture = report

    def parse():
        raise ValueError("DA3_SDPA_QUERY_CHUNK_SIZE must be an integer")

    _install_fake_da3(monkeypatch, parse=parse, bound_is_memory_bounded=True)
    function()
    [record] = capture.records
    assert record.levelno == logging.WARNING
    assert "ValueError" in record.getMessage()
    assert "DA3 attention: query_chunk=" not in record.getMessage()


def test_report_swallows_a_missing_package(report, monkeypatch):
    function, capture = report
    monkeypatch.setitem(sys.modules, "depth_anything_3", None)
    function()
    [record] = capture.records
    assert record.levelno == logging.WARNING
    assert "could not read DA3's attention configuration" in record.getMessage()


def test_report_swallows_any_exception_raised_while_importing(report, monkeypatch):
    # Not only ImportError: DA3's import can fail with whatever its module-level code
    # raises (a device or toolchain error on a fresh instance, for one).
    function, capture = report

    class _Raising:
        def find_spec(self, name, path=None, target=None):
            if name == "depth_anything_3" or name.startswith("depth_anything_3."):
                raise OSError("share went away mid-import")
            return None

    monkeypatch.setattr(sys, "meta_path", [_Raising(), *sys.meta_path])
    function()
    [record] = capture.records
    assert record.levelno == logging.WARNING
    assert "OSError" in record.getMessage()

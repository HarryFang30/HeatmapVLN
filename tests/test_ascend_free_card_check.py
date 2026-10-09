"""The Ascend launcher's "only take free cards" guard, and the per-card placement budget.

Two things stand between a run and a card that cannot hold it, and both failed
silently before.  ``scripts/ascend/npu_hbm_used_mib.awk`` reads how much HBM a chip
already has in use from ``npu-smi info``; the parser it replaced read the first
"used / total" pair of the chip row, which is Memory-Usage and "0 / 0" on this
driver, so every card read 0 MiB and the guard passed on a full card.  The
placement budget in ``scripts/ascend/run_ppa_servers_npu.sh`` refuses layouts that
put more servers on one card than 65.5 GB can hold, which the free-card read cannot
see because it runs before this run's own servers exist.

Everything here runs the real awk and the real launcher.  The awk gets the two
tables taken from the machine verbatim, plus mutations of them, and the property the
file exists for is that no input ever makes it print 0 for a card that is not at 0:
a wrong 0 is the one answer the launcher cannot tell from a free card.  The
launcher runs end to end against a throwaway tree that satisfies its file
preflight, with a stub ``npu-smi`` that prints a chosen table and a stub Python that
fails, so each run stops at the validation under test or, when it passes every
check, at the platform preflight -- before anything could be started.

The deployment runs bash 5 and gawk or mawk on Linux; on a Mac this runs bash 3.2
and BSD awk.  The awk tests run under every awk that is installed and skip the rest.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HBM_AWK = REPO / "scripts" / "ascend" / "npu_hbm_used_mib.awk"
NPU_LAUNCHER = REPO / "scripts" / "ascend" / "run_ppa_servers_npu.sh"

# Both tables are `npu-smi info` output from the 910B3 machine, verbatim.  An idle
# card reads about 3.4 GB, which is why the launcher's free threshold is 4096 MiB.
IDLE_TABLE = """\
+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.0.3                 Version: 24.1.0.3                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B3               | OK            | 101.9       33                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          3401 / 65536         |
+===========================+===============+====================================================+
| 1     910B3               | OK            | 100.2       35                0    / 0             |
| 0                         | 0000:01:00.0  | 0           0    / 0          3403 / 65536         |
+===========================+===============+====================================================+
"""

# Cards 0 and 1 each hold a server pair.  The old parser read this table as all zeros.
LOADED_TABLE = """\
+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.0.3                 Version: 24.1.0.3                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B3               | OK            | 113.7       33                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          65521/ 65536         |
+===========================+===============+====================================================+
| 1     910B3               | OK            | 100.3       35                0    / 0             |
| 0                         | 0000:01:00.0  | 0           0    / 0          65330/ 65536         |
+===========================+===============+====================================================+
| 2     910B3               | OK            | 101.0       35                0    / 0             |
| 0                         | 0000:C2:00.0  | 0           0    / 0          3401 / 65536         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| 0       0                 | 201837        | python                   | 42204                   |
| 0       0                 | 201838        | python                   | 20042                   |
+===========================+===============+====================================================+
| No running processes found in NPU 2                                                            |
+===========================+===============+====================================================+
"""

IDLE_TRUTH = {0: 3401, 1: 3403}
LOADED_TRUTH = {0: 65521, 1: 65330, 2: 3401}

LOADED_CHIP_ROW_0 = (
    "| 0                         | 0000:C1:00.0  | 0           0    / 0          65521/ 65536         |"
)
LOADED_PROCESS_ROW = (
    "| 0       0                 | 201838        | python                   | 20042                   |"
)


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"fixture edit is ambiguous or stale: {old!r}"
    return text.replace(old, new)


@pytest.fixture(params=("awk", "gawk", "mawk"))
def awk(request) -> str:
    path = shutil.which(request.param)
    if path is None:
        pytest.skip(f"{request.param} is not installed")
    return path


def _hbm_used(awk: str, table: str, npu_id: int | str) -> str:
    result = subprocess.run(
        [awk, "-v", f"id={npu_id}", "-f", str(HBM_AWK)],
        input=table,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    # The launcher only looks at stdout, so a parser that complained on stderr and
    # still exited 0 would be read as "nothing printed": pin that it does neither.
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    return result.stdout


# --------------------------------------------------------------------------- awk


def test_loaded_table_reads_the_hbm_pair_not_memory_usage(awk):
    # The regression that started this: the chip row's last field holds Memory-Usage
    # ("0 / 0") and then HBM-Usage, and the old parser took the first pair, so a card
    # holding 65521 MiB read 0 and the free-card guard waved it through.
    assert _hbm_used(awk, LOADED_TABLE, 0) == "65521\n"
    assert _hbm_used(awk, LOADED_TABLE, 1) == "65330\n"
    assert _hbm_used(awk, LOADED_TABLE, 2) == "3401\n"


def test_idle_table_reads_the_resident_few_gigabytes(awk):
    # An idle card is not at zero.  The parser before the last one concatenated every
    # digit on the row and read an idle card as 100000 MiB, so pin the exact value.
    assert _hbm_used(awk, IDLE_TABLE, 0) == "3401\n"
    assert _hbm_used(awk, IDLE_TABLE, 1) == "3403\n"


@pytest.mark.parametrize(("table", "npu_id"), [(LOADED_TABLE, 3), (IDLE_TABLE, 2), (IDLE_TABLE, 7)])
def test_an_id_that_is_not_in_the_table_prints_nothing(awk, table, npu_id):
    # A typo in PPA_NPU_DEVICES must stop the launcher, not read as a free card.
    assert _hbm_used(awk, table, npu_id) == ""


def test_a_chip_row_holding_only_the_memory_usage_pair_prints_nothing(awk):
    # If the HBM column moved or vanished, the one pair left on the row would be
    # Memory-Usage, "0 / 0" -- exactly the old misreading.  One pair is not a layout
    # this parser knows, whichever pair it is.
    table = _replace_once(
        LOADED_TABLE,
        LOADED_CHIP_ROW_0,
        "| 0                         | 0000:C1:00.0  | 0           0    / 0                               |",
    )
    assert _hbm_used(awk, table, 0) == ""


def test_a_chip_row_holding_only_the_hbm_pair_prints_nothing(awk):
    # The other single-pair shape.  The number happens to be right here, but the
    # parser cannot know which column it is reading, so it must not guess.
    table = _replace_once(
        LOADED_TABLE,
        LOADED_CHIP_ROW_0,
        "| 0                         | 0000:C1:00.0  | 0                             65521/ 65536         |",
    )
    assert _hbm_used(awk, table, 0) == ""


@pytest.mark.parametrize("npu_id", [0, 1, 2])
def test_a_table_without_the_hbm_usage_header_prints_nothing(awk, npu_id):
    # Without the header nothing says the second pair is HBM at all.
    table = _replace_once(LOADED_TABLE, "HBM-Usage(MB)", "             ")
    assert _hbm_used(awk, table, npu_id) == ""


@pytest.mark.parametrize("table", ["", "\n", "npu-smi: command failed\n"])
def test_empty_or_non_table_input_prints_nothing(awk, table):
    # What the launcher would pipe in if npu-smi printed nothing or only an error.
    assert _hbm_used(awk, table, 0) == ""


def test_a_second_chip_row_under_one_npu_prints_nothing_for_that_npu(awk):
    # A multi-chip NPU has a row per chip and this parser can only read one; reading
    # the first would ignore whatever the second chip holds.
    second_chip = "| 1                         | 0000:C3:00.0  | 0           0    / 0          60000/ 65536         |"
    table = _replace_once(LOADED_TABLE, LOADED_CHIP_ROW_0, LOADED_CHIP_ROW_0 + "\n" + second_chip)
    assert _hbm_used(awk, table, 0) == ""
    # The refusal is about that NPU, not the whole table.
    assert _hbm_used(awk, table, 1) == "65330\n"


@pytest.mark.parametrize(
    "hbm_pair",
    [
        "65521/ 0    ",  # a total of 0 is not an HBM column
        "0    / 0    ",  # the driver reporting HBM the way it reports Memory-Usage
        "65537/ 65536",  # more in use than exists
    ],
)
def test_an_impossible_hbm_pair_prints_nothing(awk, hbm_pair):
    # "0 / 0" in the HBM slot is the dangerous one: it parses cleanly and its used
    # value is the 0 that must never be printed for a card nobody measured.
    table = _replace_once(LOADED_TABLE, "65521/ 65536", hbm_pair)
    assert _hbm_used(awk, table, 0) == ""


def test_the_same_npu_id_twice_prints_nothing(awk):
    # Two blocks claiming NPU 0 is not a table this parser understands; taking either
    # one could be taking the idle one.
    block = (
        "| 0     910B3               | OK            | 101.0       35                0    / 0             |\n"
        "| 0                         | 0000:C4:00.0  | 0           0    / 0          3401 / 65536         |\n"
        "+===========================+===============+====================================================+\n"
    )
    marker = "| 2     910B3"
    table = _replace_once(LOADED_TABLE, marker, block + marker)
    assert _hbm_used(awk, table, 0) == ""


@pytest.mark.parametrize("keep_process_header", [True, False])
def test_a_process_row_with_process_id_910_does_not_change_any_answer(awk, keep_process_header):
    # The NPU row is recognised by "910" in its name column, and the process table
    # below repeats NPU and chip ids.  A process whose id is 910 must not be taken
    # for an NPU row.  With the header removed the parser cannot stop at the process
    # table, so the second case pins that the row shape alone keeps it out.
    process_910 = "| 2       0                 | 910           | python                   | 30000                   |"
    table = _replace_once(LOADED_TABLE, LOADED_PROCESS_ROW, LOADED_PROCESS_ROW + "\n" + process_910)
    if not keep_process_header:
        table = _replace_once(
            table,
            "| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |\n",
            "",
        )
    for npu_id, used in LOADED_TRUTH.items():
        assert _hbm_used(awk, table, npu_id) == f"{used}\n"
    assert _hbm_used(awk, table, 3) == ""


def test_a_card_that_really_is_at_zero_reads_zero(awk):
    # The property below forbids a wrong 0, not 0 itself.  This is what keeps it
    # from passing vacuously on a parser that never prints 0.
    table = _replace_once(LOADED_TABLE, "3401 / 65536", "0    / 65536")
    assert _hbm_used(awk, table, 2) == "0\n"


def _line_mutations(table: str):
    """Every table one line-edit away: each line dropped, doubled, swapped with the
    next, and the table cut short after it.  This is the shape a driver update or a
    truncated npu-smi run produces, not a random fuzz."""
    lines = table.splitlines(keepends=True)
    for index in range(len(lines)):
        yield f"drop line {index}", "".join(lines[:index] + lines[index + 1 :])
        yield f"double line {index}", "".join(lines[: index + 1] + lines[index:])
        yield f"cut after line {index}", "".join(lines[: index + 1])
        if index + 1 < len(lines):
            swapped = lines[:index] + [lines[index + 1], lines[index]] + lines[index + 2 :]
            yield f"swap lines {index},{index + 1}", "".join(swapped)


def _corpus():
    for name, table, truth in (("idle", IDLE_TABLE, IDLE_TRUTH), ("loaded", LOADED_TABLE, LOADED_TRUTH)):
        yield f"{name} verbatim", table, truth
        yield f"{name} CRLF", table.replace("\n", "\r\n"), truth
        yield f"{name} no Memory-Usage pairs", table.replace("0    / 0          ", "                  "), truth
        yield f"{name} no HBM header", table.replace("HBM-Usage(MB)", "             "), truth
        for mutation, mutated in _line_mutations(table):
            yield f"{name} {mutation}", mutated, truth


def test_no_input_in_the_corpus_reads_as_a_wrong_number_least_of_all_zero(awk):
    # The point of this file.  The launcher compares whatever the parser prints with
    # a threshold, so any number that is not the card's real use is a wrong launch
    # decision, and a wrong 0 is the worst of them: it is indistinguishable from a
    # free card.  For every table in the corpus and every id, the only acceptable
    # outputs are the card's true use and silence.  None of the cards in these
    # tables is really at 0, so "0" is never acceptable here.
    failures = []
    cases = 0
    for name, table, truth in _corpus():
        for npu_id in (0, 1, 2, 3):
            output = _hbm_used(awk, table, npu_id)
            allowed = {""} | ({f"{truth[npu_id]}\n"} if npu_id in truth else set())
            cases += 1
            if output not in allowed or output.strip() == "0":
                failures.append(f"{name}, id {npu_id}: printed {output!r}, allowed {sorted(allowed)}")
    assert not failures, "\n".join(failures)
    # Guard against the corpus quietly shrinking to nothing.
    assert cases > 400


def test_a_header_with_memory_and_hbm_in_the_other_order_prints_nothing(awk):
    # A layout this parser has never seen: both headers present, but HBM first.  It
    # passes every other shape check (two pairs, a non-zero second total), and taking
    # the second pair would read Memory-Usage and print 0 for a card holding 65521
    # MiB -- the exact outcome this parser exists to avoid.  So the header rule pins
    # the column ORDER, not merely the presence of an HBM column.
    table = _replace_once(
        LOADED_TABLE,
        "AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |",
        "AICore(%)   HBM-Usage(MB)     Memory-Usage(MB)     |",
    )
    table = _replace_once(
        table,
        "0           0    / 0          65521/ 65536         |",
        "0           65521/ 65536      0    / 15000         |",
    )
    assert _hbm_used(awk, table, 0) == ""


# ---------------------------------------------------------------------- launcher


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def launcher(tmp_path):
    """Run the real launcher against a tree that passes its file preflight.

    Every path is the launcher's default under PPA_EVAL_ROOT, so the tree is what a
    real instance would hold, with stubs for the parts that would do work:

    - ``npu-smi`` prints the table in $FAKE_NPU_SMI_TABLE (or fails when
      $FAKE_NPU_SMI_FAIL is set).  It has to exist: the launcher checks
      ``command -v npu-smi`` before it validates a single id.
    - ``check_amb3r_patch.sh`` says it ran and exits $FAKE_PATCH_CHECK_STATUS, so a
      test can stop a run just after the placement budget and before npu-smi.
    - the Python fails, so a run that passes every check stops at the platform
      preflight, the first step that would need the real stack.
    """
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")
    if shutil.which("awk") is None:
        pytest.skip("awk is not installed")

    root = tmp_path / "root"
    repo = root / "HeatmapVLN"
    _write(repo / "scripts" / "evaluation" / "rpc_model_server.py", "# stub\n")
    _write(repo / "scripts" / "amb3r_vo" / "rpc_amb3r_vo_server.py", "# stub\n")
    _write(repo / "scripts" / "ascend" / "npu_hbm_used_mib.awk", HBM_AWK.read_text(encoding="utf-8"))
    _write(
        repo / "scripts" / "ascend" / "check_amb3r_patch.sh",
        'echo "patch-check-ran" >&2\nexit "${FAKE_PATCH_CHECK_STATUS:-0}"\n',
    )
    _write(repo / "scripts" / "ascend" / "amb3r_npu.patch", "stub\n")
    _write(repo / "configs" / "ppa_action_refine_v2_8gpu.yaml", "stub: 1\n")
    _write(root / "weights" / "ppa_refine_v2_best.pth", "stub\n")
    _write(root / "amb3r" / "slam" / "slam_config.yaml", "stub: 1\n")
    _write(root / "amb3r" / "checkpoints" / "DA3NESTED-GIANT-LARGE" / "model.safetensors", "stub\n")
    (root / "rpc" / "src" / "vla_rpc").mkdir(parents=True)
    (root / "InternNav_Model").mkdir(parents=True)
    # The deployment serves from a clone, and the launcher refuses a repo whose commit
    # it cannot read (the client will not accept a server set it cannot identify), so
    # the fake repo has to be a checkout with one commit like the real one.
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.com"],
        ["git", "config", "user.name", "test"],
        ["git", "add", "-A"],
        ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "fake repo"],
    ):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    _write(
        root / "envs" / "ppa" / "bin" / "python",
        '#!/bin/sh\necho "python-stub-ran" >&2\nexit 1\n',
        executable=True,
    )
    ascend_env = tmp_path / "set_env.sh"
    _write(ascend_env, "export ASCEND_TOOLKIT_HOME=/nonexistent/ascend-toolkit\n")
    fake_bin = tmp_path / "bin"
    _write(
        fake_bin / "npu-smi",
        '#!/bin/sh\n[ -n "$FAKE_NPU_SMI_FAIL" ] && { echo "dcmi init failed"; exit 1; }\n'
        'cat "$FAKE_NPU_SMI_TABLE"\n',
        executable=True,
    )
    table_file = tmp_path / "npu-smi.txt"

    def run(*, table: str = IDLE_TABLE, **overrides: str) -> subprocess.CompletedProcess:
        table_file.write_text(table, encoding="utf-8")
        # A clean environment: an ASCEND_RT_VISIBLE_DEVICES or a PPA_* export in the
        # shell that runs pytest would otherwise change which check fires.
        env = {
            "PATH": f"{fake_bin}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "HOME": str(tmp_path),
            "LC_ALL": "C",
            "PPA_EVAL_ROOT": str(root),
            "PPA_NPU_ASCEND_ENV": str(ascend_env),
            "PPA_NPU_RUNTIME_DIR": str(tmp_path / "runtime"),
            "PPA_NPU_TMP_ROOT": str(tmp_path / "t"),
            "PPA_NPU_SERVER_INSTANCE": "pytest",
            "FAKE_NPU_SMI_TABLE": str(table_file),
        }
        env.update(overrides)
        return subprocess.run(
            [bash, str(NPU_LAUNCHER)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    return run


def _refused_before_launch(result: subprocess.CompletedProcess, message: str) -> None:
    assert result.returncode == 2, result.stderr
    assert f"[ppa-npu] ERROR: {message}" in result.stderr, result.stderr
    # Nothing past validation ran: neither the AMB3R check nor the Python.
    assert "patch-check-ran" not in result.stderr
    assert "python-stub-ran" not in result.stderr


def _passed_every_check(result: subprocess.CompletedProcess) -> None:
    # The stub Python is the first thing that fails once every check has passed.
    assert result.returncode == 2, result.stderr
    assert "[ppa-npu] ERROR: platform preflight failed" in result.stderr, result.stderr
    assert "python-stub-ran" in result.stderr


def test_the_fake_tree_reaches_the_platform_preflight(launcher):
    # Every refusal below is only evidence if the run would otherwise have gone on:
    # without this, a missing stub file would make every test "refuse" for the
    # wrong reason.  NPU 2 is the free card of the loaded table.
    result = launcher(table=LOADED_TABLE, PPA_NPU_DEVICES="2")
    _passed_every_check(result)
    assert "patch-check-ran" in result.stderr
    assert "[ppa-npu] npu=2 used=3401MiB free enough" in result.stdout


# -- the free-card read, end to end


def test_a_full_card_is_refused(launcher):
    # The regression, through the launcher: with the old parser NPU 0 read 0 MiB.
    result = launcher(table=LOADED_TABLE, PPA_NPU_DEVICES="0")
    assert result.returncode == 2
    assert "[ppa-npu] ERROR: NPU 0 already has 65521 MiB in use (limit 4096)" in result.stderr
    assert "python-stub-ran" not in result.stderr


def test_a_full_card_is_refused_when_only_the_vo_server_would_use_it(launcher):
    # The VO ids go through the same read; a free model card must not carry its VO
    # partner onto a full one.
    result = launcher(table=LOADED_TABLE, PPA_NPU_DEVICES="2", PPA_NPU_VO_DEVICES="1")
    assert result.returncode == 2
    assert "[ppa-npu] npu=2 used=3401MiB free enough" in result.stdout
    assert "[ppa-npu] ERROR: NPU 1 already has 65330 MiB in use (limit 4096)" in result.stderr


@pytest.mark.parametrize(("limit", "passes"), [("3400", False), ("3401", True), ("0", False)])
def test_the_limit_is_compared_with_the_parsed_number(launcher, limit, passes):
    # The idle card reads 3401: one MiB under it refuses, at it passes.  A limit of 0
    # is valid and refuses even an idle card, which a 0-reading parser would not.
    result = launcher(table=IDLE_TABLE, PPA_NPU_DEVICES="0", PPA_NPU_MAX_USED_MIB=limit)
    if passes:
        _passed_every_check(result)
    else:
        assert result.returncode == 2
        assert f"[ppa-npu] ERROR: NPU 0 already has 3401 MiB in use (limit {limit})" in result.stderr


@pytest.mark.parametrize(
    ("table", "devices"),
    [
        (LOADED_TABLE.replace("HBM-Usage(MB)", "             "), "2"),
        (LOADED_TABLE, "5"),
        ("", "0"),
    ],
    ids=["layout-changed", "id-absent", "empty-table"],
)
def test_an_unreadable_card_is_refused_and_the_table_is_shown(launcher, table, devices):
    # Silence from the parser has to stop the run, and the operator needs to see
    # what npu-smi printed to tell a layout change from a wrong id.
    result = launcher(table=table, PPA_NPU_DEVICES=devices)
    assert result.returncode == 2
    assert f"[ppa-npu] ERROR: could not read HBM use of NPU {devices} from the npu-smi table" in result.stderr
    if table:
        assert table.splitlines()[1] in result.stderr
    assert "python-stub-ran" not in result.stderr


def test_a_failing_npu_smi_is_refused_with_its_output(launcher):
    result = launcher(PPA_NPU_DEVICES="0", FAKE_NPU_SMI_FAIL="1")
    assert result.returncode == 2
    assert "[ppa-npu] ERROR: npu-smi info failed: dcmi init failed" in result.stderr


# -- id and knob validation


@pytest.mark.parametrize(
    ("devices", "vo_devices", "bad"),
    [
        ("0,00", None, "00"),
        ("01", None, "01"),
        ("0", "00", "00"),
        ("0,1", "1,01", "01"),
        ("+1", None, "+1"),
        ("-1", None, "-1"),
        (" 1", None, " 1"),
        ("0x1", None, "0x1"),
        ("1.0", None, "1.0"),
        ("0,,1", None, ""),
        ("100", None, "100"),
    ],
)
def test_npu_ids_must_be_plain_decimals(launcher, devices, vo_devices, bad):
    # "0" and "00" are one card but two strings: they used to pass the uniqueness
    # check as two slots, and the budget and free-card read then reasoned about a
    # card layout that is not the one the servers land on.
    overrides = {"PPA_NPU_DEVICES": devices}
    if vo_devices is not None:
        overrides["PPA_NPU_VO_DEVICES"] = vo_devices
    result = launcher(**overrides)
    _refused_before_launch(result, f"invalid NPU id: '{bad}' (plain decimal, no leading zeros)")


def test_a_model_id_given_twice_is_refused(launcher):
    _refused_before_launch(launcher(PPA_NPU_DEVICES="0,0"), "NPU ids must be unique")


@pytest.mark.parametrize("limit", ["0800", "0700", "08", "-1", "4k", "4096 "])
def test_the_free_threshold_must_be_a_plain_decimal(launcher, limit):
    # bash arithmetic reads a leading 0 as octal: 0800 died with "value too great
    # for base", which says nothing about the knob, and 0700 would have been taken
    # silently as 448 MiB.
    result = launcher(PPA_NPU_DEVICES="0", PPA_NPU_MAX_USED_MIB=limit)
    _refused_before_launch(
        result,
        "PPA_NPU_MAX_USED_MIB must be a non-negative integer without leading zeros (bash reads 0800 as octal)",
    )


# -- the placement budget


@pytest.mark.parametrize(
    ("devices", "vo_devices", "message"),
    [
        ("0,1", "0,0", "card 0 would hold a model server and 2 VO servers: about 81 GB"),
        ("0,1", "1,1", "card 1 would hold a model server and 2 VO servers: about 81 GB"),
        ("0,1,2", "1,1,1", "card 1 would hold a model server and 3 VO servers: about 99 GB"),
        ("0,1,2,3", "4,4,4,4", "card 4 would hold 4 VO servers: about 75 GB"),
        ("0,1,2,3,4", "5,5,5,5,5", "card 5 would hold 5 VO servers: about 93 GB"),
    ],
)
def test_a_card_that_cannot_hold_its_servers_is_refused(launcher, devices, vo_devices, message):
    # A duplicated VO id used to pass every gate and then OOM hours into a run: the
    # free-card read runs before this run's own servers exist, so only the budget
    # can see what this run itself will put on a card.
    result = launcher(PPA_NPU_DEVICES=devices, PPA_NPU_VO_DEVICES=vo_devices)
    _refused_before_launch(result, message)


@pytest.mark.parametrize(
    ("devices", "vo_devices", "card", "count"),
    [("0,1", "2,2", 2, 2), ("0,1,2", "3,3,3", 3, 3)],
)
def test_two_or_three_vo_servers_on_a_model_free_card_pass_with_a_warning(
    launcher, devices, vo_devices, card, count
):
    # Allowed, because it fits on paper, but never measured: the run goes on and the
    # log says so.  The stub patch check stops it right after the budget.
    result = launcher(PPA_NPU_DEVICES=devices, PPA_NPU_VO_DEVICES=vo_devices, FAKE_PATCH_CHECK_STATUS="1")
    assert result.returncode == 2
    assert "[ppa-npu] ERROR: AMB3R tree verification failed" in result.stderr
    assert "patch-check-ran" in result.stderr
    assert (
        f"[ppa-npu] WARNING: card {card} takes {count} VO servers (about {count * 18 + 3} GB); "
        "that layout has never been measured"
    ) in result.stderr


@pytest.mark.parametrize(("devices", "vo_devices"), [("0,1", None), ("0,1", "1,0")])
def test_one_model_and_one_vo_server_per_card_passes_silently(launcher, devices, vo_devices):
    # The default pairing, and the same pairing crossed between slots, are the
    # certified footprint: no refusal and no warning.
    overrides = {"PPA_NPU_DEVICES": devices, "FAKE_PATCH_CHECK_STATUS": "1"}
    if vo_devices is not None:
        overrides["PPA_NPU_VO_DEVICES"] = vo_devices
    result = launcher(**overrides)
    assert result.returncode == 2
    assert "[ppa-npu] ERROR: AMB3R tree verification failed" in result.stderr
    assert "WARNING" not in result.stderr

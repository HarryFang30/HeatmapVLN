"""EXP-32's two controllers, ``scripts/exp32/exp32_auto.sh`` (4090) and ``scripts/exp32/exp32_910b.sh`` (910B).

They meet only through the tunnels, so every name one side writes the other must read: the signal files, the markers
and record roots of the G0/D3 hand-off on 52413, the exports on 52417 (the sampled set's ``exp32_sampling_<pass>``,
the parity outputs, the adapters) and the server tags of P's sets.  Both scripts' ``EXP32_DRY=1`` walkthroughs print
every command; the names are taken from those prints (and from the 910B script's signal handling) and compared.
Also: B2's PASS3 goes out before pass 3 starts, ``stage_g0d3`` stages what the 910B's G0 root reads, and H's canary
result is kept only when it measured something, at the commit that measured it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AUTO = REPO / "scripts/exp32/exp32_auto.sh"
NPU = REPO / "scripts/exp32/exp32_910b.sh"
P_TAGS = {"x32r1", "x32s05", "x32e2", "x32base"}


def _linux_bash() -> bool:
    probe = 'declare -A a=([k]=1); date -d 2026-01-01T00:00:00Z +%s'
    return subprocess.run(["bash", "-c", probe], capture_output=True, text=True).returncode == 0


pytestmark = pytest.mark.skipif(not _linux_bash(), reason="needs bash >= 4 and GNU date (the 4090 container)")


def walk_4090(tmp_path: Path, name: str, **knobs: str) -> dict:
    """The 4090 walkthrough in its own sandbox: its printed commands, auto.log, SUMMARY, the signal files it wrote."""
    sandbox = tmp_path / name
    env = dict(os.environ, EXP32_DRY="1", EXP32_WT=str(REPO), EXP32_DRY_DIR=str(sandbox), EXP32_PYTHON=sys.executable)
    env.update({f"EXP32_DRY_{k}": v for k, v in knobs.items()})
    proc = subprocess.run(["bash", str(AUTO)], capture_output=True, text=True, env=env, timeout=600)
    auto = sandbox / "exp32/auto"
    summary = (auto / "SUMMARY.txt").read_text()
    assert proc.returncode == 0, proc.stderr[-2000:]
    return {
        "cmds": [line[len("DRY $ "):] for line in proc.stderr.splitlines() if line.startswith("DRY $ ")],
        "log": (auto / "auto.log").read_text(),
        "summary": summary,
        "end": [line for line in summary.splitlines() if " END: " in line][-1],
        "signals": {p.name for p in (auto / "signals_out").iterdir() if p.is_file()},
        "auto": auto,
    }


def walk_910b(tmp_path: Path) -> str:
    """The 910B walkthrough (every branch, writes nothing): its stdout."""
    root = tmp_path / "zhr_1"
    env = dict(os.environ, EXP32_DRY="1", EXP32_ROOT=str(root), EXP32_PYTHON=sys.executable)
    proc = subprocess.run(["bash", str(NPU)], capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert not root.exists(), "the 910B walkthrough wrote under its root"
    return proc.stdout


@pytest.fixture(scope="module")
def walks(tmp_path_factory) -> dict:
    tmp = tmp_path_factory.mktemp("exp32_walks")
    return {
        # B2 runs pass 3 (3,500 then 2,500), the gates ask for lora_B x 0.5, the p_repro check fails (REBASE_P)
        "scale": walk_4090(tmp, "scale", NRFT="3500 2500", NEXT="scale_0.5 val", RERUN="FAIL", EST="0 0 0 3"),
        "retrain": walk_4090(tmp, "retrain", NEXT="retrain_2_epochs val", EST="3"),
        "plain": walk_4090(tmp, "plain", EST="3"),
        "npu": walk_910b(tmp),
    }


def _all_cmds(walks: dict) -> list[str]:
    return [c for name in ("scale", "retrain", "plain") for c in walks[name]["cmds"]]


def test_both_scripts_parse():
    for script in (AUTO, NPU, REPO / "scripts/exp32/cohorts.sh"):
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert proc.returncode == 0, f"{script.name}: {proc.stderr}"


def test_walkthroughs_reach_their_ends(walks):
    assert "estimate k=4 exit 3" in walks["scale"]["end"]
    assert "estimate k=1 exit 3" in walks["retrain"]["end"] and "estimate k=1 exit 3" in walks["plain"]["end"]
    assert "STOP_SERVING (x32base stopped)" in walks["npu"]


def test_signals_sent_are_the_signals_handled(walks):
    sent = set().union(*(walks[n]["signals"] for n in ("scale", "retrain", "plain")))
    src = NPU.read_text()
    handled = set(re.findall(r"\$SIG/([A-Z][A-Z0-9_]+)", src))
    case = re.search(r'case "\$f" in ([A-Z0-9_|]+)\)', src)
    assert case, "next_signal's case list moved"
    handled |= set(case.group(1).split("|"))
    assert sent == handled == {"PASS3", "SAMPLING_DONE", "SCALE05", "RETRAIN_E2", "REBASE_P", "STOP_SERVING"}


def test_p_server_tags_match(walks):
    pulled = set(re.findall(r"EXP28_MODE=external EXP28_TAG=(\S+)", "\n".join(_all_cmds(walks))))
    served = {tag for line in walks["npu"].splitlines() if "serve_910b.sh" in line
              for tag in re.findall(r"EXP28_TAG=(\S+)", line)}
    assert pulled == served == P_TAGS | {"rfts"}


def test_exports_pulled_are_the_exports_written(walks):
    pulled = set(re.findall(r"pull_dirs\.py \S+ 52417 (\S+)", "\n".join(_all_cmds(walks))))
    out = walks["npu"]
    moved = set(re.findall(r"\[dry\] mv \S+/([A-Za-z0-9_]+)\.part \S+/\1\b", out))
    exported = moved | set(re.findall(r"\[dry\] export \S+/(exp32_sampling_p\d)\b", out))
    assert pulled == exported == {"exp32_sampling_p2", "exp32_sampling_p3", "exp32_g0_npu", "exp32_d3_npu",
                                  "adapter_rX32R1", "adapter_rX32R1E2"}
    # the files the 4090 reads from D3's export are the ones copied into it
    copied = {Path(f).name for line in out.splitlines() if line.startswith("[dry] cp ") and "exp32_d3_npu.part" in line
              for f in line.split()[2:-1]}
    cmds = "\n".join(walks["scale"]["cmds"])
    assert re.search(r"aud\.py dose --parity-dir \S+/parity_pull/exp32_d3_npu ", cmds)
    d3_calls = re.findall(r"--d3-calls \S+/parity_pull/exp32_d3_npu/(\S+)", cmds)
    assert d3_calls and set(d3_calls) <= copied and {"parity_g0.json", "parity_g0.calls.jsonl"} <= copied


def test_g0_d3_handoff_markers_and_roots(walks):
    auto = walks["plain"]["auto"]
    markers = {(auto / f"{m}_marker.txt").read_text().strip() for m in ("G0", "D3")}
    out = walks["npu"]
    awaited = set(re.findall(r"\(\S+/parity_in/([A-Z0-9_]+)\)", out))
    assert markers == awaited == {"G0_PUSHED", "D3_PUSHED"}
    # 4090: G0's records, then its marker, then D3's, then its marker, all on 52413
    pushes = [re.search(r"--files \S+/(\S+)", c).group(1) for c in walks["plain"]["cmds"] if "--ports 52413" in c]
    assert pushes == ["g0_files.txt", "G0_marker.txt", "d3_files.txt", "D3_marker.txt"]
    # 910B: G0 replays <in>/records, D3 the tag directories
    g0 = [line for line in out.splitlines() if "S2_LORA_OUTPUT_DIR=" in line and "/g0_npu " in line]
    d3 = [line for line in out.splitlines() if "S2_LORA_OUTPUT_DIR=" in line and "/d3_npu " in line]
    assert len(g0) == len(d3) == 1
    g0_root = re.search(r"S2_LORA_RECORD_ROOTS=(\S+)", g0[0]).group(1)
    assert g0_root.endswith("/parity_in/records")
    d3_roots = re.search(r"S2_LORA_RECORD_ROOTS=(\S+)", d3[0]).group(1).split(":")
    assert all(re.search(r"/parity_in/(p1|r0s|p2|p3)$", r) for r in d3_roots), d3_roots


def test_pass3_signal_and_ready_exports_come_first(walks):
    log = walks["scale"]["log"]

    def at(pattern: str) -> int:
        hit = re.search(pattern, log)
        assert hit, pattern
        return hit.start()

    assert at(r"signal PASS3") < at(r"cohorts\.sh pass3") < at(r"pull_dirs\.py \S+ 52417 exp32_sampling_p3")
    assert at(r"pull_dirs\.py \S+ 52417 exp32_sampling_p3") < at(r"EXP28_MODE=external EXP28_TAG=rfts \S+ \S+ "
                                                                  r"EXP28_OUT_DIR=\S+/exp32/p3 ")
    assert at(r"exp32_sampling_p3") < at(r"signal SAMPLING_DONE")
    p2_ready = at(r"pull_dirs\.py \S+ 52417 exp32_sampling_p2")
    assert p2_ready < at(r"EXP28_OUT_DIR=\S+/exp32/p2 ")  # neither arm of pass 2 before the 910B says so
    assert "signal PASS3" not in walks["plain"]["log"] and "exp32_sampling_p3" not in walks["plain"]["log"]
    assert re.search(r"signal SAMPLING_DONE", walks["plain"]["log"])


def _lib(tmp_path: Path, script: str, **env_extra: str) -> subprocess.CompletedProcess:
    """``exp32_auto.sh``'s functions (EXP32_LIB=1), then ``script``."""
    env = dict(os.environ, EXP32_LIB="1", EXP32_DRY_DIR=str(tmp_path / "sandbox"), EXP32_PYTHON=sys.executable)
    env.update(env_extra)
    return subprocess.run(["bash", "-c", f'source "{AUTO}" || exit 9\n{script}'], capture_output=True, text=True,
                          env=env, timeout=300)


def test_stage_g0d3_stages_what_the_910b_reads(tmp_path):
    from scripts.exp32.build_rft_stream import RECORDS_LIST, RFT_TAGS
    from scripts.tools.stream_transfer import NAME

    def record(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tar")
        path.with_name(path.name + ".sha256").write_text("x\n")
        return path

    r0g = tmp_path / "r0g/records"
    g0 = [record(r0g / f"scene{i}/ep{i:06d}.s2rec.tar") for i in range(3)]
    rft = [record(tmp_path / f"{tag}/records/sceneA/ep{i:06d}.s2rec.tar") for i, tag in enumerate(RFT_TAGS)]
    x = tmp_path / "x"
    (x / "stream_rft").mkdir(parents=True)
    rows = [f"{p}\tnpu\t{tag}" for p, tag in zip(rft, RFT_TAGS)] + [f"{rft[0]}\tcuda\tp1"]
    (x / "stream_rft" / RECORDS_LIST).write_text("\n".join(rows) + "\n")
    stage = tmp_path / "stage"
    proc = _lib(tmp_path, f'R0G="{tmp_path}/r0g"; stage_g0d3 "{stage}"', EXP32_X=str(x))
    assert proc.returncode == 0, proc.stderr
    files = proc.stdout.split()
    assert all(NAME.match(f) for f in files), files
    g0_files = [f for f in files if f.startswith("records/")]
    assert files[:len(g0_files)] == g0_files  # G0's first (pushed before G0_PUSHED)
    assert len(g0_files) == 2 * len(g0) and files[-1] == "picks.json"
    # the 910B's G0 root (<in>/records) holds every R0-G record, committed; D3's tag roots hold the npu picks
    staged = [p for p in (stage / "records").glob("*/*.s2rec.tar") if p.with_name(p.name + ".sha256").exists()]
    assert sorted(p.resolve() for p in staged) == sorted(p.resolve() for p in g0)
    picks = json.loads((stage / "picks.json").read_text())
    assert picks["g0_records"] == 3 and picks["d3_npu_pool"] == len(RFT_TAGS) == len(picks["d3"])
    for tag in RFT_TAGS:
        assert len(list((stage / tag).glob("*/*.s2rec.tar"))) == 1


CANARY_LIB = r'''
WT="$CANARY_WT"; mkdir -p "$X/logs"
x() {  # the run is a no-op; the comparison writes $CANARY_JSON (nothing when empty: a crash)
  local a out=""; for a in "$@"; do [[ "$out" == next ]] && out=$a; [[ "$a" == --out ]] && out=next; done
  if [[ "$*" == *compare_client_logs.py* ]]; then
    echo x >> "$A/compares"; [[ -n "$CANARY_JSON" ]] && printf '%s' "$CANARY_JSON" > "$out"
  fi
  return 0
}
canary noadapter "" 0
echo "KEPT=$(cat "$A/H_noadapter.result" 2>/dev/null)"
echo "COMPARES=$(wc -l < "$A/compares" 2>/dev/null || echo 0)"
'''


def _canary(tmp_path: Path, wt: Path, report: dict | None) -> tuple[str, str, int]:
    proc = _lib(tmp_path, CANARY_LIB, CANARY_WT=str(wt), CANARY_JSON=json.dumps(report) if report else "")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.splitlines()
    kept = next(line for line in out if line.startswith("KEPT="))[5:]
    compares = int(next(line for line in out if line.startswith("COMPARES="))[9:])
    return out[0], kept, compares


def _commit(wt: Path) -> str:
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-C", str(wt)]
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "c"], check=True)
    return subprocess.run(git + ["rev-parse", "--short=12", "HEAD"], capture_output=True, text=True).stdout.strip()


def test_canary_keeps_only_measurements_at_their_commit(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    subprocess.run(["git", "init", "-q", str(wt)], check=True)
    head = _commit(wt)
    full = {"shared_episodes": 3, "identical": 3, "calls_compared": 69}

    line, kept, n = _canary(tmp_path, wt, None)  # crashed twice: nothing measured, nothing kept
    assert line.startswith("FAIL no comparison (noadapter) (try 2)") and kept == "" and n == 2
    line, kept, n = _canary(tmp_path, wt, {"shared_episodes": 1, "identical": 1, "calls_compared": 20})
    assert line.startswith("FAIL 1/1") and kept == "" and n == 4  # a partial run is not a measurement either
    line, kept, n = _canary(tmp_path, wt, full)
    assert line == "PASS 3/3 episodes identical, 69 calls (try 1)" and kept == f"{line} @{head}" and n == 5
    line, kept, n = _canary(tmp_path, wt, None)  # same commit: reused, not run
    assert line == "PASS 3/3 episodes identical, 69 calls (try 1)" and n == 5
    head2 = _commit(wt)  # another commit: run again; a crash there does not read the earlier comparison
    line, kept, n = _canary(tmp_path, wt, None)
    assert line.startswith("FAIL no comparison (noadapter) (try 2)") and kept == "" and n == 7
    line, kept, n = _canary(tmp_path, wt, {"shared_episodes": 3, "identical": 2, "calls_compared": 69})
    assert line == "FAIL 2/3 episodes identical, 69 calls (try 2)" and kept == f"{line} @{head2}" and n == 9
    line, kept, n = _canary(tmp_path, wt, full)  # a measured FAIL is kept too (H allows two tries, not more)
    assert line.startswith("FAIL 2/3") and n == 9


def test_raise_marks_a_signal_sent(tmp_path):
    proc = _lib(tmp_path, 'x() { return 0; }; sent PASS3 && echo before; raise PASS3; sent PASS3 && echo after; '
                          'x() { return 1; }; raise SCALE05; sent SCALE05 || echo unsent')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["after", "unsent"]

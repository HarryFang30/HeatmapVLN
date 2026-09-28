#!/usr/bin/env bash
# EXP-19 on the RTX 4090 box: everything after a rerun, for one run.
#
#   topdown   scripts.exp18.topdown.render_topdown for the run's scenes -> <EXP19_ROOT>/topdown
#             (resumable; the EXP-18 maps the C500 figures used were on the C500)
#   render    scripts/exp19/render_views.py --run-dir runs/<run> -> renders/ (per-episode QA:
#             sensor pose <= 1e-4 and median NCC >= 0.9 against the client's own frames)
#   records   scripts.exp19.build_records (gates, bundles, metrics of THIS run)
#   timeline  scripts.exp19.build_timeline --self-check -> records_v2/
#   anim      scripts.exp19.figures.animate_v2 -> figures_v2/anim_<run>/ (MP4 + GIF, zh + en)
#
# The C500 wrappers (run_render.sh, run_figures_v2.sh, scripts/exp18/topdown/with_xvfb.sh)
# assume the AFS X11 bundle and the qwen25 / vlnce envs.  Here: one system Xvfb with
# llvmpipe (scripts/run_ppa_r2r_val_unseen_cuda.sh's recipe), /opt/conda/bin/python for
# every stage, and the per-stage flags below.  Nothing here changes what a stage computes.
#
# Differences from the C500 chain, each forced by what the box lacks:
#   * render: no collector self-check (its r2r_panoramic_data_v2 clip was on the C500);
#     the per-episode QA above still runs and fails an episode on a pose-convention error;
#   * records: no eval-log reference (the C500 main-table logs are gone), so the
#     code-equivalence gate fails and --allow-invalid is passed: this run's metrics are
#     void by construction and are never read as verdicts (the C500 runs/main are the
#     verdicts of record); the pixel-goal convention is fixed to the C500's reading
#     (field_vu, 389 ready calls) instead of re-resolved from a handful of episodes, and the
#     chain stops if this run's own auto reading is decisive and disagrees;
#   * timeline: checks 0-2 must pass; checks 3-4 need turns of both signs / turn calls,
#     which five episodes may not have: without a sample they are "not measurable";
#   * anim: ffmpeg = imageio-ffmpeg's static binary (the box has no system ffmpeg);
#     animate_v2 probes without ffprobe.
#
# Container one-time setup (see the runbook, section 8): <support>/habitat/VLN-CE = VLN-CE
# 3b0c5c0 (the collector config render_views.py builds the simulator from) with
# data/scene_datasets/mp3d -> <scenes>/mp3d; <support>/fonts = NimbusSans-*.otf +
# DroidSansFallbackFull.ttf copied from the host.
#
# Run inside fjl-habitat from the archived source:
#   cd <EXP19_ROOT>/src_<sha>
#   EXP19_ROOT=/workspace/exp19_behavior_viz_4090 EXP19_RUN=main4090 bash scripts/exp19/run_post_cuda.sh
# Env:
#   EXP19_ROOT, EXP19_RUN          required (runs/<run>/DONE must say "complete"; a merged run's DONE is
#                                  written by scripts/exp19/merge_runs.py)
#   EXP19_STAGES                   default "topdown render records timeline anim"
#   EXP19_SUPPORT                  default <EXP19_ROOT>/support
#   EXP19_SCENES_DIR               parent of mp3d/ (default /dataset)
#   EXP19_TOPDOWN_ROOT             default <EXP19_ROOT>/topdown
#   EXP19_CANDIDATES               default <EXP19_ROOT>/cases/candidates.json
#   EXP19_ANIM_EPISODES            ep_keys to animate (default: metrics.json main cases present in the run)
#   EXP19_ANIM_LANGS               default "zh en"
#   EXP19_PYTHON                   default /opt/conda/bin/python
#   EXP19_FFMPEG                   default: imageio_ffmpeg.get_ffmpeg_exe()
#   EXP19_POST_DISPLAY             default 390
#   EXP19_ALLOW_MISSING_MAIN=1     go on when a category has no main case (default: stop after records)
set -euo pipefail
trap '' PIPE

SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
EXP_ROOT="${EXP19_ROOT:?set EXP19_ROOT}"
RUN="${EXP19_RUN:?set EXP19_RUN}"
STAGES="${EXP19_STAGES:-topdown render records timeline anim}"
SUPPORT="${EXP19_SUPPORT:-$EXP_ROOT/support}"
SCENES_DIR="${EXP19_SCENES_DIR:-/dataset}"
TOPDOWN="${EXP19_TOPDOWN_ROOT:-$EXP_ROOT/topdown}"
CANDIDATES="${EXP19_CANDIDATES:-$EXP_ROOT/cases/candidates.json}"
LANGS="${EXP19_ANIM_LANGS:-zh en}"
PYTHON="${EXP19_PYTHON:-/opt/conda/bin/python}"
DISPLAY_NUM="${EXP19_POST_DISPLAY:-390}"
RUN_DIR="$EXP_ROOT/runs/$RUN"
VLNCE="$SUPPORT/habitat/VLN-CE"
FONTS="$SUPPORT/fonts"
LOG_DIR="$EXP_ROOT/logs"
XVFB_PID=""

die() { printf '[exp19-post] ERROR: %s\n' "$*" >&2 || true; exit 2; }
log() { printf '[exp19-post] %s %s\n' "$(date -u +%FT%TZ)" "$*" || true; }
cleanup() {
  local status=$?
  set +e
  trap - EXIT
  if [[ -n "$XVFB_PID" ]] && kill -0 "$XVFB_PID" 2>/dev/null; then
    kill -TERM "$XVFB_PID" 2>/dev/null
    wait "$XVFB_PID" 2>/dev/null
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

for var in EXP19_ROOT EXP19_SUPPORT EXP19_SCENES_DIR EXP19_TOPDOWN_ROOT EXP19_CANDIDATES EXP19_PYTHON; do
  if [[ -n "${!var:-}" && "${!var}" != /* ]]; then die "$var must be an absolute path, got ${!var}"; fi
done
for stage in $STAGES; do
  case "$stage" in topdown|render|records|timeline|anim) ;; *) die "unknown stage '$stage'" ;; esac
done
[[ -s "$SRC/.exp19_git_sha" ]] || die "$SRC is not an archived source copy (.exp19_git_sha missing)"
[[ -x "$PYTHON" ]] || die "missing python: $PYTHON"
[[ -s "$RUN_DIR/DONE" ]] || die "$RUN_DIR/DONE missing: the rerun has not finished"
"$PYTHON" -c 'import json, sys; d = json.load(open(sys.argv[1])); sys.exit(d.get("status") != "complete")' \
  "$RUN_DIR/DONE" || die "$RUN_DIR/DONE does not say status=complete"
[[ -d "$VLNCE/habitat_extensions" && -s "$VLNCE/habitat_extensions/config/vlnce_collect.yaml" ]] \
  || die "missing VLN-CE collector config under $VLNCE (runbook section 8)"
[[ -d "$VLNCE/data/scene_datasets/mp3d" ]] || die "$VLNCE/data/scene_datasets/mp3d must link to $SCENES_DIR/mp3d"
[[ "$(cd "$VLNCE/data/scene_datasets/mp3d" && pwd -P)" == "$(cd "$SCENES_DIR/mp3d" && pwd -P)" ]] \
  || die "$VLNCE/data/scene_datasets/mp3d is not $SCENES_DIR/mp3d"
compgen -G "$FONTS/NimbusSans-*.otf" >/dev/null || die "no NimbusSans-*.otf in $FONTS (runbook section 8)"
[[ -s "$FONTS/DroidSansFallbackFull.ttf" ]] || die "no DroidSansFallbackFull.ttf in $FONTS (runbook section 8)"
[[ -s "$CANDIDATES" ]] || die "missing $CANDIDATES (scripts/exp19/rebuild_cases.py)"
FFMPEG="${EXP19_FFMPEG:-$("$PYTHON" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)}"
if [[ " $STAGES " == *" anim "* ]]; then
  [[ -n "$FFMPEG" && -x "$FFMPEG" ]] || die "no ffmpeg (set EXP19_FFMPEG)"
fi

export PYTHONDONTWRITEBYTECODE=1
export EXP18_WORKSPACE="$SUPPORT"  # scripts/exp18/common.py: VLNCE_PROJECT = <this>/habitat/VLN-CE
export EXP18_FONT_DIR="$FONTS" EXP18_CJK_FONT="$FONTS/DroidSansFallbackFull.ttf"
export MPLCONFIGDIR="$EXP_ROOT/.mplconfig"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
GL_ENV=(DISPLAY="127.0.0.1:${DISPLAY_NUM}.0" LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
  MESA_LOADER_DRIVER_OVERRIDE=swrast LP_NUM_THREADS="${LP_NUM_THREADS:-8}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  OPENBLAS_NUM_THREADS=4 GLOG_minloglevel=2 MAGNUM_LOG=quiet CUDA_VISIBLE_DEVICES=)
mkdir -p "$LOG_DIR" "$MPLCONFIGDIR"
cd "$SRC"
log "src=$SRC git_sha=$(cat "$SRC/.exp19_git_sha") run=$RUN stages='$STAGES' ffmpeg=${FFMPEG:-none}"

start_xvfb() {
  [[ -z "$XVFB_PID" ]] || return 0
  local xvfb
  xvfb=$(command -v Xvfb) || die "no Xvfb"
  if [[ -e "/tmp/.X${DISPLAY_NUM}-lock" || -e "/tmp/.X11-unix/X${DISPLAY_NUM}" ]] \
    || (exec 3<>"/dev/tcp/127.0.0.1/$((6000 + DISPLAY_NUM))") 2>/dev/null; then
    die "display :$DISPLAY_NUM is in use (set EXP19_POST_DISPLAY)"
  fi
  env LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=swrast \
    "$xvfb" ":$DISPLAY_NUM" -screen 0 1024x768x24 -nolock -nolisten unix -listen tcp +iglx -ac \
    >"$LOG_DIR/post_${RUN}_xvfb.log" 2>&1 &
  XVFB_PID=$!
  for _ in $(seq 1 60); do
    if (exec 3<>"/dev/tcp/127.0.0.1/$((6000 + DISPLAY_NUM))") 2>/dev/null; then
      log "Xvfb :$DISPLAY_NUM ready (pid $XVFB_PID)"
      return 0
    fi
    kill -0 "$XVFB_PID" 2>/dev/null || break
    sleep 1
  done
  die "Xvfb :$DISPLAY_NUM did not start; see $LOG_DIR/post_${RUN}_xvfb.log"
}

run_scenes() {  # scene ids of the run's episodes (steps.jsonl episode_start)
  "$PYTHON" - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
scenes = set()
for f in Path(sys.argv[1]).glob("gpu*/steps/*/steps.jsonl"):
    with open(f) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("type") == "episode_start":
                scenes.add(Path(str(r["scene_id"])).name.split(".")[0])
                break
print(" ".join(sorted(scenes)))
PY
}

for stage in $STAGES; do
  case "$stage" in
    topdown)
      start_xvfb
      scenes=$(run_scenes)
      [[ -n "$scenes" ]] || die "no episodes in $RUN_DIR"
      log "topdown: $scenes -> $TOPDOWN"
      # shellcheck disable=SC2086
      env "${GL_ENV[@]}" "$PYTHON" -u -m scripts.exp18.topdown.render_topdown --scenes $scenes \
        --mp3d-root "$SCENES_DIR/mp3d" --out "$TOPDOWN"
      ;;
    render)
      start_xvfb
      log "render: $RUN_DIR -> $EXP_ROOT/renders"
      env "${GL_ENV[@]}" EXP19_ROOT="$EXP_ROOT" "$PYTHON" -u scripts/exp19/render_views.py \
        --run-dir "$RUN_DIR" --out-dir "$EXP_ROOT/renders"
      ;;
    records)
      log "records: build_records (no eval-log reference: void by construction, --allow-invalid)"
      "$PYTHON" -u -m scripts.exp19.build_records --exp-root "$EXP_ROOT" --run "$RUN" --candidates "$CANDIDATES" \
        --topdown-root "$TOPDOWN" --pixel-goal-convention field_vu --allow-invalid
      # --allow-invalid is for the two reasons this box cannot avoid (no main-table log to compare with; the
      # candidates that were not rerun).  Anything else -- trace neutrality above all -- stops the chain, and so
      # does a category without a main case (the pre-registered fallback reruns its next candidate).
      "$PYTHON" - "$EXP_ROOT/metrics/metrics.json" "${EXP19_ALLOW_MISSING_MAIN:-0}" <<'PY' || die "records gate failed"
import json, re, sys
m = json.load(open(sys.argv[1]))
ok = True
tolerated = (r"code-equivalence gate failed \(\d+/\d+\)$", r"\d+ candidate episodes not in the run: ")
for reason in m["validity"]["reasons"]:
    known = any(re.match(p, reason) for p in tolerated)
    print(f"[exp19-post] validity reason ({'tolerated: no main-table log / not rerun' if known else 'FATAL'}): {reason}")
    ok &= known
tn = m["gates"]["trace_neutrality"]
print(f"[exp19-post] trace neutrality: {tn['n_match']}/{tn['n_trajectory_calls']} trajectory calls match; "
      f"calls without a trace: {tn['calls_without_trace'] or 'none'} -> {'PASS' if tn['pass'] else 'FAIL'}")
ok &= bool(tn["pass"]) and not m["coverage"]["unfinished"]
for e in m["episodes"]:
    print(f"[exp19-post] {e['ep_key']} {e['category']}#{e['category_rank']}: predicate on this rerun "
          f"{e['predicate_holds_on_rerun']}, outcome {e['outcome']}")
conv = m["pixel_goal_convention"]
auto = conv.get("auto_resolution")
print(f"[exp19-post] pixel-goal convention {conv['convention']} (forced; this run's own auto reading: "
      f"{auto or 'ambiguous'}; " + ", ".join(f"{k} balanced agreement {c['balanced_sign_agreement']}"
                                            for k, c in conv["conventions"].items()) + ")")
ok &= auto in (None, "field_vu")  # a decisive auto reading that disagrees with the C500's is a stop
for cat, key in m["main_cases"].items():
    print(f"[exp19-post] main case {cat}: {key or 'NONE (rerun the next candidate: ledger run record 5)'}")
if any(v is None for v in m["main_cases"].values()) and sys.argv[2] != "1":
    ok = False
sys.exit(0 if ok else 1)
PY
      ;;
    timeline)
      log "timeline: build_timeline --self-check"
      rc=0
      "$PYTHON" -u -m scripts.exp19.build_timeline --exp-root "$EXP_ROOT" --run "$RUN" --candidates "$CANDIDATES" \
        --self-check || rc=$?
      [[ "$rc" -eq 0 || "$rc" -eq 5 ]] || die "build_timeline failed (exit $rc)"
      # Checks 0-2 (ring vs peaks, peak vs GT with its mirror / swap controls, key rows = the bundles) are
      # conventions and must pass.  Checks 3 and 4 (turn shifts, System 1 path signs) need executed turns of both
      # signs / turn calls, which five episodes may not have: without a sample they are "not measurable".
      "$PYTHON" - "$EXP_ROOT/records_v2/timeline_self_check.json" <<'PY' || die "timeline self-check failed"
import json, sys
checks = json.load(open(sys.argv[1]))["checks"]
ok = True
for name, c in checks.items():
    if c["pass"]:
        state = "PASS"
    elif name == "3_turn_shifts_gt" and (c["left"]["n"] == 0 or c["right"]["n"] == 0):
        state = f"not measurable (left turns {c['left']['n']}, right turns {c['right']['n']})"
    elif name == "4_system1_path" and (c["n_forward_calls"] == 0 or c["share_sign_1m_equals_turn"] is None):
        state = f"not measurable ({c['n_forward_calls']} forward calls, {c['n_turn_calls']} turn calls)"
    else:
        state, ok = "FAIL", False
    print(f"[exp19-post] timeline self-check {name}: {state}")
sys.exit(0 if ok else 1)
PY
      ;;
    anim)
      if [[ -n "${EXP19_ANIM_EPISODES:-}" ]]; then
        read -r -a episodes <<< "$EXP19_ANIM_EPISODES"
      else
        read -r -a episodes <<< "$("$PYTHON" - "$EXP_ROOT/metrics/metrics.json" "$EXP_ROOT/records" <<'PY'
import json, sys
from pathlib import Path
mc = json.load(open(sys.argv[1])).get("main_cases") or {}
print(" ".join(k for k in mc.values() if k and (Path(sys.argv[2]) / f"{k}_bundle.json").is_file()))
PY
)"
      fi
      ((${#episodes[@]})) || die "no episode to animate (no main case in metrics.json; set EXP19_ANIM_EPISODES)"
      read -r -a langs <<< "$LANGS"
      # anim_<run>: the videos carry the C500 deliverables' file names, so the directory names the run they show
      anim_dir="$EXP_ROOT/figures_v2/anim_$RUN"
      log "anim: ${episodes[*]} (${langs[*]}) -> $anim_dir"
      "$PYTHON" -u -m scripts.exp19.figures.animate_v2 --records "$EXP_ROOT/records" \
        --timelines "$EXP_ROOT/records_v2" --out-dir "$anim_dir" --run "$RUN" \
        --only "${episodes[@]}" --lang "${langs[@]}" --topdown-root "$TOPDOWN" --ffmpeg "$FFMPEG" \
        --frames-dir "$EXP_ROOT/.anim_frames" --preview-dir "$anim_dir/preview"
      ;;
  esac
done
log "DONE stages='$STAGES'"

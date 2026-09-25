#!/usr/bin/env bash
# EXP-18: render every paper figure with manifest.json and README.md, via scripts/exp18/figures/make_all.py:
#   fig1_main_case       pre-registered tier-C main-figure candidate MAIN_INDEX (default 1)
#   fig2_gallery         main-text gallery, tiers C, D, E
#   fig3_metrics         quantitative figure
#   fig4_route_*         designed routes (out-and-back, loop; tier-E picks only, anything else fails)
#   supp/candidate*      all pre-registered main-figure candidates ("Main-figure candidate k of n")
#   supp/figS_gallery_all  supplementary gallery, tiers A-E
#   supp/anim_main_case  animation of the fig1 episode (.mp4 + .gif)
# each in en and zh. Choose the main case after looking at supp/candidate*: re-run with MAIN_INDEX=k
# (ONLY=fig1,anim is enough).
#
# Needs metrics.json + the slots/episodes/rows tables (compute_metrics.py) and cases.json (select_cases.py) in
# METRICS_DIR; figures whose inputs are missing are skipped and named in the manifest. After the figures,
# make_all checks the set: the figure policy on every figure's text (all Text artists, walked before each draw /
# save, and every string drawn), the captions, the label tables, README.md and manifest.json (only pattern ids
# are stored); the numbers drawn against the slots table (D1), the elevation windows (D4), the gallery frames
# (D8), frame / view labels, zh full-width parentheses (D6), colours (D7), font size. Results are under "lint"
# in manifest.json and in README.md. CPU only, no GPU.
#
# Website / dev machine:
#   cd /mnt/afs/liwenhao/agent/370910109/<staged HeatmapVLN source>
#   export EXP18_ROOT=/mnt/afs/liwenhao/agent/370910109/model/exp18_first_person_viz
#   bash scripts/exp18/run_figures.sh
# Env:
#   EXP18_ROOT      experiment root: metrics/, topdown/ (and default figures/) under it
#                   (default: scripts/exp18/common.py EXP_ROOT)
#   METRICS_DIR     default $EXP18_ROOT/metrics
#   CASES           default $METRICS_DIR/cases.json
#   OUT_DIR         default $EXP18_ROOT/figures
#   LANGS           default en,zh
#   SRC             source tree to run (default: the tree this script is in)
#   MAIN_INDEX      main-figure candidate (1-based rank in cases.json) drawn as fig1 and animated (default 1)
#   ONLY            subset of fig1,supp,fig2,fig3,fig4,anim (default all; fig2 draws both galleries)
#   DUMPS_ROOT      <root>/<tier>/<scene>/<clip>.npz (default: the npz_path recorded in cases.json)
#   TOPDOWN_ROOT    default $EXP18_ROOT/topdown
#   CLIP_ROOT       local copy of the clips (default: the clip_dir recorded in each dump)
#   ANIM_SIZE       default 1920x1080
#   CLEAN=1         first delete the files the previous manifest.json in OUT_DIR listed
#   DRAFT=1         development run: check errors are recorded (manifest.json, README.md) but do not set the
#                   exit status (a figure that failed still does)
#   EXP18_GIT_SHA   code version for the manifest (else $SRC/.exp18_git_sha, else git)
#   EXP18_FONT_DIR / EXP18_CJK_FONT   paper fonts when /usr/share/fonts lacks them (a blank
#                   container may); the manifest records which fonts were found
#   QWEN_PYTHON     default <workspace>/envs/qwen25/bin/python
# Exit status: make_all's (0 all figures drawn and every check passed; 1 a figure failed -- an exception, a
# dropped note or an unaccounted slot -- or a check found an error; 2 bad arguments).
set -euo pipefail

WORKSPACE=${EXP18_WORKSPACE:-/mnt/afs/liwenhao/agent/370910109}
SRC=${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
QWEN_PYTHON=${QWEN_PYTHON:-${WORKSPACE}/envs/qwen25/bin/python}
[[ -f "${SRC}/scripts/exp18/figures/make_all.py" ]] || { echo "[exp18-figures] no make_all.py under SRC=${SRC}" >&2; exit 2; }
[[ -x "${QWEN_PYTHON}" ]] || { echo "[exp18-figures] python not found: ${QWEN_PYTHON}" >&2; exit 2; }

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SRC}${PYTHONPATH:+:${PYTHONPATH}}"
# matplotlib needs writable config / font caches; keep them out of the output dir
SCRATCH=${TMPDIR:-/tmp}/exp18_figures_$(id -u)
mkdir -p "${SCRATCH}/mpl" "${SCRATCH}/ttf"
export MPLCONFIGDIR=${MPLCONFIGDIR:-${SCRATCH}/mpl}
export EXP18_FONT_CACHE=${EXP18_FONT_CACHE:-${SCRATCH}/ttf}
[[ -n "${EXP18_ROOT:-}" ]] && export EXP18_ROOT

ARGS=(--langs "${LANGS:-en,zh}" --main-index "${MAIN_INDEX:-1}")
[[ -n "${EXP18_ROOT:-}" ]] && ARGS+=(--exp-root "${EXP18_ROOT}")
[[ -n "${METRICS_DIR:-}" ]] && ARGS+=(--metrics-dir "${METRICS_DIR}")
[[ -n "${CASES:-}" ]] && ARGS+=(--cases "${CASES}")
[[ -n "${OUT_DIR:-}" ]] && ARGS+=(--out-dir "${OUT_DIR}")
[[ -n "${ONLY:-}" ]] && ARGS+=(--only "${ONLY}")
[[ -n "${DUMPS_ROOT:-}" ]] && ARGS+=(--dumps-root "${DUMPS_ROOT}")
[[ -n "${TOPDOWN_ROOT:-}" ]] && ARGS+=(--topdown-root "${TOPDOWN_ROOT}")
[[ -n "${CLIP_ROOT:-}" ]] && ARGS+=(--clip-root "${CLIP_ROOT}")
[[ -n "${ANIM_SIZE:-}" ]] && ARGS+=(--anim-size "${ANIM_SIZE}")
[[ "${CLEAN:-0}" == "1" ]] && ARGS+=(--clean)
[[ "${DRAFT:-0}" == "1" ]] && ARGS+=(--draft)

echo "[exp18-figures] $(date -u +%FT%TZ) src=${SRC} git_sha=${EXP18_GIT_SHA:-$(cat "${SRC}/.exp18_git_sha" 2>/dev/null || echo from-git-or-unknown)}"
echo "[exp18-figures] ${QWEN_PYTHON} -m scripts.exp18.figures.make_all ${ARGS[*]}"
cd "${SRC}"
set +e
"${QWEN_PYTHON}" -m scripts.exp18.figures.make_all "${ARGS[@]}"
STATUS=$?
set -e
echo "[exp18-figures] $(date -u +%FT%TZ) exit ${STATUS}"
exit "${STATUS}"

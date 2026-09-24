#!/usr/bin/env bash
# EXP-18: render every paper figure (fig1-fig4, the supplementary candidates and
# the animation) with manifest.json and README.md, via scripts/exp18/figures/make_all.py.
#
# Needs metrics.json + the slots/episodes/rows tables (compute_metrics.py) and
# cases.json (select_cases.py) in METRICS_DIR; figures whose inputs are missing
# are skipped and named in the manifest. Before tier C is scored, fig1, the
# candidates and the animation are a stand-in: the main-figure rule applied to
# tier B, gallery picks excluded (README.md and the captions say so). After the
# figures, make_all checks the set (policy words in every string drawn and in the
# captions, terminology, zh parentheses, frame / view label consistency, the
# affordance-map elevation window, colours, font size, notes left out); the
# results are under "lint" in manifest.json and in README.md. CPU only, no GPU.
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
#   MAIN_INDEX      main-figure candidate (1-based rank) drawn as fig1 (default 1)
#   ONLY            subset of fig1,supp,fig2,fig3,fig4,anim (default all)
#   DUMPS_ROOT      <root>/<tier>/<scene>/<clip>.npz (default: the npz_path recorded in cases.json)
#   TOPDOWN_ROOT    default $EXP18_ROOT/topdown
#   CLIP_ROOT       local copy of the clips (default: the clip_dir recorded in each dump)
#   ANIM_SIZE       default 1920x1080
#   CASE_LAYOUT     fig_case layout of fig1 + candidates: revised (default, CaseOptions.revised():
#                   notes wrap instead of being dropped, misses numbered under the row) or approved
#   CASE_OPTIONS    single fig_case.CaseOptions fields on top, e.g. merge_notes,clamp_peaks,letters=slide
#   CLEAN=1         first delete the files the previous manifest.json in OUT_DIR listed
#   DRAFT=1         development run: check errors and notes left out of a figure are recorded
#                   (manifest.json, README.md) but do not set the exit status
#   EXP18_GIT_SHA   code version for the manifest (else $SRC/.exp18_git_sha, else git)
#   EXP18_FONT_DIR / EXP18_CJK_FONT   paper fonts when /usr/share/fonts lacks them (a blank
#                   container may); the manifest records which fonts were found
#   QWEN_PYTHON     default <workspace>/envs/qwen25/bin/python
# Exit status: make_all's (0 all figures drawn and every check passed; 1 a figure failed, a note
# was left out of a figure, or a check found an error; 2 bad arguments).
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
[[ -n "${CASE_LAYOUT:-}" ]] && ARGS+=(--case-layout "${CASE_LAYOUT}")
[[ -n "${CASE_OPTIONS:-}" ]] && ARGS+=(--case-options "${CASE_OPTIONS}")
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

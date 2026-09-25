#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# EXP-19 figures v2 (paper-grade): episode pages + overview figures.
#
# Reads <EXP19_ROOT>/records/<ep_key>_bundle.{json,npz} + <ep_key>.json (v1
# bundles and episode records, read-only), <EXP19_ROOT>/records_v2/
# <ep_key>_timeline.{json,npz} (scripts/exp19/build_timeline.py) and
# metrics/metrics.json (caption claims), and writes <EXP19_ROOT>/figures_v2/:
#   <category>/<rank>_<ep_key>_{en,zh}.{pdf,png} + _caption_{en,zh}.txt   (one page per episode, 7.0 in wide)
#   main/main_{T,F}_{en,zh}.{pdf,png} + main_{T,F}_caption_{en,zh}.txt  (overviews of the is_main cases)
#   manifest.json (source sha256s, sizes, smallest font, checks, warnings)
# Never writes into the v1 figures dir.  EXP19_FIG_ANIM=1 also runs the
# animations (scripts/exp19/figures/animate_v2.py) when that module exists.
#
# CPU only (envs/qwen25, matplotlib).  Fonts: Nimbus Sans + Droid Sans
# Fallback (exp18 style); a blank container has neither, so point
# EXP18_FONT_DIR / EXP18_CJK_FONT at copies on /mnt/afs, or run on the dev
# machine.  Missing fonts stop the run unless EXP19_FIG_ALLOW_FALLBACK_FONTS=1.
#
# Dev machine (a few minutes):
#   cd <repo>; bash scripts/exp19/run_figures_v2.sh
#
# Env: EXP19_ROOT (default <workspace>/model/exp19_behavior_viz),
#   EXP19_FIG_OUT (default <EXP19_ROOT>/figures_v2), EXP19_FIG_LANGS ("en zh"),
#   EXP19_FIG_ONLY (space-separated ep_keys; default all), EXP19_FIG_NO_VERDICTS (0),
#   EXP19_TOPDOWN_ROOT, EXP19_FIG_ALLOW_VOID (0), EXP19_FIG_ALLOW_FALLBACK_FONTS (0),
#   EXP19_FIG_ANIM (0).
# ============================================================

ROOT="/mnt/afs/liwenhao/agent/370910109"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP19_ROOT="${EXP19_ROOT:-${ROOT}/model/exp19_behavior_viz}"
PYTHON="${ROOT}/envs/qwen25/bin/python"
RECORDS="${EXP19_ROOT}/records"
TIMELINES="${EXP19_ROOT}/records_v2"
METRICS="${EXP19_ROOT}/metrics/metrics.json"
OUT="${EXP19_FIG_OUT:-${EXP19_ROOT}/figures_v2}"
LANGS="${EXP19_FIG_LANGS:-en zh}"

export PYTHONDONTWRITEBYTECODE=1

if [[ "$(basename "$OUT")" == "figures" ]]; then
  echo "[ERROR] ${OUT} looks like the v1 figures dir; v2 writes to figures_v2" >&2
  exit 1
fi
for path in "$PYTHON" "$RECORDS" "$TIMELINES" "${REPO_ROOT}/scripts/exp19/figures/fig_v2.py"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 1
  fi
done
n_bundles=$(compgen -G "${RECORDS}/*_bundle.json" | wc -l)
n_timelines=$(compgen -G "${TIMELINES}/*_timeline.npz" | wc -l || true)
if [[ "$n_bundles" -eq 0 || "$n_timelines" -lt "$n_bundles" ]]; then
  echo "[ERROR] ${n_bundles} bundles but ${n_timelines} timelines (run scripts/exp19/build_timeline.py first)" >&2
  exit 1
fi

args=(--records "$RECORDS" --timelines "$TIMELINES" --out-dir "$OUT")
if [[ "${EXP19_FIG_NO_VERDICTS:-0}" == 1 ]]; then
  args+=(--no-verdicts)
else
  if [[ ! -f "$METRICS" ]]; then
    echo "[ERROR] ${METRICS} missing: captions need the verdicts (EXP19_FIG_NO_VERDICTS=1 for a layout run)" >&2
    exit 1
  fi
  args+=(--metrics "$METRICS")
fi
# shellcheck disable=SC2206
args+=(--lang ${LANGS})
if [[ -n "${EXP19_FIG_ONLY:-}" ]]; then
  # shellcheck disable=SC2206
  args+=(--only ${EXP19_FIG_ONLY})
fi
if [[ -n "${EXP19_TOPDOWN_ROOT:-}" ]]; then
  args+=(--topdown-root "$EXP19_TOPDOWN_ROOT")
fi
if [[ "${EXP19_FIG_ALLOW_VOID:-0}" == 1 ]]; then
  args+=(--allow-void)
fi

# Paper fonts (the same lookup as scripts/exp18/figures/style.py).
LATIN_OK=0
for dir in /usr/share/fonts/opentype/urw-base35 "${EXP18_FONT_DIR:-}"; do
  if [[ -n "$dir" ]] && compgen -G "${dir}/NimbusSans-*.otf" > /dev/null; then LATIN_OK=1; fi
done
CJK_OK=0
for font in /usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf "${EXP18_CJK_FONT:-}"; do
  if [[ -n "$font" && -f "$font" ]]; then CJK_OK=1; fi
done
if [[ "$LATIN_OK" != 1 || ( "$CJK_OK" != 1 && " ${LANGS} " == *" zh "* ) ]]; then
  msg="paper fonts missing (Nimbus Sans: ${LATIN_OK}, Droid Sans Fallback: ${CJK_OK}); set EXP18_FONT_DIR / EXP18_CJK_FONT"
  if [[ "${EXP19_FIG_ALLOW_FALLBACK_FONTS:-0}" == 1 ]]; then
    echo "[WARN] ${msg}; continuing with fallback fonts" >&2
  else
    echo "[ERROR] ${msg}, or EXP19_FIG_ALLOW_FALLBACK_FONTS=1" >&2
    exit 1
  fi
fi

mkdir -p "$OUT"
LOG="${OUT}/figures_v2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "EXP-19 behaviour figures v2"
echo "============================================================"
echo "Code:      $REPO_ROOT (git sha: $(git -c safe.directory="$REPO_ROOT" -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || cat "${REPO_ROOT}/.exp19_git_sha" 2>/dev/null || echo unknown))"
echo "Bundles:   $RECORDS (${n_bundles})"
echo "Timelines: $TIMELINES (${n_timelines})"
echo "Verdicts:  $([[ "${EXP19_FIG_NO_VERDICTS:-0}" == 1 ]] && echo 'none (layout run)' || echo "$METRICS")"
echo "Output:    $OUT"
echo "Langs:     $LANGS"
echo "Log:       $LOG"
echo "============================================================"

cd "$REPO_ROOT"
"$PYTHON" -m scripts.exp19.figures.fig_v2 "${args[@]}"

if [[ "${EXP19_FIG_ANIM:-0}" == 1 ]]; then
  if [[ -f "${REPO_ROOT}/scripts/exp19/figures/animate_v2.py" ]]; then
    "$PYTHON" -m scripts.exp19.figures.animate_v2 --records "$RECORDS" --timelines "$TIMELINES" --out-dir "${OUT}/anim"
  else
    echo "[WARN] EXP19_FIG_ANIM=1 but scripts/exp19/figures/animate_v2.py does not exist; no animations" >&2
  fi
fi
echo "[DONE] figures in $OUT (manifest.json lists files, sources, sizes, smallest font and warnings)"

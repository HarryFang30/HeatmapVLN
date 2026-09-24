#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# EXP-19 [G]: behaviour figures from the figure bundles.
#
# Reads <EXP19_ROOT>/records/*_bundle.{json,npz} (build_records.py) and the
# EXP-18 top-down maps, and writes <EXP19_ROOT>/figures/:
#   <category>/<rank>_<ep_key>_{en,zh}.{pdf,png} + _caption_{en,zh}.txt  (one page per episode)
#   main/main_{T,F}_{en,zh}.{pdf,png} + main_{T,F}_caption_{en,zh}.txt
#       (the is_main cases: successes T1-T3 and failures F1-F2, one figure each)
#   manifest.json  (bundle sha256s, the verdicts the captions were built from, sizes, warnings)
# Caption claims about H1/H2/H3 come only from metrics/metrics.json "verdicts"
# (fig_behavior.claim_sentences = the pre-registered wording rules).  When
# metrics.json says the batch is void (a validity gate failed), nothing is drawn
# (exit 3) unless EXP19_FIG_ALLOW_VOID=1, which stamps every page "VOID BATCH".
#
# CPU only (envs/qwen25, matplotlib).  The paper fonts are Nimbus Sans
# (/usr/share/fonts/opentype/urw-base35) and Droid Sans Fallback for Chinese;
# a blank container has neither, so point EXP18_FONT_DIR / EXP18_CJK_FONT at
# copies on /mnt/afs, or run this on the dev machine.  Missing fonts stop the
# run unless EXP19_FIG_ALLOW_FALLBACK_FONTS=1 (DejaVu, boxes for Chinese).
#
# Dev machine (a few minutes; no GPU):
#   cd /mnt/afs/liwenhao/agent/370910109/model/exp19_behavior_viz/src_<sha>
#   bash scripts/exp19/run_figures.sh
#
# Env: EXP19_ROOT (default <workspace>/model/exp19_behavior_viz),
#   EXP19_FIG_OUT (default <EXP19_ROOT>/figures), EXP19_FIG_LANGS ("en zh"),
#   EXP19_FIG_MAIN (1 = also the main figure), EXP19_FIG_NO_VERDICTS (0; 1 =
#   layout run without metrics.json, captions carry no claims),
#   EXP19_TOPDOWN_ROOT (default: the root recorded in each bundle),
#   EXP19_FIG_ALLOW_VOID (0), EXP19_FIG_ALLOW_FALLBACK_FONTS (0).
# ============================================================

ROOT="/mnt/afs/liwenhao/agent/370910109"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP19_ROOT="${EXP19_ROOT:-${ROOT}/model/exp19_behavior_viz}"
PYTHON="${ROOT}/envs/qwen25/bin/python"
RECORDS="${EXP19_ROOT}/records"
METRICS="${EXP19_ROOT}/metrics/metrics.json"
OUT="${EXP19_FIG_OUT:-${EXP19_ROOT}/figures}"
LANGS="${EXP19_FIG_LANGS:-en zh}"

export PYTHONDONTWRITEBYTECODE=1

for path in "$PYTHON" "$RECORDS" "${REPO_ROOT}/scripts/exp19/figures/fig_behavior.py"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 1
  fi
done
if ! compgen -G "${RECORDS}/*_bundle.json" > /dev/null; then
  echo "[ERROR] no figure bundles in ${RECORDS} (run build_records.py first)" >&2
  exit 1
fi

args=(--records "$RECORDS" --out-dir "$OUT")
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
if [[ "${EXP19_FIG_MAIN:-1}" == 1 ]]; then
  args+=(--main)
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
LOG="${OUT}/figures_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "EXP-19 behaviour figures"
echo "============================================================"
echo "Code:     $REPO_ROOT (git sha: $(cat "${REPO_ROOT}/.exp19_git_sha" 2>/dev/null || echo unknown))"
echo "Bundles:  $RECORDS ($(compgen -G "${RECORDS}/*_bundle.json" | wc -l))"
echo "Verdicts: $([[ "${EXP19_FIG_NO_VERDICTS:-0}" == 1 ]] && echo 'none (layout run)' || echo "$METRICS")"
echo "Output:   $OUT"
echo "Langs:    $LANGS"
echo "Log:      $LOG"
echo "============================================================"

cd "$REPO_ROOT"
"$PYTHON" -m scripts.exp19.figures.fig_behavior "${args[@]}"
echo "[DONE] figures in $OUT (manifest.json lists files, sources and warnings)"

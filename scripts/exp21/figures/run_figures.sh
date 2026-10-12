#!/usr/bin/env bash
# EXP-21 sensitivity figure (English), from the committed results JSON.
#
# The JSON holds the 18 paired-bootstrap results (both platforms, nine points each)
# and the deployment values, so the figure rebuilds without the evaluation boxes.
# Its entries are the per-point boot_<platform>_<point>.json outputs; how those are
# made is in docs/experiments/exp21-sensitivity-runbook.md (分析).
#
# Fonts: Nimbus Sans is read from /usr/share/fonts/opentype/urw-base35 (present on
# the 4090 host, not in its container) or from $EXP18_FONT_DIR; with neither, the
# figure falls back to DejaVu Sans.
set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT="${1:-figures/exp21}"
PYTHON="${PYTHON:-python3}"
PYTHONPATH="$PWD" "$PYTHON" -m scripts.exp21.figures.fig_sensitivity \
  --results docs/experiments/exp21-sensitivity-results.json \
  --out-dir "$OUT"
echo "figures in $OUT"

#!/usr/bin/env bash
# EXP-21 sensitivity figure, both languages, from the committed results JSON.
#
# The JSON holds the 18 paired-bootstrap results (both platforms, nine points each),
# so the figure rebuilds without the evaluation boxes.  To regenerate the JSON after
# a rerun, see docs/experiments/exp21-sensitivity-report.md.
set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT="${1:-figures/exp21}"
PYTHON="${PYTHON:-python3}"
PYTHONPATH="$PWD" "$PYTHON" -m scripts.exp21.figures.fig_sensitivity \
  --results docs/experiments/exp21-sensitivity-results.json \
  --out-dir "$OUT" --lang en zh
echo "figures in $OUT"

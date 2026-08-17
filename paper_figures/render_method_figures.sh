#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHROME_BIN="${CHROME_BIN:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
RASTER_WIDTH_PX="${RASTER_WIDTH_PX:-3560}"
RASTER_DPI="${RASTER_DPI:-508}"
PAPER_WIDTH_MM="178"

python_bin="$(command -v python3 || true)"
if [[ -z "${python_bin}" ]]; then
  echo "python3 is required to form file URLs" >&2
  exit 1
fi

if [[ ! -x "${CHROME_BIN}" ]]; then
  echo "Chrome executable not found: ${CHROME_BIN}" >&2
  exit 1
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/heatmapvln-paper-figures.XXXXXX")"
trap 'rm -rf "${tmp_dir}"' EXIT

svg_dimensions() {
  local svg_path="$1"
  sed -nE 's/.*viewBox="0 0 ([0-9.]+) ([0-9.]+)".*/\1 \2/p' "${svg_path}" | head -n 1
}

file_url() {
  local file_path="$1"
  "${python_bin}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve().as_uri())' "${file_path}"
}

render_one() {
  local stem="$1"
  local svg_path="${SCRIPT_DIR}/${stem}.svg"
  local png_path="${SCRIPT_DIR}/${stem}.png"
  local pdf_path="${SCRIPT_DIR}/${stem}.pdf"
  local dimensions
  local view_width
  local view_height
  local raster_height
  local paper_height_mm
  local svg_url
  local raster_html="${tmp_dir}/${stem}_raster.html"
  local print_html="${tmp_dir}/${stem}_print.html"

  [[ -f "${svg_path}" ]] || { echo "Missing source SVG: ${svg_path}" >&2; exit 1; }
  dimensions="$(svg_dimensions "${svg_path}")"
  [[ -n "${dimensions}" ]] || { echo "Cannot parse viewBox: ${svg_path}" >&2; exit 1; }
  read -r view_width view_height <<<"${dimensions}"
  svg_url="$(file_url "${svg_path}")"
  raster_height="$(awk -v w="${view_width}" -v h="${view_height}" -v rw="${RASTER_WIDTH_PX}" 'BEGIN { printf "%d", (rw*h/w)+0.5 }')"
  paper_height_mm="$(awk -v w="${view_width}" -v h="${view_height}" -v mm="${PAPER_WIDTH_MM}" 'BEGIN { printf "%.4f", mm*h/w }')"

  printf '%s\n' \
    '<!doctype html><meta charset="utf-8">' \
    '<style>html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#fff}img{display:block;width:100%;height:100%;object-fit:fill}</style>' \
    "<img src=\"${svg_url}\">" >"${raster_html}"

  "${CHROME_BIN}" --headless=new --disable-gpu --hide-scrollbars \
    --allow-file-access-from-files --force-device-scale-factor=1 \
    --window-size="${RASTER_WIDTH_PX},${raster_height}" \
    --screenshot="${png_path}" "file://${raster_html}" >/dev/null 2>&1
  sips -s dpiWidth "${RASTER_DPI}" -s dpiHeight "${RASTER_DPI}" "${png_path}" >/dev/null

  printf '%s\n' \
    '<!doctype html><meta charset="utf-8">' \
    "<style>@page{size:${PAPER_WIDTH_MM}mm ${paper_height_mm}mm;margin:0}html,body{margin:0;width:${PAPER_WIDTH_MM}mm;height:${paper_height_mm}mm;overflow:hidden;background:#fff}img{display:block;width:${PAPER_WIDTH_MM}mm;height:${paper_height_mm}mm}</style>" \
    "<img src=\"${svg_url}\">" >"${print_html}"

  "${CHROME_BIN}" --headless=new --disable-gpu --allow-file-access-from-files \
    --no-pdf-header-footer --print-to-pdf="${pdf_path}" "file://${print_html}" >/dev/null 2>&1
}

render_one "fig_method_overview"
render_one "fig_heatmap_module"

# Keep the previous main-paper filenames synchronized for any existing LaTeX
# include paths while the canonical names above remain the source of truth.
cp "${SCRIPT_DIR}/fig_method_overview.svg" "${SCRIPT_DIR}/fig1_overall_navigation_architecture_cvpr.svg"
cp "${SCRIPT_DIR}/fig_method_overview.png" "${SCRIPT_DIR}/fig1_overall_navigation_architecture_cvpr.png"
cp "${SCRIPT_DIR}/fig_method_overview.pdf" "${SCRIPT_DIR}/fig1_overall_navigation_architecture_cvpr.pdf"
cp "${SCRIPT_DIR}/fig_heatmap_module.svg" "${SCRIPT_DIR}/fig2_amb3r_conditioned_heatmap_head_cvpr.svg"
cp "${SCRIPT_DIR}/fig_heatmap_module.png" "${SCRIPT_DIR}/fig2_amb3r_conditioned_heatmap_head_cvpr.png"
cp "${SCRIPT_DIR}/fig_heatmap_module.pdf" "${SCRIPT_DIR}/fig2_amb3r_conditioned_heatmap_head_cvpr.pdf"

gray_profile="/System/Library/ColorSync/Profiles/Generic Gray Gamma 2.2 Profile.icc"
if [[ -f "${gray_profile}" ]]; then
  for stem in fig_method_overview fig_heatmap_module; do
    gray_path="${SCRIPT_DIR}/${stem}_grayscale_preview.png"
    sips -m "${gray_profile}" "${SCRIPT_DIR}/${stem}.png" --out "${gray_path}" >/dev/null
    sips -s dpiWidth "${RASTER_DPI}" -s dpiHeight "${RASTER_DPI}" "${gray_path}" >/dev/null
  done
fi

overview_dims="$(svg_dimensions "${SCRIPT_DIR}/fig_method_overview.svg")"
module_dims="$(svg_dimensions "${SCRIPT_DIR}/fig_heatmap_module.svg")"
read -r overview_w overview_h <<<"${overview_dims}"
read -r module_w module_h <<<"${module_dims}"
overview_px_h="$(awk -v w="${overview_w}" -v h="${overview_h}" -v rw="${RASTER_WIDTH_PX}" 'BEGIN { printf "%d", (rw*h/w)+0.5 }')"
module_px_h="$(awk -v w="${module_w}" -v h="${module_h}" -v rw="${RASTER_WIDTH_PX}" 'BEGIN { printf "%d", (rw*h/w)+0.5 }')"
preview_gap=160
preview_height=$((overview_px_h + preview_gap + module_px_h))
preview_html="${tmp_dir}/figures_178mm_preview.html"
preview_png="${SCRIPT_DIR}/figures_178mm_preview.png"
colorblind_html="${tmp_dir}/figures_deuteranopia_preview.html"
colorblind_png="${SCRIPT_DIR}/figures_deuteranopia_preview.png"
overview_url="$(file_url "${SCRIPT_DIR}/fig_method_overview.svg")"
module_url="$(file_url "${SCRIPT_DIR}/fig_heatmap_module.svg")"

printf '%s\n' \
  '<!doctype html><meta charset="utf-8">' \
  "<style>html,body{margin:0;width:${RASTER_WIDTH_PX}px;height:${preview_height}px;overflow:hidden;background:#fff}.gap{height:${preview_gap}px;border-top:2px solid #d8dcdf;border-bottom:2px solid #d8dcdf;box-sizing:border-box}img{display:block;width:${RASTER_WIDTH_PX}px;object-fit:fill}</style>" \
  "<img style=\"height:${overview_px_h}px\" src=\"${overview_url}\">" \
  '<div class="gap"></div>' \
  "<img style=\"height:${module_px_h}px\" src=\"${module_url}\">" >"${preview_html}"

"${CHROME_BIN}" --headless=new --disable-gpu --hide-scrollbars \
  --allow-file-access-from-files --force-device-scale-factor=1 \
  --window-size="${RASTER_WIDTH_PX},${preview_height}" \
  --screenshot="${preview_png}" "file://${preview_html}" >/dev/null 2>&1
sips -s dpiWidth "${RASTER_DPI}" -s dpiHeight "${RASTER_DPI}" "${preview_png}" >/dev/null

module_y=$((overview_px_h + preview_gap))
printf '%s\n' \
  '<!doctype html><meta charset="utf-8"><style>html,body{margin:0;overflow:hidden;background:#fff}</style>' \
  "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"${RASTER_WIDTH_PX}\" height=\"${preview_height}\" viewBox=\"0 0 ${RASTER_WIDTH_PX} ${preview_height}\">" \
  '<defs><filter id="deuteranopia" color-interpolation-filters="sRGB"><feColorMatrix type="matrix" values="0.367 0.861 -0.228 0 0  0.280 0.673 0.047 0 0  -0.012 0.043 0.969 0 0  0 0 0 1 0"/></filter></defs>' \
  "<rect width=\"${RASTER_WIDTH_PX}\" height=\"${preview_height}\" fill=\"#fff\"/>" \
  "<image href=\"${overview_url}\" x=\"0\" y=\"0\" width=\"${RASTER_WIDTH_PX}\" height=\"${overview_px_h}\" filter=\"url(#deuteranopia)\"/>" \
  "<rect x=\"0\" y=\"${overview_px_h}\" width=\"${RASTER_WIDTH_PX}\" height=\"${preview_gap}\" fill=\"#fff\" stroke=\"#d8dcdf\" stroke-width=\"2\"/>" \
  "<image href=\"${module_url}\" x=\"0\" y=\"${module_y}\" width=\"${RASTER_WIDTH_PX}\" height=\"${module_px_h}\" filter=\"url(#deuteranopia)\"/>" \
  '</svg>' >"${colorblind_html}"

"${CHROME_BIN}" --headless=new --disable-gpu --hide-scrollbars \
  --allow-file-access-from-files --force-device-scale-factor=1 \
  --window-size="${RASTER_WIDTH_PX},${preview_height}" \
  --screenshot="${colorblind_png}" "file://${colorblind_html}" >/dev/null 2>&1
sips -s dpiWidth "${RASTER_DPI}" -s dpiHeight "${RASTER_DPI}" "${colorblind_png}" >/dev/null

echo "Rendered SVG, vector PDF, ${RASTER_DPI}-dpi PNG, 178 mm, grayscale, and deuteranopia previews in ${SCRIPT_DIR}"

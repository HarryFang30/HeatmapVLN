# HeatmapVLN paper figures

## CVPR main-paper assets

The camera-ready figure set is generated at the final double-column width of 178 mm:

- `fig_method_overview.svg` / `.pdf` / `.png` — editable system overview and exports.
- `fig_heatmap_module.svg` / `.pdf` / `.png` — editable module/injection detail and exports.
- `figures_178mm_preview.png` — both figures rendered at the same 178 mm reference width.
- `fig_method_overview_grayscale_preview.png` / `fig_heatmap_module_grayscale_preview.png` — grayscale legibility checks.
- `figures_deuteranopia_preview.png` — a deuteranopia simulation; labels and line styles remain the primary status cues.
- `architecture_figure_audit.md` — code-grounded semantics, tensor contracts, freezing state, evidence, and unresolved provenance.
- `figure_captions.md` — concise English captions.
- `render_method_figures.sh` — deterministic local export script.

Regenerate the PNG, PDF, and combined preview with:

```bash
bash paper_figures/render_method_figures.sh
```

The script uses headless Chrome to retain SVG vectors in the PDF and exports the PNG previews at 508 dpi. The SVG masters declare `178mm` width directly; normal labels are designed at no less than 8 pt at that size.

## Visual and semantic conventions

- Neutral gray/white: existing frozen InternNav or Qwen visual components.
- Blue: frozen external pose estimation (AMB3R-VO).
- Green: proposed learned modules and the residual control path.
- Solid arrows: inference-time data flow.
- Dashed arrows: training-only supervision.
- F/R/B/L: Front, Right, Back, and Left output coordinate frames. They are not four RGB inputs to the heatmap branch.

The two figures intentionally separate roles: the overview shows where explicit historical spatial grounding enters the original navigation system; the detailed figure shows how per-history pose/visual evidence produces structured heatmap tokens and how those tokens enter each frozen NextDiT block.

## Submission caveat

The figures describe the intended causal AMB3R-conditioned deployment path. The source tree establishes the architecture, interfaces, and prepared pose-adaptation recipe, but a camera-ready empirical claim must still bind the reported run to explicit AMB3R, heatmap-head, and control checkpoint provenance. See `architecture_figure_audit.md`.

## Compatibility aliases and archived drafts

The render script synchronizes the previous `_cvpr` filenames with the canonical masters so existing LaTeX include paths continue to work. The unsuffixed `fig1_*` / `fig2_*` files and the `_v2` files are earlier design iterations retained for comparison; they are not the recommended paper assets.

#!/usr/bin/env python3
"""
HTML index generation for VLN heatmap evaluation results.

Provides functions to create browsable HTML pages with thumbnails and metrics.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_html_index(
    samples_info: list[dict],
    output_dir: Path,
    metrics_summary: dict[str, float]
) -> None:
    """
    Generate HTML index page for evaluation results.

    Args:
        samples_info: List of sample metadata dictionaries
        output_dir: Output directory (eval_vis root)
        metrics_summary: Summary metrics dictionary
    """
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>VLN Heatmap Evaluation Results</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .summary {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .metric-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #007bff;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        .samples-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}
        .sample-card {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .sample-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }}
        .sample-header {{
            background: #007bff;
            color: white;
            padding: 12px 15px;
            font-weight: 600;
        }}
        .thumbnail {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            background: #eee;
        }}
        .sample-body {{
            padding: 15px;
        }}
        .instruction {{
            font-style: italic;
            color: #555;
            margin: 12px 0;
            padding: 10px;
            background: #f8f9fa;
            border-left: 3px solid #007bff;
            border-radius: 4px;
            font-size: 0.95em;
            line-height: 1.4;
        }}
        .sample-meta {{
            color: #666;
            font-size: 0.9em;
            margin: 8px 0;
        }}
        .sample-meta strong {{
            color: #333;
        }}
        .links {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #eee;
        }}
        .links a {{
            display: inline-block;
            margin-right: 10px;
            padding: 8px 15px;
            background: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 0.9em;
            transition: background 0.2s;
        }}
        .links a:hover {{
            background: #0056b3;
        }}
        .links a.secondary {{
            background: #6c757d;
        }}
        .links a.secondary:hover {{
            background: #545b62;
        }}
        .filter-bar {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .filter-bar label {{
            margin-right: 10px;
            color: #666;
        }}
        .filter-bar select {{
            padding: 6px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-right: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🗺️ VLN Heatmap Evaluation Results</h1>

        <div class="summary">
            <h2>Summary Statistics</h2>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-label">Total Samples</div>
                    <div class="metric-value">{len(samples_info)}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Avg History Loss</div>
                    <div class="metric-value">{metrics_summary.get('history_loss', 0):.4f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Avg Future Loss</div>
                    <div class="metric-value">{metrics_summary.get('future_loss', 0):.4f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Avg Total Loss</div>
                    <div class="metric-value">{metrics_summary.get('total_loss', 0):.4f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Hist Peak Error (px)</div>
                    <div class="metric-value">{metrics_summary.get('hist_peak_error', 0):.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Hist IoU</div>
                    <div class="metric-value">{metrics_summary.get('hist_iou', 0):.4f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Fut Peak Error (px)</div>
                    <div class="metric-value">{metrics_summary.get('fut_peak_error', 0):.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Fut IoU</div>
                    <div class="metric-value">{metrics_summary.get('fut_iou', 0):.4f}</div>
                </div>
            </div>
        </div>

        <div class="samples-grid">
"""

    # Generate sample cards
    for sample in samples_info:
        # Generate grid links based on overlay_mode
        grid_links = []
        if 'dual' in sample['grid_paths']:
            grid_links.append(f'<a href="{sample["grid_paths"]["dual"]}" target="_blank">View Grid (Dual)</a>')
        if 'history' in sample['grid_paths']:
            grid_links.append(f'<a href="{sample["grid_paths"]["history"]}" target="_blank" class="secondary">History Grid</a>')
        if 'future' in sample['grid_paths']:
            grid_links.append(f'<a href="{sample["grid_paths"]["future"]}" target="_blank" class="secondary">Future Grid</a>')
        if 'full' in sample['grid_paths']:
            grid_links.append(f'<a href="{sample["grid_paths"]["full"]}" target="_blank">Full Grid (7-col)</a>')

        links_html = '\n                '.join(grid_links)

        # Sample metrics
        metrics = sample.get('metrics', {})
        mae_hist = metrics.get('mae_hist', 0)
        mae_fut = metrics.get('mae_fut', 0)

        html_content += f"""
            <div class="sample-card">
                <div class="sample-header">
                    #{sample['idx']:04d}: {sample.get('scene_name', 'unknown')}
                </div>
                <img src="{sample.get('thumbnail_path', '')}" alt="Thumbnail" class="thumbnail" onerror="this.style.display='none'">
                <div class="sample-body">
                    <div class="instruction">
                        📝 {sample.get('instruction', 'No instruction provided')}
                    </div>
                    <div class="sample-meta">
                        <strong>Frames:</strong> {sample.get('num_frames', 0)} |
                        <strong>Valid Hist:</strong> {sample.get('valid_hist', 0)} |
                        <strong>Valid Fut:</strong> {sample.get('valid_fut', 0)}
                    </div>
                    <div class="sample-meta">
                        <strong>MAE Hist:</strong> {mae_hist:.4f} |
                        <strong>MAE Fut:</strong> {mae_fut:.4f}
                    </div>
                    <div class="links">
                        {links_html}
                    </div>
                </div>
            </div>
"""

    html_content += """
        </div>
    </div>
</body>
</html>
"""

    # Write HTML file
    index_path = output_dir / 'index.html'
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Save summary metrics to JSON
    metrics_path = output_dir / 'summary_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("Created HTML index: %s", index_path)
    logger.info("Saved summary metrics: %s", metrics_path)

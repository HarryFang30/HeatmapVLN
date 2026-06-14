#!/usr/bin/env python3
"""
Generate a self-contained HTML trajectory inspector from ``trajectory_steps.json``.

Usage::

    python scripts/visualization/generate_trajectory_html.py \
        --input output_path/zsNo4HB9uLZ_0001/trajectory_steps.json \
        --output output_path/zsNo4HB9uLZ_0001/trajectory.html

Batch mode::

    python scripts/visualization/generate_trajectory_html.py \
        --input-dir output_path/ \
        --output-dir output_path/html/
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


# ── helpers ────────────────────────────────────────────────────────────

def _img_to_b64(img: Image.Image, fmt: str = "JPEG", quality: int = 60) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def _load_data(json_path: str) -> dict[str, Any]:
    root = Path(json_path).parent
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    for step in data.get("steps", []):
        pano = step.get("panorama") or {}
        loaded: dict[str, str] = {}
        for view in ("front", "right", "back", "left"):
            fname = pano.get(view)
            if fname:
                img_path = root / fname
                if img_path.is_file() and Image is not None:
                    try:
                        loaded[view] = _img_to_b64(Image.open(img_path))
                    except Exception:
                        loaded[view] = ""
                else:
                    loaded[view] = ""
        step["_panorama_b64"] = loaded
    return data


# ── HTML template ──────────────────────────────────────────────────────

HTML_TPL = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VLN Trajectory Inspector – __SCENE_ID_____EPISODE_ID__</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; }
.header { padding: 12px 20px; background: #16213e; border-bottom: 2px solid #0f3460; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
.header h2 { font-size: 16px; }
.success { color: #4ade80; } .fail { color: #f87171; }
.main { display: flex; height: calc(100vh - 180px); min-height: 500px; }
.panel { padding: 10px; overflow: auto; }
.panel-left { width: 38%; border-right: 1px solid #333; }
.panel-mid { width: 32%; border-right: 1px solid #333; display: flex; flex-direction: column; align-items: center; }
.panel-right { width: 30%; }
.pano-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; width: 100%; }
.pano-grid img { width: 100%; border: 2px solid #444; border-radius: 4px; }
.pano-grid .active { border-color: #facc15; }
.view-label { text-align: center; font-size: 11px; color: #aaa; margin-top: 2px; }
.info-table { width: 100%; font-size: 13px; }
.info-table td { padding: 4px 8px; border-bottom: 1px solid #2a2a3e; vertical-align: top; }
.info-table td:first-child { color: #94a3b8; white-space: nowrap; width: 40%; }
.color-green { color: #4ade80; }
.color-red { color: #f87171; }
.color-gray { color: #9ca3af; }
.footer { height: 56px; background: #16213e; border-top: 2px solid #0f3460; display: flex; align-items: center; padding: 0 20px; gap: 12px; }
.footer input[type=range] { flex: 1; height: 8px; -webkit-appearance: none; appearance: none; background: linear-gradient(to right, var(--slider-colors, #4ade80 0%, #4ade80 50%, #f87171 50%, #f87171 100%)); border-radius: 4px; outline: none; }
.footer input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 16px; height: 16px; background: #facc15; border-radius: 50%; cursor: pointer; }
.footer button { padding: 6px 14px; border: 1px solid #444; border-radius: 4px; background: #1e293b; color: #e0e0e0; cursor: pointer; font-size: 13px; }
.footer button:hover { background: #334155; }
.footer select { padding: 4px 8px; border: 1px solid #444; border-radius: 4px; background: #1e293b; color: #e0e0e0; font-size: 13px; }
.instruction { font-size: 13px; color: #cbd5e1; padding: 8px 10px; background: #1e293b; border-radius: 4px; margin-bottom: 8px; line-height: 1.5; }
svg text { font-family: monospace; font-size: 10px; }
</style>
</head>
<body>
<div class="header">
  <div>
    <h2>Scene: __SCENE_ID__ &nbsp;|&nbsp; Episode: __EPISODE_ID__</h2>
    <span style="font-size:12px;color:#94a3b8;">
      Steps: __TOTAL_STEPS__ &nbsp;|&nbsp; VLM calls: __VLM_CALLS__ &nbsp;|&nbsp; Traj calls: __TRAJECTORY_CALLS__
    </span>
  </div>
  <div>
    <span class="__SUCCESS_CLS__" style="font-size:18px;font-weight:bold;">__SUCCESS_TEXT__</span>
    &nbsp;&nbsp; SPL: __SPL__ &nbsp;&nbsp; NE: __NE__ m
  </div>
</div>

<div class="main">
  <div class="panel panel-left" id="map-panel">
    <svg id="birdseye" width="100%" height="100%"></svg>
  </div>
  <div class="panel panel-mid" id="pano-panel">
    <div class="instruction" id="instruction-bar"></div>
    <div class="pano-grid" id="pano-grid">
      <div><img id="img-front" src=""><div class="view-label">FRONT</div></div>
      <div><img id="img-right" src=""><div class="view-label">RIGHT</div></div>
      <div><img id="img-back" src=""><div class="view-label">BACK</div></div>
      <div><img id="img-left" src=""><div class="view-label">LEFT</div></div>
    </div>
  </div>
  <div class="panel panel-right" id="info-panel">
    <table class="info-table" id="info-table"></table>
  </div>
</div>

<div class="footer">
  <button onclick="prevStep()" title="Left arrow">&#9664; Prev</button>
  <button id="play-btn" onclick="togglePlay()">&#9654; Play</button>
  <button onclick="nextStep()" title="Right arrow">Next &#9654;</button>
  <select id="speed-sel" onchange="updateSpeed()">
    <option value="500">0.5s</option>
    <option value="250" selected>1x (0.25s)</option>
    <option value="125">2x</option>
    <option value="60">4x</option>
  </select>
  <span style="font-size:13px;color:#94a3b8;" id="step-counter">Step 0/0</span>
  <input type="range" id="slider" min="0" max="0" value="0" oninput="jumpTo(parseInt(this.value))">
</div>

<script>
const DATA = __DATA_JSON__;

let current = 0;
let playing = false;
let playTimer = null;

const steps = DATA.steps || [];
const meta = DATA.metadata || {};
const total = steps.length;

function init() {
    document.getElementById('slider').max = Math.max(total - 1, 0);
    updateSliderColors();
    jumpTo(0);
}

function updateSliderColors() {
    const colors = steps.map((s, i) => {
        const dd = s.delta_dist;
        if (dd === null || dd === undefined) return '#6b7280';
        if (dd > 0.05) return '#f87171';
        if (dd < -0.05) return '#4ade80';
        return '#9ca3af';
    });
    const n = colors.length;
    if (n === 0) {
        document.getElementById('slider').style.setProperty('--slider-colors', '#6b7280 0%, #6b7280 100%');
        return;
    }
    if (n === 1) colors.push(colors[0]);
    const stops = colors.map((c, i) => c + ' ' + Math.round(i/(n-1)*100) + '%').join(', ');
    document.getElementById('slider').style.setProperty('--slider-colors', stops);
}

function _state() {
    if (total === 0) return null;
    const i = Math.max(0, Math.min(total - 1, current));
    return steps[i];
}

function jumpTo(i) {
    current = Math.max(0, Math.min(total - 1, i));
    const s = _state();
    if (!s) return;
    document.getElementById('slider').value = current;
    document.getElementById('step-counter').textContent = 'Step ' + (current+1) + '/' + total;
    renderBirdseye(s);
    renderPanorama(s);
    renderInfo(s);
}

function prevStep() { jumpTo(current - 1); }
function nextStep() { jumpTo(current + 1); }

function togglePlay() {
    playing = !playing;
    document.getElementById('play-btn').textContent = playing ? '⏸ Pause' : '▶ Play';
    if (playing) {
        playTimer = setInterval(() => {
            if (current >= total - 1) { togglePlay(); return; }
            jumpTo(current + 1);
        }, parseInt(document.getElementById('speed-sel').value));
    } else {
        clearInterval(playTimer);
    }
}

function updateSpeed() {
    if (playing) {
        clearInterval(playTimer);
        playTimer = setInterval(() => {
            if (current >= total - 1) { togglePlay(); return; }
            jumpTo(current + 1);
        }, parseInt(document.getElementById('speed-sel').value));
    }
}

// ── Bird's-eye SVG ─────────────────────────────────────────────────

function renderBirdseye(activeStep) {
    const svg = document.getElementById('birdseye');
    const W = svg.clientWidth || 600;
    const H = svg.clientHeight || 500;
    const M = 40;
    const w = W - 2*M, h = H - 2*M;

    // Collect points: GT path + agent path
    let pts = [];
    const gtPath = meta.gt_reference_path || [];
    gtPath.forEach(p => pts.push([p[0], p[2]]));
    steps.filter(s => s.position && s.position.length >= 3).forEach(s => pts.push([s.position[0], s.position[2]]));
    if (meta.goal_position) pts.push([meta.goal_position[0], meta.goal_position[2]]);
    if (meta.start_position) pts.push([meta.start_position[0], meta.start_position[2]]);

    if (pts.length === 0) { svg.innerHTML = ''; return; }

    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    pts.forEach(([x, z]) => { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z); });
    const span = Math.max(maxX - minX, maxZ - minZ, 1.0);
    const pad = span * 0.2;
    minX -= pad; maxX += pad; minZ -= pad; maxZ += pad;
    const span2 = Math.max(maxX - minX, maxZ - minZ, 1.0);

    function tx(x) { return M + (x - minX) / span2 * w; }
    function tz(z) { return M + h - (z - minZ) / span2 * h; }

    let html = '';

    // GT path
    if (gtPath.length >= 2) {
        let d = gtPath.map((p, i) => (i===0?'M':'L') + tx(p[0]).toFixed(1)+','+tz(p[2]).toFixed(1)).join(' ');
        html += '<polyline points="' + gtPath.map(p => tx(p[0]).toFixed(1)+','+tz(p[2]).toFixed(1)).join(' ') + '" fill="none" stroke="#4ade80" stroke-width="2" stroke-dasharray="6,4" opacity="0.6"/>';
    }

    // Agent path with color segments
    for (let i = 0; i < steps.length - 1; i++) {
        const a = steps[i], b = steps[i+1];
        if (!a.position || !b.position || a.position.length < 3 || b.position.length < 3) continue;
        const dd = b.delta_dist;
        let color = '#6b7280';
        if (dd !== null && dd !== undefined) {
            if (dd > 0.05) color = '#f87171';
            else if (dd < -0.05) color = '#4ade80';
        }
        html += '<line x1="'+tx(a.position[0]).toFixed(1)+'" y1="'+tz(a.position[2]).toFixed(1)+'" x2="'+tx(b.position[0]).toFixed(1)+'" y2="'+tz(b.position[2]).toFixed(1)+'" stroke="'+color+'" stroke-width="3" stroke-linecap="round"/>';
    }

    // Step circles + heading arrows
    const arrowLen = span2 * 0.025;
    for (let i = 0; i < steps.length; i++) {
        const s = steps[i];
        if (!s.position || s.position.length < 3) continue;
        const cx = tx(s.position[0]), cy = tz(s.position[2]);
        const hdg = (s.heading_deg || 0) * Math.PI / 180;
        const ax = cx + arrowLen * Math.sin(hdg) / (span2/w);
        const ay = cy - arrowLen * Math.cos(hdg) / (span2/h);
        // const ay = cy; // simplified

        const dd = s.delta_dist;
        let fill = '#6b7280';
        if (dd !== null && dd !== undefined) {
            if (dd > 0.05) fill = '#f87171';
            else if (dd < -0.05) fill = '#4ade80';
        }

        const r = (i === current) ? 5 : 3;
        html += '<circle cx="'+cx.toFixed(1)+'" cy="'+cy.toFixed(1)+'" r="'+r+'" fill="'+fill+'" stroke="'+(i===current?'#facc15':'none')+'" stroke-width="'+(i===current?2:0)+'"/>';

        // Heading arrow (only for non-action steps)
        if (s.phase === 'vlm' || i === current) {
            const headRad = (s.heading_deg || 0) * Math.PI / 180;
            const lx = cx + arrowLen * 2 * Math.sin(headRad);
            const ly = cy - arrowLen * 2 * Math.cos(headRad);
            html += '<line x1="'+cx.toFixed(1)+'" y1="'+cy.toFixed(1)+'" x2="'+lx.toFixed(1)+'" y2="'+ly.toFixed(1)+'" stroke="'+fill+'" stroke-width="1.5" opacity="0.8"/>';
        }

        // Step number label (every 5th step to avoid clutter)
        if (i % 5 === 0 || i === current) {
            html += '<text x="'+(cx+5).toFixed(0)+'" y="'+(cy-5).toFixed(0)+'" fill="#e0e0e0" font-size="9">'+(i+1)+'</text>';
        }
    }

    // Goal star
    if (meta.goal_position) {
        const gx = tx(meta.goal_position[0]), gy = tz(meta.goal_position[2]);
        html += '<polygon points="'+gx.toFixed(1)+','+(gy-8).toFixed(1)+' '+(gx+3).toFixed(1)+','+(gy-2).toFixed(1)+' '+(gx+8).toFixed(1)+','+(gy-2).toFixed(1)+' '+(gx+4).toFixed(1)+','+(gy+2).toFixed(1)+' '+(gx+5).toFixed(1)+','+(gy+7).toFixed(1)+' '+(gx).toFixed(1)+','+(gy+4).toFixed(1)+' '+(gx-5).toFixed(1)+','+(gy+7).toFixed(1)+' '+(gx-4).toFixed(1)+','+(gy+2).toFixed(1)+' '+(gx-8).toFixed(1)+','+(gy-2).toFixed(1)+' '+(gx-3).toFixed(1)+','+(gy-2).toFixed(1)+'" fill="#facc15" stroke="#eab308" stroke-width="1"/>';
        html += '<text x="'+(gx-10).toFixed(0)+'" y="'+(gy+16).toFixed(0)+'" fill="#facc15" font-size="10">GOAL</text>';
    }

    // Start marker
    if (meta.start_position) {
        const sx = tx(meta.start_position[0]), sy = tz(meta.start_position[2]);
        html += '<circle cx="'+sx.toFixed(1)+'" cy="'+sy.toFixed(1)+'" r="5" fill="none" stroke="#94a3b8" stroke-width="2"/>';
        html += '<text x="'+(sx-15).toFixed(0)+'" y="'+(sy-8).toFixed(0)+'" fill="#94a3b8" font-size="9">START</text>';
    }

    // Legend
    html += '<rect x="'+(W-130)+'" y="8" width="122" height="52" rx="4" fill="#1e293b" opacity="0.9"/>';
    html += '<line x1="'+(W-122)+'" y1="18" x2="'+(W-102)+'" y2="18" stroke="#4ade80" stroke-width="3"/>';
    html += '<text x="'+(W-98)+'" y="22" fill="#4ade80" font-size="10">toward goal</text>';
    html += '<line x1="'+(W-122)+'" y1="34" x2="'+(W-102)+'" y2="34" stroke="#f87171" stroke-width="3"/>';
    html += '<text x="'+(W-98)+'" y="38" fill="#f87171" font-size="10">away from goal</text>';
    html += '<line x1="'+(W-122)+'" y1="50" x2="'+(W-102)+'" y2="50" stroke="#6b7280" stroke-width="3"/>';
    html += '<text x="'+(W-98)+'" y="54" fill="#6b7280" font-size="10">stationary</text>';

    svg.innerHTML = html;
}

// ── Panorama ─────────────────────────────────────────────────────────

function renderPanorama(s) {
    const pano = s._panorama_b64 || {};
    document.getElementById('img-front').src = pano.front || '';
    document.getElementById('img-right').src = pano.right || '';
    document.getElementById('img-back').src = pano.back || '';
    document.getElementById('img-left').src = pano.left || '';

    // Highlight active view
    ['front','right','back','left'].forEach(v => {
        const el = document.getElementById('img-'+v);
        const label = el.parentElement.querySelector('.view-label');
        if (s.pano_goal_view && s.pano_goal_view.toLowerCase() === v) {
            el.classList.add('active');
            if (label) label.style.color = '#facc15';
        } else {
            el.classList.remove('active');
            if (label) label.style.color = '#aaa';
        }
    });
}

// ── Info panel ────────────────────────────────────────────────────────

function renderInfo(s) {
    const dd = s.delta_dist;
    let ddHtml = '';
    if (dd !== null && dd !== undefined) {
        if (dd > 0.05) ddHtml = '<span class="color-red">&#9650; +' + dd.toFixed(2) + ' m (away)</span>';
        else if (dd < -0.05) ddHtml = '<span class="color-green">&#9660; ' + dd.toFixed(2) + ' m (toward)</span>';
        else ddHtml = '<span class="color-gray">&#9644; ' + dd.toFixed(2) + ' m</span>';
    }

    let phaseLabel = s.phase || 'unknown';
    if (s.executed_action_name && s.phase !== 'vlm') {
        phaseLabel = 'action: ' + s.executed_action_name;
    }

    const rows = [
        ['Step', (current+1) + ' / ' + total + ' &nbsp;|&nbsp; phase: <b>' + phaseLabel + '</b>'],
        ['Distance to goal', (
            s.distance_to_goal !== null && s.distance_to_goal !== undefined
                ? s.distance_to_goal.toFixed(2) + ' m' : 'n/a'
        ) + ' &nbsp;' + ddHtml],
    ];

    if (s.vlm_output) {
        rows.push(['VLM output', '<code style="font-size:12px;white-space:pre-wrap;">' + escHtml(s.vlm_output || '') + '</code>']);
    }
    if (s.pixel_goal) {
        rows.push(['Pixel goal', '[' + s.pixel_goal.join(', ') + '] in <b>' + escHtml(s.pano_goal_view || '?') + '</b>']);
    }
    if (s.executed_action_name) {
        rows.push(['Executed action', '<b>' + s.executed_action_name + '</b> (' + (s.executed_action||'') + ')']);
    }
    if (s.traj_hs_total_norm !== null && s.traj_hs_total_norm !== undefined) {
        const pq = s.traj_hs_per_query || [];
        rows.push(['traj_hs norm', s.traj_hs_total_norm.toFixed(1) + ' &nbsp; per-query: [' + pq.map(v => v.toFixed(0)).join(', ') + ']']);
    }
    rows.push(['Heading', (s.heading_deg||0).toFixed(1) + '°']);
    if (s.position) {
        rows.push(['Position', '(' + s.position.map(v => v.toFixed(2)).join(', ') + ')']);
    }

    let html = '';
    rows.forEach(([k, v]) => {
        html += '<tr><td>' + k + '</td><td>' + v + '</td></tr>';
    });
    document.getElementById('info-table').innerHTML = html;

    // Instruction
    document.getElementById('instruction-bar').textContent = meta.instruction || '';
}

function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Keyboard
document.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft') { e.preventDefault(); prevStep(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); nextStep(); }
    else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
});

// Click on SVG to jump to nearest step
document.getElementById('birdseye').addEventListener('click', function(e) {
    // Simple: click anywhere advances one step
    // (More precise click-to-step would need coordinate mapping)
});

window.addEventListener('load', init);
window.addEventListener('resize', () => jumpTo(current));
</script>
</body>
</html>"""


# ── main ───────────────────────────────────────────────────────────────

def generate_html(data: dict[str, Any], output_path: str) -> None:
    meta = data.get("metadata", {})
    steps = data.get("steps", [])

    # Compute delta_dist for steps that don't have it yet
    prev_dist = None
    for s in steps:
        cur = s.get("distance_to_goal")
        if prev_dist is not None and cur is not None:
            s.setdefault("delta_dist", cur - prev_dist)
        prev_dist = cur

    ne = meta.get("total_steps", 0)
    # Use distance_to_goal from the last step if available
    if steps:
        last_dg = steps[-1].get("distance_to_goal")
        if last_dg is not None:
            ne = last_dg

    html = HTML_TPL
    html = html.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False))
    html = html.replace("__SCENE_ID__", str(meta.get("scene_id", "?")))
    html = html.replace("__EPISODE_ID__", str(meta.get("episode_id", 0)))
    html = html.replace("__TOTAL_STEPS__", str(meta.get("total_steps", len(steps))))
    html = html.replace("__VLM_CALLS__", str(meta.get("vlm_calls", 0)))
    html = html.replace("__TRAJECTORY_CALLS__", str(meta.get("trajectory_calls", 0)))
    html = html.replace("__SUCCESS_CLS__", "success" if meta.get("success") else "fail")
    html = html.replace("__SUCCESS_TEXT__", "SUCCESS" if meta.get("success") else "FAIL")
    html = html.replace("__SPL__", f"{meta.get('spl', 0.0):.4f}")
    html = html.replace("__NE__", f"{float(ne):.2f}" if ne else "?")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {len(steps)} steps → {out}")


def main() -> int:
    p = argparse.ArgumentParser(description="Generate VLN trajectory HTML inspector")
    p.add_argument("--input", default="", help="Path to trajectory_steps.json")
    p.add_argument("--input-dir", default="",
                   help="Process all trajectory_steps.json under this directory")
    p.add_argument("--output", default="", help="Output HTML path for single-file mode")
    p.add_argument("--output-dir", default="", help="Output directory for batch mode")
    args = p.parse_args()

    if args.input_dir:
        root = Path(args.input_dir)
        out_dir = Path(args.output_dir) if args.output_dir else root / "html"
        found = 0
        for json_path in sorted(root.rglob("trajectory_steps.json")):
            rel = json_path.parent.relative_to(root)
            out_path = out_dir / rel / "trajectory.html"
            data = _load_data(str(json_path))
            generate_html(data, str(out_path))
            found += 1
        print(f"Processed {found} episodes.")
        return 0

    if not args.input:
        p.error("Either --input or --input-dir is required")

    data = _load_data(args.input)
    out = args.output or str(Path(args.input).with_suffix(".html"))
    generate_html(data, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

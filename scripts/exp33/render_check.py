"""EXP-33 render check: each camera mount at one start pose, before any closed-loop run.

For every viewpoint it builds the client's own Habitat config, resets to the first episode, and
checks (client._check_camera_mount) that the RGB sensor sits at the asked height and pitch relative
to the agent's base: after reset, after the four-view panorama (set_state(reset_sensors=True)) and
after a LOOKDOWN action.  It saves the front frame and the look-down image the client would send,
as one contact sheet (a row per viewpoint), and the measured mounts as JSON.  Needs an X display
(run under Xvfb with llvmpipe, as the launcher does); no GPU.

    python scripts/exp33/render_check.py --data_path <dataset_shard_00.json.gz> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import quaternion
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.evaluation import r2r_val_unseen as client  # noqa: E402

# The pre-registered viewpoints (docs/experiments/README.md, EXP-33).
VIEWPOINTS = {
    "v0": dict(camera_height_m=None, camera_pitch_deg=None, lookdown_mode="rotate"),
    "v1": dict(camera_height_m=0.6, camera_pitch_deg=-15.0, lookdown_mode="same_frame"),
    "v2": dict(camera_height_m=0.7, camera_pitch_deg=-15.0, lookdown_mode="same_frame"),
    "v3": dict(camera_height_m=0.4, camera_pitch_deg=0.0, lookdown_mode="same_frame"),
}
DEFAULT_MOUNT = {"height_m": 1.25, "pitch_deg": 0.0, "lookdown": "rotate"}


def measure(env) -> dict:
    state = env._sim.get_agent(0).get_state()
    sensor = state.sensor_states["rgb"]
    forward = (
        quaternion.as_rotation_matrix(state.rotation).T
        @ quaternion.as_rotation_matrix(sensor.rotation)
        @ np.array([0.0, 0.0, -1.0])
    )
    return {
        "height_m": round(float(sensor.position[1]) - float(state.position[1]), 6),
        "pitch_deg": round(math.degrees(math.asin(float(np.clip(forward[1], -1.0, 1.0)))), 4),
    }


def check_viewpoint(name: str, flags: dict, data_path: str, scenes_dir: str) -> tuple[dict, Image.Image, Image.Image]:
    import habitat

    args = SimpleNamespace(dataset_split="val_unseen", scenes_dir=scenes_dir, data_path=data_path, sim_gpu_id=0, **flags)
    mount = client._camera_mount(args)
    client.ensure_vln_measures_registered()
    env = habitat.Env(config=client.build_habitat_config(args))
    try:
        observations = env.reset()
        if mount is not None:
            env.heatmapvln_lookdown_same_frame = mount["lookdown"] == "same_frame"
        expected = mount or DEFAULT_MOUNT
        record = {"viewpoint": name, "mount": expected, "episode": f"{env.current_episode.scene_id}:{env.current_episode.episode_id}"}
        client._check_camera_mount(env, expected)
        record["after_reset"] = measure(env)
        front = client._rgb_array_to_pil(client._extract_rgb_array(observations), (640, 480))
        client.capture_panoramic_views(env, image_size=(384, 384))
        client._check_camera_mount(env, expected)
        record["after_panorama"] = measure(env)
        lookdown = client.capture_lookdown_view(env, image_size=client.NATIVE_INTERNNAV_LOOKDOWN_SIZE)
        record["after_lookdown_capture"] = measure(env)
        record["lookdown_equals_front"] = bool(np.array_equal(np.asarray(lookdown), np.asarray(front)))
        observations, _ = client._apply_habitat_action(env, client.ActionCode.LOOKDOWN)
        record["after_lookdown_action"] = measure(env)
        # LOOK_DOWN then LOOK_UP returns the sensor to its mount up to float error (as the default
        # look-down capture's 2 x LOOK_DOWN + 2 x LOOK_UP does); this sizes it in pixels.
        diff = np.abs(
            np.asarray(client._rgb_array_to_pil(client._extract_rgb_array(observations), (640, 480)), dtype=np.int16)
            - np.asarray(front, dtype=np.int16)
        )
        record["lookdown_action_view_vs_front"] = {
            "max_abs": int(diff.max()), "mean_abs": round(float(diff.mean()), 4),
            "pixels_differing": round(float((diff.max(axis=-1) > 0).mean()), 6),
        }
        return record, front, lookdown
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--scenes_dir", default="/dataset")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records, rows = [], []
    for name, flags in VIEWPOINTS.items():
        record, front, lookdown = check_viewpoint(name, flags, args.data_path, args.scenes_dir)
        records.append(record)
        rows.append((name, front, lookdown))
        print(json.dumps(record, sort_keys=True), flush=True)

    sheet = Image.new("RGB", (2 * 640, len(rows) * 480), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, front, lookdown) in enumerate(rows):
        sheet.paste(front, (0, index * 480))
        sheet.paste(lookdown, (640, index * 480))
        label = f"{name} {json.dumps(records[index]['mount'])}"
        draw.rectangle((0, index * 480, 9 * len(label), index * 480 + 16), fill="black")
        draw.text((4, index * 480 + 2), label, fill="white")
    sheet.save(out / "render_check.png")
    (out / "render_check.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out / 'render_check.png'} and {out / 'render_check.json'}")


if __name__ == "__main__":
    main()

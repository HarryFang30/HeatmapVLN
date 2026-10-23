"""EXP-33 camera mounts in the RPC client: --camera_height_m, --camera_pitch_deg, --lookdown_mode.

Without the flags nothing changes: no mount, the Habitat config of build_habitat_config, the
2 x LOOKDOWN / 2 x LOOKUP look-down, the old row schema.  With ``same_frame`` the camera is one
fixed sensor, as on InternNav's Go2: the look-down image is the front frame where the sensor
points, and a LOOKDOWN action takes its step without the view ever being seen tilted.  Runs the
client end to end on the fakes of ``test_s2_record_collection``.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests import test_s2_record_collection as harness
from tests.test_s2_record_collection import client  # noqa: F401  (fixture)

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "run_ppa_r2r_val_unseen_cuda.sh"
GO2 = ("--camera_height_m", "0.6", "--camera_pitch_deg", "-15", "--lookdown_mode", "same_frame")
GO2_MOUNT = {"height_m": 0.6, "pitch_deg": -15.0, "lookdown": "same_frame"}


def _args(**overrides):
    return SimpleNamespace(**{"dataset_split": "train", "scenes_dir": "s", "data_path": "d", "sim_gpu_id": 0, **overrides})


class MountedEnv(harness.FakeEnv):
    """The fake env with an RGB sensor state at a mount, and a log of every tilt it rendered at."""

    def __init__(self, quaternion, height_m=1.25, pitch_deg=0.0):
        super().__init__(quaternion)
        self.height_m, self.pitch_deg = height_m, pitch_deg
        self.rendered_tilts: list[int] = []

    def _state(self):
        state = super()._state()
        pitch = self.q.from_rotation_vector([math.radians(self.pitch_deg), 0.0, 0.0])
        state.sensor_states = {
            "rgb": SimpleNamespace(position=state.position + np.array([0.0, self.height_m, 0.0]),
                                   rotation=self.rotation * pitch)
        }
        return state

    def _observe(self):
        self.rendered_tilts.append(self.tilt)
        return super()._observe()


def _run_mounted(client, monkeypatch, tmp_path, *, tag, height_m, pitch_deg, extra_argv=(), mismatch=True):
    monkeypatch.setattr(harness, "FakeEnv", lambda q: MountedEnv(q, height_m, pitch_deg))
    return harness._run(
        client, monkeypatch, tmp_path, mode="vo", record=False, tag=tag, mismatch=mismatch, extra_argv=extra_argv
    )


def test_without_the_flags_there_is_no_mount_and_the_config_is_untouched(client):
    assert client._camera_mount(_args()) is None
    assert client._camera_mount(_args(camera_height_m=None, camera_pitch_deg=None, lookdown_mode="rotate")) is None
    sim = client.build_habitat_config(_args()).SIMULATOR
    for sensor in (sim.RGB_SENSOR, sim.DEPTH_SENSOR):
        assert list(sensor.POSITION) == [0, 1.25, 0]
        values = getattr(sensor, "_values", None)
        if values is not None:  # habitat stubbed: the key was never written
            assert "ORIENTATION" not in values
        else:  # real habitat: its own default
            assert [float(v) for v in sensor.ORIENTATION] == [0.0, 0.0, 0.0]


def test_the_flags_move_both_sensors(client):
    args = _args(camera_height_m=0.7, camera_pitch_deg=-15.0, lookdown_mode="same_frame")
    assert client._camera_mount(args) == {"height_m": 0.7, "pitch_deg": -15.0, "lookdown": "same_frame"}
    sim = client.build_habitat_config(args).SIMULATOR
    for sensor in (sim.RGB_SENSOR, sim.DEPTH_SENSOR):
        assert list(sensor.POSITION) == [0, 0.7, 0]
        assert list(sensor.ORIENTATION) == pytest.approx([math.radians(-15.0), 0.0, 0.0])
    # Only the look-down mode: the default height and pitch, written out.
    assert client._camera_mount(_args(lookdown_mode="same_frame")) == {
        "height_m": 1.25, "pitch_deg": 0.0, "lookdown": "same_frame"}


def test_a_negative_habitat_x_orientation_looks_down():
    quaternion = pytest.importorskip("quaternion")
    rotation = quaternion.as_rotation_matrix(quaternion.from_rotation_vector([math.radians(-15.0), 0.0, 0.0]))
    forward = rotation @ np.array([0.0, 0.0, -1.0])  # Habitat cameras look down -Z, Y up
    assert forward[1] == pytest.approx(math.sin(math.radians(-15.0)))


def test_lookdown_capture_rotates_by_default_and_reuses_the_front_with_one_fixed_camera(client):
    quaternion = pytest.importorskip("quaternion")
    env = MountedEnv(quaternion)
    env.reset()
    env.trace.clear()
    tilted = client.capture_lookdown_view(env, image_size=(640, 480))
    assert [action for _, action in env.trace] == [5, 5, 4, 4] and env.tilt == 0

    env.trace.clear()
    env.heatmapvln_lookdown_same_frame = True
    front = client._rgb_array_to_pil(env._observe()["rgb"], (640, 480))
    same = client.capture_lookdown_view(env, image_size=(640, 480))
    assert env.trace == [] and env.tilt == 0
    assert np.array_equal(np.asarray(same), np.asarray(front))
    assert not np.array_equal(np.asarray(same), np.asarray(tilted))
    with pytest.raises(RuntimeError, match="no look-down depth"):
        client.capture_lookdown_view(env, image_size=(640, 480), depth_out=[])


def test_a_lookdown_action_with_one_fixed_camera_takes_its_step_and_keeps_the_view(client):
    quaternion = pytest.importorskip("quaternion")
    env = MountedEnv(quaternion)
    env.reset()
    env.heatmapvln_lookdown_same_frame = True
    env.trace.clear()
    observations, done = client._apply_habitat_action(env, client.ActionCode.LOOKDOWN)
    assert [action for _, action in env.trace] == [5, 4] and env.tilt == 0 and done is False
    assert np.array_equal(observations["rgb"], env._observe()["rgb"])


def test_the_mount_check_fails_when_the_sensor_is_elsewhere(client):
    quaternion = pytest.importorskip("quaternion")
    env = MountedEnv(quaternion, height_m=0.6, pitch_deg=-15.0)
    env.reset()
    client._check_camera_mount(env, GO2_MOUNT)
    for wrong in ({**GO2_MOUNT, "height_m": 0.7}, {**GO2_MOUNT, "pitch_deg": 15.0}, {**GO2_MOUNT, "pitch_deg": 0.0}):
        with pytest.raises(RuntimeError, match="camera mount not applied"):
            client._check_camera_mount(env, wrong)
    bare = harness.FakeEnv(quaternion)  # no sensor state at all
    bare.reset()
    with pytest.raises(RuntimeError, match="needs the RGB sensor state"):
        client._check_camera_mount(bare, GO2_MOUNT)


def test_a_fixed_camera_run_never_renders_a_tilted_view_and_labels_its_rows(client, monkeypatch, tmp_path):
    run = _run_mounted(client, monkeypatch, tmp_path, tag="go2", height_m=0.6, pitch_deg=-15.0, extra_argv=GO2)
    assert len(run.progress) == len(harness.EPISODES)
    assert all(row["camera_mount"] == GO2_MOUNT for row in run.progress)
    assert run.env.lookdown_depths == [] and -30 not in run.env.rendered_tilts  # no capture tilted the sensor
    # The fake's call 1 queues [LOOKDOWN, STOP]: it ran as LOOK_DOWN then LOOK_UP, so the only tilted
    # render is the LOOK_DOWN step's own, which _apply_habitat_action drops.
    lookdowns = sum(action == 5 for _, action, _, _ in run.env.primitive)
    assert lookdowns > 0 and run.env.rendered_tilts.count(-15) == lookdowns
    assert set(run.env.rendered_tilts) == {0, -15}
    steps = [action for _, action in run.env.trace]
    assert steps.count(5) == lookdowns
    assert all(after == 4 for before, after in zip(steps, steps[1:]) if before == 5)
    for request in run.server.requests:
        assert {"lookdown", "current/front"} <= set(request["blobs"])


def test_the_default_run_keeps_the_old_rows_and_rotates(client, monkeypatch, tmp_path):
    run = _run_mounted(client, monkeypatch, tmp_path, tag="default", height_m=1.25, pitch_deg=0.0)
    assert all("camera_mount" not in row for row in run.progress)
    assert not hasattr(run.env, "heatmapvln_lookdown_same_frame")
    assert len(run.env.lookdown_depths) == len(run.server.requests)  # one 30 deg capture per call
    assert -30 in run.env.rendered_tilts


def test_resume_refuses_rows_from_another_mount(client, monkeypatch, tmp_path):
    _run_mounted(client, monkeypatch, tmp_path, tag="mixed", height_m=0.6, pitch_deg=-15.0, extra_argv=GO2)
    with pytest.raises(ValueError, match="contract mismatch"):
        _run_mounted(client, monkeypatch, tmp_path, tag="mixed", height_m=1.25, pitch_deg=0.0, extra_argv=("--resume",))
    _run_mounted(client, monkeypatch, tmp_path, tag="plain", height_m=1.25, pitch_deg=0.0)
    with pytest.raises(ValueError, match="contract mismatch"):
        _run_mounted(client, monkeypatch, tmp_path, tag="plain", height_m=0.6, pitch_deg=-15.0,
                     extra_argv=GO2 + ("--resume",))


@pytest.mark.parametrize(
    "extra, phrase",
    [
        (("--camera_height_m", "0"), r"must be in \(0, 2\]"),
        (("--camera_pitch_deg", "-75"), r"must be in \[-60, 60\]"),
        (("--lookdown_mode", "same_frame", "--s2_record_dir", "records"), "evaluation only"),
    ],
)
def test_main_refuses_mounts_it_cannot_honour(client, monkeypatch, tmp_path, extra, phrase):
    with pytest.raises(ValueError, match=phrase):
        _run_mounted(client, monkeypatch, tmp_path, tag="refused", height_m=1.25, pitch_deg=0.0, extra_argv=extra)


def test_launcher_passes_the_camera_flags_only_when_asked():
    sh = LAUNCHER.read_text(encoding="utf-8")
    for var, flag in (("CAMERA_HEIGHT_M", "--camera_height_m"), ("CAMERA_PITCH_DEG", "--camera_pitch_deg"),
                      ("LOOKDOWN_MODE", "--lookdown_mode")):
        assert f'{var}="${{PPA_EVAL_{var}:-}}"' in sh
        assert f'[[ -n "${var}" ]] && CLIENT_EXTRA+=({flag} "${var}")' in sh
    assert sh.index("declare -a CLIENT_EXTRA=()") < sh.index("CLIENT_EXTRA+=(--camera_height_m")
    assert sh.index("CLIENT_EXTRA=(--stop_at_step_budget)") < sh.index("CLIENT_EXTRA+=(--camera_height_m")

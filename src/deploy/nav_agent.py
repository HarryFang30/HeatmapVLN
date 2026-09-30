"""Habitat-free navigation client: the deployed PPA model behind one ``act(rgb) -> action`` call.

The deployed system is two RPC servers (scripts/deploy/start_nav_servers_cuda.sh): the model
server (System 2, System 1 and the PPA heads; scripts/evaluation/rpc_model_server.py) and the
AMB3R visual-odometry server (scripts/amb3r_vo/rpc_amb3r_vo_server.py).  The model server keeps
no episode state and the VO server keeps only its map; everything else lives in the Habitat
evaluation client (scripts/evaluation/r2r_val_unseen.py, ``run_eval_rpc_panoramic``): the frame
history, the action queue, the VO frame ledger and the per-call seeds.  ``NavAgent`` is that
client with the simulator taken out, so a robot can drive the same two servers::

    agent = NavAgent("127.0.0.1:52400", "127.0.0.1:52500")
    agent.reset("Walk past the sofa and stop at the door.", scene_id="lab", episode_id=1)
    while not agent.done:
        action = agent.act(robot.front_rgb(), robot.lookdown_rgb, robot.level_rgb)
        robot.execute(action)

One ``act()`` is one executed step of the Habitat client, in the same order, and for the same
images it sends byte-identical requests (tests/test_nav_agent_habitat.py checks this against
the Habitat client's own code).  docs/deploy/navigation_interface.md is the user-facing
contract.  Deliberate differences from the Habitat client are marked "Differs from the Habitat
client" below.
"""

from __future__ import annotations

import importlib.util
import itertools
import logging
import sys
import types
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from scripts.evaluation.rpc_protocol import (
    HEATMAPVLN_RPC_DEFAULT_PROTOCOL_SEED,
    HEATMAPVLN_RPC_PROTOCOL_VERSION,
    HEATMAPVLN_RPC_SAMPLING_FIELD,
    build_rpc_sampling_metadata,
    validate_rpc_sampling_metadata,
)
from src.utils.latency import StageTimer

LOGGER = logging.getLogger("heatmapvln-nav-agent")


def _load_pure_vo_modules() -> tuple[types.ModuleType, types.ModuleType]:
    """Load src/vo/rpc_protocol.py and rpc_client.py without src/vo/__init__.py.

    The package initializer pulls the AMB3R/training stack (src.data, torch, ...); the two RPC
    modules need only NumPy.  Same trick as the Habitat client's _PURE_VO_PACKAGE_NAME loader.
    """
    package_name = "_heatmapvln_deploy_vo"
    if f"{package_name}.rpc_client" not in sys.modules:
        vo_dir = Path(__file__).resolve().parents[1] / "vo"
        package = types.ModuleType(package_name)
        package.__path__ = [str(vo_dir)]
        sys.modules[package_name] = package
        for name in ("rpc_protocol", "rpc_client"):
            spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", vo_dir / f"{name}.py")
            if spec is None or spec.loader is None:
                raise ImportError(f"could not load src/vo/{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.rpc_protocol"], sys.modules[f"{package_name}.rpc_client"]


_vo_protocol, _vo_client = _load_pure_vo_modules()
OnlineVORPCBridge = _vo_client.OnlineVORPCBridge
sample_unique_past_indices = _vo_client.sample_unique_past_indices
AMB3R_VO_POSE_PROVIDER = _vo_protocol.AMB3R_VO_POSE_PROVIDER
AMB3R_VO_RPC_PROTOCOL_VERSION = _vo_protocol.AMB3R_VO_RPC_PROTOCOL_VERSION
AMB3R_VO_MODEL_VERSION = "amb3r-vo-da3-online"
PPA_ONLINE_AMB3R_CAPABILITY = "ppa-online-amb3r-v1"

# Fixed by the deployed launcher (scripts/run_ppa_r2r_val_unseen_cuda.sh client flags) and the
# deployed config (configs/ppa_action_refine_v2_8gpu.yaml: data.image_size,
# data.trajectory.traj_image_size, system2_sft_protocol internnav + structured_pano_output false).
VLM_IMAGE_SIZE = (384, 384)
TRAJ_IMAGE_SIZE = (224, 224)
LOOKDOWN_IMAGE_SIZE = (640, 480)  # the native two-turn System 2 refuses any other size
NUM_HISTORY = 8
MAX_STEPS_PER_EPISODE = 500
MODEL_JPEG_QUALITY = 90
VO_JPEG_QUALITY = 95
MAX_QUEUED_ACTIONS = 8  # the client's MAX_STEPS guard; unreachable with 4-action chunks
PLAN_METHOD = "plan_panoramic"
VIEWS = ("front", "right", "back", "left")

FORWARD_STEP_M = 0.25
TURN_STEP_DEG = 15.0
LOOK_DOWN_DEG = 30.0


class Action(IntEnum):
    """What the robot executes before the next ``act()``; values are the server's action codes."""

    STOP = 0  # halt; the episode is over
    FORWARD = 1  # drive FORWARD_STEP_M straight ahead
    TURN_LEFT = 2  # rotate TURN_STEP_DEG counter-clockwise in place
    TURN_RIGHT = 3  # rotate TURN_STEP_DEG clockwise in place
    LOOK_DOWN = 5  # pitch the camera LOOK_DOWN_DEG down; the body does not move

    @property
    def camera_pitch_deg(self) -> float:
        """Camera pitch to set before executing this action (0 = level, negative = down).

        In Habitat, LOOKDOWN tilts the sensor only for the first capture after it: the frame
        passed to the next ``act()``.  Each panorama capture of the client ends in
        agent.set_state(..., reset_sensors=True), which levels the sensor, so a replan in the
        same step (``level_fn``), the look-down capture and the next action all start level.
        """
        return -LOOK_DOWN_DEG if self is Action.LOOK_DOWN else 0.0


@dataclass(frozen=True)
class CameraSpec:
    """The camera the model and the VO were run with in simulation (build_habitat_config).

    Only the resolution can be checked from pixels; the rest is a mounting requirement.
    """

    width: int = 640
    height: int = 480
    hfov_deg: float = 79.0
    height_m: float = 1.25
    pitch_deg: float = 0.0  # level
    lookdown_pitch_deg: float = -LOOK_DOWN_DEG

    def check_frame(self, value: Any, what: str) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{what} must be a numpy array, got {type(value).__name__}")
        expected = (self.height, self.width, 3)
        if value.dtype != np.uint8 or value.shape != expected:
            # Never resize here: a wrong camera must fail, not be squashed into this one.
            raise ValueError(f"{what} must be uint8 RGB with shape {expected}, got {value.dtype} {value.shape}")
        return value


@dataclass
class PlanCall:
    """One model-server call: what was sent, what came back, and how long it took."""

    call_index: int
    step: int
    kind: str
    llm_output: str
    actions: list[int]
    pixel_goal: Optional[list[int]]
    terminal: bool
    pose_ready: bool
    ppa_applied: bool
    vo_frame_id: int
    vo_history_frame_ids: list[int]
    vo_provider_phase: str
    vo_trajectory_revision: int
    history_capture_steps: list[int]
    response: dict[str, Any]
    client_timing_ms: dict[str, float] = field(default_factory=dict)
    server_timing_ms: Optional[dict[str, Any]] = None

    def vo_log_line(self) -> str:
        return (
            f"  [amb3r-vo] frame={self.vo_frame_id} history={self.vo_history_frame_ids} "
            f"ready={self.pose_ready} phase={self.vo_provider_phase} revision={self.vo_trajectory_revision}"
        )

    def log_lines(self) -> list[str]:
        """The Habitat client's stdout lines for this call (after its plan RPC, --no-debug_input_trace)."""
        lines = [f"  step_id: {self.step}, RPC kind={self.response.get('kind')}, VLM output: {self.llm_output}"]
        if self.response.get("trajectory_summary"):
            lines.append(f"  [debug] trajectory {self.response['trajectory_summary']}, actions={self.actions}")
        elif self.actions:
            lines.append(f"  [debug] actions={self.actions}")
        return lines


@dataclass
class StepInfo:
    """What one ``act()`` did."""

    step: int  # capture step of the frame passed in (= actions executed before it)
    action: Action
    source: str  # "queue" | "plan" | "terminal" | "empty" | "step_cap"
    call: Optional[PlanCall]  # the plan call made during this act(), if any
    timing_ms: dict[str, float] = field(default_factory=dict)
    vo_server_timing_ms: Optional[dict[str, Any]] = None


def _rgb_array_to_pil(rgb: np.ndarray, image_size: tuple[int, int]) -> Image.Image:
    """Same conversion as the Habitat client's _rgb_array_to_pil (PIL default resampling)."""
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return img.resize(image_size)


def _default_jpeg_encoder() -> Callable[..., bytes]:
    from vla_rpc.core.image import encode_rgb_to_jpeg

    return encode_rgb_to_jpeg


class _RecordingClient:
    """Passes VO calls through and keeps the last JSON response, which the bridge swallows.

    Only its ``timing_ms`` (added by the VO server when timing is on) is read.
    """

    def __init__(self, client: Any) -> None:
        self.client = client
        self.last_response: Optional[dict[str, Any]] = None

    def infer_json(self, method: str, payload: dict[str, Any], blobs: list[dict[str, Any]]):
        self.last_response = None
        result = self.client.infer_json(method, payload, blobs)
        if result is not None and isinstance(result[0], dict):
            self.last_response = result[0]
        return result

    def last_timing(self) -> Optional[dict[str, Any]]:
        return (self.last_response or {}).get("timing_ms")


def _connect(address: str, timeout_ms: int) -> Any:
    from vla_rpc.client import VLAClient

    client = VLAClient(server_addr=address, timeout_ms=int(timeout_ms))
    if not client.connect():
        raise RuntimeError(f"could not open a channel to {address}")
    return client


def _check_model_server(client: Any) -> Any:
    if not client.health_check():
        raise RuntimeError("model server is not healthy")
    info = client.get_server_info()
    if info is None or info.version != HEATMAPVLN_RPC_PROTOCOL_VERSION:
        raise RuntimeError(
            f"model server protocol mismatch: server={getattr(info, 'version', None)!r} "
            f"expected={HEATMAPVLN_RPC_PROTOCOL_VERSION!r}"
        )
    if PPA_ONLINE_AMB3R_CAPABILITY not in set(info.supported_formats):
        raise RuntimeError(
            "model server was not started with --require_ppa_online_amb3r "
            f"(formats={sorted(set(info.supported_formats))})"
        )
    return info


def _check_vo_server(client: Any) -> Any:
    if not client.health_check():
        raise RuntimeError("AMB3R VO server is not healthy")
    info = client.get_server_info()
    if info is None or info.version != AMB3R_VO_RPC_PROTOCOL_VERSION or info.model_version != AMB3R_VO_MODEL_VERSION:
        raise RuntimeError(f"unexpected AMB3R VO server identity: {info}")
    return info


class NavAgent:
    """Client-side state of one navigation episode against the deployed model and VO servers.

    ``model`` and ``vo`` are "host:port" addresses (a vla_rpc VLAClient is opened for each) or
    ready client objects with ``infer_json`` (for example src/deploy/fake_servers.py).
    ``timing=None`` follows HEATMAPVLN_TIMING; client stage times land in ``last_step`` and
    ``last_call``, and a ``timing_ms`` field in a server response is surfaced as is.
    ``on_log`` receives the Habitat client's per-call log lines (e.g. ``print``).
    """

    def __init__(
        self,
        model: Any,
        vo: Any,
        *,
        protocol_seed: int = HEATMAPVLN_RPC_DEFAULT_PROTOCOL_SEED,
        camera: CameraSpec = CameraSpec(),
        timing: Optional[bool] = None,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        rpc_timeout_ms: int = 60_000,
        jpeg_encoder: Optional[Callable[..., bytes]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        if (camera.width, camera.height) != LOOKDOWN_IMAGE_SIZE:
            # The look-down image goes to System 2 at its native size, and the server accepts
            # only 640x480 there; the VO was only run on 640x480 frames too.
            raise ValueError(f"the deployed model needs a {LOOKDOWN_IMAGE_SIZE[0]}x{LOOKDOWN_IMAGE_SIZE[1]} camera")
        if camera != CameraSpec():
            LOGGER.warning("camera %s differs from the simulated one %s: not validated", camera, CameraSpec())
        # Fail on a bad seed now rather than at the first plan call.
        build_rpc_sampling_metadata(protocol_seed=protocol_seed, scene_id="check", episode_id=0, system2_call_index=0)
        self.protocol_seed = int(protocol_seed)
        self.camera = camera
        self.timing = timing
        self.max_steps = int(max_steps)
        self._encoder = jpeg_encoder or _default_jpeg_encoder()
        self._on_log = on_log
        model_client = _connect(model, rpc_timeout_ms) if isinstance(model, str) else model
        vo_client = _connect(vo, rpc_timeout_ms) if isinstance(vo, str) else vo
        self._owned = [c for c, a in ((model_client, model), (vo_client, vo)) if isinstance(a, str)]
        if hasattr(model_client, "get_server_info"):
            self.model_server_info = _check_model_server(model_client)
        if hasattr(vo_client, "get_server_info"):
            self.vo_server_info = _check_vo_server(vo_client)
        self._model = model_client
        self._vo_client = _RecordingClient(vo_client)
        self._vo = OnlineVORPCBridge(self._vo_client, jpeg_quality=VO_JPEG_QUALITY, jpeg_encoder=self._encoder)
        placeholder = Image.new("RGB", VLM_IMAGE_SIZE, color=(0, 0, 0))
        # The right/back/left views the protocol still carries: black, like the client's own
        # fallback for a missing view (capture_panoramic_views).  The deployed server reads
        # only the front view (checked per call through native_front_only, see _check_response).
        self._placeholder_jpeg = self._encode(placeholder)
        self._instruction: Optional[str] = None
        self._vo_timing: dict[str, Any] = {}
        self.last_step: Optional[StepInfo] = None
        self.last_call: Optional[PlanCall] = None

    # ------------------------------------------------------------------ episode
    def reset(self, instruction: str, *, scene_id: str, episode_id: int) -> None:
        """Start an episode.  (scene_id, episode_id, call index) seed System 1's noise.

        The VO session is reset here (its map is lost).  Use a new episode_id per run: the same
        key with the same inputs replays the same noise.
        """
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if not isinstance(scene_id, str) or not scene_id.strip() or "/" in scene_id:
            raise ValueError("scene_id must be a non-empty string without '/'")
        if isinstance(episode_id, bool) or not isinstance(episode_id, int) or episode_id < 0:
            raise ValueError("episode_id must be a non-negative integer")
        self._instruction = None  # a failed VO reset leaves the agent un-reset
        self.scene_id = scene_id.strip()
        self.episode_id = int(episode_id)
        self._vo.reset_episode(f"{self.scene_id}/{self.episode_id:04d}", max_frames=self.max_steps + 1)
        self._step = 0
        self._queue: list[int] = []
        self._forward_count = 0
        # One entry per panorama the Habitat client appends to executed_history_*: the front view
        # (JPEG, 384x384, q90), its capture step and VO frame id.  A step can appear twice.
        self._hist_jpegs: list[bytes] = []
        self._hist_steps: list[int] = []
        self._hist_vo_ids: list[int] = []
        self._last_action: Optional[Action] = None
        self._done = False
        self._failed = False
        self.calls = 0
        self.trajectory_calls = 0
        self.ppa_applied_calls = 0
        self.ppa_warmup_calls = 0
        self.last_step = None
        self.last_call = None
        # The Habitat client's _normalize_instruction: drop one trailing . ! or ? (no strip()).
        self._instruction = instruction[:-1] if instruction.endswith((".", "!", "?")) else instruction

    @property
    def instruction(self) -> Optional[str]:
        return self._instruction

    @property
    def steps(self) -> int:
        """Actions returned so far in this episode (the Habitat client's step_id)."""
        return self._step

    @property
    def done(self) -> bool:
        return bool(self._instruction is not None and self._done)

    def act(
        self,
        front_rgb: np.ndarray,
        lookdown_fn: Callable[[], np.ndarray],
        level_fn: Callable[[], np.ndarray],
    ) -> Action:
        """Return the next action for the robot to execute.

        ``front_rgb``: the frame at the current pose, HxWx3 uint8 RGB, taken with the camera
        pitch given by the previous action's ``camera_pitch_deg`` (level except right after
        LOOK_DOWN).  ``lookdown_fn()``: called at most once, only when this step needs a plan
        call; must return the same camera's frame pitched 30 deg down from level at the current
        pose, and leave the camera level.  ``level_fn()``: called at most once, only right after
        LOOK_DOWN when this step replans after a queued STOP (a chunk such as [5, 0, 0, 0]);
        must return the same camera's frame at level pitch at the current pose, and leave the
        camera level.  Any exception ends the episode: call reset().
        """
        if self._instruction is None:
            raise RuntimeError("call reset() before act()")
        if self._failed:
            raise RuntimeError("an earlier act() failed; call reset() to start a new episode")
        if self._done:
            raise RuntimeError("the episode is over (STOP was returned); call reset()")
        timer = StageTimer(self.timing, cuda_sync=False)
        step = self._step
        self._vo_timing = {}  # the VO server's timing_ms per method, when it sends one
        try:
            with timer.stage("act_total"):
                action, source, call = self._act(front_rgb, lookdown_fn, level_fn, timer)
        except BaseException:
            self._failed = True
            raise
        timing = timer.as_dict()
        if call is not None:
            call.client_timing_ms = timing
        self.last_step = StepInfo(step, action, source, call, timing, self._vo_timing or None)
        return action

    def close(self) -> None:
        for client in self._owned:
            client.close()
        self._owned = []

    def __enter__(self) -> "NavAgent":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ internals
    def _note_vo_timing(self, method: str) -> None:
        timing = self._vo_client.last_timing()
        if timing is not None:
            self._vo_timing[method] = timing

    def _log(self, line: str) -> None:
        if self._on_log is not None:
            self._on_log(line)

    def _encode(self, image: Image.Image, quality: int = MODEL_JPEG_QUALITY) -> bytes:
        return self._encoder(np.asarray(image.convert("RGB"), dtype=np.uint8), quality=quality)

    def _blob(self, name: str, data: bytes, size: tuple[int, int]) -> dict[str, Any]:
        # Same fields as the client's _rpc_blob_from_pil.
        return {"name": name, "data": data, "mime_type": "image/jpeg", "height": int(size[1]), "width": int(size[0])}

    def _remember(self, front_jpeg: bytes, frame_id: int) -> None:
        self._hist_jpegs.append(front_jpeg)
        self._hist_steps.append(self._step)
        self._hist_vo_ids.append(frame_id)

    def _execute(self, code: int) -> Action:
        try:
            action = Action(int(code))
        except ValueError:
            # Differs from the Habitat client, which would pass any code to env.step().
            raise RuntimeError(f"model server returned an action code the robot cannot execute: {code}") from None
        self._step += 1
        self._last_action = action
        if action is Action.STOP:
            self._done = True
        return action

    def _act(self, front_rgb, lookdown_fn, level_fn, timer):
        if self._step >= self.max_steps:
            # Differs from the Habitat client: at the cap it just stops stepping (no STOP action,
            # no request); a robot needs an explicit halt.
            self._done = True
            return Action.STOP, "step_cap", None
        front = self.camera.check_frame(front_rgb, "front_rgb")
        # One VO frame per step: the native frame the robot sees now (top of the client's step loop).
        with timer.stage("vo_ingest"):
            frame_id = self._vo.ingest_rgb(front, capture_step=self._step)
        self._note_vo_timing("ingest_frame")
        # Differs from the Habitat client, which re-renders and re-encodes the panorama on every
        # loop pass and every history entry per call: here each front frame of this step (the one
        # passed in, and the level re-capture of pass_front) is encoded once.  Same pixels, same
        # encoder, so the same bytes.
        with timer.stage("front_encode"):
            front_jpeg = self._encode(_rgb_array_to_pil(front, VLM_IMAGE_SIZE))
        level: list[bytes] = []
        lookdown: list[bytes] = []

        def pass_front(loop_pass: int) -> bytes:
            # The client captures a panorama on every loop pass, and each capture ends by levelling
            # the sensor, so only the first pass of a step sees the pitch the last action left.
            # After LOOK_DOWN a later pass (a replan after a queued STOP) sees a level front.
            if loop_pass == 0 or self._last_action is not Action.LOOK_DOWN:
                return front_jpeg
            if not level:
                with timer.stage("level_capture"):
                    image = level_fn()
                image = self.camera.check_frame(image, "level_fn()")
                with timer.stage("level_encode"):
                    level.append(self._encode(_rgb_array_to_pil(image, VLM_IMAGE_SIZE)))
            return level[0]

        def lookdown_jpeg() -> bytes:
            # Differs from the Habitat client in one unreachable case: two plan calls in one step
            # (a STOP-first chunk) would capture the look-down twice there; here it is reused.
            if not lookdown:
                with timer.stage("lookdown_capture"):
                    image = lookdown_fn()
                image = self.camera.check_frame(image, "lookdown_fn()")
                with timer.stage("lookdown_encode"):
                    lookdown.append(self._encode(_rgb_array_to_pil(image, LOOKDOWN_IMAGE_SIZE)))
            return lookdown[0]

        call: Optional[PlanCall] = None
        for loop_pass in itertools.count():
            pano_front = pass_front(loop_pass)
            if self._queue:
                # Queued action (the client's `if local_actions:` branch): the panorama joins the history first.
                self._remember(pano_front, frame_id)
                action = self._queue.pop(0)
                self._forward_count += 1
                if self._forward_count > MAX_QUEUED_ACTIONS:
                    self._queue, self._forward_count = [], 0
                    continue
                if action == Action.STOP:
                    # STOP inside a chunk means "replan here", not "halt".
                    self._log("  [debug] local trajectory STOP -> replan")
                    self._queue, self._forward_count = [], 0
                    continue
                return self._execute(action), "queue", call

            call = self._plan(pano_front, frame_id, lookdown_jpeg, timer)
            actions = list(call.actions)
            # The client's handling of the response: terminal, empty, first action, rest queued.
            if call.terminal:
                return self._execute(actions[0] if actions else Action.STOP), "terminal", call
            if not actions:
                return self._execute(Action.STOP), "empty", call
            first = actions.pop(0)
            self._queue = actions
            self._forward_count = 0
            if first == Action.STOP:
                # Replan at the same step (the server's anti-deadlock makes this unreachable).
                self._queue = []
                continue
            self._forward_count += 1
            return self._execute(first), "plan", call

    def _plan(
        self, front_jpeg: bytes, frame_id: int, lookdown_jpeg: Callable[[], bytes], timer: StageTimer
    ) -> PlanCall:
        # The client's plan branch, in its order.  History is chosen before this step's
        # panorama is appended; only strictly earlier VO frames qualify.
        indices = sample_unique_past_indices(self._hist_vo_ids, current_frame_id=frame_id, max_history=NUM_HISTORY)
        history_jpegs = [self._hist_jpegs[i] for i in indices]
        history_steps = [self._hist_steps[i] for i in indices]
        with timer.stage("vo_query"):
            pose_fields = self._vo.query_model_pose_fields(
                current_frame_id=frame_id,
                history_frame_ids=[self._hist_vo_ids[i] for i in indices],
            )
        self._note_vo_timing("query_relative_poses")
        vo_line = (
            f"  [amb3r-vo] frame={pose_fields['vo_current_frame_id']} history={pose_fields['vo_history_frame_ids']} "
            f"ready={pose_fields['pose_ready']} phase={pose_fields['vo_provider_phase']} "
            f"revision={pose_fields['vo_trajectory_revision']}"
        )
        self._log(vo_line)
        lookdown = lookdown_jpeg()
        self._remember(front_jpeg, frame_id)
        call_index = self.calls
        self.calls += 1

        with timer.stage("blob_build"):
            blobs = [self._blob("current/front", front_jpeg, VLM_IMAGE_SIZE)]
            blobs += [self._blob(f"current/{view}", self._placeholder_jpeg, VLM_IMAGE_SIZE) for view in VIEWS[1:]]
            for idx, jpeg in enumerate(history_jpegs):
                blobs.append(self._blob(f"history/{idx}/front", jpeg, VLM_IMAGE_SIZE))
                blobs += [
                    self._blob(f"history/{idx}/{view}", self._placeholder_jpeg, VLM_IMAGE_SIZE) for view in VIEWS[1:]
                ]
            blobs.append(self._blob("lookdown", lookdown, LOOKDOWN_IMAGE_SIZE))
        sampling = build_rpc_sampling_metadata(
            protocol_seed=self.protocol_seed,
            scene_id=self.scene_id,
            episode_id=self.episode_id,
            system2_call_index=call_index,
        )
        # Field for field and in the same order as the client's _rpc_plan_panoramic,
        # with the deployed launcher's flags.
        payload: dict[str, Any] = {
            "instruction": self._instruction,
            "num_history": len(history_jpegs),
            "vlm_image_size": list(VLM_IMAGE_SIZE),
            "traj_image_size": list(TRAJ_IMAGE_SIZE),
            "system1_coord_order": "generated",
            "trajectory_selection": "mean",
            "trajectory_x_sign": 1.0,
            "trajectory_heading_alignment": "none",
            "require_deterministic_sampling": True,
            "phase": "joint",
            HEATMAPVLN_RPC_SAMPLING_FIELD: sampling,
        }
        payload.update(pose_fields)
        payload.update(
            {
                "current_capture_step": self._step,
                "history_capture_steps": history_steps,
                "history_age_steps": [self._step - value for value in history_steps],
            }
        )
        with timer.stage("model_rpc"):
            result = self._model.infer_json(PLAN_METHOD, payload, blobs)
        response = self._check_response(result, sampling=sampling, pose_ready=pose_fields["pose_ready"])

        actions = [int(action) for action in response.get("actions", [])]
        call = PlanCall(
            call_index=call_index,
            step=self._step,
            kind=str(response.get("kind") or ""),
            llm_output=response.get("llm_output", ""),
            actions=actions,
            pixel_goal=response.get("pixel_goal"),
            terminal=bool(response.get("terminal", False)),
            pose_ready=bool(pose_fields["pose_ready"]),
            ppa_applied=bool(response.get("ppa_applied")),
            vo_frame_id=int(pose_fields["vo_current_frame_id"]),
            vo_history_frame_ids=list(pose_fields["vo_history_frame_ids"]),
            vo_provider_phase=str(pose_fields["vo_provider_phase"]),
            vo_trajectory_revision=int(pose_fields["vo_trajectory_revision"]),
            history_capture_steps=history_steps,
            response=response,
            server_timing_ms=response.get("timing_ms"),
        )
        if response.get("trajectory_summary"):
            self.trajectory_calls += 1
        if call.kind == "trajectory":
            if call.pose_ready:
                self.ppa_applied_calls += 1
            else:
                self.ppa_warmup_calls += 1
        for line in call.log_lines():
            self._log(line)
        self.last_call = call
        return call

    def _check_response(self, result: Any, *, sampling: dict[str, Any], pose_ready: bool) -> dict[str, Any]:
        """The Habitat client's checks for this call (_rpc_plan_panoramic and the AMB3R runtime checks).

        Stricter in three ways: only the PPA runtime is accepted (the client also takes the EXP-17
        cognition arm), the server must confirm front-only inputs, and actions must be integers.
        """
        if result is None:
            # vla_rpc returns None on any gRPC error, including a timeout.
            raise RuntimeError("model server returned no response (gRPC error or timeout)")
        response = result[0]
        if not isinstance(response, dict) or not response.get("ok", False):
            raise RuntimeError(f"model server error: {response}")
        if response.get("proto_v") != HEATMAPVLN_RPC_PROTOCOL_VERSION:
            raise RuntimeError(f"model response protocol mismatch: {response.get('proto_v')!r}")
        if response.get("phase") != "joint":
            raise RuntimeError(f"model response phase mismatch: {response.get('phase')!r}")
        try:
            echoed = validate_rpc_sampling_metadata(
                response.get(HEATMAPVLN_RPC_SAMPLING_FIELD), require_deterministic=True
            )
        except ValueError as exc:
            raise RuntimeError(f"model response has an invalid sampling record: {exc}") from None
        if echoed != sampling:
            raise RuntimeError(f"model server did not echo the sampling record: {echoed!r} != {sampling!r}")
        if response.get("ppa_runtime") != PPA_ONLINE_AMB3R_CAPABILITY:
            raise RuntimeError(f"model response is not from the PPA runtime: {response.get('ppa_runtime')!r}")
        if response.get("pose_provider") != AMB3R_VO_POSE_PROVIDER:
            raise RuntimeError("model response changed the AMB3R pose provider")
        if response.get("pose_ready") is not pose_ready:
            raise RuntimeError("model response pose_ready differs from the VO query")
        if response.get("kind") == "trajectory" and response.get("ppa_applied") is not pose_ready:
            raise RuntimeError(f"trajectory call with pose_ready={pose_ready} reported ppa_applied={response.get('ppa_applied')!r}")
        # Ours: the right/back/left views are placeholders, which is only sound while the server
        # builds System 2's prompt and the PPA heads from front views alone.
        if response.get("native_front_only") is not True:
            raise RuntimeError("model server did not confirm front-only inputs (native_front_only)")
        actions = response.get("actions", [])
        if not isinstance(actions, list) or any(isinstance(a, bool) or not isinstance(a, int) for a in actions):
            raise RuntimeError(f"model response has invalid actions: {actions!r}")
        return response


__all__ = [
    "Action",
    "CameraSpec",
    "FORWARD_STEP_M",
    "LOOK_DOWN_DEG",
    "NavAgent",
    "PlanCall",
    "StepInfo",
    "TURN_STEP_DEG",
]

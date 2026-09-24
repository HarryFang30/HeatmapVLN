"""Read EXP-18 top-down maps (output of ``render_topdown.py``) and map world <-> pixel.

Importable from both dev-machine environments: ``envs/vlnce`` (Python 3.8) and
``envs/qwen25`` (Python 3.12).  numpy + PIL only.

Layout: ``<root>/<scene>/topdown.json`` plus ``level_<i>_{rgb,rgba,navmask}.png``;
``<root>`` defaults to ``$EXP18_ROOT/topdown``.  Scene names are canonical ids
(``2azQ1b91cZZ``, ``m49MsVC7BwA``); ``m49MsVC7BwA.basis``, ``00796-m49MsVC7BwA``
and full ``.glb`` paths are accepted everywhere.

Conventions (verified in the EXP-18 probe)
------------------------------------------
* World: habitat, y up.  Each map is an orthographic view from +y with +x to
  the right and +z down; it is not mirrored.
* Pixel: ``col = (x - x0)/mpp - 0.5``, ``row = (z - z0)/mpp - 0.5``; integer
  ``(row, col)`` is a pixel centre.  rgb, rgba and navmask of one level share
  this grid, and the navmask pixel ``(row, col)`` is habitat's
  ``is_navigable`` sample at exactly that centre.
* matplotlib: ``ax.imshow(img, extent=level.extent)`` (default
  ``origin="upper"``), then plot world ``(x, z)`` directly.  Call
  ``ax.set_autoscale_on(False)`` before ``add_patch`` or the view grows.
* Heading: habitat yaw ``theta`` rotates about +y; the agent/camera looks along
  ``forward = (-sin theta, -cos theta)`` in ``(x, z)``, which is also its
  ``(col, row)`` direction on the map.  ``theta = 0`` looks towards -z, i.e.
  map-up.  From a camera-to-world matrix ``R = c2w[:3, :3]``:
  ``forward = -R[:, 2]`` (the camera looks down its -z), ``right = R[:, 0]``,
  ``theta = atan2(-forward_x, -forward_z)``.  On the map ``right = (-f_z, f_x)``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

SCHEMA = "exp18-topdown-v1"


def default_root() -> Path:
    try:
        from scripts.exp18.common import EXP_ROOT
    except ImportError:  # imported by path, outside the repo package
        workspace = os.environ.get("EXP18_WORKSPACE", "/mnt/afs/liwenhao/agent/370910109")
        EXP_ROOT = Path(os.environ.get("EXP18_ROOT", workspace + "/model/exp18_first_person_viz"))
    return Path(EXP_ROOT) / "topdown"


def canonical_scene_id(name: str) -> str:
    """``hm3d/train/00796-m49MsVC7BwA/m49MsVC7BwA.basis.glb`` -> ``m49MsVC7BwA``."""
    base = os.path.basename(str(name).rstrip("/"))
    for suffix in (".glb", ".navmesh", ".basis"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return re.sub(r"^\d{5}-", "", base)


# ----------------------------------------------------------------- heading ---

def yaw_to_forward(yaw) -> np.ndarray:
    """Habitat yaw (rad, about +y) -> unit forward ``(x, z)``; vectorised over yaw."""
    yaw = np.asarray(yaw, dtype=np.float64)
    return np.stack([-np.sin(yaw), -np.cos(yaw)], axis=-1)


def forward_to_yaw(forward_xz) -> np.ndarray:
    f = np.asarray(forward_xz, dtype=np.float64)
    return np.arctan2(-f[..., 0], -f[..., 1])


def forward_from_c2w(c2w) -> np.ndarray:
    """Unit forward ``(x, z)`` of a camera-to-world 4x4 / 3x3 (camera looks down -z)."""
    rot = np.asarray(c2w, dtype=np.float64)[..., :3, :3]
    f = -rot[..., [0, 2], 2]
    return f / np.linalg.norm(f, axis=-1, keepdims=True)


def yaw_from_c2w(c2w) -> np.ndarray:
    return forward_to_yaw(forward_from_c2w(c2w))


def yaw_from_quaternion(wxyz) -> np.ndarray:
    """Yaw of a (mostly) yaw-only habitat rotation quaternion ``(w, x, y, z)``."""
    q = np.asarray(wxyz, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # forward = R @ (0, 0, -1) -> (x, z) components; exact for any rotation
    fx = -2.0 * (x * z + w * y)
    fz = -(1.0 - 2.0 * (x * x + y * y))
    return np.arctan2(-fx, -fz)


# ------------------------------------------------------------------- level ---

class Level:
    """One floor level of one scene: transform, images and rooms."""

    def __init__(self, meta: dict, scene_dir: Path):
        self.meta = meta
        self.scene_dir = Path(scene_dir)
        self.index = int(meta["index"])
        self.floor_y = float(meta["floor_y"])
        self.sub_levels = [float(y) for y in meta.get("sub_levels", [self.floor_y])]
        self.y_min = float(meta.get("y_min", self.floor_y))
        self.y_max = float(meta.get("y_max", self.floor_y))
        self.x0 = float(meta["x0"])
        self.z0 = float(meta["z0"])
        self.mpp = float(meta["mpp"])
        self.width = int(meta["width"])
        self.height = int(meta["height"])
        self.rooms = meta.get("rooms") or []
        self._cache = {}

    def __repr__(self) -> str:
        return "Level(%d, y=%.3f, %dx%d px, %.2f cm/px)" % (
            self.index, self.floor_y, self.width, self.height, self.mpp * 100)

    @property
    def extent(self) -> list:
        """matplotlib ``imshow`` extent ``[left, right, bottom, top]`` in world metres."""
        return [self.x0, self.x0 + self.width * self.mpp, self.z0 + self.height * self.mpp, self.z0]

    @property
    def shape(self) -> tuple:
        return (self.height, self.width)

    def world_to_px(self, x, z) -> tuple:
        """World ``(x, z)`` -> float ``(row, col)``; round for pixel indices."""
        col = (np.asarray(x, dtype=np.float64) - self.x0) / self.mpp - 0.5
        row = (np.asarray(z, dtype=np.float64) - self.z0) / self.mpp - 0.5
        return row, col

    def px_to_world(self, row, col) -> tuple:
        x = self.x0 + (np.asarray(col, dtype=np.float64) + 0.5) * self.mpp
        z = self.z0 + (np.asarray(row, dtype=np.float64) + 0.5) * self.mpp
        return x, z

    def contains(self, x, z) -> np.ndarray:
        row, col = self.world_to_px(x, z)
        return (row > -0.5) & (row < self.height - 0.5) & (col > -0.5) & (col < self.width - 0.5)

    def _load(self, kind: str) -> np.ndarray:
        if kind not in self._cache:
            path = self.scene_dir / self.meta["files"][kind]
            self._cache[kind] = np.asarray(Image.open(path))
        return self._cache[kind]

    def rgb(self) -> np.ndarray:
        return self._load("rgb")

    def rgba(self) -> np.ndarray:
        return self._load("rgba")

    def navmask(self) -> np.ndarray:
        return self._load("navmask") > 127

    def on_navmask(self, x, z, tol_px: int = 0) -> np.ndarray:
        """True where ``(x, z)`` hits a navigable pixel (any within ``tol_px``)."""
        mask = self.navmask()
        row, col = self.world_to_px(x, z)
        row = np.rint(np.atleast_1d(row)).astype(int)
        col = np.rint(np.atleast_1d(col)).astype(int)
        hit = np.zeros(row.shape, dtype=bool)
        for dr in range(-tol_px, tol_px + 1):
            for dc in range(-tol_px, tol_px + 1):
                r, c = row + dr, col + dc
                ok = (r >= 0) & (r < self.height) & (c >= 0) & (c < self.width)
                hit[ok] |= mask[r[ok], c[ok]]
        return hit

    def crop_window(self, img: np.ndarray, center_xz, half_size_m: float, fill=255) -> tuple:
        """Map-aligned window (-z up) around a world point, pixel-exact (no resampling).

        Returns ``(crop, extent)``; pixels outside the map are ``fill``.
        """
        row, col = self.world_to_px(center_xz[0], center_xz[1])
        half = int(math.ceil(half_size_m / self.mpp))
        r0, c0 = int(round(float(row))) - half, int(round(float(col))) - half
        size = 2 * half + 1
        out = np.empty((size, size) + img.shape[2:], dtype=img.dtype)
        out[...] = fill
        rs, cs = max(r0, 0), max(c0, 0)
        re_, ce = min(r0 + size, img.shape[0]), min(c0 + size, img.shape[1])
        if rs < re_ and cs < ce:
            out[rs - r0:re_ - r0, cs - c0:ce - c0] = img[rs:re_, cs:ce]
        x_left, z_top = self.px_to_world(r0 - 0.5, c0 - 0.5)
        extent = [float(x_left), float(x_left + size * self.mpp), float(z_top + size * self.mpp), float(z_top)]
        return out, extent

    def heading_up_crop(self, img: np.ndarray, center_xz, forward_xz=None, yaw=None,
                        half_size_m: float = 3.0, out_px: int = 512, fill=255,
                        nearest: bool = False) -> "EgoCrop":
        """Square window centred on ``center_xz`` and rotated so ``forward`` points up.

        Give either ``forward_xz`` (e.g. ``forward_from_c2w(c2w)``) or habitat
        ``yaw``.  Bilinear resampling (``nearest=True`` for masks).  Returns an
        :class:`EgoCrop`, whose ``world_to_px`` maps world points into the crop.
        """
        if forward_xz is None:
            if yaw is None:
                raise ValueError("give forward_xz or yaw")
            forward_xz = yaw_to_forward(float(yaw))
        crop = EgoCrop(center_xz, forward_xz, half_size_m, out_px)
        # world position of every output pixel centre -> source (row, col)
        j, i = np.meshgrid(np.arange(out_px), np.arange(out_px))
        wx, wz = crop.px_to_world(i, j)
        row, col = self.world_to_px(wx, wz)
        crop.image = _sample(img, row, col, fill=fill, nearest=nearest)
        return crop


class EgoCrop:
    """Heading-up window: output row 0 is ``half_size_m`` ahead, col 0 is to the left."""

    def __init__(self, center_xz, forward_xz, half_size_m: float, out_px: int):
        f = np.asarray(forward_xz, dtype=np.float64)[:2]
        self.forward = f / np.linalg.norm(f)
        self.right = np.array([-self.forward[1], self.forward[0]])
        self.center = np.asarray(center_xz, dtype=np.float64)[:2]
        self.half_size_m = float(half_size_m)
        self.size = int(out_px)
        self.mpp = 2.0 * self.half_size_m / self.size
        self.image = None

    @property
    def extent(self) -> list:
        """imshow extent in egocentric metres: x = right, y = forward (use origin='upper')."""
        h = self.half_size_m
        return [-h, h, -h, h]

    def px_to_world(self, row, col) -> tuple:
        a = (np.asarray(col, dtype=np.float64) + 0.5 - self.size / 2.0) * self.mpp   # right
        b = (self.size / 2.0 - np.asarray(row, dtype=np.float64) - 0.5) * self.mpp   # forward
        x = self.center[0] + a * self.right[0] + b * self.forward[0]
        z = self.center[1] + a * self.right[1] + b * self.forward[1]
        return x, z

    def world_to_local(self, x, z) -> tuple:
        """World ``(x, z)`` -> egocentric ``(right, forward)`` metres."""
        dx = np.asarray(x, dtype=np.float64) - self.center[0]
        dz = np.asarray(z, dtype=np.float64) - self.center[1]
        return dx * self.right[0] + dz * self.right[1], dx * self.forward[0] + dz * self.forward[1]

    def world_to_px(self, x, z) -> tuple:
        a, b = self.world_to_local(x, z)
        return self.size / 2.0 - 0.5 - b / self.mpp, a / self.mpp + self.size / 2.0 - 0.5


def _sample(img: np.ndarray, row: np.ndarray, col: np.ndarray, fill=255, nearest: bool = False) -> np.ndarray:
    """Sample ``img`` at float pixel-centre coordinates; outside -> ``fill``."""
    h, w = img.shape[:2]
    extra = img.shape[2:]
    fill_arr = np.broadcast_to(np.asarray(fill, dtype=np.float64), extra) if extra else float(fill)
    if nearest:
        r, c = np.rint(row).astype(int), np.rint(col).astype(int)
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        out = np.empty(row.shape + extra, dtype=img.dtype)
        out[...] = fill_arr
        out[ok] = img[r[ok], c[ok]]
        return out
    r0, c0 = np.floor(row).astype(int), np.floor(col).astype(int)
    fr, fc = row - r0, col - c0
    acc = np.zeros(row.shape + extra, dtype=np.float64)
    for dr, dc, wgt in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc),
                        (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
        r, c = r0 + dr, c0 + dc
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        vals = np.empty(row.shape + extra, dtype=np.float64)
        vals[...] = fill_arr
        vals[ok] = img[r[ok], c[ok]]
        acc += vals * (wgt[..., None] if extra else wgt)
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        return np.clip(np.rint(acc), info.min, info.max).astype(img.dtype)
    return acc.astype(img.dtype)


# ------------------------------------------------------------------- scene ---

class SceneTopdown:
    """All levels of one scene, loaded from ``<root>/<scene>/topdown.json``."""

    def __init__(self, meta: dict, scene_dir: Path):
        self.meta = meta
        self.scene_dir = Path(scene_dir)
        self.scene = meta["scene"]
        self.levels = [Level(m, self.scene_dir) for m in meta["levels"]]
        self.level_ys = np.array([lv.floor_y for lv in self.levels], dtype=np.float64)

    def __repr__(self) -> str:
        return "SceneTopdown(%s, %s)" % (self.scene, self.levels)

    def level_index(self, y) -> np.ndarray:
        """Nearest level for navmesh height(s) ``y``: distance to each level's
        ``[y_min, y_max]`` sub-level range (ties -> nearer main floor).  Points
        on stairs or landings between storeys go to the nearer one."""
        y = np.asarray(y, dtype=np.float64)[..., None]
        lo = np.array([lv.y_min for lv in self.levels])
        hi = np.array([lv.y_max for lv in self.levels])
        gap = np.maximum(np.maximum(lo - y, y - hi), 0.0)
        return (gap + 1e-3 * np.abs(y - self.level_ys)).argmin(axis=-1)

    def pick_level(self, y: float) -> Level:
        return self.levels[int(self.level_index(float(y)))]


def scene_dir(scene: str, root=None) -> Path:
    return Path(root if root is not None else default_root()) / canonical_scene_id(scene)


def load_topdown(scene: str, root=None) -> SceneTopdown:
    directory = scene_dir(scene, root)
    with open(directory / "topdown.json") as handle:
        meta = json.load(handle)
    if meta.get("schema") != SCHEMA:
        raise ValueError("%s: schema %r, expected %r" % (directory, meta.get("schema"), SCHEMA))
    return SceneTopdown(meta, directory)


# --------------------------------------------------------------------- CLI ---

def _episode_points(episodes_path: str, episode_id: str) -> tuple:
    with gzip.open(episodes_path, "rt") as handle:
        data = json.load(handle)
    for ep in data["episodes"]:
        if str(ep["episode_id"]) == str(episode_id):
            return np.asarray(ep["reference_path"], dtype=np.float64), canonical_scene_id(ep["scene_id"])
    raise KeyError("episode %s not in %s" % (episode_id, episodes_path))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a scene's top-down maps and check points on the navmask.")
    parser.add_argument("scene", nargs="?", default=None, help="scene id (optional with --episode-id)")
    parser.add_argument("--root", default=None, help="top-down root (default $EXP18_ROOT/topdown)")
    parser.add_argument("--points", default=None, help="JSON file with [[x, y, z], ...] world points")
    parser.add_argument("--episodes", default=None, help="R2R-style episodes .json.gz")
    parser.add_argument("--episode-id", default=None, help="check this episode's reference_path")
    parser.add_argument("--tol-px", type=int, default=0)
    args = parser.parse_args(argv)

    points, scene = None, args.scene
    if args.episode_id is not None:
        points, ep_scene = _episode_points(args.episodes, args.episode_id)
        scene = scene or ep_scene
    elif args.points:
        with open(args.points) as handle:
            points = np.asarray(json.load(handle), dtype=np.float64)
    if scene is None:
        parser.error("give a scene or --episode-id")
    td = load_topdown(scene, args.root)
    print(td)
    for lv in td.levels:
        print("  level %d: y=%.3f extent=%s rooms=%d" % (lv.index, lv.floor_y, np.round(lv.extent, 3).tolist(), len(lv.rooms)))
    if points is None:
        return 0
    idx = td.level_index(points[:, 1])
    ok_all = True
    for k, (p, li) in enumerate(zip(points, idx)):
        lv = td.levels[int(li)]
        row, col = lv.world_to_px(p[0], p[2])
        on = bool(lv.on_navmask(p[0], p[2], tol_px=args.tol_px)[0])
        ok_all &= on
        print("  pt %2d xyz=(%.3f, %.3f, %.3f) level=%d px=(row %.1f, col %.1f) navigable=%s"
              % (k, p[0], p[1], p[2], li, float(row), float(col), on))
    print("ALL_ON_NAVMASK" if ok_all else "SOME_OFF_NAVMASK")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())

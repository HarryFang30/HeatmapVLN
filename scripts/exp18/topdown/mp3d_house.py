"""Minimal Matterport3D ``.house`` parser: levels, rooms (label + floor polygon), panoramas.

The ``.house`` frame is z-up; the habitat MP3D frame is y-up with
``habitat (x, y, z) = (hx, hz, -hy)``.  A room polygon in habitat ``(x, z)`` is
therefore ``(hx, -hy)`` and its floor height is the region's ``zlo``.  Checked
in the EXP-18 probe: 97-99% of random navigable points fall inside a room
polygon of the same floor (mirrored-z control 2-55%), and the navmesh sits
0.06-0.13 m above the region floor.

Line formats used (one record per line, whitespace separated)::

    L level_idx #regions label  px py pz  xlo ylo zlo  xhi yhi zhi ...
    R region_idx level_idx 0 0 label  px py pz  xlo ylo zlo  xhi yhi zhi  height ...
    S surface_idx region_idx 0 label  px py pz  nx ny nz  bbox ...
    V vertex_idx surface_idx label  px py pz  nx ny nz ...
    P name panorama_idx region_idx 0  px py pz ...

Python 3.8 compatible, numpy only.
"""
from __future__ import annotations

import numpy as np

REGION_LABELS = {
    "a": "bathroom", "b": "bedroom", "c": "closet", "d": "dining room",
    "e": "entryway/foyer/lobby", "f": "family room", "g": "garage", "h": "hallway",
    "i": "library", "j": "laundry room/mudroom", "k": "kitchen", "l": "living room",
    "m": "meeting/conference room", "n": "lounge", "o": "office",
    "p": "porch/terrace/deck", "r": "rec/game room", "s": "stairs", "t": "toilet",
    "u": "utility/tool room", "v": "tv room", "w": "workout/gym", "x": "outdoor",
    "y": "balcony", "z": "other room", "B": "bar", "C": "classroom", "D": "dining booth",
    "S": "spa/sauna", "Z": "junk", "-": "no label",
}


def house_to_habitat(points) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def polygon_area(poly_xz: np.ndarray) -> float:
    x, z = poly_xz[:, 0], poly_xz[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1))))


def polygon_centroid(poly_xz: np.ndarray) -> np.ndarray:
    """Area centroid (falls back to the vertex mean for degenerate polygons)."""
    x, z = poly_xz[:, 0], poly_xz[:, 1]
    x1, z1 = np.roll(x, -1), np.roll(z, -1)
    cross = x * z1 - x1 * z
    area = cross.sum() / 2.0
    if abs(area) < 1e-9:
        return poly_xz.mean(axis=0)
    return np.array([((x + x1) * cross).sum(), ((z + z1) * cross).sum()]) / (6.0 * area)


def parse_house(path) -> dict:
    """Return ``{"levels": {idx: ...}, "regions": {idx: ...}, "panoramas": [...]}`` in habitat coords.

    Each region carries ``code``, ``name``, ``level`` (the house's own level index,
    which need not match navmesh floors), ``floor_y``, ``height``, ``x_range``,
    ``z_range``, ``center`` and ``polygon_xz`` (``(N, 2)`` array or ``None``).
    When a region has several floor surfaces the largest polygon is kept.
    """
    levels, regions, surfaces, verts, panos = {}, {}, {}, {}, []
    with open(path) as handle:
        for line in handle:
            t = line.split()
            if not t:
                continue
            kind = t[0]
            if kind == "L" and len(t) >= 13:
                levels[int(t[1])] = {
                    "n_regions": int(t[2]),
                    "label": t[3],
                    "center": house_to_habitat([float(v) for v in t[4:7]]),
                    "floor_y": float(t[9]),
                }
            elif kind == "R" and len(t) >= 16:
                lo = np.array([float(v) for v in t[9:12]])
                hi = np.array([float(v) for v in t[12:15]])
                regions[int(t[1])] = {
                    "level": int(t[2]),
                    "code": t[5],
                    "name": REGION_LABELS.get(t[5], t[5]),
                    "center": house_to_habitat([float(v) for v in t[6:9]]),
                    "x_range": (float(lo[0]), float(hi[0])),
                    "z_range": (float(-hi[1]), float(-lo[1])),
                    "floor_y": float(lo[2]),
                    "height": float(t[15]),
                    "polygon_xz": None,
                }
            elif kind == "S" and len(t) >= 5:
                surfaces[int(t[1])] = {"region": int(t[2]), "label": t[4]}
            elif kind == "V" and len(t) >= 7:
                verts.setdefault(int(t[2]), []).append((int(t[1]), [float(v) for v in t[4:7]]))
            elif kind == "P" and len(t) >= 8:
                panos.append({
                    "name": t[1],
                    "index": int(t[2]),
                    "region": int(t[3]),
                    "position": house_to_habitat([float(v) for v in t[5:8]]),
                })
    for sid, surface in surfaces.items():
        vs = sorted(verts.get(sid, []))
        region = regions.get(surface["region"])
        if region is None or len(vs) < 3:
            continue
        poly = house_to_habitat([v for _, v in vs])[:, [0, 2]]
        old = region["polygon_xz"]
        if old is None or polygon_area(poly) > polygon_area(old):
            region["polygon_xz"] = poly
    return {"levels": levels, "regions": regions, "panoramas": panos}


def rooms_by_level(house: dict, level_ys, max_dy: float = 1.0) -> tuple:
    """Assign rooms to navmesh floor levels by nearest floor height.

    ``level_ys[i]`` is level ``i``'s height or a list of its sub-level heights.
    Returns ``(per_level, unassigned)``: ``per_level[i]`` is a list of JSON-ready
    room dicts for level ``i``; rooms whose floor is more than ``max_dy`` from
    every level (or without a polygon) go to ``unassigned``.
    """
    groups = [np.atleast_1d(np.asarray(y, dtype=np.float64)) for y in level_ys]
    per_level = [[] for _ in range(len(groups))]
    unassigned = []
    for idx in sorted(house["regions"]):
        region = house["regions"][idx]
        poly = region["polygon_xz"]
        room = {
            "region": int(idx),
            "house_level": int(region["level"]),
            "code": region["code"],
            "label": region["name"],
            "floor_y": round(region["floor_y"], 4),
            "height": round(region["height"], 4),
            "center_xz": [round(float(region["center"][0]), 4), round(float(region["center"][2]), 4)],
            "polygon_xz": None if poly is None else np.round(poly, 4).tolist(),
            "area_m2": None if poly is None else round(polygon_area(poly), 3),
        }
        if poly is not None:
            room["center_xz"] = [round(float(v), 4) for v in polygon_centroid(poly)]
        if poly is None or not groups:
            unassigned.append(room)
            continue
        dy = np.array([np.min(np.abs(g - region["floor_y"])) for g in groups])
        best = int(np.argmin(dy))
        if dy[best] > max_dy:
            unassigned.append(room)
        else:
            per_level[best].append(room)
    return per_level, unassigned

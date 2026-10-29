"""Twin layout.yaml (RoboCasa subset) -> a layout dict RoboCasa's own scene builder places where the twin is (TWN-4).

`layout.yaml` keeps the twin contract (full sizes, absolute centre `pos`, `z_rot` = twin yaw; see docs/04-contracts.md)
and RoboCasa's `KitchenArena` reads it without an error, but it builds a different room (G6, run in real RoboCasa):

- `Wall` and `Floor` take half extents in their own rotated frame (`[half length, half height, half thickness]`), a
  `wall_side` that turns the slab and pushes it outward by its thickness, and ignore `z_rot`. A row with no
  `wall_side` is a `back` wall, so every side wall came out as a 2x-long slab across the room.
- A RoboCasa fixture with `z_rot` 0 faces -y (layouts put the back wall at the largest y); a twin fixture with yaw 0
  faces +y. Every front pointed at its wall.
- A sink is an interior fixture: the builder never moves it, the counter around it places it (`interior_obj`) and
  cuts the opening. Without that it floats inside a solid counter.
- A fixture's `pos` is its body origin, not its centre; the stove's model origin is 0.15 m above its centre.

`robocasa_layout` keeps everything else and turns each of those into RoboCasa's own form. It reads only the layout
(no RoboCasa import), so it runs offline and in tests. `origin_offsets` (fixture name -> centre minus body origin in
the fixture's frame) and `overhang` are what RoboCasa reports for the chosen style; scripts/check_robocasa_layout.py
measures them on a first build and passes them back. Without them every origin is assumed to be the centre.
"""

from __future__ import annotations

import math

# RoboCasa's scene_builder.FIXTURES keys (robocasa 1.0.1), so an unknown type becomes an explicit box, not an error.
ROBOCASA_TYPES = frozenset(
    {
        "hinge_cabinet",
        "single_cabinet",
        "open_cabinet",
        "panel_cabinet",
        "housing_cabinet",
        "drawer",
        "counter",
        "stove",
        "stove_wide",
        "stovetop",
        "oven",
        "microwave",
        "hood",
        "sink",
        "fridge_french_door",
        "fridge_side_by_side",
        "fridge_bottom_freezer",
        "dishwasher",
        "wall",
        "floor",
        "box",
        "accessory",
        "paper_towel",
        "dish_rack",
        "stool",
        "coffee_machine",
        "toaster",
        "toaster_oven",
        "blender",
        "stand_mixer",
        "electric_kettle",
        "wall_accessory",
        "window",
    }
)
FRONT_FLIP = math.pi  # twin yaw 0 faces +y; RoboCasa z_rot 0 faces -y
SINK_DOOR_TH = 0.015  # Counter._place_interior_obj adds this to the overhang ("thickness of cabinet doors")
TWIN_ONLY_KEYS = ("island",)  # twin flags no RoboCasa fixture constructor accepts


def _wrap(a: float) -> float:
    return round((a + math.pi) % (2 * math.pi) - math.pi, 6) or 0.0


def _rot(xy, a: float) -> tuple[float, float]:
    c, s = math.cos(a), math.sin(a)
    return c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]


def _r(v) -> list[float]:
    return [round(float(x), 6) for x in v]


def wall_side(row: dict, center: tuple[float, float]) -> str | None:
    """RoboCasa `wall_side` of an axis-aligned twin wall row (the side of the room it bounds), None if oblique."""
    if row.get("z_rot"):
        return None
    sx, sy = row["size"][:2]
    x, y = row["pos"][:2]
    if sx >= sy:
        return "back" if y > center[1] else "front"
    return "right" if x > center[0] else "left"


def _wall(row: dict, center: tuple[float, float]) -> dict:
    side = wall_side(row, center)
    if side is None:  # RoboCasa walls only face the four axes: keep an oblique wall as an explicit box
        return {
            "name": row["name"],
            "type": "box",
            "size": _r(row["size"]),
            "pos": _r(row["pos"]),
            "z_rot": row["z_rot"],
        }
    sx, sy, h = row["size"]
    along_x = side in ("back", "front")
    length, th = (sx, sy) if along_x else (sy, sx)
    x, y, z = row["pos"]
    out = {"back": (0, 1), "front": (0, -1), "right": (1, 0), "left": (-1, 0)}[side]
    face = [x - out[0] * th / 2, y - out[1] * th / 2, z]  # RoboCasa pushes the slab outward by its half thickness
    # default_wall_th 0: RoboCasa would otherwise add 2 cm to each half length; the twin's wall is exact.
    return {
        "name": row["name"],
        "type": "wall",
        "wall_side": side,
        "size": _r([length / 2, h / 2, th / 2]),
        "pos": _r(face),
        "default_wall_th": 0.0,
    }


def _floor(row: dict) -> dict:
    sx, sy, th = row["size"]
    x, y, _ = row["pos"]
    # Floor half extents; RoboCasa lowers it by its half thickness so the top is at pos z = 0.
    return {"name": row["name"], "type": "floor", "size": _r([sx / 2, sy / 2, th / 2]), "pos": _r([x, y, 0.0])}


def _contains(host: dict, xy) -> bool:
    lx, ly = _rot((xy[0] - host["pos"][0], xy[1] - host["pos"][1]), -host.get("z_rot", 0.0))
    return abs(lx) <= host["size"][0] / 2 + 1e-6 and abs(ly) <= host["size"][1] / 2 + 1e-6


def robocasa_layout(layout: dict, origin_offsets: dict | None = None, overhang: float = 0.05) -> tuple[dict, list[str]]:
    """Return (RoboCasa layout dict, notes). Fixture names are kept; RoboCasa appends `_<group>` to each."""
    offsets = origin_offsets or {}
    notes: list[str] = []
    room = layout.get("room") or {}
    floors = room.get("floor") or []
    walls = room.get("walls") or []
    if floors:
        center = tuple(floors[0]["pos"][:2])
    else:
        xs = [w["pos"][0] for w in walls] or [0.0]
        ys = [w["pos"][1] for w in walls] or [0.0]
        center = ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2)
    out_room: dict = {"walls": [_wall(w, center) for w in walls]}
    for w in out_room["walls"]:
        if w["type"] == "box":
            notes.append(f"wall {w['name']}: oblique, kept as a box (RoboCasa walls face the four axes)")
    if floors:
        out_room["floor"] = [_floor(f) for f in floors]
    doc: dict = {"room": out_room}

    rows = [
        dict(r)
        for k, g in layout.items()
        if k != "room"
        for key, v in g.items()
        if not key.startswith("group_")
        for r in v
    ]
    by_name = {r["name"]: r for r in rows}
    hosts: dict[str, str] = {}  # sink -> counter that holds it
    for r in rows:
        if r["type"] == "sink":
            host = next(
                (
                    c
                    for c in rows
                    if c["type"] == "counter"
                    and _contains(c, r["pos"])
                    and abs(_wrap(c.get("z_rot", 0.0) - r.get("z_rot", 0.0))) < 1e-3
                    and c["name"] not in hosts.values()
                ),
                None,
            )
            if host is None:
                notes.append(f"sink {r['name']}: no counter under it, placed on its own (no counter opening)")
            else:
                hosts[r["name"]] = host["name"]

    groups: dict[float, list[dict]] = {}
    for r in rows:
        if r["name"] in hosts.values():
            continue  # a host counter is written right after its sink
        theta = _wrap(r.get("z_rot", 0.0) + FRONT_FLIP)
        group = groups.setdefault(theta, [])
        if r["name"] in hosts:
            group.append(_fixture(r, theta, offsets, notes, interior=True))
            counter = by_name[hosts[r["name"]]]
            row = _fixture(counter, theta, offsets, notes)
            row.update(_sink_placement(r, counter, theta, overhang, offsets.get(counter["name"], (0.0, 0.0, 0.0))))
            row["interior_obj"] = r["name"]
            group.append(row)
        else:
            group.append(_fixture(r, theta, offsets, notes))
    for i, (theta, group) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        name = "main_group" if i == 0 else f"group_{i}"
        doc[name] = {"group_origin": [0, 0], "group_pos": [0.0, 0.0], "group_z_rot": theta, "bottom_row": group}
    return doc, notes


def _fixture(r: dict, theta: float, offsets: dict, notes: list[str], interior: bool = False) -> dict:
    t = r["type"]
    if t not in ROBOCASA_TYPES:
        notes.append(f"{r['name']}: type {t!r} is not a RoboCasa fixture, kept as a box")
        t = "box"
    row = {"name": r["name"], "type": t, "size": _r(r["size"])}
    for k in TWIN_ONLY_KEYS:
        if r.get(k):
            notes.append(f"{r['name']}: twin flag {k!r} has no RoboCasa fixture argument, not written")
    if "default_config_name" in r:
        row["default_config_name"] = r["default_config_name"]
    if not interior:
        # the group turns the room frame by theta about (0, 0): write the body origin in the group's frame
        ox, oy, oz = offsets.get(r["name"], (0.0, 0.0, 0.0))
        lx, ly = _rot(r["pos"][:2], -theta)
        row["pos"] = _r([lx - ox, ly - oy, r["pos"][2] - oz])
    return row


def _sink_placement(sink: dict, counter: dict, theta: float, overhang: float, counter_offset) -> dict:
    """obj_x_percent / obj_y_percent that put the sink's centre where the twin has it (Counter._place_interior_obj
    measures them from the counter's body origin, which is its centre minus its origin offset)."""
    w, d = counter["size"][:2]
    dx, dy = _rot((sink["pos"][0] - counter["pos"][0], sink["pos"][1] - counter["pos"][1]), -theta)
    dx, dy = dx + counter_offset[0], dy + counter_offset[1]
    pad = overhang + SINK_DOOR_TH
    return {"obj_x_percent": round(0.5 + dx / w, 6), "obj_y_percent": round(0.5 + (dy - pad / 2) / (d - pad), 6)}

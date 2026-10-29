"""layout.yaml -> RoboCasa-native layout (G6 / TWN-4). Runs without RoboCasa; scripts/check_robocasa_layout.py
builds the same output in real RoboCasa."""

import math
from pathlib import Path

import pytest
import yaml

from arrival.scan import roomplan
from arrival.scan.bundle import load_room
from arrival.twin.compile import compile_twin, layout_dict
from arrival.twin.robocasa_layout import FRONT_FLIP, SINK_DOOR_TH, robocasa_layout

ROOT = Path(__file__).resolve().parents[1]


def _layout(bundle):
    return yaml.safe_load(yaml.safe_dump(layout_dict(compile_twin(bundle)), sort_keys=False))


@pytest.fixture(scope="module")
def hero():
    return _layout(load_room("hero_kitchen"))


@pytest.fixture(scope="module")
def kitchen():
    p = ROOT / "tests" / "fixtures" / "roomplan" / "kitchen_real_01.json"
    return _layout(roomplan.load(p, p.stem))


def _rows(doc):
    return {r["name"]: (g, r) for k, g in doc.items() if k != "room" for r in g["bottom_row"]}


def _rot(xy, a):
    return (math.cos(a) * xy[0] - math.sin(a) * xy[1], math.sin(a) * xy[0] + math.cos(a) * xy[1])


def _slab(w):
    """Where RoboCasa's Wall puts the slab: pos pushed outward by size[2] (others.py Wall.__init__)."""
    out = {"back": (0, 1), "front": (0, -1), "right": (1, 0), "left": (-1, 0)}[w["wall_side"]]
    x, y, z = w["pos"]
    hl, hh, ht = w["size"]
    center = [x + out[0] * ht, y + out[1] * ht, z]
    full = [2 * ht, 2 * hl, 2 * hh] if out[0] else [2 * hl, 2 * ht, 2 * hh]
    return center, full


@pytest.mark.parametrize("which", ["hero", "kitchen"])
def test_walls_are_robocasa_walls_on_the_twin_slab(which, request):
    src = request.getfixturevalue(which)
    doc, notes = robocasa_layout(src)
    assert notes == []
    for w_in, w in zip(src["room"]["walls"], doc["room"]["walls"]):
        assert w["type"] == "wall" and w["default_wall_th"] == 0.0
        center, full = _slab(w)
        assert center == pytest.approx(w_in["pos"], abs=1e-5)
        assert full == pytest.approx(w_in["size"], abs=1e-5)
    # each side faces the room: back/front along x at the far/near y, right/left along y at the far/near x
    assert sorted(w["wall_side"] for w in doc["room"]["walls"]) == ["back", "front", "left", "right"]
    f_in, f = src["room"]["floor"][0], doc["room"]["floor"][0]
    assert f["size"] == pytest.approx([s / 2 for s in f_in["size"]])
    assert f["pos"][2] == 0.0


def test_hero_wall_sides():
    doc, _ = robocasa_layout(_layout(load_room("hero_kitchen")))
    sides = {w["name"]: w["wall_side"] for w in doc["room"]["walls"]}
    assert sides == {"wall_north": "front", "wall_east": "right", "wall_south": "back", "wall_west": "left"}


@pytest.mark.parametrize("which", ["hero", "kitchen"])
def test_fixtures_turned_to_face_the_twin_front_and_placed_at_the_twin_centre(which, request):
    src = request.getfixturevalue(which)
    doc, _ = robocasa_layout(src)
    rows = _rows(doc)
    src_rows = {r["name"]: r for r in src["main_group"]["bottom_row"]}
    assert set(rows) == set(src_rows)  # nothing dropped, nothing invented
    for name, r_in in src_rows.items():
        g, r = rows[name]
        theta = g["group_z_rot"]
        assert math.cos(theta - (r_in.get("z_rot", 0.0) + FRONT_FLIP)) == pytest.approx(1.0)
        assert "z_rot" not in r  # the group turns it; a per-fixture z_rot would turn a counter without its sink
        assert r["size"] == r_in["size"]
        if r["type"] == "sink":
            assert "pos" not in r  # the counter places it
            continue
        world = _rot(r["pos"][:2], theta)
        assert world == pytest.approx(tuple(r_in["pos"][:2]), abs=1e-5)
        assert r["pos"][2] == pytest.approx(r_in["pos"][2])


def test_sink_is_the_counters_interior_object_at_the_twin_spot(hero):
    doc, _ = robocasa_layout(hero, overhang=0.05)
    g = doc["main_group"]["bottom_row"]
    names = [r["name"] for r in g]
    assert names.index("sink_1") < names.index("counter_a")  # RoboCasa needs the sink built first
    counter = next(r for r in g if r["name"] == "counter_a")
    assert counter["interior_obj"] == "sink_1"
    # Counter._place_interior_obj, in the group frame, then turned by the group
    pad = 0.05 + SINK_DOOR_TH
    w, d = counter["size"][:2]
    local = (
        counter["pos"][0] + (counter["obj_x_percent"] - 0.5) * w,
        counter["pos"][1] + pad / 2 + (counter["obj_y_percent"] - 0.5) * (d - pad),
    )
    sink = next(r for r in hero["main_group"]["bottom_row"] if r["name"] == "sink_1")
    assert _rot(local, doc["main_group"]["group_z_rot"]) == pytest.approx(tuple(sink["pos"][:2]), abs=1e-5)


def test_origin_offsets_move_the_body_so_the_centre_lands_on_the_twin(hero):
    off = (0.0, 0.028, -0.146)  # what RoboCasa reports for the style-1 stove
    doc, _ = robocasa_layout(hero, origin_offsets={"stove_1": off})
    g, r = _rows(doc)["stove_1"]
    centre = [r["pos"][0] + off[0], r["pos"][1] + off[1], r["pos"][2] + off[2]]
    src = next(r for r in hero["main_group"]["bottom_row"] if r["name"] == "stove_1")
    assert _rot(centre[:2], g["group_z_rot"]) == pytest.approx(tuple(src["pos"][:2]), abs=1e-5)
    assert centre[2] == pytest.approx(src["pos"][2])


def test_kitchen_groups_by_facing(kitchen):
    doc, _ = robocasa_layout(kitchen)
    groups = {k: g for k, g in doc.items() if k != "room"}
    assert len(groups) == 3  # facing -y (yaw pi), +y (stove, yaw 0) and -x (yaw pi/2)
    assert sum(len(g["bottom_row"]) for g in groups.values()) == 6


def test_unsupported_things_stay_explicit_boxes():
    src = {
        "room": {
            "walls": [
                {"name": "w_diag", "type": "wall", "size": [2.0, 0.1, 2.4], "pos": [1.0, 1.0, 1.2], "z_rot": 0.785398}
            ]
        },
        "main_group": {
            "group_origin": [0, 0],
            "group_pos": [0.0, 0.0],
            "group_z_rot": 0.0,
            "bottom_row": [
                {"name": "bed_1", "type": "box", "size": [1.6, 2.0, 0.5], "pos": [2.0, 2.0, 0.25]},
                {"name": "odd_1", "type": "spaceship", "size": [1.0, 1.0, 1.0], "pos": [3.0, 3.0, 0.5]},
                {"name": "isl", "type": "counter", "size": [1.0, 0.6, 0.9], "pos": [1.5, 1.5, 0.45], "island": True},
                {"name": "sink_x", "type": "sink", "size": [0.5, 0.4, 0.2], "pos": [9.0, 9.0, 0.8]},
            ],
        },
    }
    doc, notes = robocasa_layout(src)
    wall = doc["room"]["walls"][0]
    assert wall == {"name": "w_diag", "type": "box", "size": [2.0, 0.1, 2.4], "pos": [1.0, 1.0, 1.2], "z_rot": 0.785398}
    rows = _rows(doc)
    assert rows["bed_1"][1]["type"] == "box" and rows["odd_1"][1]["type"] == "box"
    assert "island" not in rows["isl"][1]
    assert "pos" in rows["sink_x"][1]  # no counter under it: placed on its own
    joined = " | ".join(notes)
    for frag in ("w_diag", "odd_1", "island", "sink_x"):
        assert frag in joined

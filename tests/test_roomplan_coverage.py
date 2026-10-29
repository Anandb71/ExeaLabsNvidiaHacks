"""TWN-3: every Apple object category, doors/windows/openings, polygonCorners floors, islands, stories, curved walls."""

import copy
import json
import math

import pytest
from conftest import REPO

from arrival.catalog.categories import DEFAULT_SIZES, FIXTURE_MAP, default_size
from arrival.scan import roomplan
from arrival.twin.compile import compile_twin, layout_dict

# CapturedRoom.Object.Category cases, as Apple spells them (developer.apple.com/documentation/roomplan).
APPLE_CATEGORIES = (
    "bathtub",
    "bed",
    "chair",
    "dishwasher",
    "fireplace",
    "oven",
    "refrigerator",
    "sink",
    "sofa",
    "stairs",
    "storage",
    "stove",
    "table",
    "television",
    "toilet",
    "washerDryer",
)
FLOOR_Y = -1.4
W, D, H = 6.0, 5.0, 2.5
FIXTURES = REPO / "tests" / "fixtures" / "roomplan"


def _t(x, y, z, yaw=0.0):
    """Column-major ARKit transform: rotation about +Y by `yaw`, translation (x, y, z)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, x, y, z, 1]


def _wall(ident, x, z, length, yaw):
    return {
        "identifier": ident,
        "category": {"wall": {}},
        "confidence": {"high": {}},
        "dimensions": [length, H, 0],
        "transform": _t(x, FLOOR_Y + H / 2, z, yaw),
    }


def _room(objects=(), floors=None, **extra):
    """W by D room, floor at ARKit y = FLOOR_Y, walls centred on the origin."""
    walls = [
        _wall("north", 0, -D / 2, W, 0),
        _wall("south", 0, D / 2, W, 0),
        _wall("east", W / 2, 0, D, math.pi / 2),
        _wall("west", -W / 2, 0, D, math.pi / 2),
    ]
    if floors is None:
        floors = [
            {"identifier": "floor", "category": {"floor": {}}, "dimensions": [W, D, 0], "transform": _t(0, FLOOR_Y, 0)}
        ]
    return {
        "walls": walls,
        "floors": floors,
        "doors": [],
        "windows": [],
        "openings": [],
        "objects": list(objects),
        **extra,
    }


def _object(cat, x=0.0, z=0.0, size=None, confidence="high", ident=None):
    sx, sy, sz = size or default_size(roomplan._snake(cat))
    return {
        "identifier": ident or cat,
        "category": {cat: {}},
        "confidence": {confidence: {}},
        "dimensions": [sx, sz, sy],
        "transform": _t(x, FLOOR_Y + sz / 2, z),
    }


def _surface(kind, ident, parent, x, y, z, w, h, yaw=0.0, is_open=None):
    cat = {kind: {"isOpen": is_open}} if kind == "door" else {kind: {}}
    return {
        "identifier": ident,
        "parentIdentifier": parent,
        "category": cat,
        "confidence": {"high": {}},
        "dimensions": [w, h, 0],
        "transform": _t(x, y, z, yaw),
    }


def test_catalog_covers_every_apple_category():
    snake = {roomplan._snake(c) for c in APPLE_CATEGORIES}
    assert snake <= set(FIXTURE_MAP) and snake <= set(DEFAULT_SIZES)
    assert "washer_dryer" in snake


@pytest.mark.parametrize("cat", APPLE_CATEGORIES)
def test_category_normalizes_and_compiles(cat, tmp_path):
    b = roomplan.to_bundle(_room([_object(cat, x=1.0, z=-0.5)]), "r")
    assert [f["category"] for f in b["fixtures"]] == [roomplan._snake(cat)]
    f = b["fixtures"][0]
    assert f["size"] == pytest.approx(default_size(f["category"]))
    assert f["position"][2] - f["size"][2] / 2 == pytest.approx(0.0, abs=1e-6)
    twin = compile_twin(b)
    fx = next(t for t in twin["fixtures"] if t["id"] == cat)
    assert fx["type"] == FIXTURE_MAP[f["category"]]
    assert fx["position"][2] - fx["size"][2] / 2 == pytest.approx(0.0, abs=1e-6)
    row = layout_dict(twin)["main_group"]["bottom_row"]
    assert any(e["name"] == cat and e["type"] == fx["type"] for e in row)


def test_missing_dimensions_use_default_size():
    o = _object("sofa")
    del o["dimensions"]
    b = roomplan.to_bundle(_room([o]), "r")
    assert b["fixtures"][0]["size"] == default_size("sofa")


def test_doors_windows_openings():
    rp = _room()
    # On the north wall (ARKit z = -D/2), which normalizes to y = D along +X after the min-corner shift.
    rp["doors"] = [_surface("door", "d1", "north", -1.0, FLOOR_Y + 1.05, -D / 2, 0.9, 2.1, is_open=True)]
    rp["windows"] = [_surface("window", "w1", "north", 1.5, FLOOR_Y + 1.5, -D / 2, 1.2, 1.0)]
    rp["openings"] = [_surface("opening", "o1", "nowhere", W / 2, FLOOR_Y + 1.0, 0.0, 1.0, 2.0, yaw=math.pi / 2)]
    b = roomplan.to_bundle(rp, "r")
    d, w, o = b["doors"][0], b["windows"][0], b["openings"][0]
    assert d["wall_id"] == "north" and d["is_open"] is True
    assert d["position"] == pytest.approx([2.0, D, 1.05]) and d["size"] == pytest.approx([0.9, 0.0, 2.1])
    assert w["wall_id"] == "north" and "is_open" not in w
    assert w["position"] == pytest.approx([4.5, D, 1.5]) and w["yaw"] == 0.0
    assert o["wall_id"] is None  # parent not among the walls
    assert o["position"] == pytest.approx([6.0, 2.5, 1.0]) and abs(o["yaw"]) == pytest.approx(math.pi / 2)
    twin = compile_twin(b)
    assert twin["doors"][0]["id"] == "d1" and len(twin["windows"]) == 1 and len(twin["openings"]) == 1


def _seg_dist(p, w):
    (x0, y0), (x1, y1) = w["start"], w["end"]
    dx, dy = x1 - x0, y1 - y0
    t = max(0.0, min(1.0, ((p[0] - x0) * dx + (p[1] - y0) * dy) / (dx * dx + dy * dy)))
    return math.dist(p, [x0 + t * dx, y0 + t * dy])


@pytest.mark.parametrize("name", ["office", "study"])
def test_authored_fixture_portals_sit_on_their_walls(name):
    raw = json.loads((FIXTURES / f"{name}.json").read_text())
    b = roomplan.load(FIXTURES / f"{name}.json", name)
    walls = {w["id"]: w for w in b["walls"]}
    for key in ("doors", "windows", "openings"):
        assert len(b[key]) == len(raw[key])
        for p in b[key]:
            assert p["wall_id"] in walls
            assert _seg_dist(p["position"][:2], walls[p["wall_id"]]) <= 0.02
            assert 0 < p["position"][2] - p["size"][2] / 2 + 0.05  # sill at or above the floor
    if name == "office":
        assert [d["is_open"] for d in b["doors"]] == [False, True]


def test_polygon_floor_sets_extent():
    # L-shaped floor polygon in local plane coordinates, wider than the walls on +X.
    corners = [
        [-W / 2, -D / 2, 0],
        [W / 2 + 1, -D / 2, 0],
        [W / 2 + 1, 0, 0],
        [W / 2, 0, 0],
        [W / 2, D / 2, 0],
        [-W / 2, D / 2, 0],
    ]
    # Floor transform: local x -> ARKit x, local y -> ARKit z, local z -> ARKit y.
    m = [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, FLOOR_Y, 0, 1]
    floor = {
        "identifier": "fl",
        "category": {"floor": {}},
        "dimensions": [W + 1, D, 0],
        "transform": m,
        "polygonCorners": corners,
    }
    b = roomplan.to_bundle(_room(floors=[floor]), "r")
    assert len(b["floors"]) == 1 and b["floors"][0]["z"] == 0.0
    xs = [c[0] for c in b["floors"][0]["corners"]]
    ys = [c[1] for c in b["floors"][0]["corners"]]
    assert len(xs) == 6 and max(xs) == pytest.approx(W + 1) and min(xs) == pytest.approx(0.0)
    assert min(ys) == pytest.approx(0.0) and max(ys) == pytest.approx(D)
    doc = layout_dict(compile_twin(b))
    assert doc["room"]["floor"][0]["size"] == pytest.approx([W + 1, D, 0.02])
    assert doc["room"]["floor"][0]["pos"] == pytest.approx([(W + 1) / 2, D / 2, 0.0])


@pytest.mark.parametrize("name", ["office", "study"])
def test_authored_fixture_polygon_floor_matches_walls(name):
    b = roomplan.load(FIXTURES / f"{name}.json", name)
    assert len(b["floors"]) == 1 and len(b["floors"][0]["corners"]) == 4
    xs = [c[0] for c in b["floors"][0]["corners"]]
    ys = [c[1] for c in b["floors"][0]["corners"]]
    wx = [p[0] for w in b["walls"] for p in (w["start"], w["end"])]
    wy = [p[1] for w in b["walls"] for p in (w["start"], w["end"])]
    assert [min(xs), max(xs), min(ys), max(ys)] == pytest.approx([min(wx), max(wx), min(wy), max(wy)], abs=0.05)


def test_floor_without_polygon_keeps_wall_extent():
    b = roomplan.to_bundle(_room(), "r")
    assert b["floors"] == []
    assert layout_dict(compile_twin(b))["room"]["floor"][0]["size"] == pytest.approx([W, D, 0.02])


def test_island():
    island = _object("table", x=0.0, z=0.0, size=[2.0, 0.9, 0.92], ident="island")
    dining = _object("table", x=0.0, z=1.5, size=[1.4, 0.8, 0.75], ident="dining")
    against_wall = _object("storage", x=0.0, z=-D / 2 + 0.3, size=[1.0, 0.6, 0.9], ident="cab")
    twin = compile_twin(roomplan.to_bundle(_room([island, dining, against_wall]), "r"))
    fx = {f["id"]: f for f in twin["fixtures"]}
    assert fx["island"]["type"] == "counter" and fx["island"]["island"] is True
    assert "default_config_name" not in fx["island"]
    assert fx["dining"]["type"] == "counter" and fx["dining"].get("default_config_name") == "dining"
    assert "island" not in fx["dining"]
    assert fx["cab"]["type"] == "hinge_cabinet" and "island" not in fx["cab"]
    row = {e["name"]: e for e in layout_dict(twin)["main_group"]["bottom_row"]}
    assert row["island"].get("island") is True and "island" not in row["dining"]


def test_freestanding_storage_is_not_island():
    sideboard = _object("storage", x=0.0, z=0.0, size=[1.6, 0.5, 1.1], ident="sideboard")
    table = _object("table", x=0.0, z=1.5, size=[1.6, 0.9, 1.1], ident="island")
    twin = compile_twin(roomplan.to_bundle(_room([sideboard, table]), "r"))
    fx = {f["id"]: f for f in twin["fixtures"]}
    assert fx["sideboard"]["type"] == "hinge_cabinet" and "island" not in fx["sideboard"]
    assert fx["island"]["type"] == "counter" and fx["island"]["island"] is True


def test_island_needs_walls():
    rp = _room([_object("table", size=[2.0, 0.9, 0.92], ident="island")])
    rp["walls"] = []
    twin = compile_twin(roomplan.to_bundle(rp, "r"))
    assert "island" not in twin["fixtures"][0]


def _two_story_structure():
    ground = _room([_object("sofa", x=1.0)])
    upper = copy.deepcopy(
        _room(
            [
                _object("bed", x=0.5, ident="bed"),
                _object("chair", x=-1.5, ident="c1"),
                _object("chair", x=-2.0, z=1.0, ident="c2"),
            ]
        )
    )
    for item in upper["walls"] + upper["floors"] + upper["objects"]:  # second story 3 m up
        item["transform"] = list(item["transform"])
        item["transform"][13] += 3.0
    ground["story"], upper["story"] = 0, 1
    return ground, upper


def test_multi_story_structure_rooms():
    ground, upper = _two_story_structure()
    b = roomplan.to_bundle({"rooms": [ground, upper], "version": 2}, "r")
    assert b["story"] == 1 and b["story_count"] == 2  # most objects
    assert sorted(f["id"] for f in b["fixtures"]) == ["bed", "c1", "c2"]
    for f in b["fixtures"]:
        assert f["position"][2] - f["size"][2] / 2 == pytest.approx(0.0, abs=1e-6)  # measured from story 1 floor
    b0 = roomplan.to_bundle({"rooms": [ground, upper]}, "r", story=0)
    assert b0["story"] == 0 and [f["category"] for f in b0["fixtures"]] == ["sofa"]
    assert b0["fixtures"][0]["position"][2] == pytest.approx(default_size("sofa")[2] / 2)
    with pytest.raises(ValueError):
        roomplan.to_bundle({"rooms": [ground, upper]}, "r", story=2)


def test_multi_story_flat_export():
    ground, upper = _two_story_structure()
    flat = {k: [] for k in ("walls", "floors", "doors", "windows", "openings", "objects")}
    for room in (ground, upper):
        for k in flat:
            flat[k] += [{**item, "story": room["story"]} for item in room[k]]
    b = roomplan.to_bundle(flat, "r")
    assert b["story"] == 1 and b["story_count"] == 2 and len(b["fixtures"]) == 3 and len(b["walls"]) == 4
    assert compile_twin(b)["story_count"] == 2


def test_single_story_defaults():
    b = roomplan.load(REPO / "scans" / "hero_kitchen" / "roomplan.json", "hero_kitchen")
    assert b["story"] == 0 and b["story_count"] == 1
    assert b["doors"] == [] and b["windows"] == [] and b["openings"] == [] and b["floors"] == []


def _circumcenter(a, b, c):
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return ux, uy


@pytest.mark.parametrize("sweep,n_seg", [(math.pi / 2, 6), (math.radians(10), 2)])
def test_curved_wall_is_approximated_by_chords(sweep, n_seg):
    """A wall with curve {startAngle, endAngle, radius} becomes equal chords every 15°, at least two."""
    radius = 4.0
    rp = _room()
    rp["walls"][0]["curve"] = {"startAngle": 0.0, "endAngle": sweep, "radius": radius}
    b = roomplan.to_bundle(rp, "r")
    segs = [w for w in b["walls"] if w["id"] == "north" or w["id"].startswith("north_")]
    assert len(segs) == n_seg
    for i, w in enumerate(segs[:-1]):
        assert w["end"] == pytest.approx(segs[i + 1]["start"], abs=1e-6)
    chord = 2 * radius * math.sin(sweep / (2 * n_seg))
    for w in segs:
        assert math.dist(w["start"], w["end"]) == pytest.approx(chord, abs=1e-6)
        assert w["height"] == pytest.approx(H)
        assert w["thickness"] > 0
    pts = [segs[0]["start"]] + [w["end"] for w in segs]
    cx, cy = _circumcenter(pts[0], pts[1], pts[2])
    assert all(math.hypot(p[0] - cx, p[1] - cy) == pytest.approx(radius, abs=1e-6) for p in pts)
    assert math.dist(pts[0], pts[-1]) == pytest.approx(2 * radius * math.sin(sweep / 2), abs=1e-6)
    assert compile_twin(b)["walls"]


def test_portal_on_a_curved_wall_names_the_chord_it_sits_on():
    """A window 80% of the way round a quarter-circle wall: its parentIdentifier is the wall, which is now six
    chords; wall_id must be the chord under it, not the first one (3.8 m away, so the sim drew no window)."""
    rp = _room()
    radius, a = 4.0, math.radians(72)
    rp["walls"][0]["curve"] = {"startAngle": 0.0, "endAngle": math.pi / 2, "radius": radius}
    rp["windows"] = [
        _surface(
            "window",
            "w1",
            "north",
            radius * math.cos(a),
            FLOOR_Y + 1.5,
            -D / 2 + radius * math.sin(a),
            0.8,
            1.0,
            yaw=-(a + math.pi / 2),
        )
    ]
    b = roomplan.to_bundle(rp, "r")
    w = b["windows"][0]
    chords = [s for s in b["walls"] if s["id"] == "north" or s["id"].startswith("north_")]
    assert len(chords) == 6
    assert w["wall_id"] == "north_5"
    assert _seg_dist(w["position"][:2], next(s for s in chords if s["id"] == "north_5")) < 0.05
    assert all(_seg_dist(w["position"][:2], s) > 0.15 for s in chords if s["id"] != "north_5")


@pytest.mark.parametrize("dimensions", ["missing", "zero", "negative", "undersized"])
@pytest.mark.parametrize("encoding", ["xy", "xyz", "dict"])
@pytest.mark.parametrize("with_floor", [False, True])
def test_wall_polygon_supplies_bounds_and_asymmetric_centres(dimensions, encoding, with_floor):
    raw = _room(floors=None if with_floor else [])
    expected = roomplan.to_bundle(raw, "polygon_walls")
    for wall in raw["walls"]:
        length = wall["dimensions"][0]
        # A non-centred local polygon represents exactly the original world wall, including yawed walls.
        cx, cy = 0.75, 0.3
        points = [[-length / 2 + cx, -H / 2 + cy, 0], [length / 2 + cx, H / 2 + cy, 0]]
        wall["polygonCorners"] = (
            [p[:2] for p in points]
            if encoding == "xy"
            else [dict(zip(("x", "y", "z"), p)) for p in points]
            if encoding == "dict"
            else points
        )
        m = wall["transform"]
        for row in range(3):
            m[12 + row] -= m[row] * cx + m[4 + row] * cy
        if dimensions == "missing":
            del wall["dimensions"]
        else:
            wall["dimensions"] = {"zero": [0, 0, 0], "negative": [-1, -1, 0], "undersized": [length / 2, H / 2, 0]}[
                dimensions
            ]
    untouched = copy.deepcopy(raw)
    bundle = roomplan.to_bundle(raw, "polygon_walls")
    assert raw == untouched
    assert bundle == expected
    twin = compile_twin(bundle)
    assert twin["walls"] == compile_twin(expected)["walls"]
    ends = [p for wall in twin["walls"] for p in (wall["start"], wall["end"])]
    assert all(sum(math.dist(p, q) < 1e-8 for q in ends) == 2 for p in ends)
    assert all(w["height"] == H and w["thickness"] == roomplan.MIN_WALL_THICKNESS for w in twin["walls"])


@pytest.mark.parametrize("axis", [0, 1])
def test_wall_polygon_refines_each_axis_independently(axis):
    raw = _room()
    wall = raw["walls"][0]
    wall["dimensions"][axis] /= 2
    wall["polygonCorners"] = [[-W / 2, -H / 2, 0], [W / 2, H / 2, 0]]
    assert roomplan.to_bundle(raw, "r") == roomplan.to_bundle(_room(), "r")


@pytest.mark.parametrize("delta", [0.0, 1e-10, -0.5])
def test_matching_or_smaller_wall_polygons_preserve_dimensions_exactly(delta):
    raw = _room()
    expected = roomplan.to_bundle(raw, "r")
    for wall in raw["walls"]:
        length = wall["dimensions"][0]
        wall["polygonCorners"] = [
            [-(length + delta) / 2, -(H + delta) / 2, 0],
            [(length + delta) / 2, (H + delta) / 2, 0],
        ]
    assert roomplan.to_bundle(raw, "r") == expected


def test_equal_span_asymmetric_polygon_moves_the_wall_without_changing_its_size():
    raw = _room()
    wall = raw["walls"][0]
    wall["polygonCorners"] = [[-W / 2 + 0.5, -H / 2 + 0.25], [W / 2 + 0.5, H / 2 + 0.25]]
    w = next(w for w in roomplan.to_bundle(raw, "r")["walls"] if w["id"] == "north")
    assert w["start"] == [0.5, D] and w["end"] == [W + 0.5, D]
    assert w["height"] == H


@pytest.mark.parametrize(
    "corner",
    [
        [float("nan"), 0, 0],
        [0, float("inf"), 0],
        [0, 0, float("-inf")],
        ["bad", 0],
        [True, 1, 0],
        [0],
        [0, 0, 0, 0],
        {"x": 0},
        {"x": 0, "y": 0, "z": "bad"},
        "not-a-point",
    ],
)
def test_invalid_wall_polygon_points_name_the_wall_and_field(corner):
    raw = _room()
    raw["walls"][0]["polygonCorners"] = [[-3, -1], [3, 1], corner]
    with pytest.raises(ValueError, match=r"wall 0 .*north.*polygonCorners point 2.*finite numeric"):
        roomplan.to_bundle(raw, "r")


def test_invalid_wall_polygon_container_is_rejected():
    raw = _room()
    raw["walls"][0]["polygonCorners"] = "invalid"
    with pytest.raises(ValueError, match="polygonCorners must be a list"):
        roomplan.to_bundle(raw, "r")


def test_degenerate_polygon_cannot_supply_a_missing_height():
    raw = _room()
    raw["walls"][0]["dimensions"] = [0, 0, 0]
    raw["walls"][0]["polygonCorners"] = [[-3, 0], [3, 0]]
    with pytest.raises(ValueError, match="positive length and height"):
        roomplan.to_bundle(raw, "r")


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        (["a", "a", "a_2"], ["a", "a_3", "a_2"]),
        (["a", "a_2", "a", "a_3"], ["a", "a_2", "a_4", "a_3"]),
        (["a", "a", "a"], ["a", "a_2", "a_3"]),
        (["a", "a", "a_2", "a_2"], ["a", "a_3", "a_2", "a_2_2"]),
    ],
)
def test_duplicate_ids_do_not_claim_existing_suffixes(ids, expected):
    items = [{"id": ident} for ident in ids]
    roomplan._unique(items)
    assert [item["id"] for item in items] == expected


def test_roomplan_duplicate_wall_ids_preserve_suffixed_parent():
    raw = _room()
    for wall, ident in zip(raw["walls"], ["a", "a", "a_2", "west"]):
        wall["identifier"] = ident
    raw["windows"] = [_surface("window", "window", "a_2", W / 2, FLOOR_Y + 1.5, 0, 0.8, 1.0, yaw=math.pi / 2)]
    bundle = roomplan.to_bundle(raw, "r")
    assert [wall["id"] for wall in bundle["walls"]] == ["a", "a_3", "a_2", "west"]
    assert bundle["windows"][0]["wall_id"] == "a_2"
    assert len(compile_twin(bundle)["walls"]) == 4

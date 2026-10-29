from arrival.tasks.propose import LIGHTING, propose


def test_families_and_sources(hero_twin):
    tasks = propose(hero_twin)["tasks"]
    assert {t["family"] for t in tasks} == {"counter_to_sink", "counter_to_counter"}
    assert {t["object_id"] for t in tasks} == {"mug_1", "bowl_1"}
    assert "sponge_1" not in {t["object_id"] for t in tasks}
    assert {t["lighting"] for t in tasks} <= set(LIGHTING)
    for t in tasks:
        assert "people" not in t and "cosmos" not in t
        assert t["source_fixture_id"] == "counter_a"
    sink = [t for t in tasks if t["family"] == "counter_to_sink"]
    assert {t["target_fixture_id"] for t in sink} == {"sink_1"}
    assert {t["mimicgen_template"] for t in sink} == {"PnPCounterToSink"}
    c2c = [t for t in tasks if t["family"] == "counter_to_counter"]
    assert {t["target_fixture_id"] for t in c2c} == {"counter_b"}
    assert {t["mimicgen_template"] for t in c2c} == {None}


def test_four_wordings_for_mug_sink(hero_twin):
    tasks = propose(hero_twin)["tasks"]
    mug = [
        t
        for t in tasks
        if t["family"] == "counter_to_sink"
        and t["object_id"] == "mug_1"
        and t["pose_index"] == 0
        and t["lighting"] == "neutral"
    ]
    assert [t["instruction"] for t in mug] == [
        "Pick up the mug from the counter and place it in the sink.",
        "Put the mug in the sink.",
        "Move the mug off the counter into the sink.",
        "Take the mug and drop it in the sink.",
    ]
    assert mug[0]["start_position"] == [1.0, 0.35, 0.97]


def test_counter_to_counter_wording(hero_twin):
    tasks = propose(hero_twin)["tasks"]
    w = {t["instruction"] for t in tasks if t["family"] == "counter_to_counter" and t["object_id"] == "bowl_1"}
    assert "Put the bowl onto the other counter." in w
    assert len(w) == 4


def test_start_poses(hero_twin):
    tasks = propose(hero_twin)["tasks"]
    poses = {tuple(t["start_position"]) for t in tasks if t["object_id"] == "mug_1"}
    assert poses == {(1.0, 0.35, 0.97), (1.08, 0.35, 0.97), (0.92, 0.35, 0.97), (1.0, 0.43, 0.97)}


def _apple_obj(ident, cat, dims, x, y_up, z, yaw_deg=0.0):
    import math

    if cat != "wall":  # a RoomPlan object's front is its local +Z (measured on a real scan, kitchen_real_01)
        yaw_deg += 180.0
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return {
        "identifier": ident,
        "category": {cat: {}},
        "confidence": {"high": {}},
        "dimensions": dims,
        "transform": [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, x, y_up, z, 1],
    }


def test_real_format_kitchen_produces_tasks():
    """Synthetic kitchen in Apple's keyed-enum encoding, floor at ARKit y=-1.4, rotated 30 degrees."""
    import math

    from arrival.scan import roomplan
    from arrival.twin.compile import compile_twin, place_default_objects

    rot = 30.0
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))

    def place(x, z):  # rotate an ARKit (x, z) floor point about Y
        return c * x + s * z, -s * x + c * z

    walls = []
    for i, (x, z, length, yaw) in enumerate([(2, 0, 4, 0), (2, -3, 4, 0), (0, -1.5, 3, 90), (4, -1.5, 3, 90)]):
        px, pz = place(x, z)
        walls.append({**_apple_obj(f"w{i}", "wall", [length, 2.5, 0], px, -0.15, pz, yaw + rot)})
    sx, sz = place(1.5, -0.35)
    tx, tz = place(3.0, -0.35)
    rp = {
        "floors": [{"transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1.4, 0, 1]}],
        "walls": walls,
        "objects": [
            _apple_obj("sink", "sink", [0.6, 0.25, 0.5], sx, -0.55, sz, rot),
            _apple_obj("stove", "stove", [0.7, 0.9, 0.65], tx, -0.95, tz, rot),
            _apple_obj("ghost", "dishwasher", [0.6, 0.8, 0.6], 1, -1, -1, rot) | {"confidence": {"low": {}}},
        ],
    }
    b = roomplan.to_bundle(rp, "k")
    assert {f["category"] for f in b["fixtures"]} == {"sink", "stove"}
    assert all(f["yaw"] == 0.0 for f in b["fixtures"])  # aligned + snapped
    twin = compile_twin(b)
    assert {o["label"] for o in place_default_objects(twin, seed=0)} >= {"mug", "bowl"}  # plus whatever else fits
    tasks = propose(twin)["tasks"]
    assert {t["family"] for t in tasks} == {"counter_to_sink", "counter_to_counter"}

import math

import pytest
import yaml

from arrival.twin.compile import compile_twin, layout_dict, write_twin


def _bundle(fixtures, walls=()):
    return {
        "schema_version": "1",
        "room_id": "t",
        "units": "meters",
        "frame": "z_up",
        "geometry_source": "roomplan",
        "blocks_claims": [],
        "walls": list(walls),
        "fixtures": fixtures,
        "objects": [],
    }


def test_hero_fixture_types(hero_twin):
    fx = {f["id"]: f for f in hero_twin["fixtures"]}
    assert fx["sink_1"]["type"] == "sink"
    assert fx["sink_1"]["position"][2] == 0.85
    assert fx["fridge_1"]["type"] == "fridge_bottom_freezer"
    assert fx["stove_1"]["type"] == "stove"
    assert not any(f["id"].startswith("counter_synth") for f in hero_twin["fixtures"])
    assert {o["catalog_id"] for o in hero_twin["objects"]} == {"mug", "bowl", "sponge"}


def test_unknown_category_is_box():
    twin = compile_twin(
        _bundle([{"id": "tv", "category": "television", "size": [1, 0.1, 0.6], "position": [1, 1, 1], "yaw": 0}])
    )
    assert twin["fixtures"][0]["type"] == "box"
    assert layout_dict(twin)["main_group"]["bottom_row"][0]["type"] == "box"


def test_table_is_dining_counter():
    twin = compile_twin(
        _bundle([{"id": "t1", "category": "table", "size": [1, 1, 0.75], "position": [1, 1, 0.375], "yaw": 0}])
    )
    assert twin["fixtures"][0]["type"] == "counter"
    assert twin["fixtures"][0]["default_config_name"] == "dining"


def test_sink_only_synthesizes_counter():
    twin = compile_twin(
        _bundle(
            [{"id": "sink_1", "category": "sink", "size": [0.6, 0.5, 0.25], "position": [1.6, 0.4, 0.85], "yaw": 0}]
        )
    )
    counters = [f for f in twin["fixtures"] if f["type"] == "counter"]
    assert len(counters) == 1
    c = counters[0]
    assert c["size"][2] == 0.92 and c["position"][2] == 0.46
    assert c["size"][0] == 1.8 and c["position"][:2] == [1.6, 0.4]


def test_sink_on_counter_height_storage_does_not_synthesize_overlapping_counter():
    import math

    sink = {
        "id": "sink",
        "category": "sink",
        "size": [0.6, 0.5, 0.25],
        "position": [2.0, 1.5, 0.85],
        "yaw": -math.pi / 2,
    }
    base = {
        "id": "base",
        "category": "storage",
        "size": [1.0, 0.6, 0.9],
        "position": [2.0, 1.5, 0.45],
        "yaw": -math.pi / 2,
    }
    twin = compile_twin(_bundle([sink, base]))
    assert [f["id"] for f in twin["fixtures"]] == ["sink", "base"]
    assert twin["fixtures"][1]["type"] == "hinge_cabinet"

    tall_base = {**base, "size": [1.0, 0.6, 2.0], "position": [2.0, 1.5, 1.0]}
    twin = compile_twin(_bundle([sink, tall_base]))
    assert "counter_synth_sink" in {f["id"] for f in twin["fixtures"]}


def test_layout_yaml(tmp_path, hero_twin):
    _, layout_path = write_twin(hero_twin, tmp_path)
    doc = yaml.safe_load(layout_path.read_text())
    assert "main_group" in doc and "bottom_row" in doc["main_group"]
    walls = {w["name"]: w for w in doc["room"]["walls"]}
    assert walls["wall_north"]["size"] == [4.0, 0.02, 2.4]
    assert walls["wall_north"]["pos"] == [2.0, 0.0, 1.2]
    assert walls["wall_east"]["size"] == [0.02, 3.0, 2.4]
    assert doc["room"]["floor"][0]["size"] == [4.0, 3.0, 0.02]
    assert doc["room"]["floor"][0]["pos"] == [2.0, 1.5, 0.0]


def test_yawed_sink_counter_and_containment():
    import math

    from arrival.tasks.propose import in_top_rect

    yaw = math.pi / 2
    twin = compile_twin(
        _bundle([{"id": "s", "category": "sink", "size": [0.6, 0.5, 0.25], "position": [0.4, 2.0, 0.85], "yaw": yaw}])
    )
    c = next(f for f in twin["fixtures"] if f["type"] == "counter")
    assert c["yaw"] == yaw
    assert in_top_rect([0.4, 2.8], c) and not in_top_rect([1.2, 2.0], c)  # long axis runs along world Y


def test_oblique_wall_gets_z_rot():
    import math

    w = {"id": "w", "start": [0, 0], "end": [3, 3], "height": 2.4, "thickness": 0.1}
    doc = layout_dict(compile_twin(_bundle([], [w])))
    entry = doc["room"]["walls"][0]
    assert entry["z_rot"] == round(math.pi / 4, 6)
    assert entry["size"][0] == round(math.hypot(3, 3), 6)


def test_place_default_objects_seeded():
    from arrival.tasks.propose import on_counter
    from arrival.twin.compile import place_default_objects

    fx = [{"id": "s", "category": "sink", "size": [0.6, 0.5, 0.25], "position": [1.6, 0.4, 0.85], "yaw": 0}]
    a, b = compile_twin(_bundle(fx)), compile_twin(_bundle(fx))
    pa, pb = place_default_objects(a, seed=3), place_default_objects(b, seed=3)
    assert pa == pb and {o["label"] for o in pa} == {"mug", "bowl", "plate", "cup", "apple"}
    assert pa[0]["size"] != [0.08, 0.08, 0.1] and 0.07 < pa[0]["size"][0] < 0.09  # seeded size, not the nominal
    counter = next(f for f in a["fixtures"] if f["type"] == "counter")
    assert all(on_counter(o, counter) for o in pa)


def test_stove_between_counters_gets_no_synthesized_counter():
    from arrival.twin.compile import compile_twin

    bundle = {
        "schema_version": "1",
        "room_id": "r",
        "units": "meters",
        "frame": "z_up",
        "geometry_source": "authored_sample",
        "blocks_claims": [],
        "walls": [],
        "objects": [],
        "fixtures": [
            {"id": "left", "category": "counter", "size": [1.0, 0.65, 0.92], "position": [0.5, 0.4, 0.46], "yaw": 0},
            {"id": "stove", "category": "stove", "size": [0.7, 0.65, 0.9], "position": [1.35, 0.4, 0.45], "yaw": 0},
            {"id": "right", "category": "counter", "size": [1.0, 0.65, 0.92], "position": [2.2, 0.4, 0.46], "yaw": 0},
            {"id": "lone", "category": "stove", "size": [0.7, 0.65, 0.9], "position": [1.35, 2.5, 0.45], "yaw": 0},
        ],
    }
    ids = [f["id"] for f in compile_twin(bundle)["fixtures"]]
    assert "counter_synth_stove" not in ids  # flanked by counters: no phantom counter over the neighbours
    assert "counter_synth_lone" in ids  # a stove with nothing around it still gets a counter


# ---- L10: the U-shaped real scan from issue #42 (aligned frame, metres; sizes are RoomPlan-typical) ----
U_YAW = -math.pi / 2


def _u_fixture(fid, cat, x, y, z, size, yaw):
    return {"id": fid, "category": cat, "size": size, "position": [x, y, z], "yaw": yaw}


def u_kitchen_bundle():
    """Sink, dishwasher, oven, fridge and four storage cabinets in a U, three walls (x 0..7.55, y 0..3.82), the east
    side open. Positions and yaws are the #42 table; sizes are what RoomPlan reports for such cabinets."""
    fixtures = [
        _u_fixture("sink", "sink", 7.22, 2.43, 0.85, [0.6, 0.5, 0.25], U_YAW),
        _u_fixture("storage_a", "storage", 7.22, 2.47, 0.45, [0.9, 0.6, 0.9], U_YAW),
        _u_fixture("dishwasher", "dishwasher", 7.21, 1.62, 0.43, [0.6, 0.6, 0.86], U_YAW),
        _u_fixture("oven", "stove", 5.91, 3.45, 0.45, [0.7, 0.65, 0.9], 0.0),
        _u_fixture("storage_b", "storage", 5.0, 3.47, 0.45, [1.0, 0.6, 0.9], 0.0),
        _u_fixture("storage_c", "storage", 6.93, 3.42, 0.45, [0.9, 0.6, 0.9], 0.0),
        _u_fixture("fridge", "refrigerator", 6.95, 0.47, 0.9, [0.9, 0.7, 1.8], U_YAW),
    ]
    walls = [
        {"id": "w_s", "start": [0, 0], "end": [7.55, 0], "height": 2.5, "thickness": 0.1},
        {"id": "w_w", "start": [0, 0], "end": [0, 3.82], "height": 2.5, "thickness": 0.1},
        {"id": "w_n", "start": [0, 3.82], "end": [7.55, 3.82], "height": 2.5, "thickness": 0.1},
    ]
    return _bundle(fixtures, walls)


def _real_overlaps(twin):
    """Pairs of fixtures whose footprints cross, except a sink or stove inset in a counter or worktop cabinet."""
    from test_scans import _allowed, _overlap

    fx = twin["fixtures"]
    return sorted(
        (a["id"], b["id"]) for i, a in enumerate(fx) for b in fx[i + 1 :] if _overlap(a, b) and not _allowed(a, b)
    )


def test_u_kitchen_sink_in_scanned_cabinet_gets_no_second_counter():
    """#42/#49: RoomPlan 'storage' at counter height already is the counter; before, a 1.8 m counter_synth_sink was
    stacked over it and the physics exploded."""
    twin = compile_twin(u_kitchen_bundle())
    ids = [f["id"] for f in twin["fixtures"]]
    assert "counter_synth_sink" not in ids
    assert "counter_synth_oven" in ids  # the oven has no cabinet at its own position: it still gets a worktop
    assert _real_overlaps(twin) == []


def test_u_kitchen_builds_a_scene_with_the_sink_in_its_cabinet():
    """Tasks on the scanned cabinets are PR #77's job; the twin must at least build and keep the sink basin."""
    mujoco = pytest.importorskip("mujoco")
    from arrival.sim.scene import build_mjcf

    twin = compile_twin(u_kitchen_bundle())
    model = mujoco.MjModel.from_xml_string(build_mjcf(twin, (3.0, 1.5, 0.75)))
    assert model.geom("sink_bottom") is not None and model.joint("sink_faucet") is not None


def _wall_end_bundle(fridge_x=1.55):
    sink = {"id": "sink", "category": "sink", "size": [0.6, 0.5, 0.25], "position": [0.7, 0.4, 0.85], "yaw": 0.0}
    fridge = {
        "id": "fridge",
        "category": "refrigerator",
        "size": [0.9, 0.7, 1.8],
        "position": [fridge_x, 0.45, 0.9],
        "yaw": 0.0,
    }
    walls = [
        {"id": "w_s", "start": [0, 0], "end": [4, 0], "height": 2.5, "thickness": 0.1},
        {"id": "w_w", "start": [0, 0], "end": [0, 3], "height": 2.5, "thickness": 0.1},
        {"id": "w_n", "start": [0, 3], "end": [4, 3], "height": 2.5, "thickness": 0.1},
    ]
    return _bundle([sink, fridge], walls)


def test_synthesized_counter_stops_at_the_fridge_beside_the_sink():
    """A sink near a wall end with the fridge next to it: the default 1.8 m counter ran 0.4 m into the fridge."""
    twin = compile_twin(_wall_end_bundle())
    fx = {f["id"]: f for f in twin["fixtures"]}
    counter, fridge, sink = fx["counter_synth_sink"], fx["fridge"], fx["sink"]
    assert _real_overlaps(twin) == []
    assert counter["position"][0] + counter["size"][0] / 2 <= fridge["position"][0] - fridge["size"][0] / 2 + 0.021
    assert (
        counter["position"][0] - counter["size"][0] / 2 <= sink["position"][0] - sink["size"][0] / 2
    )  # holds the sink
    assert counter["position"][0] + counter["size"][0] / 2 >= sink["position"][0] + sink["size"][0] / 2
    assert counter["size"][0] < 1.8 and counter["position"][2] == 0.46 and counter["size"][1] == 0.65


def test_synthesized_counter_keeps_its_default_length_when_nothing_is_beside_it():
    far = compile_twin(_wall_end_bundle(fridge_x=3.4))
    counter = next(f for f in far["fixtures"] if f["id"] == "counter_synth_sink")
    assert counter["size"][0] == 1.8 and counter["position"][:2] == [0.7, 0.4]


def test_sinks_in_a_row_share_one_counter():
    """Five sinks 0.9 m apart used to get stacked counters with a sink straddling each seam."""
    sinks = [
        {
            "id": f"s{i}",
            "category": "sink",
            "size": [0.6, 0.5, 0.25],
            "position": [1.0 + 0.9 * i, 0.4, 0.85],
            "yaw": 0.0,
        }
        for i in range(5)
    ]
    twin = compile_twin(_bundle(sinks))
    counters = [f for f in twin["fixtures"] if f["type"] == "counter"]
    assert len(counters) == 1
    c = counters[0]
    for s in sinks:
        assert abs(s["position"][0] - c["position"][0]) + 0.3 <= c["size"][0] / 2 + 1e-9
    assert _real_overlaps(twin) == []


def test_island_clear_of_walls_in_an_open_u_room():
    """A counter-height table the cabinets do not touch is an island; one against the wall stays a plain table."""
    bundle = u_kitchen_bundle()
    bundle["fixtures"] += [
        {"id": "island", "category": "table", "size": [1.6, 0.9, 0.92], "position": [3.0, 1.9, 0.46], "yaw": 0.0},
        {
            "id": "table_on_wall",
            "category": "table",
            "size": [1.6, 0.6, 0.92],
            "position": [2.0, 3.45, 0.46],
            "yaw": 0.0,
        },
    ]
    fx = {f["id"]: f for f in compile_twin(bundle)["fixtures"]}
    assert fx["island"].get("island") is True
    assert "island" not in fx["table_on_wall"]


def test_island_beside_an_oblique_wall_is_measured_to_the_wall_not_the_axes():
    walls = [{"id": "d", "start": [0, 0], "end": [4, 4], "height": 2.5, "thickness": 0.1}]
    near = {"id": "near", "category": "table", "size": [0.8, 0.8, 0.92], "position": [2.0, 2.4, 0.46], "yaw": 0.0}
    far = {"id": "far", "category": "table", "size": [0.8, 0.8, 0.92], "position": [3.0, 0.4, 0.46], "yaw": 0.0}
    fx = {f["id"]: f for f in compile_twin(_bundle([near, far], walls))["fixtures"]}
    assert "island" not in fx["near"] and fx["far"].get("island") is True

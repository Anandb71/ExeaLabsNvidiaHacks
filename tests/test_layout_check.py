"""Offline layout.yaml vs twin checks. Nothing here loads RoboCasa."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest
import yaml

from arrival.scan import roomplan
from arrival.twin.compile import compile_twin, layout_dict

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_layout", ROOT / "scripts" / "check_layout.py")
cl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cl)


def _layout(twin):
    return yaml.safe_load(yaml.safe_dump(layout_dict(twin), sort_keys=False))


def test_hero_passes(hero_twin):
    res = cl.check_layout(hero_twin, _layout(hero_twin))
    assert res["ok"], res["errors"]
    assert res["fixtures"] == len(hero_twin["fixtures"])


@pytest.mark.parametrize("path", sorted((ROOT / "tests" / "fixtures" / "roomplan").glob("*.json")))
def test_roomplan_fixtures_pass_and_declare_omissions(path):
    twin = compile_twin(roomplan.load(path, path.stem))
    res = cl.check_layout(twin, _layout(twin))
    assert res["ok"], res["errors"]
    assert len(_layout(twin)["main_group"]["bottom_row"]) == len(twin["fixtures"])  # nothing silently dropped
    for key, n in res["declared_omissions"].items():
        assert n == len(twin[key])


def test_oblique_wall_and_yawed_fixture_round_trip():
    bundle = {
        "schema_version": "1",
        "room_id": "r",
        "units": "meters",
        "frame": "z_up",
        "geometry_source": "authored_sample",
        "blocks_claims": [],
        "objects": [],
        "walls": [
            {"id": "w0", "start": [0, 0], "end": [4, 0], "height": 2.4, "thickness": 0.1},
            {"id": "w1", "start": [4, 0], "end": [4, 3], "height": 2.4, "thickness": 0.1},
            {"id": "w2", "start": [4, 3], "end": [0, 0], "height": 2.4, "thickness": 0.1},
        ],
        "fixtures": [
            {"id": "bed1", "category": "bed", "size": [1.6, 2.0, 0.5], "position": [2, 1, 0.25], "yaw": 1.0},
            {"id": "thing", "category": "mystery", "size": [0.4, 0.3, 0.2], "position": [1, 1, 0.1]},
        ],
    }
    twin = compile_twin(bundle)
    assert cl.check_layout(twin, _layout(twin))["ok"]


def _mutated(twin, fn):
    lay = _layout(twin)
    fn(lay)
    return cl.check_layout(twin, lay)["errors"]


def test_checker_catches_each_defect(hero_twin):
    row = lambda lay: lay["main_group"]["bottom_row"]  # noqa: E731
    assert any("dropped" in e for e in _mutated(hero_twin, lambda l: row(l).pop()))
    assert any(
        "half-extent" in e
        for e in _mutated(hero_twin, lambda l: row(l)[0].update(size=[v / 2 for v in row(l)[0]["size"]]))
    )
    assert any(
        "5 cm" in e
        for e in _mutated(
            hero_twin, lambda l: row(l)[0].update(pos=[row(l)[0]["pos"][0] + 0.06] + row(l)[0]["pos"][1:])
        )
    )
    assert any("z_rot" in e for e in _mutated(hero_twin, lambda l: row(l)[0].update(z_rot=0.7)))
    assert any(
        "ends" in e for e in _mutated(hero_twin, lambda l: l["room"]["walls"][0].update(size=[0.02, 4.0, 2.4]))
    )  # length and thickness swapped
    assert any(
        "thickness" in e for e in _mutated(hero_twin, lambda l: l["room"]["walls"][0].update(size=[4.0, 0.3, 2.4]))
    )
    assert any("dropped" in e for e in _mutated(hero_twin, lambda l: l["room"]["walls"].pop()))
    assert any("ends" in e for e in _mutated(hero_twin, lambda l: l["room"]["walls"][0].update(pos=[3.0, 0.0, 1.2])))
    bad = copy.deepcopy(hero_twin)
    bad["units"] = "centimeters"
    assert any("units" in e for e in cl.check_layout(bad, _layout(hero_twin))["errors"])
    cw = copy.deepcopy(hero_twin)
    for w in cw["walls"]:
        w["start"], w["end"] = w["end"], w["start"]
    res = cl.check_layout(cw, _layout(cw))
    assert res["ok"] and any("clockwise" in w for w in res["warnings"])


def test_cli_room(capsys):
    assert cl.main(["--room", "hero_kitchen"]) == 0
    assert "not loaded in RoboCasa" in capsys.readouterr().out
    assert math.isfinite(cl.TOL)


@pytest.mark.parametrize("argv", [["--room", "no_such_room"], ["{tmp}"]])
def test_cli_missing_input_is_a_usage_error_not_a_layout_fail(argv, tmp_path, capsys):
    """Exit 1 means the layout failed the check, so a missing room or twin folder must not exit 1 too."""
    with pytest.raises(SystemExit) as e:
        cl.main([x.format(tmp=tmp_path) for x in argv])
    assert e.value.code == 2
    assert "No such file" in capsys.readouterr().err


def test_cli_room_with_scans_folder(capsys):
    assert cl.main(["--room", "rotated_kitchen", "--scans", "tests/fixtures/scans"]) == 0
    assert "not loaded in RoboCasa" in capsys.readouterr().out


def test_cli_roomplan_fixture(capsys):
    fixture = str(ROOT / "tests" / "fixtures" / "roomplan" / "kitchen_real_01.json")
    assert cl.main(["--roomplan", fixture]) == 0
    assert "not loaded in RoboCasa" in capsys.readouterr().out


def test_cli_multiple_inputs_rejected(capsys):
    with pytest.raises(SystemExit) as e:
        cl.main(["--room", "hero_kitchen", "--roomplan", "tests/fixtures/roomplan/office.json"])
    assert e.value.code == 2
    assert "give exactly one of" in capsys.readouterr().err


def test_cli_room_hint_on_missing_room(capsys):
    with pytest.raises(SystemExit) as e:
        cl.main(["--room", "missing_room"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "rooms under scans: hero_kitchen" in err

"""The GR00T modality config must agree with the meta/modality.json our exporter writes.

GR00T itself needs a GPU box, so its modules are stubbed: the config file is executed for real and the
ModalityConfig objects it builds are compared against a freshly written dataset.
"""

import json
import runpy
import sys
import types
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim.collect import collect  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "gr00t"
CONFIG = CONFIGS / "arrival_panda_config.py"
MOBILE_CONFIG = CONFIGS / "arrival_panda_mobile_config.py"


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _load_config(monkeypatch, path=CONFIG):
    registered = []
    enums = {
        n: types.SimpleNamespace(RELATIVE="rel", ABSOLUTE="abs", NON_EEF="joint", DEFAULT="default")
        for n in ("ActionRepresentation", "ActionType", "ActionFormat")
    }
    mods = {
        "gr00t": types.ModuleType("gr00t"),
        "gr00t.configs": types.ModuleType("gr00t.configs"),
        "gr00t.configs.data": types.ModuleType("gr00t.configs.data"),
        "gr00t.configs.data.embodiment_configs": types.SimpleNamespace(
            register_modality_config=lambda cfg, embodiment_tag: registered.append((cfg, embodiment_tag))
        ),
        "gr00t.data": types.ModuleType("gr00t.data"),
        "gr00t.data.embodiment_tags": types.SimpleNamespace(EmbodimentTag=types.SimpleNamespace(NEW_EMBODIMENT="NEW")),
        "gr00t.data.types": types.SimpleNamespace(ActionConfig=_Cfg, ModalityConfig=_Cfg, **enums),
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    runpy.run_path(str(path))
    assert len(registered) == 1 and registered[0][1] == "NEW"
    return registered[0][0]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    twin = compile_twin(load_room("hero_kitchen", "scans"))
    res = collect(twin, ["open_drawer"], 1, tmp_path_factory.mktemp("ds"), people=0, res=64, log=lambda *_: None)
    return res["dataset"]


@pytest.fixture(scope="module")
def mobile_dataset(tmp_path_factory):
    twin = compile_twin(load_room("hero_kitchen", "scans"))
    res = collect(
        twin,
        ["counter_to_counter"],
        1,
        tmp_path_factory.mktemp("dsm"),
        people=0,
        res=32,
        base="mobile",
        log=lambda *_: None,
    )
    assert res["report_data"]["successes"] == 1
    return res["dataset"]


def _modality(ds):
    return json.loads((ds / "meta" / "modality.json").read_text())


@pytest.mark.parametrize("which", ["dataset", "mobile_dataset"])
def test_arm_config_keys_are_the_dataset_keys_without_the_base(monkeypatch, request, which):
    """The default config reads the first 8 floats of the 11-float rows, fixed or mobile; the base key is not in it."""
    cfg = _load_config(monkeypatch)
    modality = _modality(request.getfixturevalue(which))
    assert cfg["video"].modality_keys == list(modality["video"])
    assert cfg["state"].modality_keys == [k for k in modality["state"] if k != "base"]
    assert cfg["action"].modality_keys == [k for k in modality["action"] if k != "base"]
    assert cfg["language"].modality_keys == ["annotation." + k for k in modality["annotation"]]
    assert "base" in modality["state"] and "base" in modality["action"]  # the data has it; this config just skips it


def test_config_action_setup(monkeypatch, dataset):
    cfg = _load_config(monkeypatch)
    act = cfg["action"]
    assert len(act.action_configs) == len(act.modality_keys)
    assert [c.rep for c in act.action_configs] == ["rel", "abs"]  # arm as deltas, gripper absolute
    assert act.delta_indices == list(range(16))
    assert cfg["video"].delta_indices == cfg["state"].delta_indices == [0]
    # dims the config relies on: 7-dof arm + 1 gripper first, same layout for state and action, then the base slice
    modality = _modality(dataset)
    for group in ("state", "action"):
        assert [(v["start"], v["end"]) for v in modality[group].values()] == [(0, 7), (7, 8), (8, 11)]
    info = json.loads((dataset / "meta" / "info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == info["features"]["action"]["shape"] == [11]


def test_mobile_config_adds_the_base_key_at_8_to_11(monkeypatch, mobile_dataset):
    cfg = _load_config(monkeypatch, MOBILE_CONFIG)
    modality = _modality(mobile_dataset)
    assert cfg["video"].modality_keys == list(modality["video"])
    assert cfg["state"].modality_keys == list(modality["state"]) == ["single_arm", "gripper", "base"]
    assert cfg["action"].modality_keys == list(modality["action"])
    act = cfg["action"]
    assert len(act.action_configs) == len(act.modality_keys)
    assert [c.rep for c in act.action_configs] == ["rel", "abs", "abs"]  # the base is a commanded absolute pose
    assert act.delta_indices == list(range(16))
    for group in ("state", "action"):
        assert modality[group]["base"]["start"] == 8 and modality[group]["base"]["end"] == 11
    assert modality["action"]["base"]["absolute"] is True
    emb = json.loads((mobile_dataset / "meta" / "embodiment.json").read_text())
    assert emb["base_actuated"] is True and emb["dof_base"] == 3


def test_multiview_config_matches_paired_dataset(monkeypatch, tmp_path):
    twin = compile_twin(load_room("hero_kitchen", "scans"))
    result = collect(twin, ["open_drawer"], 1, tmp_path, people=0, res=32, exocentric=True, log=lambda *_: None)
    cfg = _load_config(monkeypatch, CONFIGS / "arrival_panda_multiview_config.py")
    modality = _modality(result["dataset"])
    assert cfg["video"].modality_keys == list(modality["video"]) == ["ego_view", "wrist", "overview"]
    assert cfg["video"].delta_indices == [0]
    assert cfg["state"].modality_keys == ["single_arm", "gripper"]
    assert cfg["action"].modality_keys == ["single_arm", "gripper"]
    assert [c.rep for c in cfg["action"].action_configs] == ["rel", "abs"]

"""family_fixtures replaced the fixture lists make_tasks built inline (counters, cabinets, sinks, then a filter per
family). OLD freezes, for the three planner-benchmark twins, the fixture ids those inline lists gave each family at
upstream 897aa05, so a change to the map that would change these twins' tasks fails here."""

import importlib.util

import pytest
from conftest import REPO

from arrival.sim import tasks
from arrival.sim.tasks import SIM_FAMILIES, family_fixtures, make_tasks

_K = [
    "11793D80-8BE3-45F9-AF2F-34068D45F7C2",
    "CC64C90B-5AD3-4C42-8E56-3D8B4FE9C9B9",
    "F9337652-45E5-419A-B727-FE171FB57F90",
]  # kitchen_real_01's three cabinets; the middle one is a narrow door
_K_DRAWERS, _K_STOVE, _K_DISHWASHER, _K_FRIDGE = (
    [_K[0], _K[2]],
    ["807FD3A5-AF23-4620-A58F-D4FDD13850A3"],
    ["F8CDA831-6DB6-48C1-9482-A5DA79652AEC"],
    ["F068E3AE-575B-49A0-BD35-8E233153BE91"],
)
_HERO = ["counter_a", "counter_b"]
OLD = {
    "hero_kitchen": {
        "counter_to_sink": ["sink_1"],
        "counter_to_counter": _HERO,
        "open_drawer": _HERO,
        "close_drawer": _HERO,
        "put_in_drawer": _HERO,
        "open_door": _HERO,
        "close_door": _HERO,
        "wipe_counter": _HERO,
        "move_on_table": [],
        "open_oven": ["stove_1"],
        "open_dishwasher": [],
        "load_dishwasher": [],
        "unload_dishwasher": [],
        "turn_on_faucet": ["sink_1"],
        "open_fridge": ["fridge_1"],
    },
    "kitchen_real_01": {
        "counter_to_sink": [],
        "counter_to_counter": _K,
        "open_drawer": _K_DRAWERS,
        "close_drawer": _K_DRAWERS,
        "put_in_drawer": _K_DRAWERS,
        "open_door": _K,
        "close_door": _K,
        "wipe_counter": _K,
        "move_on_table": [],
        "open_oven": _K_STOVE,
        "open_dishwasher": _K_DISHWASHER,
        "load_dishwasher": _K_DISHWASHER,
        "unload_dishwasher": _K_DISHWASHER,
        "turn_on_faucet": [],
        "open_fridge": _K_FRIDGE,
    },
    "office": {
        **{fam: [] for fam in SIM_FAMILIES},
        "counter_to_counter": ["TABLE-01"],
        "wipe_counter": ["TABLE-01"],
        "move_on_table": ["TABLE-01"],
    },
}


@pytest.fixture(scope="module")
def twins():
    pytest.importorskip("mujoco")  # place_default_objects for the RoomPlan rooms, and the reach pre-flight
    spec = importlib.util.spec_from_file_location("planner_benchmark", REPO / "scripts" / "planner_benchmark.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return bench.room_twins()


def test_family_fixtures_gives_the_lists_make_tasks_built_inline(twins):
    assert set(twins) == set(OLD)
    for room, twin in twins.items():
        assert {fam: [f["id"] for f in fxs] for fam, fxs in family_fixtures(twin).items()} == OLD[room], room


@pytest.mark.parametrize("room", sorted(OLD))
@pytest.mark.parametrize("mobile", [False, True])
def test_make_tasks_makes_the_same_tasks_from_the_old_lists(twins, room, mobile, monkeypatch):
    twin = twins[room]
    quiet = dict(seed=0, mobile=mobile, log=lambda *_: None)
    now_pre, old_pre = {}, {}
    now = make_tasks(twin, list(SIM_FAMILIES), 60, preflight=now_pre, **quiet)
    by_id = {f["id"]: f for f in twin["fixtures"]}
    monkeypatch.setattr(
        tasks, "family_fixtures", lambda t: {fam: [by_id[i] for i in ids] for fam, ids in OLD[room].items()}
    )
    old = make_tasks(twin, list(SIM_FAMILIES), 60, preflight=old_pre, **quiet)
    assert now and now == old and now_pre == old_pre

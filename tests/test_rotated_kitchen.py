"""The sim must not assume every counter is on the north wall facing +y (issue #42). `rotated_kitchen` puts runs
on the west (facing +x), south (facing -y) and east (facing -x) walls: the base parks in front of each fixture,
the walkers get a floor strip that exists, and the scripted expert still succeeds."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim.collect import SimEnv, run_episode  # noqa: E402
from arrival.sim.people import person_spec  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402
from arrival.sim.stand import front  # noqa: E402
from arrival.sim.tasks import SIM_FAMILIES, finish_task, make_tasks  # noqa: E402
from arrival.twin.compile import compile_twin, rect_distance  # noqa: E402

SCANS = "tests/fixtures/scans"
RUNS = ("counter_w", "counter_s", "counter_e")


@pytest.fixture(scope="module")
def twin():
    return compile_twin(load_room("rotated_kitchen", SCANS))


def _tasks(twin, families, n, people=0):
    tasks = make_tasks(twin, list(families), n, seed=5, people=people, log=lambda *_: None)
    return [finish_task(t) for t in tasks]  # collect() fills the joint and handle names before an episode runs


def _footprint_clearance(twin, xy):
    return min(rect_distance(list(xy), f) for f in twin["fixtures"])


def test_every_fixture_type_is_built(twin):
    env = SimEnv(twin, _tasks(twin, ["open_drawer"], 1)[0], res=32)
    bodies = {env.model.body(i).name for i in range(env.model.nbody)}
    assert {"fridge_w", "counter_w", "stove_s", "counter_s", "counter_e"} <= bodies
    assert any(env.model.geom(i).name.startswith("sink_w") for i in range(env.model.ngeom))  # the basin is cut in


def test_base_parks_in_front_of_fixtures_on_every_wall(twin):
    tasks = _tasks(twin, SIM_FAMILIES, 14)
    assert {t.get("fixture") or t["source"] for t in tasks} >= set(RUNS)
    for t in tasks:
        env = SimEnv(twin, t, res=32)
        base = env.data.body("robot_base")
        xy = base.xpos[:2]
        anchor = env.anchor
        assert _footprint_clearance(twin, xy) >= 0.25, (t["task_id"], xy)
        assert 0.4 < xy[0] < 4.1 and 0.4 < xy[1] < 3.1, (t["task_id"], xy)
        heading = base.xmat.reshape(3, 3)[:, 0][:2]
        assert np.dot(heading, front(anchor)[:2]) < -0.99, t["task_id"]  # faces the fixture's front
        near = 1.0 if t["family"] in ("open_oven", "open_dishwasher") else 0.8  # the flap door's length is in between
        assert rect_distance(list(xy), anchor) < near, t["task_id"]  # and stands at it, not across the room


def test_sink_tasks_only_from_the_sink_run(twin):
    tasks = _tasks(twin, ["counter_to_sink"], 8)
    assert {t["source"] for t in tasks} == {"counter_w"} and "mug_2" not in {t["object_id"] for t in tasks}
    assert not _tasks(twin, ["counter_to_counter"], 4)  # the runs are on different walls, none continues another


@pytest.mark.parametrize("fixture", RUNS)
def test_walkers_get_a_strip_in_front_of_each_run(twin, fixture):
    t = next(t for t in _tasks(twin, ["open_drawer"], 6, people=2) if t["fixture"] == fixture)
    env = SimEnv(twin, t, res=32)
    assert env.people_note is None
    floor, (bx, by), front, note = env._free_floor(env.anchor, env.data.body("robot_base").xpos)
    assert note is None and floor[2] - floor[0] >= 1.0 and floor[3] - floor[1] >= 0.6
    w = env.walkers
    for i, path in enumerate(w.paths):
        radius = person_spec(t["seed"], i)["radius"]
        for p in path:  # world waypoints: in the room and clear of every footprint by the person's radius
            assert 0.0 < p[0] < 4.5 and 0.0 < p[1] < 3.5, (fixture, p)
            assert _footprint_clearance(twin, p) >= radius - 1e-9, (fixture, p)
        if w.visit and i == w.visit["person"]:
            continue  # the fridge visitor's loop starts with the trip to the fridge
        k = int(np.argmax(w.pauses[i]))  # the counter stop, in the work frame beside the base and in front of the run
        sx, sy = w.M @ path[k]
        assert abs(sy - (front + 0.35)) < 1e-9 and abs(sx - bx) >= 0.8, (fixture, sx, sy)
    env.step(np.array(list(env.data.ctrl[:7]) + [0.04]))
    assert all(0.0 < x < 4.5 and 0.0 < y < 3.5 for x, y in env.people_xy)


@pytest.mark.parametrize(
    "family,fixture,obj",
    [
        ("counter_to_sink", "counter_w", "mug_1"),
        ("open_drawer", "counter_e", None),
        ("open_door", "counter_s", None),
        ("put_in_drawer", "counter_w", "mug_1"),
    ],
)
def test_expert_succeeds_with_people_on_a_rotated_run(twin, family, fixture, obj):
    t = next(
        t
        for t in _tasks(twin, [family], 8, people=2)
        if (t.get("fixture") or t["source"]) == fixture and (obj is None or t["object_id"] == obj)
    )
    env = SimEnv(twin, t, res=32)
    ep = run_episode(env, ScriptedExpert(), t)
    assert ep["success"] and not ep["safety"]["human_contact"], (t["task_id"], ep["detail"])
    assert ep["ik_error"] < 0.02


def test_closed_pads_squeeze_a_bowl_wall_and_a_plate_rim(twin):
    """The pads close to a fixed gap, so a rim thinner than it sits loose between them and the grasp holds only
    where an octagon panel crosses the pads at an angle (the bowl on counter_w missed 3 of 4 with a 6 mm wall)."""
    from arrival.sim.scene import BOWL_WALL, PLATE_RIM

    env = SimEnv(twin, _tasks(twin, ["counter_to_sink"], 1)[0], res=32)
    m, d = env.model, env.data
    for j in ("finger_l", "finger_r"):
        d.joint(j).qpos[0] = 0.0
    mujoco.mj_forward(m, d)
    pads = [
        i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name in ("finger_l", "finger_r") and m.geom_contype[i]
    ]
    a, b = d.geom_xpos[pads[0]], d.geom_xpos[pads[1]]
    axis = (b - a) / np.linalg.norm(b - a)
    half = [float(np.abs(d.geom_xmat[i].reshape(3, 3).T @ axis) @ m.geom_size[i]) for i in pads]
    gap = float(np.linalg.norm(b - a)) - sum(half)
    assert 0.0 < gap < BOWL_WALL and gap < PLATE_RIM, (gap, BOWL_WALL, PLATE_RIM)


@pytest.mark.parametrize("k", [0, 2])  # the bowl at its spot and at a start offset
def test_expert_carries_the_bowl_on_the_west_run(twin, k):
    tasks = [finish_task(t) for t in make_tasks(twin, ["counter_to_sink"], 3, seed=0, log=lambda *_: None)]
    t = tasks[k]
    assert t["object_id"] == "bowl_1" and t["source"] == "counter_w"
    ep = run_episode(SimEnv(twin, t, res=32), ScriptedExpert(), t)
    assert ep["success"], (t["task_id"], ep["detail"])

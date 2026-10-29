"""MAN-5 appliance families: the scripted expert opens the oven and dishwasher doors, lifts the faucet lever, and
loads and unloads the dishwasher's upper rack; the dishwasher is hollow with a rack that rolls out over the door."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.sim.collect import DOOR_FLAT, FAUCET_ON, FLAP_OPEN, RACK_OUT, SimEnv, run_episode  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402
from arrival.sim.scene import RACK_Z, build_mjcf, part_names  # noqa: E402
from arrival.sim.tasks import SIM_FAMILIES, finish_task, make_tasks  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

WALLS = [
    {"id": "n", "start": [0.0, 0.0], "end": [4.0, 0.0], "height": 2.4, "thickness": 0.02},
    {"id": "e", "start": [4.0, 0.0], "end": [4.0, 3.0], "height": 2.4, "thickness": 0.02},
    {"id": "s", "start": [4.0, 3.0], "end": [0.0, 3.0], "height": 2.4, "thickness": 0.02},
    {"id": "w", "start": [0.0, 3.0], "end": [0.0, 0.0], "height": 2.4, "thickness": 0.02},
]


@pytest.fixture(scope="module")
def dishwasher_twin():
    """A counter with a sink, a cup on it, and a dishwasher at the counter's end (the hero scan has none)."""
    fixtures = [
        {"id": "c_1", "category": "counter", "size": [2.0, 0.65, 0.92], "position": [1.0, 0.40, 0.46], "yaw": 0.0},
        {"id": "sink_1", "category": "sink", "size": [0.6, 0.5, 0.25], "position": [0.8, 0.4, 0.85], "yaw": 0.0},
        {"id": "dw_1", "category": "dishwasher", "size": [0.6, 0.6, 0.86], "position": [2.3, 0.40, 0.43], "yaw": 0.0},
    ]
    cup = {"id": "cup_1", "label": "cup", "size": [0.07, 0.07, 0.09], "position": [1.7, 0.40, 0.965], "yaw": 0.0}
    return compile_twin(
        {
            "schema_version": "1",
            "room_id": "t",
            "units": "meters",
            "frame": "z_up",
            "geometry_source": "authored_sample",
            "blocks_claims": [],
            "walls": WALLS,
            "fixtures": fixtures,
            "objects": [cup],
        }
    )


def _task(twin, family, seed=1, people=0):
    return finish_task(make_tasks(twin, [family], 1, seed=seed, people=people, log=lambda *_: None)[0])


def test_families_are_registered_and_pooled(hero_twin, dishwasher_twin):
    assert {"open_oven", "open_dishwasher", "turn_on_faucet", "load_dishwasher", "unload_dishwasher"} <= set(
        SIM_FAMILIES
    )
    hero = make_tasks(hero_twin, ["open_oven", "turn_on_faucet", "open_dishwasher"], 6, seed=0, log=lambda *_: None)
    assert {t["family"] for t in hero} == {"open_oven", "turn_on_faucet"}  # no dishwasher in the hero scan
    assert {t["fixture"] for t in hero} == {"stove_1", "sink_1"}
    dw = make_tasks(
        dishwasher_twin, ["open_dishwasher", "load_dishwasher", "unload_dishwasher"], 6, seed=0, log=lambda *_: None
    )
    assert {t["family"] for t in dw} == {"open_dishwasher", "load_dishwasher", "unload_dishwasher"}
    assert all(t["object_id"] == "cup_1" and t["source"] == "c_1" for t in dw if "object_id" in t)


def test_dishwasher_is_hollow_with_a_rack_that_rolls_out(dishwasher_twin):
    m = mujoco.MjModel.from_xml_string(build_mjcf(dishwasher_twin, (1.3, 1.2, 0.75)))
    d = mujoco.MjData(m)
    names = {m.geom(i).name for i in range(m.ngeom)}
    assert {"dw_1_bottom", "dw_1_top", "dw_1_back", "dw_1_left", "dw_1_right", "dw_1_rack_tray"} <= names
    assert "dw_1_box" not in names
    rack = next(p for p in part_names(dishwasher_twin) if p["joint"] == "dw_1_rack")
    assert rack["kind"] == "slide" and rack["open"] == RACK_OUT
    mujoco.mj_forward(m, d)
    y0 = d.geom("dw_1_rack_tray").xpos[1]
    d.qpos[m.joint("dw_1_rack").qposadr[0]] = RACK_OUT
    mujoco.mj_forward(m, d)
    assert d.geom("dw_1_rack_tray").xpos[1] - y0 == pytest.approx(RACK_OUT, abs=1e-6)
    assert d.geom("dw_1_rack_tray").xpos[2] == pytest.approx(RACK_Z, abs=1e-6)


@pytest.mark.parametrize("family", ["open_oven", "turn_on_faucet"])
def test_hero_kitchen_oven_and_faucet(hero_twin, family):
    task = _task(hero_twin, family)
    env = SimEnv(hero_twin, task, res=32)
    assert float(env.data.joint(task["joint"]).qpos[0]) < 0.01  # starts closed / off
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"], ep["detail"]
    assert ep["detail"]["qpos"] > (FLAP_OPEN if family == "open_oven" else FAUCET_ON)
    assert ep["ik_error"] < 0.03
    a = np.array(list(env.data.ctrl[:7]) + [env.data.ctrl[7]])
    for _ in range(400):  # 20 s with the arm parked: the door stays down and the lever stays up
        env.step(a)
    assert float(env.data.joint(task["joint"]).qpos[0]) > (FLAP_OPEN if family == "open_oven" else FAUCET_ON)


def test_open_oven_stand_keeps_every_joint_off_its_stop(hero_twin):
    """Straight in front of the bar, joint 5 sat on its stop for 71 steps and the episode validator flagged both G1
    open_oven episodes; from OVEN_ALONG to the side of the bar no joint stays in the 3% band for LIMIT_STEPS."""
    from arrival.export.quality import check_trace
    from arrival.sim import robot
    from arrival.sim.stand import OVEN_ALONG, base_pose
    from arrival.twin.compile import to_local

    task = _task(hero_twin, "open_oven")
    (x, y, _), _, src = base_pose(hero_twin, task)
    assert to_local((x, y), src)[0] == pytest.approx(OVEN_ALONG)
    ep = run_episode(SimEnv(hero_twin, task, res=32), ScriptedExpert(), task)
    assert ep["success"], ep["detail"]
    assert ep["ik_error"] < 0.005
    flags = check_trace(ep["states"], ep["actions"], list(robot.RANGES))
    assert not [f for f in flags if f["check"] == "joint_at_limit"], flags


def test_open_oven_stands_on_the_other_side_when_one_side_is_against_a_wall(hero_twin):
    """The hero layout with stove_1 and counter_b swapped puts the stove's +x end 1 cm from wall_east, where the
    OVEN_ALONG spot is too close to the wall; the base stands on the -x side instead and the episode still clears."""
    import copy

    from arrival.export.quality import check_trace
    from arrival.sim import robot
    from arrival.sim.reach import check_task
    from arrival.sim.stand import OVEN_ALONG, base_pose
    from arrival.twin.compile import to_local

    twin = copy.deepcopy(hero_twin)
    fx = {f["id"]: f for f in twin["fixtures"]}
    fx["stove_1"]["position"][0] = 4.0 - 0.01 - 0.01 - 0.35
    fx["counter_b"]["position"][0] = fx["stove_1"]["position"][0] - 0.35 - 0.55
    task = _task(twin, "open_oven")
    assert check_task(twin, task) is None
    (x, y, _), _, src = base_pose(twin, task)
    assert to_local((x, y), src)[0] == pytest.approx(-OVEN_ALONG)
    ep = run_episode(SimEnv(twin, task, res=32), ScriptedExpert(), task)
    assert ep["success"], ep["detail"]
    flags = check_trace(ep["states"], ep["actions"], list(robot.RANGES))
    assert not [f for f in flags if f["check"] == "joint_at_limit"], flags


def test_faucet_lever_is_pushed_without_driving_it_through_its_stop(hero_twin):
    task = _task(hero_twin, "turn_on_faucet")
    env = SimEnv(hero_twin, task, res=32)
    pol = ScriptedExpert()
    pol.reset(env, task)
    qs = []
    for _ in range(len(pol.plan) + 10):
        env.step(pol.act(None))
        qs.append(float(env.data.joint(task["joint"]).qpos[0]))
    hi = float(env.model.jnt_range[env.model.joint(task["joint"]).id][1])
    assert max(qs) < hi + 0.05, max(qs)  # never forced past the 1.2 rad range by the fingers
    assert FAUCET_ON < qs[-1] <= hi + 0.01


def test_oven_door_is_pulled_before_it_falls(hero_twin):
    """The expert pulls the door through the first part of its arc and only then lets gravity finish; the door
    must be moving with the hand while the gripper is closed, not knocked open."""
    task = _task(hero_twin, "open_oven")
    env = SimEnv(hero_twin, task, res=32)
    pol = ScriptedExpert()
    pol.reset(env, task)
    closed_q = []
    for _ in range(len(pol.plan)):
        a = pol.act(None)
        env.step(a)
        if a[7] < 0.005:
            closed_q.append(float(env.data.joint(task["joint"]).qpos[0]))
    assert closed_q and max(closed_q) > 0.4  # pulled well past 20 deg while held
    for _ in range(20):  # the door is still falling when the plan ends; run_episode's tail covers this
        env.step(a)
    assert float(env.data.joint(task["joint"]).qpos[0]) > FLAP_OPEN


@pytest.mark.parametrize("family", ["open_dishwasher", "load_dishwasher", "unload_dishwasher"])
def test_dishwasher_families(dishwasher_twin, family):
    task = _task(dishwasher_twin, family, people=2)
    env = SimEnv(dishwasher_twin, task, res=32)
    if family != "open_dishwasher":  # door flat and rack out at the start, as a person would leave it
        assert float(env.data.joint("dw_1_door").qpos[0]) == pytest.approx(DOOR_FLAT, abs=0.01)  # settled on its stop
        assert float(env.data.joint("dw_1_rack").qpos[0]) == pytest.approx(RACK_OUT, abs=0.01)
    if family == "unload_dishwasher":
        p = env.body_pos("cup_1")
        assert abs(p[2] - (RACK_Z + 0.01 + 0.045)) < 0.01 and p[1] > 0.7  # resting in the rolled-out part of the rack
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"], ep["detail"]
    if family == "unload_dishwasher":
        assert abs(ep["detail"]["object_pos"][2] - 0.965) < 0.01  # back on the counter top
    assert ep["ik_error"] < 0.07
    assert all(int(w) == 0 for w in env.data.warning.number)


def test_walkers_and_people_stream_still_work_with_the_rack_out(dishwasher_twin):
    task = _task(dishwasher_twin, "load_dishwasher", people=2)
    env = SimEnv(dishwasher_twin, task, res=32)
    env.step(np.array(list(env.data.ctrl[:7]) + [0.05]))
    assert len(env.people_xy) == 2


# ---- one handle/joint table for collect() and eval, and the east-wall layout (#42's lesson applied here) ----

APPLIANCE = ("open_oven", "open_dishwasher", "turn_on_faucet", "load_dishwasher", "unload_dishwasher")
QUIET = lambda *_: None  # noqa: E731


def test_collect_and_eval_assign_handles_and_joints_with_one_function(hero_twin, dishwasher_twin):
    """`python -m arrival eval` once carried its own copy of the table and raised on every family it did not know.
    collect() and eval both call arrival.sim.tasks.finish_task, and every name it writes must exist in the model
    the task runs in."""
    import inspect

    import arrival.eval.manifest as manifest_mod

    assert "finish_task" in inspect.getsource(
        manifest_mod.make_manifest
    )  # eval builds its tasks through the same function
    for twin, fams in (
        (hero_twin, ("open_oven", "turn_on_faucet")),
        (dishwasher_twin, ("open_dishwasher", "load_dishwasher", "unload_dishwasher", "turn_on_faucet")),
    ):
        model = mujoco.MjModel.from_xml_string(build_mjcf(twin, (1.3, 1.2, 0.75)))
        for fam in fams:
            t = _task(twin, fam)
            assert model.joint(t["joint"]).id >= 0  # raises KeyError-like if the name is not in the model
            if fam in ("open_oven", "open_dishwasher", "turn_on_faucet"):
                assert model.site(t["handle"]).id >= 0


def test_collect_writes_an_appliance_episode(dishwasher_twin, tmp_path):
    """The same path through collect(): a faucet task and a dishwasher load, one episode each, written and counted."""
    from arrival.sim.collect import collect

    rep = collect(
        dishwasher_twin, ["turn_on_faucet", "load_dishwasher"], 2, out_dir=tmp_path, people=0, res=32, seed=1, log=QUIET
    )["report_data"]
    assert rep["attempts"] == 2 and rep["successes"] == 2 and rep["skipped"] == 0, rep
    assert set(rep["families"]) == {"turn_on_faucet", "load_dishwasher"}


def test_eval_runs_every_appliance_family_one_episode_each(hero_twin, dishwasher_twin, tmp_path):
    """evaluate() turns a raised episode into a counted failure, so success and an empty error list both matter."""
    from arrival.eval.harness import evaluate

    for twin, fams in (
        (hero_twin, ["open_oven", "turn_on_faucet"]),
        (dishwasher_twin, ["open_dishwasher", "load_dishwasher", "unload_dishwasher"]),
    ):
        rep = evaluate(twin, fams, 1, res=32, people=0, out_dir=tmp_path, log=QUIET)
        assert set(rep["families"]) == set(fams)
        for fam, f in rep["families"].items():
            assert f["episodes"] == 1 and f["errors"] == [] and f["successes"] == 1, (fam, f)


def _east_wall_twin():
    """The dishwasher layout turned to face -x (counter on the east wall, yaw 90 deg) with a stove added: the run's
    local x points along +y, so any world-x assumption in the base pose, the success check or the plans fails."""
    walls = [
        {"id": "n", "start": [0.0, 0.0], "end": [4.0, 0.0], "height": 2.4, "thickness": 0.02},
        {"id": "e", "start": [4.0, 0.0], "end": [4.0, 5.0], "height": 2.4, "thickness": 0.02},
        {"id": "s", "start": [4.0, 5.0], "end": [0.0, 5.0], "height": 2.4, "thickness": 0.02},
        {"id": "w", "start": [0.0, 5.0], "end": [0.0, 0.0], "height": 2.4, "thickness": 0.02},
    ]
    x, yaw = 3.67, np.pi / 2

    def at(local):  # a position along the run, whose centre is (3.67, 2.5)
        return [x, 2.5 + local]

    fixtures = [
        {"id": "c_1", "category": "counter", "size": [2.0, 0.65, 0.92], "position": [*at(0.0), 0.46], "yaw": yaw},
        {"id": "sink_1", "category": "sink", "size": [0.6, 0.5, 0.25], "position": [*at(-0.2), 0.85], "yaw": yaw},
        {"id": "dw_1", "category": "dishwasher", "size": [0.6, 0.6, 0.86], "position": [*at(1.3), 0.43], "yaw": yaw},
        {"id": "stove_1", "category": "stove", "size": [0.70, 0.65, 0.90], "position": [*at(-1.4), 0.45], "yaw": yaw},
    ]
    cup = {"id": "cup_1", "label": "cup", "size": [0.07, 0.07, 0.09], "position": [*at(0.7), 0.965], "yaw": 0.0}
    return compile_twin(
        {
            "schema_version": "1",
            "room_id": "t_east",
            "units": "meters",
            "frame": "z_up",
            "geometry_source": "authored_sample",
            "blocks_claims": [],
            "walls": walls,
            "fixtures": fixtures,
            "objects": [cup],
        }
    )


@pytest.mark.parametrize("family", APPLIANCE)
def test_appliance_families_on_an_east_wall_run(family):
    twin = _east_wall_twin()
    task = _task(twin, family)
    assert task["family"] == family
    env = SimEnv(twin, task, res=32)
    assert env.base_pose()[0] < 3.3  # the base stands in the room in front of the run, not behind or inside it
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"], ep["detail"]

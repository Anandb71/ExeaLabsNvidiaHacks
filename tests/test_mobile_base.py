"""MAN-3: the pedestal drives behind --base mobile; the base pose is state[8:11] and action[8:11] in both modes."""

import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.sim import robot  # noqa: E402
from arrival.sim.collect import SimEnv, collect, run_episode  # noqa: E402
from arrival.sim.export import STATE_DIM  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402
from arrival.sim.robot import FPS  # noqa: E402
from arrival.sim.scene import build_mjcf  # noqa: E402
from arrival.sim.tasks import make_tasks  # noqa: E402


def _task(twin, family, seed=1, people=0, mobile=False):
    t = make_tasks(twin, [family], 1, seed=seed, people=people, mobile=mobile, log=lambda *_: None)[0]
    if family in ("open_drawer", "close_drawer", "put_in_drawer"):
        t["handle"], t["joint"] = f"{t['fixture']}_drawer0_handle", f"{t['fixture']}_drawer0"
    elif family in ("open_door", "close_door"):
        t["joint"] = f"{t['fixture']}_door"
    return t


def test_fixed_base_scene_is_unchanged_and_mobile_adds_three_joints(hero_twin):
    fixed = build_mjcf(hero_twin, (1.3, 1.2, 0.75), base="fixed")
    assert fixed == build_mjcf(hero_twin, (1.3, 1.2, 0.75)) and "base_x" not in fixed
    assert robot.arm_xml((1.3, 1.2, 0.75), -90.0) == robot.arm_xml((1.3, 1.2, 0.75), -90.0, mobile=False)
    m = mujoco.MjModel.from_xml_string(build_mjcf(hero_twin, (1.3, 1.2, 0.75), base="mobile"))
    assert [m.joint(j).name for j in range(m.njnt)][:3] != [] and all(m.joint(j).id >= 0 for j in robot.BASE_JOINTS)
    assert [m.actuator(i).name for i in range(m.nu)][-3:] == ["b_x", "b_y", "b_yaw"]
    d = mujoco.MjData(m)
    d.qpos[m.joint("base_x").qposadr[0]] = 0.5  # +x in the room frame whatever the parked yaw
    mujoco.mj_forward(m, d)
    assert np.allclose(d.body("robot_base").xpos[:2], [1.8, 1.2], atol=1e-6)
    with pytest.raises(ValueError):
        SimEnv(hero_twin, _task(hero_twin, "open_drawer"), res=32, base="rolling")


def test_state_and_action_carry_the_base_pose_in_both_modes(hero_twin):
    task = _task(hero_twin, "open_drawer")
    env = SimEnv(hero_twin, task, res=32)
    s = env.state()
    assert s.shape == (STATE_DIM,) and np.allclose(s[8:11], env.base0)
    env.step(np.array(list(robot.HOME) + [robot.FINGER_OPEN]))  # arm-only actions still drive the fixed base
    assert np.allclose(env.state()[8:11], env.base0)
    env = SimEnv(hero_twin, task, res=32, base="mobile")
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"], ep["detail"]
    assert ep["states"].shape[1] == STATE_DIM and ep["actions"].shape[1] == STATE_DIM
    assert np.ptp(ep["states"][:, 8:11], axis=0).max() < 0.01  # the base holds while the arm pulls the drawer
    assert np.ptp(ep["actions"][:, 8:11], axis=0).max() == 0.0


def test_collect_fixed_base_writes_the_base_slice(hero_twin, tmp_path):
    from arrival.export.validate import validate

    res = collect(hero_twin, ["counter_to_sink"], 1, tmp_path, people=0, res=32, log=lambda *_: None)
    ds = res["dataset"]
    assert res["report_data"]["base"] == "fixed" and res["report_data"]["successes"] == 1
    m = json.loads((ds / "meta" / "modality.json").read_text())
    assert m["state"]["base"] == {"start": 8, "end": 11} and m["action"]["base"] == {
        "start": 8,
        "end": 11,
        "absolute": True,
    }
    e = json.loads((ds / "meta" / "embodiment.json").read_text())
    assert e["dof_base"] == 3 and e["base"] == "planar" and e["base_frame"] == "room" and e["base_actuated"] is False
    info = json.loads((ds / "meta" / "info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [11] and info["features"]["action"]["shape"] == [11]
    assert validate(ds) == []


def test_mobile_base_reaches_the_far_counter(hero_twin):
    """counter_to_counter in the hero kitchen: about 2 m between pick and target, skipped on a fixed base."""
    assert make_tasks(hero_twin, ["counter_to_counter"], 2, seed=1, log=lambda *_: None) == []
    task = _task(hero_twin, "counter_to_counter", seed=1, mobile=True)
    assert abs(task["start"][0] - task["target_x"]) > 1.5
    env = SimEnv(hero_twin, task, res=32, base="mobile")
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"], ep["detail"]
    assert ep["ik_error"] < 0.01
    b = ep["states"][:, 8:11]
    assert np.sum(np.linalg.norm(np.diff(b[:, :2], axis=0), axis=1)) > 1.2  # it drove
    assert np.allclose(b[-1, :2], env.drive_to[:2], atol=0.02)
    for arr in (ep["states"], ep["actions"]):
        v = np.linalg.norm(np.diff(arr[:, 8:10], axis=0), axis=1) * FPS
        w = np.abs(np.diff(arr[:, 10])) * FPS
        assert v.max() <= robot.BASE_SPEED + 0.01 and w.max() <= robot.BASE_TURN + 0.01
    g = ep["actions"][:, 7]
    moving = np.linalg.norm(np.diff(ep["actions"][:, 8:10], axis=0), axis=1) > 1e-6
    assert g[1:][moving].max() < 0.01  # the base only moves with the gripper closed on the object


def test_people_keep_off_the_drive_route(hero_twin):
    task = _task(hero_twin, "counter_to_counter", seed=3, people=2, mobile=True)
    env = SimEnv(hero_twin, task, res=32, base="mobile")
    pol = ScriptedExpert()
    ep = run_episode(env, pol, task)
    assert ep["success"] and not ep["safety"]["human_contact"], (ep["detail"], ep["safety"])
    assert ep["safety"]["min_human_distance_m"] > 0.0
    for path in env.walkers.paths:  # no waypoint on the route from the parked spot to the drive target
        a, b = env.base0[:2], env.drive_to[:2]
        for p in path:
            t = float(np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0))
            assert np.linalg.norm(a + t * (b - a) - p) >= 0.6 - 1e-6
    # The crossing lanes run beside the route, so nobody stands in the drive lane and the base never waits here (on
    # main it waited 49 steps for a walker 0.69 m off the drive line); a person on the route still holds it below
    assert pol.held == 0


def test_a_parked_mobile_base_never_waits_for_people(hero_twin):
    """Only a drive in the plan makes the expert wait for people. The base servos sag under the arm's pull, and
    kitchen_real_01's mobile open_oven, open_fridge and open_dishwasher read that sag as a drive: a person standing
    still within STOP_CLEAR froze the arm until the step budget ran out. Here a ring of people 0.5 m around the base
    is in front of any sag."""
    from arrival.sim.tasks import finish_task

    task = finish_task(_task(hero_twin, "open_oven", seed=1, mobile=True))
    env = SimEnv(hero_twin, task, res=32, base="mobile", cameras=())
    pol = ScriptedExpert()
    pol.reset(env, task)
    sag = 0.0
    for _ in range(len(pol.plan) + 20):
        b = env.base_pose()
        sag = max(sag, float(np.linalg.norm(b[:2] - env.base0[:2])))
        env.people_xy = [
            b[:2] + 0.5 * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)
        ]
        env.step(np.asarray(pol.act(env.obs()), dtype=float))
    ok, detail = env.success(task)
    env.close()
    assert sag > 1e-5  # the base did sag, so the old measured-pose check would have read a drive
    assert pol.held == 0 and ok, (pol.held, detail)


def test_the_base_waits_for_a_person_on_the_route_not_beside_it(hero_twin):
    """A person within STOP_CLEAR ahead and within DRIVE_LANE of the drive line holds the drive; one as close but
    0.65 m off the line (a walker in its crossing lane) does not."""
    from arrival.sim.policy import DRIVE_LANE, STOP_CLEAR

    task = _task(hero_twin, "counter_to_counter", seed=1, mobile=True)
    held = {}
    for name, (ahead, side) in {"on the route": (0.5, 0.0), "beside it": (0.2, 0.65)}.items():
        assert np.hypot(ahead, side) < STOP_CLEAR and (side < DRIVE_LANE) == (name == "on the route")
        env = SimEnv(hero_twin, task, res=32, base="mobile", cameras=())
        pol = ScriptedExpert()
        pol.reset(env, task)
        u = env.drive_to[:2] - env.base0[:2]
        u /= np.linalg.norm(u)
        n = np.array([-u[1], u[0]])
        for _ in range(len(pol.plan) + 20):
            env.people_xy = [env.base_pose()[:2] + ahead * u + side * n]
            env.step(np.asarray(pol.act(env.obs()), dtype=float))
        held[name] = (pol.held, pol.t, len(pol.plan))
        env.close()
    assert held["on the route"][0] > 0 and held["on the route"][1] < held["on the route"][2]  # never got to drive
    assert held["beside it"][0] == 0 and held["beside it"][1] >= held["beside it"][2], held  # drove and finished


def test_mobile_base_opens_door_without_human_contact(hero_twin):
    task = _task(hero_twin, "open_door", seed=3, people=2, mobile=True)
    with SimEnv(hero_twin, task, res=32, base="mobile") as env:
        ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"] and not ep["safety"]["human_contact"], (ep["detail"], ep["safety"])
    assert ep["safety"]["min_human_distance_m"] > 0.0


# ---- on top of the any-wall pre-flight (L1, L2) ----


def test_preflight_checks_the_drive_spot_and_the_whole_path(hero_twin):
    import copy

    from arrival.sim.reach import check_task
    from arrival.sim.stand import base_pose, drive_spot, place_point

    task = _task(hero_twin, "counter_to_counter", seed=1, mobile=True)
    assert check_task(hero_twin, task, mobile=True) is None
    assert check_task(hero_twin, task) is not None  # the fixed base cannot reach the place point from where it parks

    start, _, _ = base_pose(hero_twin, task, mobile=True)
    end = drive_spot(hero_twin, task)
    blocked = copy.deepcopy(hero_twin)  # a column on the straight line between the two stand spots, ends untouched
    blocked["fixtures"].append(
        {
            "id": "pillar",
            "type": "storage",
            "yaw": 0.0,
            "size": [0.3, 0.3, 1.5],
            "position": [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, 0.75],
        }
    )
    why = check_task(blocked, task, mobile=True)
    assert why and "drive path" in why and "pillar" in why

    far = dict(task)  # a place point farther from the drive spot than the hand reaches
    tgt = next(f for f in hero_twin["fixtures"] if f["id"] == task["target"])
    away = 1.0 if tgt["position"][1] > end[1] else -1.0
    far["target_y"] = place_point(hero_twin, task)[1] + away * 0.9
    why = check_task(hero_twin, far, mobile=True)
    assert why and "from the drive spot" in why and "beyond the arm's reach" in why and "IK solve misses" in why


def test_mobile_counter_to_counter_on_a_south_wall_run(tmp_path):
    """The u_kitchen_42 scan's storage run faces -y (yaw pi): the stand spot, the drive and the walkers' strip all
    work in that run's frame, not world x. A fixed base finds no pair there."""
    from arrival.scan.bundle import load_room
    from arrival.twin.compile import compile_twin

    twin = compile_twin(load_room("u_kitchen_42", "tests/fixtures/scans"))
    assert make_tasks(twin, ["counter_to_counter"], 2, seed=5, log=lambda *_: None) == []
    task = _task(twin, "counter_to_counter", seed=5, people=1, mobile=True)
    src = next(f for f in twin["fixtures"] if f["id"] == task["source"])
    assert abs(abs(src["yaw"]) - np.pi) < 1e-6  # not the hero kitchen's north wall
    env = SimEnv(twin, task, res=32, base="mobile")
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"] and not ep["safety"]["human_contact"], (ep["detail"], ep["safety"])
    assert np.allclose(ep["states"][-1, 8:10], env.drive_to[:2], atol=0.02)
    assert np.hypot(*(env.drive_to[:2] - env.base0[:2])) > 0.8  # it drove along the run (0.92 m here)


def test_parked_mobile_base_does_not_turn_under_the_wipe_strokes():
    """u_kitchen_42, mobile wipe_counter: the strokes turn the parked base. With a yaw servo as soft as joint 1's
    it turned up to 0.053 rad against each stroke and the sponge travelled under the 0.4 m the check needs (0.62 m
    on the fixed base)."""
    from arrival.scan.bundle import load_room
    from arrival.sim.collect import WIPE_MIN
    from arrival.twin.compile import compile_twin

    twin = compile_twin(load_room("u_kitchen_42", "tests/fixtures/scans"))
    task = _task(twin, "wipe_counter", seed=1, mobile=True)
    env = SimEnv(twin, task, res=32, base="mobile", cameras=())
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"], ep["detail"]
    assert ep["detail"]["wiped_m"] > WIPE_MIN + 0.05
    assert np.abs(ep["states"][:, 10] - env.base0[2]).max() < 0.01

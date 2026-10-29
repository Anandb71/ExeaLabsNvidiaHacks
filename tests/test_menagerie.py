import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim import menagerie, robot  # noqa: E402
from arrival.sim.collect import SimEnv, run_episode  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402
from arrival.sim.scene import build_mjcf  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

needs_asset = pytest.mark.skipif(not menagerie.available(), reason="run python scripts/fetch_menagerie.py")
SINK_TASK = {
    "family": "counter_to_sink",
    "object_id": "mug_1",
    "source": "counter_a",
    "target": "sink_1",
    "target_x": 1.6,
    "start": [1.0, 0.35, 0.97],
    "lighting": "neutral",
    "people": 0,
    "seed": 0,
}


@pytest.fixture(scope="module")
def twin():
    return compile_twin(load_room("hero_kitchen", "scans"))


def test_procedural_arm_is_the_default_and_unknown_arms_are_rejected(twin):
    xml = build_mjcf(twin, (1.3, 1.0, 0.75))
    assert xml == build_mjcf(twin, (1.3, 1.0, 0.75), arm="procedural")
    assert "meshdir" not in xml and "panda_" not in xml
    with pytest.raises(ValueError):
        build_mjcf(twin, (1.3, 1.0, 0.75), arm="ur5")


def test_missing_asset_is_a_clear_error(monkeypatch, twin):
    monkeypatch.setattr(menagerie, "PANDA_DIR", menagerie.PANDA_DIR / "nowhere")
    with pytest.raises(FileNotFoundError, match="fetch_menagerie"):
        build_mjcf(twin, (1.3, 1.0, 0.75), arm="menagerie")


@needs_asset
def test_menagerie_arm_has_every_name_the_sim_relies_on(twin):
    m = mujoco.MjModel.from_xml_string(build_mjcf(twin, (1.3, 1.0, 0.75), n_people=1, arm="menagerie"))
    joints = {m.joint(i).name for i in range(m.njnt)}
    assert set(robot.ARM_JOINTS) | {"finger_l", "finger_r", "counter_a_drawer0", "mug_1"} <= joints
    bodies = {m.body(i).name for i in range(m.nbody)}
    assert {"robot_base", "link1", "link7", "hand", "finger_l", "finger_r"} <= bodies
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "eef") >= 0
    assert {m.camera(i).name for i in range(m.ncam)} >= {"ego_view", "wrist", "overview"}
    assert [m.actuator(i).name for i in range(m.nu)] == [f"a{i}" for i in range(1, 8)] + ["g_l", "g_r"]
    assert m.nmesh > 50
    for j, (lo, hi) in zip(robot.ARM_JOINTS, robot.RANGES):
        assert abs(m.joint(j).range[0] - lo) < 0.01 and abs(m.joint(j).range[1] - hi) < 0.01
    assert list(m.joint("finger_l").range) == [0, menagerie.FINGER_OPEN]
    assert m.actuator_ctrlrange[m.actuator("g_l").id][1] == menagerie.FINGER_OPEN


@needs_asset
def test_menagerie_collision_geoms_use_the_robot_classes(twin):
    m = mujoco.MjModel.from_xml_string(build_mjcf(twin, (1.3, 1.0, 0.75), arm="menagerie"))
    arm_bodies = {m.body(b).id for b in ("link1", "link4", "link7", "hand", "finger_l", "finger_r")}
    classes = {
        (int(m.geom_contype[g]), int(m.geom_conaffinity[g]), int(m.geom_group[g]))
        for g in range(m.ngeom)
        if m.geom_bodyid[g] in arm_bodies
    }
    assert classes == {(2, 1, 3), (0, 0, 2)}  # hidden collision meshes in the robot class, visual meshes only


@needs_asset
def test_menagerie_arm_picks_the_mug_and_opens_the_drawer(twin):
    env = SimEnv(twin, dict(SINK_TASK), res=32, arm="menagerie")
    ep = run_episode(env, ScriptedExpert(), dict(SINK_TASK))
    assert ep["success"], ep["detail"]
    s = np.array(ep["states"])
    assert (
        s.shape[1] == 11 and s[:, 7].min() < 0.036 < s[:, 7].max() <= menagerie.FINGER_OPEN + 5e-4
    )  # soft joint limit
    task = {
        "family": "open_drawer",
        "fixture": "counter_a",
        "handle": "counter_a_drawer0_handle",
        "joint": "counter_a_drawer0",
        "lighting": "neutral",
        "people": 0,
        "seed": 0,
    }
    env = SimEnv(twin, task, res=32, arm="menagerie")
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"] and ep["detail"]["qpos"] > 0.25, ep["detail"]


# ---- rebased onto the mobile-base and appliance chain (L25) ----


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / ".git").exists(), reason="needs a checkout with its .git"
)
def test_menagerie_asset_is_not_in_git_and_the_fetch_target_is_ignored():
    """The 34 MB of meshes must never be committed: the fetch destination is gitignored and no tracked file sits in it."""
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    assert "arrival/assets/menagerie/" in (repo / ".gitignore").read_text().splitlines()
    ignored = subprocess.run(["git", "check-ignore", "-q", str(menagerie.PANDA_DIR / "panda.xml")], cwd=repo)
    assert ignored.returncode == 0
    tracked = subprocess.run(["git", "ls-files", "arrival/assets"], cwd=repo, capture_output=True, text=True).stdout
    assert tracked.strip() == ""


def test_cli_arm_menagerie_without_the_asset_is_a_usage_error(tmp_path, capsys, monkeypatch):
    """Offline and in CI the asset is absent: the sim and eval commands say how to fetch it instead of a traceback."""
    import argparse

    from arrival import cli

    monkeypatch.setattr(menagerie, "PANDA_DIR", menagerie.PANDA_DIR / "nowhere")
    ns = argparse.Namespace(
        room="hero_kitchen",
        scans="scans",
        out=str(tmp_path),
        families="counter_to_sink",
        episodes=1,
        people=0,
        policy="scripted",
        res=32,
        seed=0,
        save_preview=False,
        planner="rule",
        split="train",
        instructions="template",
        tavily=False,
        holdout=False,
        arm="menagerie",
    )
    for cmd in (cli.cmd_sim, cli.cmd_eval):
        assert cmd(ns) == cli.EXIT_USAGE
        assert "fetch_menagerie" in capsys.readouterr().err
    parser = cli.build_parser()
    assert parser.parse_args(["sim", "--arm", "menagerie"]).arm == "menagerie"
    assert parser.parse_args(["eval", "--arm", "menagerie"]).arm == "menagerie"
    assert parser.parse_args(["sim"]).arm == "procedural"


def test_eval_hands_every_episode_the_requested_arm(twin, tmp_path, monkeypatch):
    """The eval loop kept a local called `arm` that shadowed the argument after the first episode."""
    from arrival.eval import harness

    seen = []

    class Boom:
        def __init__(self, twin, task, **kw):
            seen.append(kw.get("arm"))
            raise RuntimeError("no sim here")

    monkeypatch.setattr(harness, "SimEnv", Boom)
    rep = harness.evaluate(
        twin, ["open_drawer"], 3, res=32, people=0, out_dir=tmp_path, arm="menagerie", log=lambda *_: None
    )
    assert seen == ["menagerie"] * 3
    assert rep["arm"] == "menagerie"


def test_faucet_stand_spot_is_closer_for_the_real_hand(twin):
    """The real hand's closed fingertips reach 8 mm past the eef, the procedural fingers 23 mm."""
    from arrival.sim.stand import ARM_STANDOFF_TRIM, base_pose
    from arrival.sim.tasks import make_tasks

    t = make_tasks(twin, ["turn_on_faucet"], 1, seed=0, people=0, log=lambda *_: None)[0]
    proc, yaw_p, _ = base_pose(twin, t, None, False)
    real, yaw_m, _ = base_pose(twin, t, None, False, "menagerie")
    assert yaw_p == yaw_m and ARM_STANDOFF_TRIM["menagerie"] == 0.02
    assert abs(np.hypot(proc[0] - real[0], proc[1] - real[1]) - 0.02) < 1e-9
    assert base_pose(twin, t, None, False, "procedural")[0] == proc  # the default arm's spot is untouched


@needs_asset
def test_menagerie_mobile_base_has_the_planar_joints_first(twin):
    m = mujoco.MjModel.from_xml_string(build_mjcf(twin, (1.3, 1.0, 0.75), base="mobile", arm="menagerie"))
    assert [m.actuator(i).name for i in range(m.nu)] == (
        [f"a{i}" for i in range(1, 8)] + ["g_l", "g_r", "b_x", "b_y", "b_yaw"]
    )
    base = m.body("robot_base")
    assert [m.joint(j).name for j in range(base.jntadr[0], base.jntadr[0] + base.jntnum[0])] == list(robot.BASE_JOINTS)
    fixed = mujoco.MjModel.from_xml_string(build_mjcf(twin, (1.3, 1.0, 0.75), arm="menagerie"))
    assert "base_x" not in {fixed.joint(i).name for i in range(fixed.njnt)}


@needs_asset
@pytest.mark.parametrize(
    "family,joint,floor", [("open_oven", "stove_1_oven_door", 1.4), ("turn_on_faucet", "sink_1_faucet", 0.9)]
)
def test_menagerie_hand_runs_the_appliance_plans(twin, family, joint, floor):
    """Oven: the real pads need the bar 8 mm outside the eef (not 15) and a higher back-off to clear the falling door.
    Faucet: the real fingertips are 15 mm shorter, so the base stands 2 cm closer."""
    from arrival.sim.tasks import finish_task, make_tasks

    task = finish_task(make_tasks(twin, [family], 1, seed=0, people=0, log=lambda *_: None)[0])
    env = SimEnv(twin, dict(task), res=32, arm="menagerie")
    ep = run_episode(env, ScriptedExpert(), dict(task))
    assert ep["success"] and ep["detail"]["joint"] == joint and ep["detail"]["qpos"] > floor, ep["detail"]


@needs_asset
def test_menagerie_hand_closes_the_counter_b_door(twin):
    """The closed real fingertips reach 15 mm less than the procedural ones: with the procedural push standoff
    (0.03) the counter_b door stalled at 0.45 rad (seed 5); the Menagerie standoff shuts it."""
    from arrival.sim.tasks import finish_task, make_tasks

    tasks = [finish_task(t) for t in make_tasks(twin, ["close_door"], 4, seed=5, people=0, log=lambda *_: None)]
    task = next(t for t in tasks if "counter_b" in str(t))
    env = SimEnv(twin, dict(task), res=32, arm="menagerie")
    ep = run_episode(env, ScriptedExpert(), dict(task))
    assert ep["success"] and ep["detail"]["joint"] == "counter_b_door" and ep["detail"]["qpos"] < 0.05, ep["detail"]
    assert menagerie.HAND["door_push"] < robot.HAND["door_push"]


@needs_asset
def test_menagerie_dataset_says_which_arm_made_it(twin, tmp_path):
    from arrival.sim.collect import collect

    res = collect(twin, ["open_drawer"], 1, tmp_path, people=0, res=32, arm="menagerie", log=lambda *_: None)
    assert res["report_data"]["arm"] == "menagerie" and res["report_data"]["successes"] == 1
    emb = json.loads((res["dataset"] / "meta" / "embodiment.json").read_text())
    assert emb["arm"] == "menagerie" and emb["gripper_open"] == menagerie.FINGER_OPEN
    assert "Menagerie" in emb["note"] and f"0..{menagerie.FINGER_OPEN}" in emb["note"]
    state = np.array(
        pq.read_table(next(res["dataset"].rglob("episode_000000.parquet"))).column("observation.state").to_pylist()
    )
    assert state.shape[1] == 11 and state[:, 7].max() <= menagerie.FINGER_OPEN + 5e-4


@needs_asset
@pytest.mark.parametrize("seed", [0, 3])
def test_menagerie_arm_opens_the_oven(twin, seed):
    """At the steepest flap tilt IK reaches the oven bar, but the real hand's body meets the stove front 3.4 cm short
    of it and the door never moved (0 of 6 on main). The real hand leaves that tilt out for the oven."""
    from arrival.sim.tasks import finish_task, make_tasks

    task = finish_task(
        dict(
            next(
                t
                for t in make_tasks(twin, ["open_oven"], 2, seed=seed, log=lambda *a: None, arm="menagerie")
                if t["family"] == "open_oven"
            )
        )
    )
    ep = run_episode(SimEnv(twin, task, res=32, arm="menagerie", cameras=()), ScriptedExpert(), task)
    assert ep["success"] and ep["detail"]["qpos"] > 1.4, ep["detail"]


def test_only_the_real_hand_narrows_the_flap_tilts():
    from arrival.sim.policy import FLAP_TILTS

    assert "flap_tilts" not in robot.HAND  # the procedural arm keeps every tilt, so its plans are unchanged
    assert (
        set(menagerie.HAND["flap_tilts"]) == {"open_oven"}
        and max(FLAP_TILTS) not in menagerie.HAND["flap_tilts"]["open_oven"]
    )

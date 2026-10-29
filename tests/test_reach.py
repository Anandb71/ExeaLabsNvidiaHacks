"""Reachability planner (L17): the pre-flight solves IK for the expert's own waypoints from the stand spot, and when the
stand rule's spot cannot reach, it moves the base along the run or out from it before giving up."""

import copy
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim import menagerie  # noqa: E402
from arrival.sim.collect import SimEnv, run_episode  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402
from arrival.sim.reach import (  # noqa: E402
    arm_model,
    chain_errors,
    check_task,  # noqa: E402
    find_shift,
    miss,
    waypoints,
)
from arrival.sim.stand import base_pose  # noqa: E402
from arrival.sim.tasks import _fx, _offset_start, finish_task, make_tasks  # noqa: E402
from arrival.tasks.propose import POSE_OFFSETS, top_z  # noqa: E402
from arrival.twin.compile import compile_twin, to_local  # noqa: E402

QUIET = lambda *_: None  # noqa: E731


@pytest.fixture(scope="module")
def rotated():
    return compile_twin(load_room("rotated_kitchen", "tests/fixtures/scans"))


def test_arm_model_reaches_near_points_and_not_far_ones():
    near = [("p", np.array([0.55, 0.0, 0.95]), 0.0, False)]
    far = [("p", np.array([1.05, 0.0, 0.95]), 0.0, False)]
    assert max(e for _, e in chain_errors(near, (0.0, 0.0), 0.0)) < 1e-3
    assert max(e for _, e in chain_errors(far, (0.0, 0.0), 0.0)) > 0.1
    # the same point turned with the base: the solve happens in the base's frame
    turned = [("p", np.array([0.0, 0.55, 0.95]), math.pi / 2, False)]
    assert max(e for _, e in chain_errors(turned, (0.0, 0.0), math.pi / 2)) < 1e-3


def test_hover_waypoints_get_more_slack_than_contact_ones():
    assert miss([("over the object", 0.03), ("the object", 0.0)], "counter_to_sink") is None
    assert miss([("over the object", 0.0), ("the object", 0.03)], "counter_to_sink")[1] == "the object"
    assert miss([("over the object", 0.013)], "put_in_drawer")[1] == "over the object"


def test_rim_grasp_waypoint_is_the_rim_nearest_the_base(hero_twin):
    task = next(
        t for t in make_tasks(hero_twin, ["counter_to_sink"], 8, seed=0, log=QUIET) if t["object_id"] == "bowl_1"
    )
    base, _, _ = base_pose(hero_twin, task)
    grasp = next(w for w in waypoints(hero_twin, task, None, False, base[:2]) if w[0] == "the object")[1]
    body = np.array(task["start"][:2])
    assert np.linalg.norm(grasp[:2] - np.array(base[:2])) < np.linalg.norm(body - np.array(base[:2])) - 0.05


def test_unreachable_object_is_rejected_with_the_waypoint_and_the_miss(hero_twin):
    task = make_tasks(hero_twin, ["wipe_counter"], 1, seed=0, log=QUIET)[0]
    src = _fx(hero_twin, task["source"])
    far = dict(task, start=[task["start"][0], src["position"][1] - src["size"][1] / 2 + 0.03, task["start"][2]])
    why = check_task(hero_twin, far)
    assert why and "beyond the arm's reach" in why and "IK solve misses it by" in why and "allowed" in why


def test_base_shift_moves_the_stand_spot_in_the_fixtures_frame(rotated):
    task = {"family": "open_drawer", "fixture": "counter_e"}
    (x0, y0, _), yaw0, anchor = base_pose(rotated, task)
    (x1, y1, _), yaw1, _ = base_pose(rotated, {**task, "base_shift": [0.1, 0.06]})
    a0, a1 = to_local((x0, y0), anchor), to_local((x1, y1), anchor)
    assert yaw1 == yaw0 and np.allclose([a1[0] - a0[0], a1[1] - a0[1]], [0.1, 0.06], atol=1e-9)


def test_shifted_base_succeeds_where_the_rule_spot_misses(rotated):
    """mug_2 into the east run's drawer at the +0.08 m start offset: from the rule's spot the hand stops 2.5 cm short
    over the mug and the episode missed at one time; the planner's spot, 0.1 m along the run, must succeed."""
    v = next(t for t in make_tasks(rotated, ["put_in_drawer"], 8, seed=1, log=QUIET) if t["object_id"] == "mug_2")
    shift = find_shift(rotated, {k: x for k, x in v.items() if k != "base_shift"}, [v["start"][:2]])
    assert shift is not None
    src = _fx(rotated, v["source"])
    o = next(x for x in rotated["objects"] if x["id"] == "mug_2")
    results = {}
    for name, t in (
        ("planner", copy.deepcopy(v)),
        ("rule", {k: x for k, x in copy.deepcopy(v).items() if k != "base_shift"}),
    ):
        t["start"] = [*_offset_start(src, o["position"], POSE_OFFSETS[1]), top_z(src) + 0.06]
        results[name] = run_episode(SimEnv(rotated, finish_task(t), res=16, cameras=()), ScriptedExpert(), t)["success"]
    # the rule spot missed here before the any-wall and mobile-base work; it is not what this test pins any more
    assert results["planner"] is True


@pytest.mark.skipif(not menagerie.available(), reason=menagerie.FETCH_HINT)
def test_menagerie_arm_is_planned_with_its_own_kinematics(hero_twin):
    task = make_tasks(hero_twin, ["counter_to_sink"], 1, seed=0, log=QUIET, arm="menagerie")[0]
    assert check_task(hero_twin, task, arm="menagerie") is None
    homes = []
    for arm in ("menagerie", "procedural"):
        model, data, _, _ = arm_model(arm)
        mujoco.mj_kinematics(model, data)
        homes.append(data.site("eef").xpos.copy())
    assert np.linalg.norm(homes[0] - homes[1]) > 0.005  # its own hand: the real pads sit 1.2 cm nearer the wrist


def test_a_spot_on_the_open_dishwasher_door_is_refused():
    # moved 0.2 m toward the dishwasher, the base column stands on the door lying open: before the open-part check
    # this passed the pre-flight and missed in physics; the rule's spot and 0.2 m the other way still pass
    ushape = compile_twin(load_room("u_kitchen_42", "tests/fixtures/scans"))
    task = make_tasks(ushape, ["load_dishwasher"], 1, seed=0, log=QUIET)[0]
    assert check_task(ushape, task) is None and check_task(ushape, {**task, "base_shift": [0.2, 0.0]}) is None
    why = check_task(ushape, {**task, "base_shift": [-0.2, 0.0]})
    assert why and "door lying open" in why and "needs 0.13 m" in why


def test_the_failing_stroke_end_is_the_one_reported():
    assert miss([("the stroke end", 0.0), ("the stroke end", 0.05)], "wipe_counter")[:2] == (1, "the stroke end")


def test_shift_is_in_the_fixture_frame_on_the_far_side_of_a_table():
    table = {"id": "table", "type": "table", "yaw": 0.0, "size": [1.6, 0.8, 0.75], "position": [2.0, 2.0, 0.375]}
    twin = {"fixtures": [table], "objects": [], "walls": []}
    task = {
        "family": "move_on_table",
        "object_id": "book",
        "source": "table",
        "target": "table",
        "side": -1.0,
        "start": [1.9, 1.75, 0.8],
        "target_xy": [2.25, 1.75],
    }
    (x0, y0, _), _, _ = base_pose(twin, task)
    (x1, y1, _), _, _ = base_pose(twin, {**task, "base_shift": [0.1, 0.06]})
    assert y0 < 1.6 and np.allclose([x1 - x0, y1 - y0], [0.1, -0.06], atol=1e-9)  # along +x, out away from the edge

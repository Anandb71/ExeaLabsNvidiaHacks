"""A fridge on a side wall: the expert's push after the hooked pull follows the fixture's yaw. u_kitchen_42's fridge
stands at yaw +90 deg; the push used to be planned for a fridge at yaw 0, 1.5 m from the door, and failed 0 of 4."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim.collect import SimEnv, collect, run_episode  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402
from arrival.sim.tasks import finish_task, make_tasks  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

QUIET = lambda *a, **k: None  # noqa: E731


@pytest.fixture(scope="module")
def u_kitchen():
    twin = compile_twin(load_room("u_kitchen_42", "tests/fixtures/scans"))
    assert next(f for f in twin["fixtures"] if f["id"] == "fridge")["yaw"] == pytest.approx(np.pi / 2)
    return twin


def test_push_plan_follows_the_fridge_yaw(u_kitchen):
    task = make_tasks(u_kitchen, ["open_fridge"], 1, seed=0, people=0, log=QUIET)[0]
    finish_task(task)
    env = SimEnv(u_kitchen, task, res=16)
    expert = ScriptedExpert()
    ep = run_episode(env, expert, task)
    env.close()
    assert expert.ik_error < 0.02, expert.ik_error  # was 1.5 m
    assert ep["success"], ep["detail"]
    assert np.degrees(ep["detail"]["qpos"]) > 75.0


@pytest.mark.parametrize("people,seed", [(0, 5), (2, 3)])
def test_expert_opens_a_side_wall_fridge(u_kitchen, tmp_path, people, seed):
    r = collect(u_kitchen, ["open_fridge"], 2, tmp_path, people=people, res=32, seed=seed, log=QUIET)["report_data"]
    assert (r["attempts"], r["successes"], r["skipped_unreachable"]) == (2, 2, 0)
    assert r["human_contacts"] == 0 and r["door_contacts"] == 0

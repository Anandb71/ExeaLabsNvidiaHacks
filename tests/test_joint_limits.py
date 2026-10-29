"""The scripted expert's joint plan stays off the joint stops: no arm joint sits within the quality check's 3% band
for longer than its flag length. The first IK plan for open_oven on the hero kitchen parked joint 5 on its stop for
71 steps; the expert now re-plans (grasp yaw, wrist roll, jaw heading) when a plan does that. close_door is not
re-planned (see ScriptedExpert.reset) and is not asserted here."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.export.quality import LIMIT_STEPS  # noqa: E402
from arrival.sim.collect import SimEnv  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402
from arrival.sim.tasks import finish_task, make_tasks  # noqa: E402


def _planned_run(twin, family, seed):
    task = finish_task(make_tasks(twin, [family], 1, seed=seed, people=0, log=lambda *_: None)[0])
    env = SimEnv(twin, task, res=32)
    pol = ScriptedExpert()
    pol.reset(env, task)
    return env, pol, ScriptedExpert._limit_run(env, pol.plan)


def test_limit_run_counts_the_longest_stretch_in_the_band(hero_twin):
    env, pol, _ = _planned_run(hero_twin, "open_drawer", 0)
    lo, hi = env.model.jnt_range[[env.model.dof_jntid[i] for i in env.dofadr]].T
    plan = np.tile(np.concatenate([(lo + hi) / 2, np.zeros(4)]), (60, 1))
    assert ScriptedExpert._limit_run(env, plan) == 0
    plan[10:25, 4] = hi[4]  # joint 5 on its stop for 15 steps
    plan[30:34, 1] = lo[1] + 0.01 * (hi[1] - lo[1])  # a shorter stretch on another joint
    assert ScriptedExpert._limit_run(env, plan) == 15


@pytest.mark.parametrize("family", ["open_oven", "open_door", "open_fridge", "open_drawer"])
def test_expert_plan_stays_off_the_joint_stops(hero_twin, family):
    for seed in (0, 1):
        env, pol, run = _planned_run(hero_twin, family, seed)
        assert run <= LIMIT_STEPS, f"{family} seed {seed}: a joint sits on its stop for {run} planned steps"

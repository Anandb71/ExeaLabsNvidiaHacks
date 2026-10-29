"""Reachability planner (L17) and task pre-flight: can the base stand where the task needs it, and can the arm put
its hand where the scripted expert will ask from there? `check_task` answers both before any scene is built.

`chain_errors` solves `ik.solve` for the task's own waypoints (over the object, at the grasp, lifted, over the
target, at the target, the wipe strokes, the drawer tray, the dishwasher rack) on an arm-only model built once per
arm, with the target moved into the base's frame, so a check costs no scene build. The hand points down with the
yaw the expert uses, and the half-turned grasp is tried as well, as the expert does. The stand rules in
`stand.base_pose` stay the first choice; `find_shift` is a fallback grid used only when the rule's spot fails:
it tries spots moved along the fixture's run and out from its face, nearest first, and returns the first one that
passes `check_task` (walls, fixtures, open doors and drawers, every waypoint) at every start offset.
It is kinematics only: joint limits are respected, collisions between the arm and the room are not checked.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from arrival.catalog.categories import HANDLE_HELD, RIM_HELD
from arrival.sim import ik, menagerie, robot
from arrival.sim.geometry import drawer_x
from arrival.sim.robot import HANDLE_ROOT, PAD_BELOW_EEF
from arrival.sim.scene import BOWL_WALL, CARCASS_H, PLATE_FLOOR, PLATE_RIM, RACK_Z
from arrival.sim.stand import (
    DRAWER_OPEN_PUT,
    base_pose,
    drive_spot,
    mobile_drive,
    place_point,
    rack_spot,
    stand_problem,
)
from arrival.twin.compile import to_world

TOL_CONTACT = 0.025  # m: IK position error allowed where the hand grasps, sets down, presses or strokes
TOL_HOVER = 0.05  # m: allowed in free air, where the hand only hovers or carries lower than planned
TOL_HOVER_FAMILY = {"put_in_drawer": 0.01}  # its misses (1.3 and 4.0 cm) overlap its successes (3.1 to 4.0 cm)
# the three tolerances come from 117 physics runs over pushed-back starts on three kitchens (BUILD-LOG, L17)
SHIFTS = sorted(
    ((da, do) for do in (0.0, -0.06, 0.06, 0.12) for da in (0.0, 0.1, -0.1, 0.2, -0.2)), key=lambda d: math.hypot(*d)
)[1:]
# (along the fixture's local x, out from its face) in metres, nearest first; the rule's own spot (0, 0) is tried before these and left out
WIDE_SHIFTS = sorted(
    (
        (da, do)
        for do in (0.0, -0.1, 0.1, -0.2, 0.2, 0.3)
        for da in (round(k * 0.05, 2) for k in range(-20, 21))
        if abs(da) > 0.2 or abs(do) > 0.12
    ),
    key=lambda d: (math.hypot(*d), d),
)
# the second grid, in the same frame: used only when no SHIFTS entry passes, so a fixture whose rule spot is in a wall
# or on its neighbour (a corner cabinet beside the sink run, a fridge whose free edge is at the room's corner) can be
# stood at from further along its own run. The planner's checks are the same; the expert's run decides the rest.
IK_FAMILIES = (
    "counter_to_sink",
    "counter_to_counter",
    "move_on_table",
    "put_in_drawer",
    "wipe_counter",
    "load_dishwasher",
    "unload_dishwasher",
)  # the families whose hand targets come from the twin alone
WIDE_FAMILIES = IK_FAMILIES + (
    "open_door",
)  # close_door is left out: its two-phase push is not modelled by the waypoints here
# the families the second grid serves: the IK families (their reach is solved at each spot) and the cabinet door tasks,
# whose handle is reached from beside the door (the corner cabinet of u_kitchen_42, 8 of 8 runs from 0.2 m along). A drawer
# is pulled from straight in front of it (0 of 36 spots on the one drawer a sink blocks) and the fridge's hook-and-push
# plan needs the free-edge side (IK misses it by 1.4 m from the hinge side), so those keep their rejection.
STROKE = 0.15  # wipe_counter strokes run this far either side of the sponge, along the counter (policy.py)


@lru_cache(maxsize=2)
def arm_model(arm: str = "procedural"):
    """(model, data, qadr, dofadr) of the arm alone on its pedestal at the origin, facing +x."""
    import mujoco

    if arm == "menagerie":
        f = menagerie.arm_fragments((0.0, 0.0, 0.75), 0.0)
        xml = (
            f'<mujoco><compiler angle="radian" autolimits="true" meshdir="{f["meshdir"]}"/>'
            f"<default>{f['defaults']}</default><asset>{f['assets']}</asset><worldbody>{f['body']}</worldbody></mujoco>"
        )
    else:
        xml = (
            f'<mujoco><compiler angle="radian" autolimits="true"/>'
            f"<worldbody>{robot.arm_xml((0.0, 0.0, 0.75), 0.0)}</worldbody></mujoco>"
        )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    qadr = [int(model.joint(j).qposadr[0]) for j in robot.ARM_JOINTS]
    dofadr = [int(model.joint(j).dofadr[0]) for j in robot.ARM_JOINTS]
    data.qpos[qadr] = robot.HOME
    return model, data, qadr, dofadr


def _top(f: dict) -> float:
    return f["position"][2] + f["size"][2] / 2


def grasp(o: dict, body: np.ndarray, base_xy) -> tuple[np.ndarray, float | None, float]:
    """(eef xy offset from the body centre, yaw or None for the run's, eef height) as SimEnv.grasp holds `o` with the
    body centre at `body`: a bowl or plate by the rim nearest the base, a pan by its handle, anything else by the body."""
    cid, sz = o["catalog_id"], o["size"][2]
    radius = min(o["size"][0], o["size"][1]) / 2
    if cid in RIM_HELD:
        d = np.asarray(base_xy[:2], float) - body[:2]
        d = d / (np.linalg.norm(d) + 1e-9)
        wall = BOWL_WALL if cid == "bowl" else PLATE_RIM
        z = body[2] + sz / 2 - 0.02 if cid == "bowl" else body[2] - sz / 2 + PLATE_FLOOR + PAD_BELOW_EEF + 0.003
        return d * (radius - wall / 2), float(np.arctan2(d[0], -d[1])), z
    if cid in HANDLE_HELD:
        return np.array([radius + HANDLE_ROOT, 0.0]), 0.0, body[2] + sz / 2 - 0.008
    return np.zeros(2), None, body[2] + min(0.01, sz / 2 * 0.2)


def waypoints(
    twin: dict, task: dict, obj_xy=None, mobile: bool = False, base_xy=None
) -> list[tuple[str, np.ndarray, float, bool]]:
    """(name, eef position in the world, gripper yaw in the world, from the drive spot) for each waypoint the
    expert solves. Heights follow ScriptedExpert._waypoints; empty for the families IK_FAMILIES leaves out.
    `base_xy` is where the base stands (a rim grasp takes the rim nearest it)."""
    fam = task["family"]
    if fam not in IK_FAMILIES:
        return []
    fx = {f["id"]: f for f in twin["fixtures"]}
    if obj_xy is None:
        obj_xy = task["start"][:2]
    o = next(x for x in twin["objects"] if x["id"] == task["object_id"])
    hh = o["size"][2] / 2
    src = fx[task["source"]]
    run = src.get("yaw", 0.0)
    if fam == "unload_dishwasher":  # the object starts in the rolled-out rack; obj_xy is its spot on the counter
        body = np.array([*rack_spot(fx[task["fixture"]]), RACK_Z + 0.01 + hh + 0.003])
    else:
        body = np.array([obj_xy[0], obj_xy[1], _top(src) + hh + 0.003])
    gz = body[2] + min(0.01, hh * 0.2)
    p = np.array([body[0], body[1], gz])
    up = np.array([0.0, 0.0, 1.0])
    pick = [("over the object", p + 0.18 * up), ("the object", p)]
    if fam in ("counter_to_sink", "counter_to_counter", "move_on_table"):
        off, yaw, gz = grasp(o, body, body[:2] if base_xy is None else base_xy)
        yaw = run if yaw is None else yaw
        p = np.array([body[0] + off[0], body[1] + off[1], gz])
        tgt = place_point(twin, task) + np.array([off[0], off[1], gz - body[2]])
        drive = mobile_drive(task, mobile)
        out = [
            (n, q, yaw, False)
            for n, q in [("over the object", p + 0.18 * up), ("the object", p), ("the lifted object", p + 0.22 * up)]
        ]
        return out + [
            ("over the place point", np.array([tgt[0], tgt[1], gz + 0.22]), yaw, drive),
            ("the place point", tgt, yaw, drive),
        ]
    if fam == "put_in_drawer":
        c = fx[task["fixture"]]
        tray = to_world((drawer_x(c, twin["fixtures"]), c["size"][1] / 2 + DRAWER_OPEN_PUT / 2), c)
        t = np.array([tray[0], tray[1], CARCASS_H - 0.07 + hh + 0.01])
        lift = gz + 0.15
        return [
            (n, q, run + math.pi / 2, False)
            for n, q in pick
            + [
                ("the lifted object", np.array([p[0], p[1], lift])),
                ("over the drawer tray", np.array([t[0], t[1], lift])),
                ("the drawer tray", t),
            ]
        ]
    if fam == "wipe_counter":
        u = np.array([math.cos(run), math.sin(run), 0.0])
        press = np.array([p[0], p[1], _top(src) + hh - 0.004])
        return [
            (n, q, run, False)
            for n, q in [
                ("over the sponge", p + 0.15 * up),
                ("the sponge", p),
                ("the stroke end", press + STROKE * u),
                ("the stroke end", press - STROKE * u),
            ]
        ]
    # load_dishwasher / unload_dishwasher
    rack = np.array([*rack_spot(fx[task["fixture"]]), RACK_Z + 0.01 + hh + 0.01])
    counter = np.array([obj_xy[0], obj_xy[1], _top(src) + hh + 0.012])
    tgt = rack if fam == "load_dishwasher" else counter
    lift = max(gz, tgt[2]) + 0.15
    return [
        (n, q, run, False)
        for n, q in [
            ("over the object", np.array([p[0], p[1], lift])),
            ("the object", p),
            ("over the target", np.array([tgt[0], tgt[1], lift])),
            ("the rack" if fam == "load_dishwasher" else "the counter spot", tgt),
        ]
    ]


def is_hover(name: str) -> bool:
    """A waypoint in free air (over the object, the lift, over the target), not one where the hand touches."""
    return name.startswith(("over ", "the lifted"))


def chain_errors(wps, base, base_yaw: float, arm: str = "procedural") -> list[tuple[str, float]]:
    """(waypoint name, position error) solving the waypoints in order from HOME, as the expert does, for the better of
    the two half-turned grasps (the smaller worst error, as the expert keeps). `base` is the base (x, y), `base_yaw`
    in radians."""
    model, data, qadr, dofadr = arm_model(arm)
    c, s = math.cos(base_yaw), math.sin(base_yaw)
    best = None
    for flip in (0.0, math.pi):
        q, errs = np.array(robot.HOME), []
        for name, pos, yaw, _ in wps:
            dx, dy = pos[0] - base[0], pos[1] - base[1]
            local = np.array([c * dx + s * dy, -s * dx + c * dy, pos[2]])
            q, err = ik.solve(model, data, local, ik.down(yaw - base_yaw + flip), q, qadr, dofadr)
            errs.append((name, err))
        if best is None or max(e for _, e in errs) < max(e for _, e in best):
            best = errs
    return best


def miss(errs: list[tuple[str, float]], family: str) -> tuple[int, str, float, float] | None:
    """(index, waypoint, its error, the tolerance it breaks) for the worst waypoint over its tolerance, or None."""
    hover = TOL_HOVER_FAMILY.get(family, TOL_HOVER)
    over = [(e - (hover if is_hover(n) else TOL_CONTACT), k, n, e) for k, (n, e) in enumerate(errs)]
    worst = max(over, default=(0.0, -1, "", 0.0))
    return None if worst[0] <= 0 else (worst[1], worst[2], worst[3], hover if is_hover(worst[2]) else TOL_CONTACT)


def find_shift(
    twin: dict, v: dict, starts: list, mobile: bool = False, arm: str = "procedural"
) -> tuple[float, float] | None:
    """The nearest SHIFTS entry (along the run, out from it) from which the pool variation `v` (with its "family")
    passes `check_task` at every start in `starts`, or None; then the WIDE_SHIFTS the same way. Only for WIDE_FAMILIES:
    the other families' stand spots are fitted to the part they open."""
    if v["family"] not in WIDE_FAMILIES:
        return None
    for shift in (SHIFTS if v["family"] in IK_FAMILIES else []) + WIDE_SHIFTS:
        t = {**v, "base_shift": list(shift)}
        if all(check_task(twin, t, xy, mobile, arm) is None for xy in starts):
            return shift
    return None


def check_task(twin: dict, task: dict, obj_xy=None, mobile: bool = False, arm: str = "procedural") -> str | None:
    """None when the base can stand where the task needs it and reach what it needs; otherwise one sentence saying
    why not. `task` is a pool entry (fixture, or object_id, source and start) or a finished task. On a mobile base
    a counter_to_counter task is checked at both stand spots and along the straight drive between them. Reach is an
    IK solve of the expert's waypoints from the stand spot; families whose hand targets come
    from the built model (doors, drawers, the fridge, flap doors, the faucet) get the stand checks only."""
    try:
        base, yaw, anchor = base_pose(twin, task, obj_xy, mobile, arm)
    except (KeyError, StopIteration, IndexError, TypeError, ValueError) as e:  # a task naming a fixture the twin lacks
        return f"no stand spot: the task does not match the twin ({type(e).__name__}: {e})"
    b = np.array(base[:2])
    why = stand_problem(twin, b, anchor, task=task)
    if why:
        return why
    wps = waypoints(twin, task, obj_xy, mobile, b)
    spots = [(b, "the base", [w for w in wps if not w[3]])]
    if mobile_drive(task, mobile):
        end = np.array(drive_spot(twin, task)[:2])
        for k in range(1, 9):  # the drive is one straight line: its whole length must be clear, not just its ends
            why = stand_problem(twin, b + (end - b) * k / 8, anchor, "base drive path", task)
            if why:
                return why
        spots.append((end, "the drive spot", [w for w in wps if w[3]]))
    for xy, name, chain in spots:
        if not chain:
            continue
        m = miss(chain_errors(chain, xy, math.radians(yaw), arm), task["family"])
        if m:
            k, at, err, tol = m
            d = float(np.linalg.norm(chain[k][1][:2] - xy))
            return (
                f"{at} ({d:.2f} m out) is beyond the arm's reach from {name} at {anchor['id']}: "
                f"the IK solve misses it by {err * 100:.1f} cm (allowed {tol * 100:.1f} cm)"
            )
    return None

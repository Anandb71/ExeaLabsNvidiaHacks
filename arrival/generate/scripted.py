"""Scripted retarget generator. NOT physics and NOT MimicGen: seed cartesian trajectories
translated to new start poses, with the target point recomputed from the fixtures. A distance oracle (`judge`)
drops attempts whose object did not reach the target."""

from __future__ import annotations

import math

from arrival.tasks.propose import shuffled, top_z

FPS = 20
SEG = 16
N_FRAMES = 5 * SEG
HOVER = 0.18
GRASP = 0.02
LIFT = 0.20
RISE = 0.15
TOLERANCE = 0.05  # m: how close the object must end to the target for the oracle to keep the attempt


def is_miss(attempt_index: int) -> bool:
    """Attempt indexes 9, 19, 29, ... are scripted misses (gripper never closes)."""
    return attempt_index % 10 == 9


def task_slot(attempt_index: int) -> int:
    """Index into the family-interleaved order. Skipping one slot every 10 attempts shifts parity so the
    misses (9, 19, 29, ...) alternate between the two families instead of always hitting the same one."""
    return attempt_index + attempt_index // 10


def target_point(task: dict, twin: dict) -> list[float]:
    """Where the scripted trajectory sets the object down: just above the sink for counter_to_sink, resting on the
    target counter top otherwise."""
    fixtures = {f["id"]: f for f in twin["fixtures"]}
    target = fixtures[task["target_fixture_id"]]
    if task["family"] == "counter_to_sink":
        return [target["position"][0], target["position"][1], top_z(target) + 0.02]
    return [target["position"][0], target["position"][1], top_z(target) + task["object_size"][2] / 2]


def _lerp(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(len(a))]


def seed_trajectory(start: list[float], target: list[float], miss: bool = False) -> list[dict]:
    """80 frames, piecewise linear in cartesian space, 16 frames per segment.
    While grasped the object keeps a constant offset GRASP below the eef; on release it is lowered onto
    the target (no teleport), the gripper opens, then the eef rises."""
    sx, sy, sz = start
    hover, low = [sx, sy, sz + HOVER], [sx, sy, sz + GRASP]
    lifted_eef = [sx, sy, sz + LIFT]
    carried = [target[0], target[1], sz + LIFT]
    placed = [target[0], target[1], target[2] + GRASP]
    risen = [target[0], target[1], target[2] + GRASP + RISE]

    def held(eef):
        return [eef[0], eef[1], eef[2] - GRASP]

    frames = []
    for i in range(N_FRAMES):
        seg, k = divmod(i, SEG)
        t = k / (SEG - 1)
        if seg == 0:
            eef, grip, obj = _lerp(hover, low, t), 0.0, list(start)
        elif seg == 1:
            eef, grip, obj = list(low), t, list(start)
        elif seg == 2:
            eef, grip = _lerp(low, lifted_eef, t), 1.0
            obj = held(eef)
        elif seg == 3:
            eef, grip = _lerp(lifted_eef, carried, t), 1.0
            obj = held(eef)
        elif k < 8:  # lower onto the target while still grasped
            eef, grip = _lerp(carried, placed, k / 7), 1.0
            obj = held(eef)
        elif k < 11:  # open in place
            eef, grip, obj = list(placed), 1.0 - (k - 7) / 3, list(target)
        else:  # retreat
            eef, grip, obj = _lerp(placed, risen, (k - 10) / 5), 0.0, list(target)
        if miss:
            grip, obj = 0.0, list(start)
        frames.append(
            {
                "t": round(i / FPS, 6),
                "joint": [0.0] * 7,
                "gripper": round(grip, 6),
                "eef": [round(v, 6) for v in eef],
                "object_position": [round(v, 6) for v in obj],
            }
        )
    return frames


def retarget(
    seed_frames: list[dict], seed_start: list[float], task: dict, target: list[float], miss: bool = False
) -> list[dict]:
    """Translate the seed's approach (hover/grasp/lift) by start delta; recompute carry/release to target."""
    d = [task["start_position"][i] - seed_start[i] for i in range(3)]
    if not any(abs(v) > 1e-12 for v in d) and not miss:
        return [dict(f) for f in seed_frames]
    start = [seed_start[i] + d[i] for i in range(3)]
    fresh = seed_trajectory(start, target, miss=miss)
    # Approach segments are a pure translation of the seed: copy them translated to keep the seed's shape.
    if not miss:
        for i in range(3 * SEG):
            f = seed_frames[i]
            fresh[i]["eef"] = [round(f["eef"][j] + d[j], 6) for j in range(3)]
            fresh[i]["object_position"] = [round(f["object_position"][j] + d[j], 6) for j in range(3)]
    return fresh


def seed_key(task: dict) -> tuple:
    return (task["family"], task["object_id"])


def build_seeds(tasks: list[dict], twin: dict) -> dict:
    """Seeds: scanned pose (pose_index 0), neutral lighting, wording template 1."""
    seeds = {}
    # Prefer the scanned pose; if the counter inset rejected it, fall back to the lowest surviving pose.
    for t in sorted(tasks, key=lambda t: t["pose_index"]):
        if t["lighting"] == "neutral" and t["wording_index"] == 1 and seed_key(t) not in seeds:
            seeds[seed_key(t)] = (
                list(t["start_position"]),
                seed_trajectory(t["start_position"], target_point(t, twin)),
            )
    return seeds


def judge(attempt: dict) -> bool:
    """The oracle: True when the gripper closed and the object ended within TOLERANCE of the target (in 3D, so it
    must be lowered onto the target, not left hovering)."""
    frames, target = attempt["frames"], attempt["target"]
    closed = max(f["gripper"] for f in frames) > 0.5
    return math.dist(frames[-1]["object_position"], target) <= TOLERANCE and closed


def make_attempt(attempt_index: int, task: dict, twin: dict, seeds: dict) -> dict:
    """One scripted attempt: the task's seed trajectory retargeted to its start pose and target (a scripted miss at
    every 10th index), judged by the oracle."""
    miss = is_miss(attempt_index)
    target = target_point(task, twin)
    seed_start, seed_frames = seeds[seed_key(task)]
    frames = retarget(seed_frames, seed_start, task, target, miss=miss)
    attempt = {"attempt_index": attempt_index, "task": task, "fps": FPS, "frames": frames, "target": target}
    attempt["success"] = judge(attempt)
    return attempt


def generate(tasks: list[dict], twin: dict, episodes: int, seed: int = 0):
    """Yield attempts until `episodes` successes or `2 * episodes` attempts."""
    order = shuffled(tasks, seed)
    if not order:
        raise ValueError("no tasks to generate from")
    seeds = build_seeds(tasks, twin)
    successes = 0
    i = 0
    while successes < episodes and i < 2 * episodes:
        att = make_attempt(i, order[task_slot(i) % len(order)], twin, seeds)
        successes += att["success"]
        yield att
        i += 1


def count_attempts(episodes: int) -> dict:
    """Attempt/success/drop counts for N successes under the miss pattern, without generating frames."""
    successes = attempts = 0
    while successes < episodes and attempts < 2 * episodes:
        successes += not is_miss(attempts)
        attempts += 1
    return {"attempts": attempts, "successes": successes, "dropped": attempts - successes}

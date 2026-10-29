"""Human seeds for SE(3) retargeting: the P3 keyboard-teleop demos (issue #62) instead of the scripted expert.

    python -m arrival retarget --teleop docs/results/p3-teleop-2026-10-21 ...   (after extracting the P3 archive there)

A teleop run (arrival.sim.teleop, --record) keeps the joint targets the operator's keys produced, the attempt log and
session.json. Each successful demo is replayed here by its recorded actions in the twin, rebuilt with the run's own
seed, and becomes a source only if the physics oracle passes again with no human contact: the arm follows the recorded
targets exactly, but grasp contact is chaotic, so a demo recorded on another machine or MuJoCo build can drop its
object on replay (15 of the 20 P3 demos replay on MuJoCo 3.17). Demos whose length is a quality-check outlier
(over LENGTH_RATIO times the median) are skipped: retargeting copies a source's length into every episode.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from arrival.export.lerobot import list_column
from arrival.export.quality import LENGTH_RATIO
from arrival.generate.se3_retarget import record_source
from arrival.provenance import canonical_sha256
from arrival.sim.collect import SimEnv
from arrival.sim.policy import Policy
from arrival.sim.tasks import finish_task, make_tasks

GENERATOR = "keyboard_teleop_replay"
MOVED = (
    0.02  # m: the object anchor hands over to the sink anchor once the object has moved this far (a push or a grasp)
)


class TeleopReplay(Policy):
    """Plays back one recorded demo's actions; the base, which a fixed-arm dataset does not record, holds still."""

    def __init__(self, actions):
        self.plan = [np.asarray(a, dtype=float) for a in actions]

    def reset(self, env, task):
        self.env, self.t = env, 0

    def act(self, obs):
        a = self.plan[min(self.t, len(self.plan) - 1)].copy()
        self.t += 1
        a[7] = min(a[7], self.env.finger_open)  # float32 storage turns the 0.05 m open command into 0.0500000007
        return np.r_[a[:8], self.env.base_pose()] if len(a) == 8 else a

    def done(self):
        return self.t >= len(self.plan)


def _arg(args: list, flag: str, default):
    return type(default)(args[args.index(flag) + 1]) if flag in args else default


def load_demos(root, room: str) -> tuple[list[dict], dict]:
    """The successful attempts of a recorded teleop run under `root`, in dataset episode order, with their actions.
    Returns (demos, settings), settings being the seed, people and task count teleop.run drew its tasks with."""
    root = Path(root)
    log = [
        json.loads(line)
        for line in (root / "reports" / f"{room}_teleop_attempts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    data = root / "datasets" / f"{room}_teleop"
    episodes = [
        json.loads(line) for line in (data / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()
    ]
    wins = [r for r in log if r["success"]]
    if len(wins) != len(episodes):
        raise ValueError(
            f"{len(wins)} successful attempts in the log but {len(episodes)} recorded episodes; "
            "only a run recorded with --record has its demos"
        )
    session = json.loads((root / "session.json").read_text()) if (root / "session.json").exists() else {}
    args = [str(x) for x in session.get("arguments", [])]
    settings = {
        "seed": _arg(args, "--seed", 0),
        "people": _arg(args, "--people", 0),
        "n": max(_arg(args, "--successes", 20) * 3, 1),
    }  # teleop.run draws target_successes * 3 tasks
    demos = []
    for ep, row in zip(episodes, wins):
        if ep["tasks"][0] != row["instruction"]:
            raise ValueError(
                f"episode {ep['episode_index']} says {ep['tasks'][0]!r}, attempt {row['attempt']} "
                f"{row['instruction']!r}: the log and the dataset are from different runs"
            )
        chunk = ep["episode_index"] // 1000
        table = pq.read_table(data / "data" / f"chunk-{chunk:03d}" / f"episode_{ep['episode_index']:06d}.parquet")
        demos.append(
            {
                "episode_index": ep["episode_index"],
                "attempt": row["attempt"],
                "task_id": row["task_id"],
                "instruction": row["instruction"],
                "actions": list_column(table.column("action")),
            }
        )
    return demos, settings


def teleop_sources(twin: dict, root, log=print) -> tuple[list[dict], list[dict]]:
    """Retargeting sources from a teleop run's successful demos. Returns (sources, rows): one row per demo with
    its outcome (source, skip_length_outlier, skip_replay_fail, skip_never_moved), so nothing is dropped silently."""
    demos, settings = load_demos(root, twin["room_id"])
    tasks = {
        t["task_id"]: finish_task(t)
        for t in make_tasks(
            twin,
            ["counter_to_sink"],
            settings["n"],
            seed=settings["seed"],
            people=settings["people"],
            log=lambda *a: None,
        )
    }
    median = float(np.median([len(d["actions"]) for d in demos])) if demos else 0.0
    sources, rows = [], []
    for d in demos:
        row = {"episode_index": d["episode_index"], "attempt": d["attempt"], "steps": len(d["actions"])}
        task = tasks.get(d["task_id"])
        if task is None or task["instruction"] != d["instruction"]:
            raise ValueError(
                f"{d['task_id']} ({d['instruction']!r}) is not a task of {twin['room_id']} with seed "
                f"{settings['seed']}: the twin is not the one the demos were recorded in"
            )
        if len(d["actions"]) > median * LENGTH_RATIO:
            row["outcome"] = "skip_length_outlier"
        else:
            try:
                with SimEnv(twin, task, res=32, cameras=()) as env:
                    source = record_source(env, task, TeleopReplay(d["actions"]), generator=GENERATOR, moved=MOVED)
            except ValueError as error:
                row["outcome"] = "skip_never_moved" if "never moved" in str(error) else "skip_replay_fail"
            else:
                del source["source_id"]  # re-hashed with the demo it came from
                source["teleop"] = {"episode_index": d["episode_index"], "attempt": d["attempt"]}
                source["source_id"] = canonical_sha256(source)
                sources.append(source)
                row.update(outcome="source", source_id=source["source_id"], object=task["object_id"])
        rows.append(row)
        log(f"[teleop_seeds] episode {d['episode_index']} (attempt {d['attempt']}): {row['outcome']}")
    return sources, rows

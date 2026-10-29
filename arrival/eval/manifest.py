"""Frozen inputs for replaying the same evaluation trials across policies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from arrival.sim.robot import FPS

EVAL_SEED_BASE = 10_000
SCHEMA_VERSION = 1
MAX_STEPS = 400


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def make_manifest(
    twin: dict,
    families: list[str],
    episodes: int,
    seed: int = 0,
    people: int = 2,
    holdout: bool = False,
    res: int = 128,
    log=print,
    base: str = "fixed",
    arm: str = "procedural",
) -> dict:
    """Freeze the inputs of an evaluation: the twin, the settings and every trial's finished task, each family drawn
    from its own held-out seed. Families with no reachable task are listed as omitted. The result carries
    manifest_id, the hash of everything else, so a replay can prove it ran the same trials."""
    from arrival.sim.tasks import finish_task, make_tasks

    if (
        any(type(value) is not int or value < minimum for value, minimum in ((episodes, 1), (people, 0), (res, 1)))
        or type(seed) is not int
        or type(holdout) is not bool
    ):
        raise ValueError("evaluation needs integer episodes/resolution/people/seed and a boolean holdout")
    if (
        not isinstance(families, list)
        or not families
        or any(not isinstance(family, str) or not family.strip() for family in families)
        or len(set(families)) != len(families)
    ):
        raise ValueError("evaluation needs distinct, nonempty family names")
    if base not in ("fixed", "mobile") or arm not in ("procedural", "menagerie"):
        raise ValueError("evaluation needs a supported base and arm")
    tasks, omitted = [], []
    for i, family in enumerate(families):
        batch = make_tasks(
            twin,
            [family],
            episodes,
            seed=seed + EVAL_SEED_BASE + i,
            people=people,
            holdout=holdout,
            mobile=base == "mobile",
            log=log,
            arm=arm,
        )
        tasks.extend(finish_task(t) for t in batch)
        if not batch:
            omitted.append(family)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "twin": twin,
        "twin_sha256": digest(twin),
        "settings": {
            "families": families,
            "episodes_per_family": episodes,
            "seed": seed,
            "people": people,
            "holdout": holdout,
            "resolution": res,
            "base": base,
            "arm": arm,
            "fps": FPS,
            "max_steps": MAX_STEPS,
        },
        "tasks": tasks,
        "omitted_families": omitted,
    }
    manifest["manifest_id"] = digest(manifest)
    return validate_manifest(manifest)


def validate_manifest(manifest: dict) -> dict:
    """Check that a manifest is the current schema, that its hashes match its contents and that its settings and tasks
    are well formed; returns it unchanged. Raises ValueError on the first problem."""
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("unsupported evaluation manifest schema")
    payload = {k: v for k, v in manifest.items() if k != "manifest_id"}
    if manifest.get("manifest_id") != digest(payload):
        raise ValueError("evaluation manifest hash does not match its contents")
    try:
        twin, cfg, tasks = manifest["twin"], manifest["settings"], manifest["tasks"]
        if not isinstance(twin, dict) or not isinstance(twin["room_id"], str) or not twin["room_id"].strip():
            raise ValueError("evaluation manifest needs a twin with a nonempty room_id")
        if digest(twin) != manifest["twin_sha256"]:
            raise ValueError("evaluation twin hash does not match")
        if (
            cfg["base"] not in ("fixed", "mobile")
            or cfg.get("arm", "procedural") not in ("procedural", "menagerie")
            or cfg["fps"] != FPS
            or cfg["max_steps"] != MAX_STEPS
        ):
            raise ValueError("evaluation manifest uses unsupported base, arm, FPS or rollout length")
        for key, minimum in (("episodes_per_family", 1), ("people", 0), ("resolution", 1)):
            if type(cfg[key]) is not int or cfg[key] < minimum:
                raise ValueError(f"invalid evaluation setting {key}")
        if type(cfg["seed"]) is not int or type(cfg["holdout"]) is not bool:
            raise ValueError("invalid evaluation seed or holdout")
        families = cfg["families"]
        if not isinstance(families, list) or not families or not all(isinstance(f, str) for f in families):
            raise ValueError("evaluation manifest needs named families")
        if len(set(families)) != len(families) or not isinstance(tasks, list):
            raise ValueError("evaluation families must be distinct and tasks must be a list")
        seen = set()
        for task in tasks:
            tid = task["task_id"]
            if not isinstance(tid, str) or not tid or tid in seen:
                raise ValueError("evaluation task IDs must be nonempty and unique")
            seen.add(tid)
            if (
                task["family"] not in families
                or not isinstance(task["instruction"], str)
                or not task["instruction"].strip()
            ):
                raise ValueError(f"invalid evaluation task {tid}")
            if type(task["seed"]) is not int or type(task["people"]) is not int or type(task["holdout"]) is not bool:
                raise ValueError(f"invalid evaluation task {tid}: seed, people or holdout type")
            if task["people"] != cfg["people"] or task["holdout"] != cfg["holdout"]:
                raise ValueError(f"evaluation task {tid} disagrees with the manifest settings")
        omitted = [f for f in families if not any(t["family"] == f for t in tasks)]
        if omitted != manifest["omitted_families"]:
            raise ValueError("evaluation manifest has inconsistent omitted families")
        for family in families:
            count = sum(t["family"] == family for t in tasks)
            if count and count != cfg["episodes_per_family"]:
                raise ValueError(f"evaluation family {family} has an inconsistent trial count")
    except (KeyError, TypeError) as e:
        raise ValueError(f"invalid evaluation manifest structure: {e}") from e
    return json.loads(canonical(manifest))


def load_manifest(path: str | Path) -> dict:
    return validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))

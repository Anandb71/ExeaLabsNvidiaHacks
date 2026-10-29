"""Task proposer: two families x objects x start poses x lighting x wordings. No language model."""

from __future__ import annotations

import json
import random
from pathlib import Path

from arrival.twin.compile import is_counter_host, to_local

LIGHTING = ("dim", "neutral", "bright")
POSE_OFFSETS = ([0.0, 0.0], [0.08, 0.0], [-0.08, 0.0], [0.0, 0.08])
INSET = 0.05

WORDINGS = {
    "counter_to_sink": [
        "Pick up the {name} from the counter and place it in the sink.",
        "Put the {name} in the sink.",
        "Move the {name} off the counter into the sink.",
        "Take the {name} and drop it in the sink.",
    ],
    "counter_to_counter": [
        "Pick up the {name} from the counter and place it onto the other counter.",
        "Put the {name} onto the other counter.",
        "Move the {name} off the counter onto the other counter.",
        "Take the {name} and drop it onto the other counter.",
    ],
}
MIMICGEN_TEMPLATE = {"counter_to_sink": "PnPCounterToSink", "counter_to_counter": None}
FAMILIES = ("counter_to_sink", "counter_to_counter")


def top_z(fixture: dict) -> float:
    return fixture["position"][2] + fixture["size"][2] / 2


def in_top_rect(xy, fixture: dict, inset: float = 0.0) -> bool:
    """Is (x, y) inside the fixture's top rectangle (in its yawed local frame), shrunk by `inset`?"""
    lx, ly = to_local(xy, fixture)
    sx, sy, _ = fixture["size"]
    return abs(lx) <= sx / 2 - inset + 1e-9 and abs(ly) <= sy / 2 - inset + 1e-9


def on_counter(obj: dict, counter: dict) -> bool:
    z = obj["position"][2]
    return in_top_rect(obj["position"], counter) and 0.0 <= z - top_z(counter) <= 0.25


def _fixture(twin: dict, fid: str) -> dict | None:
    return next((f for f in twin["fixtures"] if f["id"] == fid), None)


def propose(twin: dict) -> dict:
    """The placeholder pipeline's tasks for a twin: counter_to_sink and counter_to_counter for every object on the
    source counter, crossed with start-pose offsets, lighting and wordings. No language model."""
    counters = [f for f in twin["fixtures"] if is_counter_host(f)]
    counter_a = (
        _fixture(twin, "counter_a")
        or next((c for c in counters if any(on_counter(o, c) for o in twin["objects"])), None)
        or next(iter(counters), None)
    )
    counter_b = _fixture(twin, "counter_b") or next(
        (f for f in twin["fixtures"] if is_counter_host(f) and f is not counter_a), None
    )
    sink = next((f for f in twin["fixtures"] if f["type"] == "sink"), None)
    targets = {"counter_to_sink": sink, "counter_to_counter": counter_b}
    tasks = []
    if counter_a is not None:
        sources = [o for o in twin["objects"] if on_counter(o, counter_a)]
        for family in FAMILIES:
            target = targets[family]
            if target is None:
                continue
            for obj in sources:
                name = obj.get("label", obj["catalog_id"])
                for pi, off in enumerate(POSE_OFFSETS):
                    x, y, z = obj["position"]
                    start = [round(x + off[0], 6), round(y + off[1], 6), z]
                    if not in_top_rect(start, counter_a, INSET):
                        continue
                    for light in LIGHTING:
                        for wi, wording in enumerate(WORDINGS[family]):
                            tasks.append(
                                {
                                    "task_id": f"{family}__{obj['id']}__p{pi}__{light}__w{wi + 1}",
                                    "family": family,
                                    "object_id": obj["id"],
                                    "object_label": name,
                                    "object_size": obj["size"],
                                    "source_fixture_id": counter_a["id"],
                                    "target_fixture_id": target["id"],
                                    "start_position": start,
                                    "pose_index": pi,
                                    "lighting": light,
                                    "wording_index": wi + 1,
                                    "instruction": wording.format(name=name),
                                    "mimicgen_template": MIMICGEN_TEMPLATE[family],
                                }
                            )
    return {"room_id": twin["room_id"], "tasks": tasks}


def shuffled(tasks: list[dict], seed: int = 0) -> list[dict]:
    """Shuffle with random.Random(seed), then interleave families (keeping shuffled order within each)
    so any prefix of >= 2 attempts covers both families. See docs/BUILD-LOG.md."""
    order = list(tasks)
    random.Random(seed).shuffle(order)
    by_family: dict[str, list[dict]] = {}
    for t in order:
        by_family.setdefault(t["family"], []).append(t)
    queues = [by_family[f] for f in FAMILIES if f in by_family]
    out = []
    for i in range(max((len(q) for q in queues), default=0)):
        out += [q[i] for q in queues if i < len(q)]
    return out


def write_tasks(tasks: dict, twin_dir: str | Path) -> Path:
    path = Path(twin_dir) / "tasks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, indent=2))
    return path

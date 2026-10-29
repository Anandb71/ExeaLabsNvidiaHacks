"""merge-shards refuses shards simulated from different scans (the scan was edited between the shard runs) instead of
joining them under shard 0's provenance. Synthetic shard datasets, no MuJoCo."""

import copy
import json

import numpy as np
import pytest
from conftest import SCANS

from arrival import cli
from arrival.export.validate import validate
from arrival.jobs.shards import ShardError, merge_shards, shard_range
from arrival.provenance import run_record
from arrival.scan.bundle import load_room
from arrival.sim.export import SimDatasetWriter
from arrival.twin.compile import compile_twin

CAMS = ("ego_view", "wrist")


def _shard(out, idx, count, total, bundle):
    """A finished `sim --shard idx/count` dataset of one task per shard, with the provenance a twin of `bundle` gives."""
    lo, hi = shard_range(total, idx, count)
    root = out / "shards" / f"{idx}of{count}"
    prov = run_record(compile_twin(bundle), seed=0, generator="mujoco_scripted_ik", policy="scripted")
    w = SimDatasetWriter(root / "datasets" / "hero_kitchen_sim", "hero_kitchen", 20, 16, CAMS, provenance=prov)
    rng = np.random.default_rng(idx)
    for k in range(lo, hi):
        n = 5 + k
        w.add_episode(
            {
                "states": rng.normal(size=(n, 11)),
                "actions": rng.normal(size=(n, 11)),
                "videos": {c: rng.integers(0, 255, (n, 16, 16, 3), dtype=np.uint8) for c in CAMS},
                "detail": {"joint": "counter_b_drawer0", "qpos": 0.3},
                "safety": {"min_human_distance_m": None, "human_contact": 0},
                "people_xy": [[] for _ in range(n)],
            },
            {
                "task_id": f"sim_open_drawer_{k:04d}",
                "family": "open_drawer",
                "lighting": "neutral",
                "people": 0,
                "instruction": "Open the top drawer.",
            },
        )
    meta = {
        "index": idx,
        "count": count,
        "task_range": [lo, hi],
        "total_tasks": total,
        "episodes": total,
        "seed": 0,
        "families": ["open_drawer"],
        "people": 0,
        "policy": "scripted",
        "planner": "rule",
        "split": "train",
        "instructions": False,
        "base": "fixed",
        "arm": "procedural",
        "res": 16,
    }
    w.finish(hi - lo, None, {"shard": meta})
    return root


def test_merge_refuses_shards_simulated_from_different_scans(tmp_path, capsys):
    scan = load_room("hero_kitchen", SCANS)
    edited = copy.deepcopy(scan)  # the scan was edited after shard 0 ran: mug_1 25 cm further along counter_a
    next(o for o in edited["objects"] if o["id"] == "mug_1")["position"][0] += 0.25
    _shard(tmp_path / "run", 0, 2, 2, scan)
    _shard(tmp_path / "run", 1, 2, 2, edited)
    out = tmp_path / "merged"
    with pytest.raises(ShardError, match="shard 1 was simulated from another scan than shard 0"):
        merge_shards([tmp_path / "run" / "shards"], out)
    assert not out.exists()
    assert cli.main(["merge-shards", str(tmp_path / "run" / "shards"), "--out", str(out)]) == cli.EXIT_USAGE
    assert "another scan" in capsys.readouterr().err


def test_merge_takes_shards_of_one_scan_saved_twice_or_compiled_elsewhere(tmp_path):
    scan = load_room("hero_kitchen", SCANS)
    resaved = copy.deepcopy(scan)  # the same content written again: another file hash, the same scan
    resaved["input_sha256"] = "0" * 64
    _shard(tmp_path / "run", 0, 2, 2, scan)
    one = _shard(tmp_path / "run", 1, 2, 2, resaved)
    info_p = one / "datasets" / "hero_kitchen_sim" / "meta" / "info.json"
    info = json.loads(info_p.read_text())
    info["arrival_provenance"]["twin_code"] = info["arrival_provenance"]["run_code"] = {
        "sha": "f" * 40,
        "dirty": True,
        "source": "git",
    }
    info_p.write_text(json.dumps(info))
    res = merge_shards([tmp_path / "run" / "shards"], tmp_path / "merged")
    assert validate(res["dataset"]) == [] and res["info"]["total_episodes"] == 2
    assert (
        res["info"]["arrival_provenance"]["input"]["bundle_sha256"]
        == run_record(compile_twin(scan))["input"]["bundle_sha256"]
    )

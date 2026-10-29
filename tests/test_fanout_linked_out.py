"""`arrival fanout --backend local --out <dir>` where the out folder is a symbolic link (an `out` that points at a data
disk, or a path under /tmp on macOS, which is /private/tmp). LocalBackend kept the store path as given but resolves each
object's path, so listing a shard's objects in the collect step raised ValueError ("... is not in the subpath of ...")
after every shard had run, and again on every rerun, since the done markers skip the shards straight to the collect."""

import sys

import pytest

from arrival.jobs import fanout
from arrival.jobs.fanout import FanoutConfig, LocalBackend


def _linked_out(tmp_path):
    (tmp_path / "disk").mkdir()
    try:
        (tmp_path / "out").symlink_to(tmp_path / "disk", target_is_directory=True)
    except OSError as e:  # Windows without Developer Mode or admin rights: WinError 1314, no symlink to test
        pytest.skip(f"this machine cannot create symlinks ({e.__class__.__name__}: {e.strerror or e})")
    return tmp_path / "out"


def test_objects_are_listed_under_a_linked_out(tmp_path):
    out = _linked_out(tmp_path)
    be = LocalBackend(out / "fanout_store", out / "fanout_scratch")
    be.put_object("r1/shards/0of1/meta/info.json", b"{}")
    be.put_object("r1/shards/0of1/data/episode_000000.parquet", b"x")
    assert be.list_objects("r1/shards/0of1") == [
        "r1/shards/0of1/data/episode_000000.parquet",
        "r1/shards/0of1/meta/info.json",
    ]
    assert (tmp_path / "disk" / "fanout_store" / "r1" / "shards" / "0of1" / "meta" / "info.json").is_file()


def _shard(cfg, i, shard_out):
    """Stands in for `arrival sim --shard i/n` (no MuJoCo): writes the two files a done marker needs."""
    meta = shard_out / "shards" / f"{i}of{cfg.shards}" / "datasets" / f"{cfg.room}_sim" / "meta"
    return [
        sys.executable,
        "-c",
        f"import pathlib; m = pathlib.Path({str(meta)!r}); m.mkdir(parents=True); "
        f"(m / 'info.json').write_text('{{}}'); (m / 'episodes.jsonl').write_text('')",
    ]


def test_fanout_collects_every_shard_under_a_linked_out(tmp_path, monkeypatch):
    out = _linked_out(tmp_path)
    merged = []  # merge_shards itself is covered by test_shards.py; here it only records what was collected
    monkeypatch.setattr(fanout, "merge_shards", lambda sources, out_dir, log: merged.append(sources) or {})
    cfg = FanoutConfig("r1", 2, episodes=2, res=32)
    for rerun in (False, True):  # the rerun skips both shards (done markers) and collects again
        be = LocalBackend(out / "fanout_store", out / "fanout_scratch", command=_shard)
        res = fanout.run_fanout(be, cfg, out, poll_seconds=0.01, log=lambda msg: None)
        assert (res["ran"], res["skipped"]) == (([], [0, 1]) if rerun else ([0, 1], []))
        got = sorted(p.relative_to(out / "collected").as_posix() for p in (out / "collected").rglob("*") if p.is_file())
        assert got == [
            f"shards/{i}of2/datasets/hero_kitchen_sim/meta/{name}"
            for i in (0, 1)
            for name in ("episodes.jsonl", "info.json")
        ]
    assert merged == [[out / "collected" / "shards"]] * 2

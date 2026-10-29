"""Paired robot and external views must describe the same simulated observations."""

import json

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import pytest

pytest.importorskip("mujoco")

from arrival import cli  # noqa: E402
from arrival.export.validate import validate  # noqa: E402
from arrival.jobs.shards import Journal, ShardError, merge_shards  # noqa: E402
from arrival.sim.collect import SimEnv, collect, run_episode  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402

CAMS = ("ego_view", "wrist", "overview")


def test_cli_exocentric_is_opt_in():
    parser = cli.build_parser()
    assert not parser.parse_args(["sim"]).exocentric
    assert parser.parse_args(["sim", "--exocentric"]).exocentric


def test_observations_render_all_views_before_stepping(hero_twin):
    from arrival.sim.tasks import make_tasks

    task = make_tasks(hero_twin, ["open_drawer"], 1, people=0)[0]
    from arrival.sim.tasks import finish_task

    finish_task(task)
    with SimEnv(hero_twin, task, res=32, cameras=CAMS) as env:
        before = env.data.qpos.copy()
        time = env.data.time
        obs = env.obs()
        assert set(obs) == {"state", *CAMS}
        np.testing.assert_array_equal(env.data.qpos, before)
        assert env.data.time == time
        for cam in CAMS:
            np.testing.assert_array_equal(obs[cam], env.render(cam))
        assert not np.array_equal(obs["ego_view"], obs["overview"])


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    from arrival.scan.bundle import load_room
    from arrival.twin.compile import compile_twin

    twin = compile_twin(load_room("hero_kitchen", "scans"))
    root = tmp_path_factory.mktemp("paired")
    result = collect(
        twin,
        ["open_drawer"],
        1,
        root,
        people=0,
        res=64,
        seed=3,
        exocentric=True,
        save_preview=True,
        log=lambda *_: None,
    )
    assert result["report_data"]["successes"] == 1
    return twin, root, result


def test_paired_export_is_synchronized_and_has_preview(paired, tmp_path):
    twin, root, result = paired
    ds = result["dataset"]
    assert validate(ds) == []
    row = json.loads((ds / "meta/episodes.jsonl").read_text())
    info = result["info"]
    assert info["total_videos"] == 3
    assert info["arrival_camera_capture"]["phase"] == "pre_action"
    assert info["arrival_camera_capture"]["roles"]["overview"] == "exocentric"
    clips = {}
    for cam in CAMS:
        with imageio.get_reader(ds / "videos/chunk-000" / f"observation.images.{cam}/episode_000000.mp4") as reader:
            clips[cam] = np.stack([frame for frame in reader])
        assert clips[cam].shape == (row["length"], 64, 64, 3)
        assert clips[cam].std() > 1
    assert not np.array_equal(clips["ego_view"], clips["overview"])
    # Capturing another camera must leave both the control sequence and original videos untouched.
    original = collect(twin, ["open_drawer"], 1, tmp_path, people=0, res=64, seed=3, log=lambda *_: None)
    a = pq.read_table(ds / "data/chunk-000/episode_000000.parquet")
    b = pq.read_table(original["dataset"] / "data/chunk-000/episode_000000.parquet")
    assert a.equals(b)
    for cam in ("ego_view", "wrist"):
        path = f"videos/chunk-000/observation.images.{cam}/episode_000000.mp4"
        assert (ds / path).read_bytes() == (original["dataset"] / path).read_bytes()
    preview = root / "previews/hero_kitchen_sim_ego_exo.mp4"
    with imageio.get_reader(preview) as reader:
        frames = [frame for frame in reader]
    assert len(frames) == row["length"] and frames[0].shape == (96, 128, 3)
    assert (root / "previews/hero_kitchen_sim_ego_exo.png").is_file()


def test_resume_cannot_mix_camera_sets(paired):
    twin, root, _ = paired
    with pytest.raises(ShardError, match="cameras"):
        collect(twin, ["open_drawer"], 1, root, people=0, res=64, seed=3, resume=True, log=lambda *_: None)
    resumed = collect(
        twin, ["open_drawer"], 1, root, people=0, res=64, seed=3, exocentric=True, resume=True, log=lambda *_: None
    )
    assert resumed["report_data"]["resumed_tasks"] == 1 and validate(resumed["dataset"]) == []


def test_legacy_journal_defaults_to_two_cameras(tmp_path):
    from arrival.jobs.shards import FINGERPRINT_KEYS

    fp = {key: None for key in FINGERPRINT_KEYS if key != "cameras"}
    p = tmp_path / "journal.jsonl"
    p.write_text(json.dumps({"fingerprint": fp}) + "\n")
    assert Journal(p).start(fp, resume=True) == []
    with pytest.raises(ShardError, match="cameras"):
        Journal(p).start({**fp, "cameras": list(CAMS)}, resume=True)


def test_writer_rejects_desynchronized_video_before_writing(tmp_path):
    from arrival.sim.export import SimDatasetWriter

    writer = SimDatasetWriter(tmp_path / "ds", "room", 20, 32, CAMS)
    ep = {
        "states": np.zeros((3, 11)),
        "actions": np.zeros((3, 11)),
        "videos": {c: np.zeros((3 if c != "overview" else 2, 32, 32, 3), dtype=np.uint8) for c in CAMS},
    }
    with pytest.raises(ValueError, match="overview"):
        writer.add_episode(ep, {"instruction": "open", "family": "open_drawer"})
    assert not list(writer.root.glob("data/**/*.parquet")) and not writer.episodes and not writer.stats


def test_run_episode_preserves_same_number_of_frames_per_camera(hero_twin):
    from arrival.sim.tasks import make_tasks

    task = make_tasks(hero_twin, ["open_drawer"], 1, people=0)[0]
    from arrival.sim.tasks import finish_task

    finish_task(task)
    ep = run_episode(SimEnv(hero_twin, task, res=32, cameras=CAMS), ScriptedExpert(), task)
    assert {v.shape[0] for v in ep["videos"].values()} == {len(ep["states"])}
    assert len(ep["actions"]) == len(ep["states"])


def test_merge_keeps_exocentric_video_and_metadata(hero_twin, tmp_path):
    for i in range(2):
        result = collect(
            hero_twin,
            ["open_drawer"],
            2,
            tmp_path,
            people=0,
            res=32,
            seed=3,
            exocentric=True,
            shard=(i, 2),
            log=lambda *_: None,
        )
        assert result["report_data"]["successes"] == 1
    merged = merge_shards([tmp_path / "shards"], tmp_path / "merged", log=lambda *_: None)
    assert validate(merged["dataset"]) == [] and merged["info"]["total_videos"] == 6
    assert merged["info"]["arrival_camera_capture"]["roles"]["overview"] == "exocentric"


@pytest.mark.parametrize("fault", ["missing", "size", "dtype"])
def test_writer_rejects_bad_camera_arrays_before_writing(tmp_path, fault):
    from arrival.sim.export import SimDatasetWriter

    writer = SimDatasetWriter(tmp_path / "ds", "room", 20, 32, CAMS)
    videos = {cam: np.zeros((3, 32, 32, 3), dtype=np.uint8) for cam in CAMS}
    if fault == "missing":
        del videos["overview"]
    elif fault == "size":
        videos["overview"] = np.zeros((3, 16, 32, 3), dtype=np.uint8)
    else:
        videos["overview"] = videos["overview"].astype(float)
    with pytest.raises(ValueError, match="overview"):
        writer.add_episode(
            {"states": np.zeros((3, 11)), "actions": np.zeros((3, 11)), "videos": videos},
            {"instruction": "open", "family": "open_drawer"},
        )
    assert not list(writer.root.glob("data/**/*.parquet")) and not writer.stats and not writer.episodes


def test_paired_preview_png_preserves_matching_frame_pixels(tmp_path):
    from PIL import Image

    from arrival.render.multiview import write_paired_preview

    ego = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    exo = ego.copy()
    ego[1, :, :, 0] = 160
    exo[1, :, :, 2] = 210
    write_paired_preview(tmp_path / "room.v1_pair", ego, exo, 20)
    with Image.open(tmp_path / "room.v1_pair.png") as image:
        pair = np.asarray(image)
    np.testing.assert_array_equal(pair[32:, :32], ego[1])
    np.testing.assert_array_equal(pair[32:, 32:], exo[1])
    np.testing.assert_array_equal(ego[1, 0, 0], [160, 0, 0])


def test_overview_preview_keeps_its_480x360_size(paired):
    """`sim --save-preview` used to resize the 480x360 overview to 480x368 and print an imageio WARNING."""
    _, root, _ = paired
    with imageio.get_reader(root / "previews/hero_kitchen_sim_overview.mp4") as reader:
        frame = reader.get_data(0)
    assert frame.shape == (360, 480, 3)

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

mujoco = pytest.importorskip("mujoco")

from conftest import REPO, SCANS  # noqa: E402

from arrival.cli import run_demo  # noqa: E402
from arrival.export.loader_check import check_loader, format_report  # noqa: E402
from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim.collect import collect  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from check_dataset_loader import main as check_main  # noqa: E402


@pytest.fixture(scope="module")
def sim_ds(tmp_path_factory):
    twin = compile_twin(load_room("hero_kitchen", "scans"))
    out = tmp_path_factory.mktemp("sim")
    res = collect(twin, ["counter_to_sink"], 2, out, people=0, res=64, log=lambda *_: None)
    return res["dataset"]


@pytest.fixture
def copy(sim_ds, tmp_path):
    dst = tmp_path / "ds"
    shutil.copytree(sim_ds, dst)
    return dst


def codes(res):
    return {e["code"] for e in res["errors"]}


def edit_json(path: Path, fn):
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))


def test_fresh_sim_dataset_passes(sim_ds):
    res = check_loader(sim_ds)
    assert res["ok"], res["errors"]
    i = res["info"]
    assert i["episodes"] == 2 and i["cameras"] == ["ego_view", "wrist"] and i["videos_decoded"] == 4
    assert i["padded_windows"] == 15 * 2 and i["short_episodes"] == 0
    assert i["relative_action_range"] is not None
    assert set(res["stats"]["action"]) == {"mean", "std", "min", "max", "q01", "q99"}
    assert "PASS" in format_report(sim_ds, res)


def test_stats_json_has_the_quantiles_gr00t_normalises_with(sim_ds):
    # GR00T's state/action processor reads q01/q99 from meta/stats.json for percentile normalisation
    stats = json.loads((sim_ds / "meta" / "stats.json").read_text())
    for col, st in stats.items():
        assert {"mean", "std", "min", "max", "q01", "q99"} <= set(st), col
        assert len(st["q01"]) == len(st["q99"]) == len(st["mean"]), col
        assert np.all(np.asarray(st["min"]) <= st["q01"]) and np.all(np.asarray(st["q99"]) <= st["max"]), col
    res = check_loader(sim_ds)
    assert "stats_no_quantiles" not in {w["code"] for w in res["warnings"]}
    assert np.allclose(stats["action"]["q99"], res["stats"]["action"]["q99"])


def test_wrong_quantile_is_a_mismatch(copy):
    edit_json(copy / "meta" / "stats.json", lambda d: d["action"].update(q01=[v - 1.0 for v in d["action"]["q01"]]))
    assert "stats_mismatch" in codes(check_loader(copy))


def test_placeholder_demo_is_refused(tmp_path):
    out = tmp_path / "out"
    ds = run_demo("hero_kitchen", 2, "all", out_dir=out, scans_dir=SCANS, log=lambda *a: None)["dataset"]
    res = check_loader(ds)
    assert not res["ok"] and codes(res) == {"placeholder"}
    assert "placeholder" in res["errors"][0]["msg"]


def test_nan_action(copy):
    p = copy / "data" / "chunk-000" / "episode_000000.parquet"
    t = pq.read_table(p)
    a = np.asarray(t.column("action").to_pylist(), dtype=np.float32)
    a[3, 2] = np.nan
    t = t.set_column(
        t.schema.get_field_index("action"), t.schema.field("action"), pa.array(list(a), type=pa.list_(pa.float32()))
    )
    pq.write_table(t, p)
    assert "nonfinite" in codes(check_loader(copy))


def test_wrong_fps_in_info(copy):
    edit_json(copy / "meta" / "info.json", lambda d: d.update(fps=30))
    c = codes(check_loader(copy))
    assert "timestamp_fps" in c and "video_fps" in c


def test_missing_wrist_mp4(copy):
    (copy / "videos" / "chunk-000" / "observation.images.wrist" / "episode_000001.mp4").unlink()
    res = check_loader(copy)
    assert codes(res) == {"missing_video"} and "wrist" in res["errors"][0]["msg"]


def test_stats_std_zero_on_moving_dim(copy):
    def zero(d):
        d["action"]["std"][0] = 0.0

    edit_json(copy / "meta" / "stats.json", zero)
    assert codes(check_loader(copy)) == {"stats_std_zero"}


def test_stats_mean_mismatch(copy):
    def shift(d):
        d["observation.state"]["mean"][1] += 1.0

    edit_json(copy / "meta" / "stats.json", shift)
    assert codes(check_loader(copy)) == {"stats_mismatch"}


def test_task_index_without_task_row(copy):
    p = copy / "data" / "chunk-000" / "episode_000001.parquet"
    t = pq.read_table(p)
    t = t.set_column(t.schema.get_field_index("task_index"), "task_index", pa.array(np.full(t.num_rows, 99)))
    pq.write_table(t, p)
    assert codes(check_loader(copy)) == {"task_index_unresolved"}


def test_cli_exit_codes(sim_ds, copy, capsys):
    assert check_main([str(sim_ds)]) == 0
    assert "PASS" in capsys.readouterr().out
    assert check_main([str(sim_ds), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    edit_json(copy / "meta" / "info.json", lambda d: d.update(fps=30))
    assert check_main([str(copy)]) == 1
    assert "FAIL" in capsys.readouterr().out


# ---- the 11-float schema: base x, y, yaw at state/action[8:11] (MAN-3, L13) ----


@pytest.fixture(scope="module")
def mobile_ds(tmp_path_factory):
    twin = compile_twin(load_room("hero_kitchen", "scans"))
    res = collect(
        twin,
        ["counter_to_counter"],
        1,
        tmp_path_factory.mktemp("mobile"),
        people=0,
        res=64,
        base="mobile",
        log=lambda *_: None,
    )
    assert res["report_data"]["successes"] == 1
    return res["dataset"]


@pytest.fixture
def mcopy(mobile_ds, tmp_path):
    dst = tmp_path / "mds"
    shutil.copytree(mobile_ds, dst)
    return dst


def _set_col(path: Path, col: str, fn):
    t = pq.read_table(path)
    x = np.asarray(t.column(col).to_pylist(), dtype=np.float32)
    fn(x)
    pq.write_table(
        t.set_column(
            t.schema.get_field_index(col), t.schema.field(col), pa.array(list(x), type=pa.list_(pa.float32()))
        ),
        path,
    )


def test_fixed_base_dataset_is_11_wide_and_the_parked_base_is_a_warning_not_a_defect(sim_ds):
    res = check_loader(sim_ds)
    assert res["ok"], res["errors"]
    assert res["info"]["state_dim"] == res["info"]["action_dim"] == 11 and res["info"]["base"] == "fixed"
    warn = {w["code"] for w in res["warnings"]}
    assert "fixed_base_constant" in warn and "constant_dim" not in warn
    assert "fixed base in state/action[8:11]" in format_report(sim_ds, res)


def test_mobile_dataset_passes_and_its_base_columns_move(mobile_ds):
    res = check_loader(mobile_ds)
    assert res["ok"], res["errors"]
    assert res["info"]["base"] == "mobile"
    assert "fixed_base_constant" not in {w["code"] for w in res["warnings"]}
    std = np.asarray(res["stats"]["action"]["std"])
    assert std[8:10].max() > 0.05  # it drove metres


def test_a_base_that_moves_on_a_fixed_embodiment_is_an_error(mcopy):
    edit_json(mcopy / "meta" / "embodiment.json", lambda d: d.update(base_actuated=False))
    assert "base_fixed_moves" in codes(check_loader(mcopy))


def test_a_teleporting_base_is_an_error(mcopy):
    p = mcopy / "data" / "chunk-000" / "episode_000000.parquet"

    def jump(x):
        x[30:, 8] += 1.0  # one metre in a single step

    _set_col(p, "observation.state", jump)
    assert "base_speed" in codes(check_loader(mcopy))


def test_base_slice_must_be_8_to_11(mcopy):
    def shrink(d):
        d["action"]["base"]["end"] = 10

    edit_json(mcopy / "meta" / "modality.json", shrink)
    assert "base_slice" in codes(check_loader(mcopy))


def test_embodiment_with_a_base_needs_the_slice(mcopy):
    def drop(d):
        d["state"].pop("base")
        d["action"].pop("base")

    edit_json(mcopy / "meta" / "modality.json", drop)
    assert "base_slice" in codes(check_loader(mcopy))  # dof_base 3 in embodiment.json, no base slice in modality.json


@pytest.mark.parametrize("shape", ["truncated", "broadcast"])
def test_episode_stats_wrong_shape_is_reported(copy, shape):
    path = copy / "meta" / "episodes_stats.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mean = rows[0]["stats"]["action"]["mean"]
    rows[0]["stats"]["action"]["mean"] = mean[:3] if shape == "truncated" else [mean]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    res = check_loader(copy)
    assert not res["ok"]
    assert any(
        e["code"] == "stats_mismatch" and "episodes_stats.jsonl episode 0 action.mean" in e["msg"]
        for e in res["errors"]
    )

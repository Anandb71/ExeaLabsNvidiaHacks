"""sim --split eval with the template wordings (the default --instructions) ran the train split: the same tasks,
wordings and episodes as --split train, with "split": "eval" in the dataset's provenance. Only the generated wordings
(--instructions nemotron) have an eval split; the template ones are held out by eval --holdout and never go into a sim
dataset. sim now refuses the combination before anything runs."""

import pytest

from arrival import cli

pytest.importorskip("mujoco")  # cmd_sim checks the families and loads the twin first

ARGS = ["sim", "--episodes", "1", "--families", "open_drawer", "--people", "0", "--res", "16"]


@pytest.fixture
def collect_calls(monkeypatch):
    """collect records how it was called and raises instead of simulating, so a test sees whether a run started."""
    calls = []

    def collect(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("collect ran")

    monkeypatch.setattr("arrival.sim.collect.collect", collect)
    return calls


def test_eval_split_with_template_wordings_is_a_usage_error(tmp_path, capsys, monkeypatch, collect_calls):
    monkeypatch.chdir(tmp_path)  # resolve_scans falls back to the repo's scans/
    assert cli.main([*ARGS, "--split", "eval", "--out", str(tmp_path)]) == cli.EXIT_USAGE
    assert "--split eval needs --instructions nemotron" in capsys.readouterr().err
    assert collect_calls == [] and not (tmp_path / "datasets").exists()


@pytest.mark.parametrize(
    "extra,split,generated",
    [
        ([], "train", False),
        (["--split", "train"], "train", False),
        (["--instructions", "nemotron"], "train", True),
        (["--instructions", "nemotron", "--split", "eval"], "eval", True),
    ],
)
def test_other_split_and_wording_choices_still_run(extra, split, generated, tmp_path, monkeypatch, collect_calls):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="collect ran"):
        cli.main([*ARGS, *extra, "--out", str(tmp_path)])
    assert [(c["split"], c["use_generated_instructions"]) for c in collect_calls] == [(split, generated)]

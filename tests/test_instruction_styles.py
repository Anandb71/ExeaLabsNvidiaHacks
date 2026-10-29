import json

import pytest

from arrival.tasks.instructions import INSTRUCTION_STYLES, STYLE_PROMPTS, _cache_key, generate_instructions


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.delenv("TOKEN_FACTORY_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)


def test_invalid_instruction_style(offline):
    with pytest.raises(ValueError, match="style must be one of"):
        generate_instructions("counter_to_sink", "mug", style="invalid_style")


@pytest.mark.parametrize("style", INSTRUCTION_STYLES)
def test_each_instruction_style_fallback_split(offline, style):
    res = generate_instructions("counter_to_sink", "mug", style=style)
    assert res["source"] == "template_fallback"
    assert res["style"] == style
    assert len(res["train"]) > 0
    assert len(res["eval"]) > 0
    assert set(res["train"]).isdisjoint(res["eval"])

    if style == "polite":
        assert all(s.startswith("Please") for s in res["train"] + res["eval"])
    elif style == "urgent":
        assert all(s.startswith("Quickly") for s in res["train"] + res["eval"])
    elif style == "descriptive":
        assert all(s.startswith("Carefully") for s in res["train"] + res["eval"])
    elif style == "goal_oriented":
        assert all(s.startswith("To tidy up,") for s in res["train"] + res["eval"])


def test_cache_key_isolation_by_style():
    mixed_key = _cache_key("counter_to_sink", "mug", None, style="mixed")
    assert "::style=" not in mixed_key  # default preserves exact legacy key format
    assert mixed_key == "counter_to_sink::obj=mug::fx=none"

    concise_key = _cache_key("counter_to_sink", "mug", None, style="concise")
    assert concise_key.endswith("::style=concise")
    assert mixed_key != concise_key

    polite_key = _cache_key("counter_to_sink", "mug", None, style="polite")
    assert polite_key.endswith("::style=polite")
    assert concise_key != polite_key


def test_token_factory_prompt_injects_style_guidance(monkeypatch, tmp_path):
    prompts = []

    def fake_complete(messages, role="lightning", **_):
        prompts.append(messages[1]["content"])
        return json.dumps([f"Instruction {i}" for i in range(8)])

    monkeypatch.setattr("arrival.models.token_factory.complete", fake_complete)

    res = generate_instructions("counter_to_sink", "mug", style="urgent", cache_path=tmp_path / "cache.json")
    assert res["source"] == "token_factory"
    assert res["style"] == "urgent"
    assert len(prompts) == 1
    assert STYLE_PROMPTS["urgent"] in prompts[0]
    assert "Task family: counter_to_sink" in prompts[0]


def test_make_tasks_propagates_instruction_style(hero_twin, offline):
    from arrival.sim.tasks import make_tasks

    tasks = make_tasks(
        hero_twin,
        ["counter_to_sink"],
        2,
        use_generated_instructions=True,
        instruction_style="polite",
        log=lambda *_: None,
    )
    assert len(tasks) == 2
    for t in tasks:
        assert t.get("instruction_style") == "polite"
        assert t["instruction"].startswith("Please")


def test_cli_instruction_style_requires_nemotron(capsys):
    from arrival.cli import main

    code = main(["sim", "--instruction-style", "concise"])
    assert code != 0
    err = capsys.readouterr().err
    assert "--instruction-style needs --instructions nemotron" in err

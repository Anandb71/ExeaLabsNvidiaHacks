import json

from arrival.tasks.instructions import generate_instructions


def test_model_wordings_are_unique_across_train_and_eval(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "arrival.models.token_factory.complete",
        lambda **_: json.dumps(["Pick the mug", "Move the mug", "Pick the mug", "Set down the mug"]),
    )
    result = generate_instructions("counter_to_sink", "mug", n=4, cache_path=tmp_path / "cache.json")
    assert result["source"] == "token_factory"
    assert result["train"] + result["eval"] == ["Pick the mug", "Move the mug", "Set down the mug"]
    assert set(result["train"]).isdisjoint(result["eval"])


def test_single_repeated_model_wording_keeps_train_only(monkeypatch, tmp_path):
    monkeypatch.setattr("arrival.models.token_factory.complete", lambda **_: json.dumps(["same"] * 8))
    result = generate_instructions("counter_to_sink", "mug", cache_path=tmp_path / "cache.json")
    assert result["source"] == "token_factory"
    assert result["train"] == ["same"]
    assert result["eval"] == []

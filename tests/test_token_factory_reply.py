import io
import json

import pytest

from arrival.models import token_factory
from arrival.tasks.instructions import generate_instructions


@pytest.fixture
def reply(monkeypatch):
    """Point complete() at a stubbed urlopen that answers with the given JSON body; no request leaves the test."""
    monkeypatch.setenv("TOKEN_FACTORY_API_KEY", "tf-key")
    monkeypatch.setenv("NEMOTRON_LIGHTNING_MODEL", "nvidia/lightning")

    def answer(body):
        monkeypatch.setattr(
            token_factory.urllib.request, "urlopen", lambda req, timeout: io.BytesIO(json.dumps(body).encode())
        )

    return answer


def test_a_reply_with_text_returns_it(reply):
    reply({"choices": [{"message": {"content": '["Move the mug"]'}, "finish_reason": "stop"}]})
    assert token_factory.complete([{"role": "user", "content": "hi"}], role="lightning") == '["Move the mug"]'


@pytest.mark.parametrize(
    "body",
    [
        {"choices": []},
        {"choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "length"}]},
        {"error": {"message": "upstream overloaded"}},
    ],
)
def test_a_reply_with_no_text_says_so(reply, body):
    reply(body)
    with pytest.raises(ValueError, match="no message text"):
        token_factory.complete([{"role": "user", "content": "hi"}], role="lightning")


def test_wordings_fallback_names_the_empty_reply(reply, tmp_path):
    reply({"choices": [{"message": {"content": None}, "finish_reason": "length"}]})
    result = generate_instructions("counter_to_sink", "mug", cache_path=tmp_path / "cache.json")
    assert result["source"] == "template_fallback"
    assert "no message text" in result["error"] and "'length'" in result["error"]

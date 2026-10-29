import json
import os
from pathlib import Path

import pytest

from arrival.models.tavily import TavilyNotConfigured
from arrival.models.tavily import search as tavily_search
from arrival.tasks.tavily_research import TavilyResearchAgent


def test_tavily_not_configured_raises(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(TavilyNotConfigured):
        tavily_search("test query")


def test_tavily_research_agent_cache_hit(tmp_path):
    cache_file = tmp_path / "test_tavily_cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "test appliance query": {
                    "source": "tavily_api",
                    "answer": "Test answer for appliance latch and clearance.",
                    "results": [{"title": "Test Guide", "url": "https://example.com/test", "content": "..."}],
                }
            }
        )
    )

    agent = TavilyResearchAgent(log=lambda *_: None, cache_path=cache_file)
    res = agent.search("test appliance query")
    assert agent.last_source == "cache"
    assert "Test answer" in res["answer"]
    assert len(res["results"]) == 1


def test_tavily_research_agent_fallback_when_unconfigured(monkeypatch, tmp_path):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    empty_cache = tmp_path / "empty.json"

    agent = TavilyResearchAgent(log=lambda *_: None, cache_path=empty_cache)
    info = agent.research_appliance("dishwasher")

    assert info["fixture_type"] == "dishwasher"
    assert info["canonical_type"] == "dishwasher"
    assert len(info["safety_guidelines"]) >= 1
    assert len(info["operating_steps"]) >= 1
    assert "counterbalanced" in info["summary"] or "hinge" in info["summary"]


def test_tavily_research_agent_mock_live_search(monkeypatch, tmp_path):
    cache_file = tmp_path / "save_cache.json"

    def mock_tavily_search(query, search_depth="basic", include_answer=True):
        return {
            "answer": "Mock live Tavily answer for microwave button.",
            "results": [{"title": "Microwave Manual", "url": "https://example.com/mw", "content": "specs"}],
        }

    monkeypatch.setattr("arrival.tasks.tavily_research.tavily_search", mock_tavily_search)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-dummy-key")

    agent = TavilyResearchAgent(log=lambda *_: None, cache_path=cache_file)
    info = agent.research_appliance("microwave")

    assert agent.last_source == "tavily_api"
    assert "Mock live Tavily answer" in info["summary"]
    assert "https://example.com/mw" in info["sources"]
    assert cache_file.is_file()

    # Second query should hit cache
    agent2 = TavilyResearchAgent(log=lambda *_: None, cache_path=cache_file)
    info2 = agent2.research_appliance("microwave")
    assert agent2.last_source == "cache"
    assert info2["summary"] == info["summary"]


def test_enrich_twin_with_tavily_research(tmp_path):
    agent = TavilyResearchAgent(log=lambda *_: None)
    twin = {
        "room_id": "test_room",
        "fixtures": [
            {"id": "c1", "type": "counter"},
            {"id": "dw1", "type": "dishwasher"},
            {"id": "st1", "type": "gas_range"},
        ],
        "objects": [],
    }

    enriched = agent.enrich_twin(twin)
    assert "tavily_research" in enriched
    research = enriched["tavily_research"]
    assert "dishwasher" in research
    assert "gas_range" in research
    assert len(research["dishwasher"]["safety_guidelines"]) >= 1


def test_cli_research_command(tmp_path):
    import argparse

    from arrival.cli import cmd_research

    args = argparse.Namespace(
        room="hero_kitchen",
        query=None,
        out=str(tmp_path),
        scans="scans",
    )
    code = cmd_research(args)
    assert code == 0
    report_file = tmp_path / "research" / "hero_kitchen_research.json"
    assert report_file.is_file()
    data = json.loads(report_file.read_text())
    assert "sink" in data or "counter" in data or "stove" in data


def test_enrich_twin_skips_non_appliances_and_negative_caches(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    agent = TavilyResearchAgent(log=lambda *_: None, cache_path=tmp_path / "c.json")
    twin = {
        "fixtures": [
            {"id": "b", "type": "box"},
            {"id": "c", "type": "counter"},
            {"id": "h", "type": "hinge_cabinet"},
            {"id": "s", "type": "sink"},
        ]
    }
    research = agent.enrich_twin(twin)["tavily_research"]
    assert set(research) == {"sink"}

    calls = []

    def boom(*a, **k):
        calls.append(a)
        raise RuntimeError("network down")

    monkeypatch.setattr("arrival.tasks.tavily_research.tavily_search", boom)
    agent.research_appliance("fridge")
    agent.research_appliance("fridge")
    assert len(calls) == 1


def test_load_dotenv_reads_every_file(monkeypatch, tmp_path):
    from arrival.models import tavily, token_factory

    (tmp_path / "a.env").write_text("UNRELATED=1\n")
    (tmp_path / "b.env").write_text("TAVILY_KEY_TEST=real\n")
    monkeypatch.delenv("TAVILY_KEY_TEST", raising=False)
    for mod in (tavily, token_factory):
        monkeypatch.delenv("TAVILY_KEY_TEST", raising=False)
        mod.load_dotenv([tmp_path / "a.env", tmp_path / "b.env"])
        assert os.environ["TAVILY_KEY_TEST"] == "real"
    monkeypatch.delenv("TAVILY_KEY_TEST", raising=False)
    os.environ.pop("UNRELATED", None)


def test_default_cache_is_under_out_not_in_the_package(tmp_path, monkeypatch):
    # a cache in the package answered the standard appliance queries, so a run with a TAVILY_API_KEY never
    # called Tavily for them; the default now lives in the run's out/ folder
    import arrival

    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(
        "arrival.tasks.tavily_research.tavily_search",
        lambda q, **k: calls.append(q) or {"answer": "live", "results": []},
    )
    agent = TavilyResearchAgent(log=lambda *_: None)
    for fixture in ("dishwasher", "stove", "fridge", "microwave", "sink"):
        assert agent.research_appliance(fixture)["source"] == "tavily_api"
    assert len(calls) == 5
    assert (tmp_path / "out" / "cache" / "tavily_cache.json").is_file()
    assert not list((Path(arrival.__file__).parent / "models").glob("*cache*.json"))


@pytest.fixture
def http_reply(monkeypatch):
    """An HTTP-200 byte stream in process: no listener, API keys or real connection needed."""
    import io

    from arrival.models import tavily

    calls = []
    monkeypatch.setattr(tavily, "config", lambda: {"api_key": "test-only", "base_url": "https://example.test/search"})

    def install(payload):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

        def reply(request, timeout):
            calls.append((request, timeout))
            return io.BytesIO(body)

        monkeypatch.setattr(tavily.urllib.request, "urlopen", reply)
        return calls

    return install


@pytest.mark.parametrize(
    "payload",
    [
        b"<html>proxy response</html>",
        b"\xff",
        [1, 2],
        "ok",
        None,
        {"results": None},
        {"results": {}},
        {"results": "wrong"},
        {"error": "unavailable"},
        {"results": [None, {"title": None, "url": None, "content": None}]},
        {"answer": {}, "results": []},
    ],
)
def test_bad_http_200_replies_fall_back_without_poisoning_cache(tmp_path, http_reply, payload):
    from arrival.tasks.tavily_research import FALLBACK_KNOWLEDGE

    calls = http_reply(payload)
    with pytest.raises(ValueError, match="Tavily reply") as error:
        tavily_search("query")
    assert not isinstance(error.value, json.JSONDecodeError)
    path = tmp_path / "cache.json"
    agent = TavilyResearchAgent(cache_path=path, log=lambda *_: None)
    for _ in range(2):  # unsuccessful queries are suppressed only for this agent's lifetime
        info = agent.research_appliance("microwave")
        assert info["source"] == "fallback" and info["sources"] == []
        assert info["guidelines_source"] == "built_in_table"
        assert info["summary"] == FALLBACK_KNOWLEDGE["microwave"]["summary"]
        assert info["safety_guidelines"] == FALLBACK_KNOWLEDGE["microwave"]["safety_guidelines"]
    assert len(calls) == 2  # direct client + first agent lookup, no repeated failed request
    assert not agent._cache and not path.exists()


@pytest.mark.parametrize("bad_text", [None, 42, ["bad"], {"bad": "text"}])
def test_mixed_result_rows_are_sanitized_before_research_and_cache(tmp_path, http_reply, bad_text):
    payload = {
        "answer": bad_text,
        "results": [
            None,
            7,
            {"title": bad_text, "url": bad_text, "content": "Maintain 15 cm clearance."},
            {"title": "Manual", "url": "https://example.test/manual", "content": bad_text},
        ],
    }
    http_reply(payload)
    path = tmp_path / "cache.json"
    agent = TavilyResearchAgent(cache_path=path, log=lambda *_: None)
    info = agent.research_appliance("microwave")
    assert info["source"] == "tavily_api"
    assert info["sources"] == ["https://example.test/manual"]
    assert info["guidelines_source"] == "built_in_table"
    cached = next(iter(json.loads(path.read_text()).values()))
    assert cached["answer"] == "" and len(cached["results"]) == 2
    assert all(isinstance(row[field], str) for row in cached["results"] for field in ("title", "url", "content"))
    assert cached["results"][0]["content"] == "Maintain 15 cm clearance."
    replay = TavilyResearchAgent(cache_path=path, log=lambda *_: None)
    assert replay.research_appliance("microwave")["source"] == "cache"


@pytest.mark.parametrize(
    "cache_body",
    [
        '{"q":',
        "[1, 2]",
        '"ok"',
        "null",
        b"\xff",
        '{"q": {"source": "tavily_api", "answer": null, "results": []}}',
        '{"q": {"source": "tavily_api", "answer": "old", "results": null}}',
        '{"q": {"source": "tavily_api", "answer": "old", "results": [null]}}',
        '{"q": {"source": "tavily_api", "answer": "old", "results": [{"title": "Guide", "url": null}]}}',
        '{"q": {"source": "fallback", "answer": "old", "results": []}}',
        '{"q": {}}',
    ],
)
def test_corrupt_or_malformed_cache_is_a_miss_and_recovers(tmp_path, http_reply, cache_body):
    path = tmp_path / "cache.json"
    path.write_bytes(cache_body if isinstance(cache_body, bytes) else cache_body.encode())
    calls = http_reply({"answer": "Fresh answer", "results": []})
    agent = TavilyResearchAgent(cache_path=path, log=lambda *_: None)
    assert agent.search("q") == {"source": "tavily_api", "answer": "Fresh answer", "results": []}
    assert agent.last_source == "tavily_api" and len(calls) == 1
    assert json.loads(path.read_text()) == {"q": {"source": "tavily_api", "answer": "Fresh answer", "results": []}}
    replay = TavilyResearchAgent(cache_path=path, log=lambda *_: None)
    assert replay.search("Q")["answer"] == "Fresh answer"
    assert replay.last_source == "cache" and len(calls) == 1


def test_valid_http_response_and_request_stay_unchanged(http_reply):
    payload = {
        "query": "clearance",
        "answer": "Allow 15 cm.",
        "results": [
            {"title": "Guide", "url": "https://example.test/guide", "content": "15 cm clearance", "score": 0.9}
        ],
        "response_time": 0.1,
    }
    calls = http_reply(payload)
    assert tavily_search("clearance", search_depth="advanced", timeout=2) == payload
    request, timeout = calls[0]
    assert request.get_method() == "POST" and timeout == 2
    assert json.loads(request.data) == {
        "api_key": "test-only",
        "query": "clearance",
        "search_depth": "advanced",
        "include_answer": True,
        "max_results": 5,
    }


def test_bad_lookup_preserves_other_valid_cached_findings(tmp_path, monkeypatch):
    path = tmp_path / "cache.json"
    original = json.dumps({"kept": {"source": "tavily_api", "answer": "Keep this", "results": []}})
    path.write_text(original)
    # In-process clients can bypass the HTTP validator; the agent must validate these replies too.
    monkeypatch.setattr("arrival.tasks.tavily_research.tavily_search", lambda *a, **k: {"results": None})
    agent = TavilyResearchAgent(cache_path=path, log=lambda *_: None)
    assert agent.search("new")["source"] == "fallback"
    assert path.read_text() == original
    assert agent.search("kept")["answer"] == "Keep this" and agent.last_source == "cache"


def test_missing_result_text_fields_are_safe_strings(http_reply):
    http_reply({"answer": None, "results": [{"url": "https://example.test/manual"}]})
    assert tavily_search("query") == {
        "answer": "",
        "results": [{"title": "", "url": "https://example.test/manual", "content": ""}],
    }

import pytest

from arrival.tasks.tavily_research import TavilyResearchAgent


@pytest.mark.parametrize("cache_body", ['["query"]', '{"query": [1, 2]}', "null"])
def test_malformed_cache_entries_are_ignored(tmp_path, monkeypatch, cache_body):
    path = tmp_path / "cache.json"
    path.write_text(cache_body)
    calls = []

    def live(query, **kwargs):
        calls.append(query)
        return {"answer": "verified", "results": []}

    monkeypatch.setattr("arrival.tasks.tavily_research.tavily_search", live)
    agent = TavilyResearchAgent(cache_path=path, log=lambda *_: None)
    assert agent.search("query")["answer"] == "verified"
    assert calls == ["query"]

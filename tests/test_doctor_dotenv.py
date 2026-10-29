"""arrival doctor reports a key from .env as set only when config.get, which every module reads it through, returns it."""

import pytest

from arrival import config, doctor

KEY = "TAVILY_API_KEY"


@pytest.fixture
def dotenv(monkeypatch, tmp_path):
    """No known setting set, the cwd an empty dir whose .env sets KEY and the Token Factory key, .env lookup on
    (conftest turns it off for every other test), and no repo-root .env for config or doctor."""
    for name in config.SETTINGS:  # setenv first so teardown removes whatever a .env load adds during the test
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("ARRIVAL_DOTENV", "1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path / "no_repo_root")
    monkeypatch.setattr(doctor, "REPO", tmp_path / "no_repo_root")
    (tmp_path / ".env").write_text(
        f"{KEY}=tvly-from-dotenv-0123456789\nTOKEN_FACTORY_API_KEY=nvapi-from-dotenv-0123\n", encoding="utf-8"
    )
    return tmp_path


def _doctor(name: str) -> doctor.Check:
    """Doctor's row for `name`. Each test asks doctor first: config.get copies the .env values into os.environ."""
    return next(c for c in doctor.check_env() if c.name == name)


def test_a_key_in_dotenv_is_set(dotenv):
    check = _doctor(KEY)
    assert check.status == "ok" and "set in .env" in check.detail
    assert config.get(KEY) == "tvly-from-dotenv-0123456789"


def test_dotenv_turned_off(dotenv, monkeypatch):
    monkeypatch.setenv("ARRIVAL_DOTENV", "0")
    checks = doctor.check_env()
    check = next(c for c in checks if c.name == KEY)
    assert check.status == "info"
    assert check.detail.startswith("not set") and "ARRIVAL_DOTENV=0" in check.detail
    # the Token Factory key is not read either, so its model ids are not worth a warning
    assert not [c.name for c in checks if c.status == "warn"]
    assert config.get(KEY) is None
    assert config.get("TOKEN_FACTORY_API_KEY") is None


def test_an_empty_variable_does_not_hide_the_dotenv_value(dotenv, monkeypatch):
    monkeypatch.setenv(KEY, "")
    check = _doctor(KEY)
    assert check.status == "ok" and "set in .env" in check.detail
    assert "tvly-from-dotenv" not in check.detail
    assert config.get(KEY) == "tvly-from-dotenv-0123456789"


def test_an_empty_line_in_the_cwd_dotenv_does_not_hide_the_repo_one(dotenv, monkeypatch):
    repo = dotenv / "repo"
    repo.mkdir()
    (repo / ".env").write_text(f"{KEY}=tvly-from-repo-0123456789\n", encoding="utf-8")
    (dotenv / ".env").write_text(f"{KEY}=\n", encoding="utf-8")
    monkeypatch.setattr(config, "REPO_ROOT", repo)
    monkeypatch.setattr(doctor, "REPO", repo)
    assert _doctor(KEY).status == "ok"
    assert config.get(KEY) == "tvly-from-repo-0123456789"

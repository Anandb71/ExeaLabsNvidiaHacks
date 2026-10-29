"""A comment after a value in .env is not part of the value (python-dotenv and a shell read it the same way)."""

import pytest

from arrival import config


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for name in config.SETTINGS:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("ARRIVAL_DOTENV", "1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path / "no_repo_root")
    return tmp_path


def test_comment_after_a_value_is_dropped(clean_env):
    (clean_env / ".env").write_text(
        "GR00T_POLICY_URL=http://gpu-box:8000  # the A100 box\n"
        'TAVILY_API_KEY="tvly-abc123def456" # from the console\n'
        "NEBIUS_BUCKET=arrival-runs\t# made on Oct 20\n"
        "NEBIUS_REGION='eu-north1'# no space before the hash\n"
    )
    assert config.get("GR00T_POLICY_URL") == "http://gpu-box:8000"
    assert config.get("TAVILY_API_KEY") == "tvly-abc123def456"
    assert config.get("NEBIUS_BUCKET") == "arrival-runs"
    assert config.get("NEBIUS_REGION") == "eu-north1"


def test_hash_inside_a_value_is_kept(clean_env):
    (clean_env / ".env").write_text(
        "COSMOS_URL=http://cosmos:9000/#frag\n"
        'NEBIUS_JOB_IMAGE="registry/arrival # tag"\n'
        "NEBIUS_PROJECT_ID='abc' # it's \"quoted\"\n"
    )
    assert config.get("COSMOS_URL") == "http://cosmos:9000/#frag"
    assert config.get("NEBIUS_JOB_IMAGE") == "registry/arrival # tag"
    assert config.get("NEBIUS_PROJECT_ID") == "abc"

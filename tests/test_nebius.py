"""C1: NebiusBackend against in-process fakes of the Serverless AI Jobs REST API and of S3 (tests/fake_nebius.py).
Nothing here talks to Nebius."""

import dataclasses
import importlib.util
import itertools
import json
import sys

import pytest
from conftest import REPO, SCANS
from fake_nebius import FakeJobs, FakeS3

from arrival import config
from arrival.jobs.fanout import FAILED, FanoutConfig, LocalBackend, run_fanout
from arrival.jobs.fanout_status import StatusRecorder, status_key
from arrival.jobs.nebius import NebiusBackend, NebiusNotConfigured, ObjectStore, sigv4
from arrival.jobs.shards import ShardError

needs_sim = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="needs mujoco")
CFG = FanoutConfig("r1", 2, "hero_kitchen", "counter_to_sink,open_drawer", 4, 0, 32, 3, str(SCANS))
CFG1 = dataclasses.replace(CFG, run_id="r2", shards=1)
TOKEN, KEY_ID, SECRET = "fake-iam-token-0123456789", "FAKEKEYID0123", "fake/secret+key0123456789"
quiet = lambda *_: None  # noqa: E731


@pytest.fixture
def nebius(monkeypatch, tmp_path):
    """Start both fakes and point every NEBIUS_* setting at them; `make(**fake_jobs_options)` restarts the job fake."""
    for name in config.SETTINGS:
        if name.startswith("NEBIUS_"):
            monkeypatch.delenv(name, raising=False)
    s3 = FakeS3("arrival-test", KEY_ID, SECRET, "eu-north1")
    secrets = {"mbsec-test": {"NEBIUS_ACCESS_KEY_ID": KEY_ID, "NEBIUS_SECRET_ACCESS_KEY": SECRET}}
    fakes = {"s3": s3}

    def make(**kw):
        if "jobs" in fakes:
            fakes["jobs"].close()
        logs = tmp_path / f"jobs{len(list(tmp_path.glob('jobs*')))}"
        logs.mkdir()
        fakes["jobs"] = FakeJobs(TOKEN, "project-test", secrets, REPO, logs, **kw)
        monkeypatch.setenv("NEBIUS_API_URL", fakes["jobs"].url)
        return fakes["jobs"]

    for name, value in {
        "NEBIUS_IAM_TOKEN": TOKEN,
        "NEBIUS_PROJECT_ID": "project-test",
        "NEBIUS_JOB_IMAGE": "cr.example/arrival:test",
        "NEBIUS_JOB_S3_SECRET_ID": "mbsec-test",
        "NEBIUS_BUCKET": "arrival-test",
        "NEBIUS_ACCESS_KEY_ID": KEY_ID,
        "NEBIUS_SECRET_ACCESS_KEY": SECRET,
        "NEBIUS_S3_ENDPOINT": s3.url,
    }.items():
        monkeypatch.setenv(name, value)
    fakes["make"] = make
    make()
    yield fakes
    fakes["jobs"].close()
    s3.close()


def _rows(ds):
    return [json.loads(x) for x in (ds / "meta" / "episodes.jsonl").read_text().splitlines()]


def _frames(ds):
    import pyarrow.parquet as pq

    return [pq.read_table(p).to_pydict() for p in sorted(ds.glob("data/chunk-*/episode_*.parquet"))]


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    pytest.importorskip("mujoco")
    root = tmp_path_factory.mktemp("ref")
    return run_fanout(LocalBackend(root / "store", root / "scratch"), CFG, root / "out", log=quiet)


def _same_episodes(res, reference):
    from arrival.export.validate import validate

    rows = _rows(res["dataset"])
    ids = [r["episode_index"] for r in rows]
    assert validate(res["dataset"]) == []
    assert len(ids) == len(set(ids)) == len(_rows(reference["dataset"]))  # no duplicate or missing episode
    assert rows == _rows(reference["dataset"])
    assert _frames(res["dataset"]) == _frames(reference["dataset"])


def test_sigv4_matches_the_aws_signing_test_suite():
    # awslabs/aws-c-auth tests/aws-signing-test-suite/v4: get-vanilla-query-order-key-case, post-x-www-form-urlencoded
    key, secret, when = "AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", "20150830T123600Z"
    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    auth, canonical = sigv4(
        "GET",
        "/",
        [("Param2", "value2"), ("Param1", "value1")],
        {"Host": "example.amazonaws.com", "X-Amz-Date": when},
        empty,
        key,
        secret,
        "us-east-1",
        "service",
        when,
    )
    assert canonical == (
        "GET\n/\nParam1=value1&Param2=value2\nhost:example.amazonaws.com\nx-amz-date:"
        + when
        + "\n\nhost;x-amz-date\n"
        + empty
    )
    assert auth.endswith("Signature=b97d918cfa904a5beff61c982a1b6f458b799221646efd99d3219ec94cdf2500")
    body = "9095672bbd1f56dfc5b65f3e153adc8731a4a654192329106275f4c7b24d0b6e"
    auth, _ = sigv4(
        "POST",
        "/",
        [],
        {
            "Content-Type": "application/x-www-form-urlencoded",
            "Content-Length": "13",
            "Host": "example.amazonaws.com",
            "x-amz-content-sha256": body,
            "X-Amz-Date": when,
        },
        body,
        key,
        secret,
        "us-east-1",
        "service",
        when,
    )
    assert auth == (
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, SignedHeaders="
        "content-length;content-type;host;x-amz-content-sha256;x-amz-date, "
        "Signature=d3875051da38690788ef43de4db0d8f280229d82040bfac253562e56c3f20e0b"
    )


def test_missing_settings_are_named_without_values(monkeypatch):
    for name in config.SETTINGS:
        if name.startswith("NEBIUS_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NEBIUS_SECRET_ACCESS_KEY", "do-not-print-me-123")
    with pytest.raises(NebiusNotConfigured) as e:
        NebiusBackend()
    msg = str(e.value)
    for name in (
        "NEBIUS_IAM_TOKEN",
        "NEBIUS_PROJECT_ID",
        "NEBIUS_JOB_IMAGE",
        "NEBIUS_JOB_S3_SECRET_ID",
        "NEBIUS_BUCKET",
        "NEBIUS_ACCESS_KEY_ID",
    ):
        assert name in msg
    assert "NEBIUS_SECRET_ACCESS_KEY" not in msg and "do-not-print-me" not in msg
    assert config.redact("token do-not-print-me-123") == "token [redacted]"


def test_object_store_pages_signs_and_keeps_prefixes_apart(nebius):
    s3 = nebius["s3"]
    store = ObjectStore(s3.url, "eu-north1", "arrival-test", KEY_ID, SECRET)
    for k in [
        "r/shards/1of2/a b.json",
        "r/shards/1of2/x/y.parquet",
        "r/shards/1of20/z",
        "r/done/1of2.json",
        "r/shards/1of2/c+d",
        "r/shards/1of2/e",
        "r/shards/1of2/f",
    ]:
        store.put(k, k.encode())
    assert store.list("r/shards/1of2/") == sorted(
        [
            "r/shards/1of2/a b.json",
            "r/shards/1of2/x/y.parquet",
            "r/shards/1of2/c+d",
            "r/shards/1of2/e",
            "r/shards/1of2/f",
        ]
    )
    assert store.get("r/shards/1of2/a b.json") == b"r/shards/1of2/a b.json"
    assert store.exists("r/done/1of2.json") and not store.exists("r/done/0of2.json")
    assert s3.rejected == []
    with pytest.raises(ShardError, match="SignatureDoesNotMatch"):
        ObjectStore(s3.url, "eu-north1", "arrival-test", KEY_ID, "wrong-secret").get("r/done/1of2.json")
    with pytest.raises(ShardError, match="NoSuchKey"):
        store.get("r/missing")


def test_job_spec_carries_no_secret_and_open_jobs_cancel(nebius):
    jobs = nebius["make"](hold={0, 1})
    be = NebiusBackend()
    job_id = be.submit(CFG, 0)
    method, path, body = jobs.requests[0]
    assert (method, path) == ("POST", "/ai/v1/jobs")
    assert SECRET.encode() not in body and KEY_ID.encode() not in body and TOKEN.encode() not in body
    req = json.loads(body)
    spec = req["spec"]
    assert req["metadata"]["parentId"] == "project-test" and spec["image"] == "cr.example/arrival:test"
    assert (spec["containerCommand"], spec["platform"], spec["preset"]) == ("python", "cpu-d3", "4vcpu-16gb")
    assert spec["args"].split()[:4] == ["-m", "arrival.jobs.nebius", "run-shard", "--dest"]
    assert "--shard 0/2 --resume" in spec["args"]
    secret_env = {e["name"]: e["mysteryboxSecret"] for e in spec["environmentVariables"] if "mysteryboxSecret" in e}
    assert secret_env == {n: {"secretId": "mbsec-test"} for n in ("NEBIUS_ACCESS_KEY_ID", "NEBIUS_SECRET_ACCESS_KEY")}
    assert [be.poll(job_id) for _ in range(3)] == ["running"] * 3  # PROVISIONING, IMAGE_PULLING, RUNNING
    other = be.submit(CFG, 1)
    assert be.cancel_open() == [] and sorted(jobs.cancelled) == sorted([job_id, other])
    assert be.poll(job_id) == FAILED and be.jobs[job_id]["state"] == "CANCELLED"


def test_job_states_map_and_unknown_state_fails_clearly(nebius):
    nebius["make"](states={0: "ERROR", 1: "QUEUED_SOMEWHERE"})
    be = NebiusBackend()
    a, b = be.submit(CFG, 0), be.submit(CFG, 1)
    assert [be.poll(a) for _ in range(3)] == ["running", "running", FAILED]
    be.poll(b), be.poll(b)
    with pytest.raises(ShardError, match="state 'QUEUED_SOMEWHERE', which this client does not know"):
        be.poll(b)


def test_success_without_upload_leaves_no_marker_and_open_jobs_cancel(nebius, tmp_path):
    jobs = nebius["make"](hold={0, 1}, states={0: "COMPLETED"})
    be = NebiusBackend()
    with pytest.raises(ShardError, match="uploaded nothing"):
        run_fanout(be, CFG, tmp_path / "out", poll_seconds=0, log=quiet)
    assert not [k for k in nebius["s3"].objects if k.startswith("r1/done/")]
    assert be.cancel_open() == [] and jobs.cancelled == [j for s, j in jobs.created if s == 1]


def test_bad_token_is_an_error_without_the_token(nebius, monkeypatch):
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "wrong-token-abcdefgh")
    with pytest.raises(ShardError, match="HTTP 401") as e:
        NebiusBackend().submit(CFG, 0)
    assert "wrong-token" not in str(e.value)


def test_a_reply_that_is_not_a_json_object_is_a_shard_error(nebius, monkeypatch):
    # a proxy's HTML page with HTTP 200 on one cancel must not stop cancel_open before the other paid job
    import arrival.jobs.nebius as neb

    jobs = nebius["make"](hold={0, 1})
    be = NebiusBackend()
    first, second = be.submit(CFG, 0), be.submit(CFG, 1)
    real, garbled = neb._http, iter([b"<html>gateway</html>"])

    def http(method, url, body=None, headers=None, timeout=60.0):
        if url.endswith("/ai/v1/jobs/cancel"):
            reply = next(garbled, None)
            if reply is not None:
                return 200, reply
        return real(method, url, body, headers, timeout)

    monkeypatch.setattr(neb, "_http", http)
    assert be.cancel_open() == [first] and jobs.cancelled == [second]
    monkeypatch.setattr(neb, "_http", lambda *a, **k: (200, b"[]"))
    with pytest.raises(ShardError, match="not a JSON object"):
        be.poll(first)


@needs_sim
@pytest.mark.render  # `reference` runs shards as subprocesses, where --no-render's patch does not reach
def test_failed_shard_resumes_to_the_uninterrupted_episode_set(nebius, reference, tmp_path):
    jobs = nebius["make"](fail_first={1})
    be = NebiusBackend()
    res = run_fanout(be, CFG, tmp_path / "out", log=quiet)
    assert be.submissions == [(0, 1), (1, 1), (1, 2)]  # only shard 1 was retried
    assert [s for s, _ in jobs.created] == [0, 1, 1] and res["ran"] == [0, 1]
    assert all(nebius["s3"].objects.get(f"r1/done/{i}of2.json") for i in (0, 1))
    _same_episodes(res, reference)
    assert nebius["s3"].rejected == []


@needs_sim
@pytest.mark.render  # `reference` runs shards as subprocesses, where --no-render's patch does not reach
def test_killed_run_reruns_only_unmarked_shards(nebius, reference, tmp_path):
    jobs = nebius["make"]()

    def kill(i):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_fanout(NebiusBackend(), CFG, tmp_path / "out", parallel=1, on_shard_done=kill, log=quiet)
    assert "r1/done/0of2.json" in nebius["s3"].objects and "r1/done/1of2.json" not in nebius["s3"].objects
    assert [s for s, _ in jobs.created] == [0]
    jobs = nebius["make"]()  # a new orchestrating process; the bucket is the same
    be2 = NebiusBackend()
    res = run_fanout(be2, CFG, tmp_path / "out", log=quiet)
    assert res["skipped"] == [0] and res["ran"] == [1]
    assert [s for s, _ in jobs.created] == [1]  # shard 0 was not run again
    _same_episodes(res, reference)


def _ticks(step=1000.0):
    """A monotonic clock that moves `step` seconds every time the backend reads it."""
    return itertools.count(0.0, step).__next__


def _status(s3, cfg, i):
    return json.loads(s3.objects[status_key(cfg, i)])


def test_staging_prefix_is_new_for_every_attempt_and_every_process(nebius):
    jobs = nebius["make"](hold={0})
    be = NebiusBackend()
    be.submit(CFG, 0), be.submit(CFG, 0)
    NebiusBackend().submit(CFG, 0)  # a rerun after a kill counts its attempts from 1 again
    dests = [
        json.loads(body)["spec"]["args"].split()[4]
        for method, path, body in jobs.requests
        if (method, path) == ("POST", "/ai/v1/jobs")
    ]
    assert len(dests) == len(set(dests)) == 3
    assert all(d.startswith("r1/attempts/0of2/") for d in dests)


@pytest.mark.parametrize(
    "fake, states, reason",
    [
        (
            {"hold": {0}},
            ["PROVISIONING", "IMAGE_PULLING", "RUNNING", "CANCELLING"],
            "not finished 7200 s after it was created",
        ),
        ({"silent": {0}}, ["STATE_UNSPECIFIED", "CANCELLING"], "reported no state for 1800 s"),
    ],
)
def test_stuck_or_stateless_job_is_cancelled_and_retried_up_to_max_attempts(nebius, tmp_path, fake, states, reason):
    jobs = nebius["make"](**fake)
    be = NebiusBackend(clock=_ticks())
    with pytest.raises(ShardError, match="failed 2 times"):
        run_fanout(StatusRecorder(be), CFG1, tmp_path / "out", max_attempts=2, poll_seconds=0, log=quiet)
    created = [j for _, j in jobs.created]
    assert len(created) == 2 and jobs.cancelled == created
    # each job was cancelled before the next attempt was created
    assert [p for m, p, _ in jobs.requests if m == "POST"] == ["/ai/v1/jobs", "/ai/v1/jobs/cancel"] * 2
    assert be.cancel_open() == [] and jobs.cancelled == created
    attempts = _status(nebius["s3"], CFG1, 0)["attempts"]
    assert [(a["job_id"], a["outcome"], a["failed_stage"], a["reason"]) for a in attempts] == [
        (j, "failed", "run", f"{reason}; cancelled") for j in created
    ]
    assert all([s["state"] for s in a["states"]] == states for a in attempts)


def test_limits_leave_a_job_alone_until_they_are_passed(nebius):
    nebius["make"](hold={0})
    be = NebiusBackend(clock=_ticks(600.0))
    job = be.submit(CFG, 0)
    assert {be.poll(job) for _ in range(12)} == {"running"}  # 600 s to 7200 s after creation
    assert be.poll(job) == FAILED and be.jobs[job]["state"] == "CANCELLING"  # 7800 s


def test_limit_settings_are_checked_against_the_job_timeout(nebius, monkeypatch):
    for value, match in [
        ("3600", "longer than the job timeout"),
        ("abc", "NEBIUS_MAX_JOB_SECONDS must be"),
        ("0", "NEBIUS_MAX_JOB_SECONDS must be"),
    ]:
        monkeypatch.setenv("NEBIUS_MAX_JOB_SECONDS", value)
        with pytest.raises(NebiusNotConfigured, match=match):
            NebiusBackend()
    monkeypatch.setenv("NEBIUS_MAX_JOB_SECONDS", "7200")
    monkeypatch.setenv("NEBIUS_MAX_NO_STATE_SECONDS", "-1")
    with pytest.raises(NebiusNotConfigured, match="NEBIUS_MAX_NO_STATE_SECONDS"):
        NebiusBackend()


def test_status_record_names_the_failure_stage(nebius, monkeypatch, tmp_path):
    s3 = nebius["s3"]
    nebius["make"](hold={0}, states={0: "ERROR"})
    with pytest.raises(ShardError, match="failed 1 times"):
        run_fanout(StatusRecorder(NebiusBackend()), CFG1, tmp_path / "a", max_attempts=1, poll_seconds=0, log=quiet)
    [run] = _status(s3, CFG1, 0)["attempts"]
    assert (run["outcome"], run["failed_stage"], run["reason"]) == ("failed", "run", "ERROR")
    assert [s["state"] for s in run["states"]] == ["PROVISIONING", "IMAGE_PULLING", "ERROR"]
    assert run["resources"] == {
        "platform": "cpu-d3",
        "preset": "4vcpu-16gb",
        "disk_bytes": 64 * 1024**3,
        "timeout": "3600s",
    }
    assert run["job_id"] and run["wall_seconds"] >= 0 and run["finished_at"] >= run["submitted_at"]

    nebius["make"](hold={0}, states={0: "COMPLETED"})
    cfg = dataclasses.replace(CFG1, run_id="r3")
    with pytest.raises(ShardError, match="uploaded nothing"):
        run_fanout(StatusRecorder(NebiusBackend()), cfg, tmp_path / "b", poll_seconds=0, log=quiet)
    [fetch] = _status(s3, cfg, 0)["attempts"]
    assert (fetch["outcome"], fetch["failed_stage"]) == ("failed", "fetch") and "uploaded nothing" in fetch["reason"]
    assert fetch["states"][-1]["state"] == "COMPLETED"

    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "wrong-token-abcdefgh")
    cfg = dataclasses.replace(CFG1, run_id="r4")
    with pytest.raises(ShardError, match="HTTP 401"):
        run_fanout(StatusRecorder(NebiusBackend()), cfg, tmp_path / "c", poll_seconds=0, log=quiet)
    [submit] = _status(s3, cfg, 0)["attempts"]
    assert (submit["job_id"], submit["outcome"], submit["failed_stage"]) == (None, "failed", "submit")
    assert "HTTP 401" in submit["reason"] and b"wrong-token" not in s3.objects[status_key(cfg, 0)]


def test_local_backend_records_attempts_and_a_failed_marker(tmp_path, monkeypatch):
    def command(cfg, i, out):
        return [
            sys.executable,
            "-c",
            f"import pathlib; p = pathlib.Path(r'{out}') / 'shards' / '{i}of{cfg.shards}'; "
            "m = p / 'datasets' / 'hero_kitchen_sim' / 'meta'; m.mkdir(parents=True, exist_ok=True); "
            "(m / 'info.json').write_text('{}'); (m / 'episodes.jsonl').write_text('')",
        ]

    def kill(i):
        raise KeyboardInterrupt  # stop before the merge, which these stand-in shards cannot pass

    be = LocalBackend(tmp_path / "store", tmp_path / "scratch", fail_first={0}, command=command)
    with pytest.raises(KeyboardInterrupt):
        run_fanout(StatusRecorder(be), CFG1, tmp_path / "out", poll_seconds=0, on_shard_done=kill, log=quiet)
    rec = json.loads(be.get_object(status_key(CFG1, 0)))
    first, second = rec["attempts"]
    assert (first["job_id"], first["outcome"], first["failed_stage"]) == ("r2-0-1", "failed", "run")
    assert first["reason"].startswith("exit code 3, log ") and first["states"][-1]["state"] == "exited 3"
    assert (second["job_id"], second["outcome"], second["failed_stage"], second["reason"]) == (
        "r2-0-2",
        "succeeded",
        None,
        None,
    )
    assert second["states"][-1]["state"] == "exited 0" and second["resources"] is None and second["wall_seconds"] > 0

    class MarkerFails(LocalBackend):
        def put_object(self, key, data):
            if "/done/" in key:
                raise OSError("disk full, key do-not-store-me-123")
            super().put_object(key, data)

    monkeypatch.setenv("NEBIUS_SECRET_ACCESS_KEY", "do-not-store-me-123")
    be = MarkerFails(tmp_path / "store", tmp_path / "scratch", command=command)
    cfg = dataclasses.replace(CFG1, run_id="r5")
    with pytest.raises(OSError, match="disk full"):
        run_fanout(StatusRecorder(be), cfg, tmp_path / "out", poll_seconds=0, log=quiet)
    [marker] = json.loads(be.get_object(status_key(cfg, 0)))["attempts"]
    assert (marker["outcome"], marker["failed_stage"], marker["reason"]) == (
        "failed",
        "marker",
        "disk full, key [redacted]",
    )


def test_stopping_after_a_job_succeeded_is_not_recorded_as_a_cancel(nebius, tmp_path):
    class StopsInFetch(NebiusBackend):
        def fetch_result(self, job_id, prefix):
            raise KeyboardInterrupt

    nebius["make"](hold={0}, states={0: "COMPLETED"})
    rec = StatusRecorder(StopsInFetch())
    with pytest.raises(KeyboardInterrupt):
        run_fanout(rec, CFG1, tmp_path / "out", poll_seconds=0, log=quiet)
    assert rec.cancel_open() == [] and nebius["jobs"].cancelled == []
    [a] = _status(nebius["s3"], CFG1, 0)["attempts"]
    assert (a["outcome"], a["failed_stage"]) == ("failed", "fetch") and a["states"][-1]["state"] == "COMPLETED"


@needs_sim
@pytest.mark.render  # `reference` runs shards as subprocesses, where --no-render's patch does not reach
def test_status_record_survives_a_kill_and_rerun(nebius, reference, tmp_path):
    s3 = nebius["s3"]
    jobs = nebius["make"](hold={1})

    def kill(i):
        raise KeyboardInterrupt  # the orchestrator dies here, so nothing cancels shard 1's job

    with pytest.raises(KeyboardInterrupt):
        run_fanout(StatusRecorder(NebiusBackend()), CFG, tmp_path / "out", on_shard_done=kill, log=quiet)
    first = dict(jobs.created)
    done, left = _status(s3, CFG, 0), _status(s3, CFG, 1)
    [a] = done["attempts"]
    assert (a["job_id"], a["outcome"], a["failed_stage"]) == (first[0], "succeeded", None)
    states = [s["state"] for s in a["states"]]
    assert states[:2] == ["PROVISIONING", "IMAGE_PULLING"] and states[-1] == "COMPLETED"
    assert a["resources"]["preset"] == "4vcpu-16gb" and a["wall_seconds"] > 0
    assert [(b["job_id"], b["outcome"]) for b in left["attempts"]] == [(first[1], "running")]

    jobs = nebius["make"]()  # a new orchestrating process; the bucket is the same
    res = run_fanout(StatusRecorder(NebiusBackend()), CFG, tmp_path / "out", log=quiet)
    assert res["skipped"] == [0] and res["ran"] == [1]
    assert _status(s3, CFG, 0) == done
    after = _status(s3, CFG, 1)["attempts"]
    assert [(b["attempt"], b["job_id"], b["outcome"]) for b in after] == [
        (1, first[1], "abandoned"),
        (2, dict(jobs.created)[1], "succeeded"),
    ]
    _same_episodes(res, reference)

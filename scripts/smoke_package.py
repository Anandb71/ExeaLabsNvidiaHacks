"""Packaging smoke test: build the sdist and wheel, install the wheel, run it from outside the repo.

    python scripts/smoke_package.py            # installs dependencies from the package index
    python scripts/smoke_package.py --offline  # no index: reuse this interpreter's installed packages

Steps: build both archives with setuptools (no network, no build isolation); check what is inside
them; install the wheel into a fresh temporary venv; then, from an empty temp directory, import the
installed copy (not the checkout), run `arrival --help`, and run the offline `demo` against the
checkout's `scans/` with an explicit --scans, then `validate` the dataset it wrote.
The wheel ships only the `arrival` package. `scans/` and `configs/` live in the repo root, so an
installed copy needs --scans (the last check proves the failure without it is one clear line).
Prints one line per check, exit 0 when all pass and 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


# Paths travel in the environment: setuptools' build hooks reinterpret sys.argv.
BUILD_SNIPPET = (
    "import os; from setuptools import build_meta as b; d = os.environ['SMOKE_DIST']; "
    "os.chdir(os.environ['SMOKE_SRC']); s = b.build_sdist(d); w = b.build_wheel(d); "
    "print('BUILT', s, w)"
)


def build(dist: Path, python: str = sys.executable) -> tuple[Path, Path]:
    """(sdist, wheel) built with setuptools' PEP 517 hooks from the repo root: no network, no isolation.

    Runs in a child interpreter so `--build-python` can name any Python that has setuptools.
    """
    env = {**os.environ, "SMOKE_SRC": str(REPO), "SMOKE_DIST": str(dist)}
    r = subprocess.run([python, "-c", BUILD_SNIPPET], env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"build failed (does {python} have setuptools?):\n" + r.stderr[-800:])
    sdist, wheel = next(ln for ln in r.stdout.splitlines() if ln.startswith("BUILT ")).split()[1:]
    return dist / sdist, dist / wheel


def check_archives(sdist: Path, wheel: Path) -> list[str]:
    probs = []
    with tarfile.open(sdist) as t:
        names = t.getnames()
    for need in ("pyproject.toml", "README.md", "arrival/cli.py", "arrival/__init__.py"):
        if not any(n.endswith("/" + need) for n in names):
            probs.append(f"sdist is missing {need}")
    with zipfile.ZipFile(wheel) as z:
        wn = z.namelist()
        entry = next((n for n in wn if n.endswith("entry_points.txt")), None)
        eps = z.read(entry).decode() if entry else ""
    for need in ("arrival/cli.py", "arrival/__init__.py"):
        if need not in wn:
            probs.append(f"wheel is missing {need}")
    if "arrival = arrival.cli:main" not in eps:
        probs.append("wheel has no `arrival` console script")
    stray = [n for n in wn if n.split("/")[0] in ("tests", "scripts", "scans", "configs", "out")]
    if stray:
        probs.append(f"wheel contains repo-only paths: {stray[:3]}")
    return probs


def run(cmd: list[str], cwd: Path, env: dict | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def install(wheel: Path, work: Path, offline: bool) -> tuple[list[str], dict]:
    """(python command, env) for an interpreter that imports the installed wheel."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    if not offline:
        venv.EnvBuilder(with_pip=True, clear=True).create(work / "venv")
        py = work / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        r = run([str(py), "-m", "pip", "install", "--disable-pip-version-check", str(wheel)], work)
        if r.returncode:
            raise RuntimeError("pip install of the wheel failed:\n" + r.stderr[-800:])
        return [str(py)], env
    # offline: unpack only the wheel into a target dir and borrow this interpreter's packages for deps
    target = work / "site"
    with zipfile.ZipFile(wheel) as z:
        z.extractall(target)
    deps = [p for p in sys.path if p.endswith("site-packages") and Path(p).is_dir()]
    env["PYTHONPATH"] = os.pathsep.join([str(target)] + deps)
    return [sys.executable], env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--offline", action="store_true", help="do not create a venv or touch the package index")
    ap.add_argument("--build-python", default=sys.executable, help="interpreter with setuptools (default: this one)")
    ap.add_argument("--keep", help="build into this directory and keep it (default: a temp dir)")
    a = ap.parse_args(argv)
    failed = 0

    def report(ok: bool, what: str, detail: str = ""):
        nonlocal failed
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {what}" + (f": {detail}" if detail and not ok else ""), flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(a.keep or tmp)
        work.mkdir(parents=True, exist_ok=True)
        dist = work / "dist"
        dist.mkdir(exist_ok=True)
        sdist, wheel = build(dist, a.build_python)
        probs = check_archives(sdist, wheel)
        report(not probs, f"build {sdist.name} and {wheel.name}", "; ".join(probs))
        if probs:
            return 1
        py, env = install(wheel, work, a.offline)
        empty = work / "elsewhere"
        empty.mkdir(exist_ok=True)

        r = run(py + ["-c", "import arrival, sys; print(arrival.__file__)"], empty, env)
        where = Path(r.stdout.strip()).resolve() if r.returncode == 0 and r.stdout.strip() else None
        outside = where is not None and REPO not in where.parents
        report(outside, "import arrival from outside the checkout", f"{r.stdout.strip()} {r.stderr[-300:]}")

        r = run(py + ["-m", "arrival", "--help"], empty, env)
        report(r.returncode == 0 and "demo" in r.stdout, "python -m arrival --help", r.stderr[-300:])

        out = empty / "out"
        r = run(
            py
            + [
                "-m",
                "arrival",
                "demo",
                "--room",
                "hero_kitchen",
                "--scans",
                str(REPO / "scans"),
                "--episodes",
                "1",
                "--videos",
                "none",
                "--out",
                str(out),
            ],
            empty,
            env,
        )
        report(r.returncode == 0, "arrival demo with --scans", r.stderr[-400:] + r.stdout[-300:])
        ds = out / "datasets" / "hero_kitchen"
        r = run(py + ["-m", "arrival", "validate", str(ds)], empty, env)
        report(r.returncode == 0, "arrival validate on that dataset", r.stderr[-400:] + r.stdout[-300:])

        r = run(
            py
            + [
                "-m",
                "arrival",
                "demo",
                "--room",
                "hero_kitchen",
                "--episodes",
                "1",
                "--videos",
                "none",
                "--out",
                str(empty / "out2"),
            ],
            empty,
            env,
        )
        one_line = r.returncode != 0 and "Traceback" not in r.stderr and "not found under" in (r.stderr + r.stdout)
        report(one_line, "demo without --scans fails with one clear line (scans are not packaged)", r.stderr[-300:])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

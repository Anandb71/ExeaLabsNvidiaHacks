# Repo map

Create these paths. Do not create packages outside this list unless a task says to.

```
LICENSE                          Apache-2.0
README.md                        runner-facing, task T13
pyproject.toml
.gitignore                       includes out/
scans/hero_kitchen/scan.json
scans/hero_kitchen/roomplan.json
arrival/__init__.py
arrival/__main__.py              calls arrival.cli:main
arrival/cli.py
arrival/scan/bundle.py
arrival/scan/roomplan.py
arrival/scan/four_photo.py
arrival/twin/compile.py
arrival/catalog/categories.py
arrival/tasks/propose.py
arrival/generate/scripted.py
arrival/generate/mimicgen.py
arrival/render/sim_texture.py
arrival/render/splat.py
arrival/export/lerobot.py
arrival/export/validate.py
arrival/eval/cost.py
arrival/models/token_factory.py
arrival/jobs/queue.py
arrival/web/app.py
tests/test_roomplan.py
tests/test_compile.py
tests/test_tasks.py
tests/test_dataset.py
tests/test_web.py
```

`pyproject.toml`:

- Name `arrival-sim`, version `0.1.0`.
- Package is the `arrival` directory.
- Script is optional. `python -m arrival` must work via `__main__.py`.
- Dependencies listed in [01-decisions.md](01-decisions.md).
- Pytest `testpaths = ["tests"]`.

`.gitignore` includes `out/`, `.venv/`, `__pycache__/`, `*.pyc`, `.env`.

Since the MVP the repo also has `Makefile`, `Dockerfile`, `CONTRIBUTING.md`, `.github/` (CI workflow and Dependabot) and `scripts/` (demo, docs and packaging checks, scan intake); `.gitignore` grew to cover build output and editor files. See [REFERENCE.md](REFERENCE.md).

Each package directory needs `__init__.py`. Keep them empty.

## CLI

```
python -m arrival demo [--episodes N] [--videos all|sparse|none] [--room hero_kitchen]
python -m arrival validate <dataset_dir>
python -m arrival serve [--port 8000]
```

This is the original MVP surface. The CLI has since gained `sim`, `eval`, `work` and `research`, and `serve --host`; their flags are in [REFERENCE.md](REFERENCE.md).

Defaults: `--episodes 4`, `--videos all` (one mp4 per episode, which GR00T's loader needs). `--videos sparse` or `none` skips the per-episode videos. The 1,000-episode command in REFERENCE.md uses the default.

Exit codes: 0 success, 1 validation failure, 2 usage error.

## Sample data

`demo` reads `scans/<room>/scan.json` only. It does not read `roomplan.json` unless a flag `--from-roomplan` is passed. Implement that flag. It normalizes `scans/<room>/roomplan.json` and then compiles. The hero `roomplan.json` is incomplete relative to `scan.json`, so `--from-roomplan` on the hero room is allowed to produce a smaller twin. Tests cover the sink coordinate. The done command does not use the flag.

## Secrets

Read environment variables only inside `arrival.models.token_factory`:

| Variable | Purpose |
|---|---|
| `TOKEN_FACTORY_API_KEY` | Bearer token. If unset, `complete()` raises `TokenFactoryNotConfigured` |
| `TOKEN_FACTORY_BASE_URL` | Default `https://api.tokenfactory.nebius.com/v1`. Confirm against the live docs before relying on it |
| `NEMOTRON_OMNI_MODEL` | Optional model id |
| `NEMOTRON_ULTRA_MODEL` | Optional model id |
| `NEMOTRON_SUPER_MODEL` | Optional model id |
| `NEMOTRON_LIGHTNING_MODEL` | Optional model id |

No key is committed. No `.env` is committed. The demo path does not read these variables.

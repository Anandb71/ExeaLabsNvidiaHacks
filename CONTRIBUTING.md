# Contributing

The team plan, task IDs and the definition of done for a PR are in the README
([Join in](README.md#join-in-team-plan-for-30-people)). This page is the practical part: how to run
the checks CI runs, before you push.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev,sim,lint]"             # or: make install
```

Python 3.11 or 3.12. No GPU and no API keys are needed for anything below.

## Before you push

| Check | Command | What it covers |
|---|---|---|
| Tests | `make test` | `python -m pytest -q -p no:cacheprovider` |
| Offline demo | `make demo` | scan, twin, sim dataset, validate, eval, contact sheet, `out/index.html`; about 20 s |
| Lint | `make lint` | `ruff check` (bugs, sorted imports, no unused imports or variables) and `ruff format --check`; `make format` fixes both |
| Docs | `make docs-check` | `scripts/check_docs.py` (commands parse, paths and labels agree) and `scripts/check_links.py` (relative links and anchors in every Markdown file) |
| Packaging | `make package-smoke` | builds the sdist and wheel and runs the installed wheel from another directory |

CI runs all of them on every PR (`.github/workflows/ci.yml`). Ubuntu is the gate; the macOS and
Windows rows are reported but do not block a merge, because hosted runners have no GPU.

## Rules that are easy to break

- **Honest labels.** `arrival_physically_valid`, `arrival_generator` and `joints_are_placeholder` must
  describe what really happened. Scripted data is never "MimicGen", "trained" or "real-robot".
- **Offline.** Tests and `make demo` make no network calls and need no keys. A test that needs a
  network service must skip itself, and a new client must have a mock-server test.
- **Memory.** Keep tests small: 64 px renders, 2 episodes or fewer, 2 people or fewer.
- **Docs.** One paragraph per change in `docs/BUILD-LOG.md` (blank line between paragraphs), and any new
  command or flag in `docs/REFERENCE.md`. Link to files that exist: `make docs-check` fails otherwise.
  Files that are planned but not written yet belong under a `## ... to-do` heading in the README, where
  `check_docs.py` does not demand that they exist.
- **No secrets or personal data.** Never commit `.env`, `out/`, or anything from a real scan of a
  person's home. `.gitignore` covers the usual places; check `git status` before you commit.
- **Do not edit another person's open PR or branch.** To build on it, cherry-pick its commits onto your
  own branch and credit it in your PR body.

## Opening a PR

Branch from `main` (`feat/<TASK-ID>-short-name`), keep it small, and fill in What, Why, How to test and
the test output. Attach a before/after image for anything visual (`--save-preview` writes one).

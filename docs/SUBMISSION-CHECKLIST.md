# What is left before submission, as of 2026-10-28

Deadline: **Friday, October 30, 2026, 10:00am Pacific** (https://nebiusglobalaihackathon.devpost.com/). Every item below was checked against the [official rules](https://nebiusglobalaihackathon.devpost.com/rules) on Oct 25. Judging is Dec 1 to 15. After the deadline the submission cannot be edited, and the project has to stay free to test until judging ends.

`main` has 415 merged pull requests, through #478. Open: #101, which ignores `site/` (close it), and #472. The to-do list is `site/data/todo.json`; run `python scripts/render_todo_readme.py` after editing it.

Setup note: this machine has a stray `tests` package in `~/miniconda3/lib/python3.13/site-packages`. Use a venv (`python -m venv .venv && .venv/bin/pip install -e ".[dev,sim]"`) before running the suite.

## 1. Blocks Stage One (pass/fail). Do these first.

1. **A real runtime call on Nebius (C6): done Oct 26.** Nemotron Super and Lightning answered live through Token Factory, and so did Tavily (C7). The evidence is `docs/results/live-apis-2026-10-26.json` and `docs/results/planner-benchmark-2026-10-26.json`. Rerun `python scripts/smoke_live_apis.py` on the day if anything changes.
2. **The video (P4).** It must be on public YouTube, under 3 minutes, and for Physical AI with no hardware it must have **at least one minute of the key modules in action**. `site/motion/pitch.mp4` is 24.4 s, which is not enough. Record the screen capture in `docs/PITCH-SCRIPT.md` (about 2:30). Show the live Nemotron call from step 1. Use no copyrighted music, and no third-party trademarks without permission (keep logos off screen).
3. **Unlicensed data (P5).** Done: replaced unlicensed `amal.json` and `shopyseek.json` with clean authored fixtures `office.json` and `study.json` under Apache 2.0.
4. **Consent for the real kitchen.** `kitchen_real_01.json` is a teammate's real kitchen. Confirm the owner agreed to publish it. It holds geometry only, with no frames, but say so in the fixture README.

## 2. The form (SUB)

Text to paste is in `docs/DEVPOST-SUBMISSION.md`; fill or delete every **[bracket]**.

- Track: **Physical AI**.
- Description: what it is, why, and how it works.
- Public repo URL: https://github.com/AvnehSBhatia/ExeaLabsNvidiaHacks. The Apache 2.0 licence is detected and shows in About. Also set a real About description and the website: `gh repo edit --description "Scan a room once, get GR00T robot training data for that exact room" --homepage https://arrival.exealabs.org` (it currently says "For the Exea team :D").
- README with setup steps and where NVIDIA models and Nebius services are used: done ("NVIDIA and Nebius in Arrival"). Its one "Not yet" cell, Nebius Serverless Jobs (C1), is accurate; change it only if C1 runs.
- YouTube link (step 2).
- Feedback on Token Factory, AI Cloud and the NVIDIA models: `docs/DEVPOST-FEEDBACK.md`. Two bracketed lines are still unfilled: the Cosmos line and the Token Factory sign-up time and price; fill or delete them. This feedback is also judged for the Most Valuable Feedback prize (10 × $100).
- Demo URL: optional on this track. https://arrival.exealabs.org is live (the static site). Redeploy it after the last merge with `cd site && vercel --prod`: the live page is behind `main` (it still shows the Oct 19 text).
- "Existed before the submission period?": no. The first commit is Oct 12, 2026, after Aug 26.
- City: if anyone attended a Builders & Brews event, pick that city (City Winner, $500).
- One person submits as the team's Representative. Every teammate needs to be at least the age of majority, and not resident where the rules exclude. Teammates should join the Devpost project.

## 3. Raises the score (equal weights: technology, design, impact, idea)

- **C1, one real Serverless Jobs run.** The Physical AI track text asks for "Nebius Serverless Jobs to run simulations, generate synthetic data, evaluate robot policies". One small `arrival fanout --backend nebius --shards 2` run is the strongest single proof for the Nebius judges. It needs `NEBIUS_IAM_TOKEN`, `NEBIUS_PROJECT_ID`, a job image, a bucket and keys (`docs/DEPLOY.md`, `.env.example`). Put the job ids and the time in the README.
- **C7, a real Tavily call** for Best Use of Tavily ($3,000; it needs "a functional, runtime call to the Tavily API"): done Oct 26 (step 1's evidence files). The prize rules allow one Overall *or* Track award *plus* one Bonus award.
- **C3, measured cost per demo** against the 20 teleop demos: time is measured (15.2 s per retargeted demo on one CPU core against 61.5 operator seconds, #439); dollars still need real cloud usage from C1: `python -m arrival cost`.
- Close done issues: #37 and #63 (G1), #74 and #79 (fridge work), #62 (#439) and #69 (#418).

## 4. Not needed for submission (say "not done" if asked)

Done since this list was first written, so do not call them not done: G2 (real GR00T server, Oct 21), G6 (RoboCasa, #418), C4 (VLM detection on a real call, Oct 26), and Cosmos Transfer on three episodes (C5, Oct 27). Still not done: G3 (three-cell eval), G4 (generic-RoboCasa baseline), G5 (GPU image never run), upstream MimicGen in G7 (the 104-episode retarget benchmark did run, #439), G8 (splats), C5's run over a whole dataset, P2 (three measured scans). The README says which are which; the video and the form should match it.

## 5. Last checks on the day

- `make install && make test` on a fresh clone. The suite takes about 20 minutes on one core.
- `python scripts/check_docs.py`, `python scripts/check_links.py` and `python scripts/render_todo_readme.py --check` all exit 0.
- Everything the video and the description claim is reproducible from the README ("must function as depicted in the video").
- Nothing in the repo, the video or the form contains a key, a teammate's address or a face.

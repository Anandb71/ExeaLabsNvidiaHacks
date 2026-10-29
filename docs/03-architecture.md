# Architecture

One scan becomes a dataset. Tonight every box below exists as a Python module. Only the solid path runs.

```
scan bundle
  -> normalize (RoomPlan JSON or four-photo slot)
  -> twin (fixture boxes + RoboCasa-subset YAML)
  -> tasks (two families, four variation axes)
  -> seeds + retarget
  -> oracle (drop failures)
  -> sim_texture frames
  -> GR00T LeRobot v2
  -> local page + queued upload
```

On the MVP night the later boxes were only modules that raised a named error or returned a skipped probe: Token Factory calls, splat composite, MimicGen, Cosmos, GR00T fine-tune, Serverless fan-out. Since then the Token Factory calls run live, the splat compositor, the Cosmos client and the Serverless fan-out are built, and GR00T was fine-tuned off Nebius; MimicGen is still only a probe. The Nebius mapping below says how far each was run.

## Modules

| Module | Responsibility |
|---|---|
| `arrival.scan.bundle` | Load and validate a ScanBundle |
| `arrival.scan.roomplan` | CapturedRoom JSON → ScanBundle, Y-up to Z-up |
| `arrival.scan.four_photo` | Four images → ScanBundle flagged unmeasured |
| `arrival.twin.compile` | ScanBundle → layout dict + YAML |
| `arrival.twin.detect` | VLM-based object detection in scan frames |
| `arrival.catalog.categories` | RoomPlan category → RoboCasa fixture type or `box`, default sizes; object label → catalog id and physical spec |
| `arrival.tasks.propose` | Tasks, variations, wordings. No network |
| `arrival.generate.scripted` | Seed trajectories and pose retargets |
| `arrival.generate.scripted` (`judge`) | Drop attempts that miss the goal |
| `arrival.generate.mimicgen` | Import probe. Checks if RoboCasa is installed; wiring not yet implemented |
| `arrival.render.sim_texture` | 64×64 RGB frames from episode cartesian traces |
| `arrival.render.splat` | Depth-based composite of static splat RGB-D and dynamic sim RGB-D |
| `arrival.export.lerobot` | Parquet, mp4, meta |
| `arrival.export.validate` | Structural checker. Does not import GR00T |
| `arrival.eval.cost` | Speed and cost report |
| `arrival.models.token_factory` | OpenAI-compatible client. Used for VLM planning and wordings |
| `arrival.jobs.queue` | Write `out/jobs/<id>/status.json` |
| `arrival.web.app` | `ThreadingHTTPServer` routes |
| `arrival.cli` | `demo`, `validate`, `serve`, `sim`, `retarget`, `research` |

## Data flow for `arrival demo`

1. Load `scans/hero_kitchen/scan.json`.
2. Compile `out/twins/hero_kitchen/layout.yaml` and `twin.json`.
3. Propose tasks to `out/twins/hero_kitchen/tasks.json`.
4. Generate attempts. Every 10th attempt (indexes 9, 19, …) is a miss: the gripper stays open. The oracle drops misses.
5. Keep going until `successes == --episodes` or attempts hit `episodes * 2`, whichever first. Attempt indexes 9, 19, 29, … are misses. Reaching 1,000 successes takes 1,111 attempts and 111 drops. Use RNG seed 0.
6. Render frames for episodes included by the video policy.
7. Write the LeRobot folder and `out/reports/hero_kitchen.json`.

Video policy:

- `--videos all`: an mp4 for every successful episode.
- `--videos sparse` (required for the 1,000 run): an mp4 when `episode_index % 125 == 0`, so episodes 0, 125, …, 875 (8 files).
- `--videos none`: no mp4. Validator then requires `info.json` field `arrival_video_policy = "none"`. The done command does not use this.

## Composite renderer (depth rule implemented, full pipeline not run)

`arrival.render.splat.render` implements the depth selection rule below for
aligned static and dynamic RGB-D layers. A trained Gaussian splat, camera-pose
rendering, and a dynamic-only simulator pass still need to supply those layers
before an episode can use the composite. No dataset is labelled as splat output.

For each robot camera pose at each frame:

- Splat supplies RGB and depth for the static room.
- The simulator supplies RGB and depth for the robot, the grasped object, and any moving fixture. Its background is empty.
- Output pixel is the simulator pixel where simulator depth is closer than splat depth by more than `0.01` m. Otherwise the splat pixel.

`sim_texture` draws the whole frame and records `appearance = "sim_texture"`.

## Web

Single process. No authentication. Bind `127.0.0.1` only. Uploaded bytes are stored under `out/jobs/<id>/upload.bin` with a max size of 200 MB. Oversize returns HTTP 413. The status JSON is the whole API. There is no worker loop that drains the queue tonight.

## Nebius mapping

The spec assigns products. On the MVP night none of them were called; the last column says where each step stands now. Token Factory is the one Nebius product called live.

| Step | Product | MVP night | Now (Oct 27) |
|---|---|---|---|
| Read the scan | Token Factory, Nemotron 3 Nano Omni | Deterministic catalog | Walls and fixtures: the catalog. Small objects in scan frames: `arrival/twin/detect.py` with the omni model (`NEMOTRON_OMNI_MODEL`), live |
| Assemble the twin, fill gaps | Token Factory, Nemotron 3 Ultra | Compiler only, no gap fill | Unchanged: the ultra role is configured but never called |
| Propose tasks | Token Factory, Nemotron 3 Super | Templates | Templates propose; the super model plans each task from a request and the room inventory (`sim --planner vlm`, `VLMPlanner`), live |
| Wordings | Nemotron 3.5 Lightning | Four templates | Train and held-out wordings from the lightning model (`--instructions nemotron`, `arrival/tasks/instructions.py`), live; templates offline |
| People and episode labels | Nemotron 3.5 Lightning | Not built | People are scripted walkers (`arrival/sim/people.py`), no model; labels come from the physics check |
| Build the twin and appearance | Serverless AI Job, 1 GPU | Local compiler + sim_texture | Local |
| Generate demos | Serverless AI Jobs, many | Local scripted retarget | Local or local shards; `fanout --backend nebius` is built and tested against in-process fakes, not run on Nebius |
| Restyle frames | Serverless Endpoint, Cosmos 3 | Not built | Client built (`arrival/render/variation.py`, SIM-7a), run only against a local mock server |
| Fine-tune | Serverless AI Job, H100, GR00T N1.7 | Not built | GR00T N1.7-3B fine-tunes ran on one 96 GB GPU on Cosmos, not Nebius (`docs/07-eval.md`) |
| Store artifacts | Object Storage | Local `out/` | Local `out/`; the Nebius fan-out's bucket code is tested against a fake |
| Host the app | Serverless Endpoint, CPU | Local `serve` | Static demo site hosted at https://arrival.exealabs.org; the upload-to-dataset app runs locally |

## Coordinates

Internal frame is Z-up, meters, right-handed. Positions are box centers. Yaw is radians about Z. RoomPlan input is ARKit Y-up. Conversion is in [04-contracts.md](04-contracts.md) and is tested.

"""Build a twin's layout.yaml with RoboCasa's own scene builder and compare the room it makes with the twin.

Run it with an interpreter that has RoboCasa (github.com/robocasa/robocasa, 1.0.1) and robosuite (master) installed,
and RoboCasa's texture and fixture assets downloaded (`download_kitchen_assets --type tex fixtures_lw`):

    MUJOCO_GL=osmesa <robocasa-venv>/bin/python scripts/check_robocasa_layout.py out/twins/hero_kitchen \
        out/twins/kitchen_real_01 [--style 1] [--json report.json] [--save-layouts DIR]

Each folder needs the twin.json and layout.yaml that `python -m arrival demo` writes. For each twin it builds two
layouts through RoboCasa's `KitchenArena` (which runs `scene_builder.create_fixtures`) and compiles the scene to a
MuJoCo model with robosuite's `ManipulationTask`, with no robot:

- `as_written`: layout.yaml exactly as the compiler writes it. Reported, never fails the run.
- `robocasa`: `arrival.twin.robocasa_layout.robocasa_layout(layout.yaml)`, built once to read each fixture's
  origin offset and the counter overhang RoboCasa chose for the style, then rebuilt with them.

For every twin fixture it reports what RoboCasa built: the centre and size of RoboCasa's own exterior box of the
fixture (`get_ext_sites`, its "main" region, the box RoboCasa aligns and places by) against the twin's centre and
size, and the yaw (RoboCasa faces -y at z_rot 0, the twin faces +y at yaw 0); the collision geometry's bounding box is
in the JSON too (a stove's includes its backsplash). Walls and the floor are compared by their collision box. Overlaps are MuJoCo contacts deeper than 1 cm between two different fixtures or walls, listed next to the
overlaps the twin's own boxes already have. Exit 1 when the `robocasa` layout does not build, a twin fixture or wall
is missing, a fixture centre is more than --tol (5 cm) off horizontally or vertically (a sink's height is set by
RoboCasa, flush with its counter top, and is reported, not checked), a yaw is wrong, or RoboCasa adds an overlap
the twin does not have. Exit 0 with a SKIP line when RoboCasa is not importable (use --require to make that a 1).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrival.twin.robocasa_layout import FRONT_FLIP, robocasa_layout  # noqa: E402

OVERLAP = 0.01  # a contact or box intersection deeper than this is an overlap


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------- the twin's own boxes ----------


def twin_boxes(twin: dict) -> dict:
    """name -> {kind, center, size (in its own frame), yaw} for every fixture and wall of the twin."""
    out = {}
    for f in twin["fixtures"]:
        out[f["id"]] = {
            "kind": f["type"],
            "center": list(f["position"]),
            "size": list(f["size"]),
            "yaw": float(f.get("yaw") or 0.0),
        }
    for w in twin.get("walls", []):
        (x0, y0), (x1, y1) = w["start"], w["end"]
        out[w["id"]] = {
            "kind": "wall",
            "center": [(x0 + x1) / 2, (y0 + y1) / 2, w["height"] / 2],
            "size": [math.hypot(x1 - x0, y1 - y0), w["thickness"], w["height"]],
            "yaw": math.atan2(y1 - y0, x1 - x0),
        }
    return out


def box_penetration(a: dict, b: dict) -> float:
    """Depth two yawed boxes interpenetrate (separating axes in the plane, plus z); <= 0 when apart."""
    dz = (a["size"][2] + b["size"][2]) / 2 - abs(a["center"][2] - b["center"][2])
    depth = dz
    for yaw in (a["yaw"], a["yaw"] + math.pi / 2, b["yaw"], b["yaw"] + math.pi / 2):
        ax = (math.cos(yaw), math.sin(yaw))

        def half(box):
            c, s = math.cos(box["yaw"] - yaw), math.sin(box["yaw"] - yaw)
            return abs(c) * box["size"][0] / 2 + abs(s) * box["size"][1] / 2

        d = abs((a["center"][0] - b["center"][0]) * ax[0] + (a["center"][1] - b["center"][1]) * ax[1])
        depth = min(depth, half(a) + half(b) - d)
    return depth


def twin_overlaps(boxes: dict) -> dict:
    names = sorted(boxes)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if boxes[a]["kind"] == boxes[b]["kind"] == "wall":
                continue  # walls meet at corners by construction
            d = box_penetration(boxes[a], boxes[b])
            if d > OVERLAP:
                out[tuple(sorted((a, b)))] = round(d, 4)
    return out


# ---------- RoboCasa ----------


def build(doc: dict, style: int, seed: int):
    """RoboCasa's KitchenArena (scene_builder.create_fixtures) -> robosuite ManipulationTask -> MuJoCo model."""
    import mujoco
    import numpy as np
    from robocasa.models.scenes.kitchen_arena import KitchenArena
    from robosuite.models.tasks import ManipulationTask

    arena = KitchenArena(layout_id=copy.deepcopy(doc), style_id=style, rng=np.random.default_rng(seed))
    arena.set_origin([0, 0, 0])
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[], mujoco_objects=list(arena.fixtures.values()))
    model = task.get_model(mode="mujoco")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return arena, model, data


def fixture_names(doc: dict) -> dict:
    """RoboCasa's fixture key (name + '_' + group) -> the twin name, for every row the layout has."""
    out = {}
    for group, g in doc.items():
        for key, rows in g.items():
            if not key.startswith("group_"):
                for r in rows:
                    out[f"{r['name']}_{group}"] = r["name"]
    return out


def measure(arena, model, data, doc: dict) -> tuple[dict, dict, dict]:
    """(twin name -> what RoboCasa built, geom id -> twin name, extras) from the compiled model."""
    import mujoco
    import numpy as np

    names = fixture_names(doc)
    built, owner = {}, {}
    for key, fx in arena.fixtures.items():
        name = names.get(key)
        if name is None:
            continue
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fx.root_body)
        sub = {root}
        for b in range(root + 1, model.nbody):  # MuJoCo numbers a subtree after its root
            if int(model.body_parentid[b]) in sub:
                sub.add(b)
        geoms = [
            g
            for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in sub
            and (model.geom_contype[g] or model.geom_conaffinity[g])
            and model.geom_rgba[g][3] > 0
        ]
        # (alpha 0 drops the parts RoboCasa parks out of sight, e.g. a counter's unused base panels at z + 10 m)
        for g in geoms:
            owner[g] = name
        pts = []
        for g in geoms:
            c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        pts.append(data.geom_xpos[g] + data.geom_xmat[g].reshape(3, 3) @ (c + h * [sx, sy, sz]))
        pts = np.array(pts)
        lo, hi = pts.min(0), pts.max(0)
        rot = float(fx.rot) if hasattr(fx, "rot") else 0.0
        body = np.array(data.xpos[root])
        ext = ext_size = None
        if hasattr(fx, "get_ext_sites") and type(fx).__name__ not in ("Wall", "Floor", "Box"):
            # RoboCasa's own exterior box of the fixture (its "main" region), the box it aligns and places with
            e = np.array(fx.get_ext_sites(all_points=True, relative=False))
            ext = ((e.min(0) + e.max(0)) / 2).tolist()
            r_ = np.array(fx.get_ext_sites(all_points=True, relative=True))
            ext_size = (r_.max(0) - r_.min(0)).tolist()
        c, s = math.cos(rot), math.sin(rot)
        world = hi - lo
        built[name] = {
            "class": type(fx).__name__,
            "aabb_center": ((lo + hi) / 2).tolist(),
            "aabb_size": world.tolist(),
            "center": ext if ext is not None else ((lo + hi) / 2).tolist(),
            "ext_size": ext_size,
            "rot": rot,
            "body": body.tolist(),
            "origin_offset": [float(x) for x in getattr(fx, "origin_offset", (0, 0, 0))],
            "overhang": getattr(fx, "overhang", None),
            # the AABB's extent along the fixture's own x and y (exact for multiples of 90 degrees)
            "size": [abs(c) * world[0] + abs(s) * world[1], abs(s) * world[0] + abs(c) * world[1], float(world[2])]
            if abs(round(s * c, 6)) < 1e-6
            else None,
        }
    return built, owner, {"nbody": int(model.nbody), "ngeom": int(model.ngeom), "ncon": int(data.ncon)}


def contact_overlaps(model, data, owner: dict) -> dict:
    out: dict = {}
    for i in range(data.ncon):
        con = data.contact[i]
        a, b = owner.get(int(con.geom1)), owner.get(int(con.geom2))
        if a is None or b is None or a == b or con.dist > -OVERLAP:
            continue
        k = tuple(sorted((a, b)))
        out[k] = round(max(out.get(k, 0.0), -float(con.dist)), 4)
    return out


def compare(twin: dict, built: dict, overlaps: dict, sinks_in_counter: set, tol: float, floor: dict | None) -> dict:
    boxes = twin_boxes(twin)
    rows, errors = [], []
    for name, tb in boxes.items():
        b = built.get(name)
        if b is None:
            errors.append(f"{name}: in the twin, not built by RoboCasa")
            rows.append({"name": name, "kind": tb["kind"], "built": None})
            continue
        d = [b["center"][i] - tb["center"][i] for i in range(3)]
        dxy, dz = math.hypot(d[0], d[1]), abs(d[2])
        row = {
            "name": name,
            "kind": tb["kind"],
            "class": b["class"],
            "twin_center": [round(x, 4) for x in tb["center"]],
            "built_center": [round(x, 4) for x in b["center"]],
            "dxy": round(dxy, 4),
            "dz": round(dz, 4),
        }
        if tb["kind"] == "wall":
            ext = sorted(b["aabb_size"][:2])
            want = sorted(
                [
                    abs(math.cos(tb["yaw"])) * tb["size"][0] + abs(math.sin(tb["yaw"])) * tb["size"][1],
                    abs(math.sin(tb["yaw"])) * tb["size"][0] + abs(math.cos(tb["yaw"])) * tb["size"][1],
                ]
            )
            row["size_err"] = round(
                max(abs(ext[0] - want[0]), abs(ext[1] - want[1]), abs(b["aabb_size"][2] - tb["size"][2])), 4
            )
            if dxy > tol or dz > tol or row["size_err"] > tol:
                errors.append(
                    f"wall {name}: centre off {dxy:.3f} m (xy) {dz:.3f} m (z), size off {row['size_err']:.3f} m"
                )
        else:
            want = _wrap(tb["yaw"] + FRONT_FLIP)
            row["yaw_err"] = round(abs(_wrap(b["rot"] - want)), 4)
            size = b["ext_size"] or b["size"]
            if size is not None:
                row["size_err"] = round(max(abs(size[i] - tb["size"][i]) for i in range(3)), 4)
            z_checked = name not in sinks_in_counter
            row["z_checked"] = z_checked
            row["collision_center"] = [round(x, 4) for x in b["aabb_center"]]
            row["collision_size"] = [round(x, 4) for x in (b["size"] or b["aabb_size"])]
            if dxy > tol or (z_checked and dz > tol):
                errors.append(f"{name} ({b['class']}): centre off {dxy:.3f} m (xy) {dz:.3f} m (z), tol {tol}")
            if row["yaw_err"] > 1e-3:
                errors.append(f"{name} ({b['class']}): faces {row['yaw_err']:.3f} rad off the twin's front")
        rows.append(row)
    twin_ov = twin_overlaps(boxes)
    new = {f"{a} / {b}": d for (a, b), d in overlaps.items() if (a, b) not in twin_ov}
    for k, d in new.items():
        errors.append(f"overlap RoboCasa added: {k} ({d:.3f} m deep)")
    floor_row = None
    if floor is not None:
        floor_row = {
            "top_z": round(floor["aabb_center"][2] + floor["aabb_size"][2] / 2, 4),
            "size_xy": [round(x, 4) for x in floor["aabb_size"][:2]],
        }
    return {
        "fixtures": rows,
        "errors": errors,
        "floor": floor_row,
        "overlaps_robocasa": {f"{a} / {b}": d for (a, b), d in overlaps.items()},
        "overlaps_twin": {f"{a} / {b}": d for (a, b), d in twin_ov.items()},
        "overlaps_added": new,
    }


def check_twin(twin_dir: Path, style: int, seed: int, tol: float, save: Path | None) -> dict:
    twin = json.loads((twin_dir / "twin.json").read_text())
    layout = yaml.safe_load((twin_dir / "layout.yaml").read_text())
    room = twin_dir.name
    res: dict = {"room": room, "twin_fixtures": len(twin["fixtures"]), "twin_walls": len(twin.get("walls", []))}

    # 1. layout.yaml as the compiler writes it
    try:
        arena, model, data = build(layout, style, seed)
        built, owner, info = measure(arena, model, data, layout)
        cmp_ = compare(twin, built, contact_overlaps(model, data, owner), set(), tol, built.get("floor"))
        res["as_written"] = {"built": True, "robocasa_fixtures": len(arena.fixtures), **info, **cmp_}
    except Exception as e:  # noqa: BLE001 - this half only reports
        res["as_written"] = {"built": False, "error": f"{type(e).__name__}: {e}"}

    # 2. the RoboCasa-native layout: build, read origin offsets and overhang, rebuild with them
    doc, notes = robocasa_layout(layout)
    sinks = {
        r["interior_obj"]
        for g in doc.values()
        for k, rows in g.items()
        if not k.startswith("group_")
        for r in rows
        if r.get("interior_obj")
    }
    try:
        arena, model, data = build(doc, style, seed)
        first, _, _ = measure(arena, model, data, doc)
        offsets = {n: b["origin_offset"] for n, b in first.items() if b["class"] not in ("Wall", "Floor", "Box")}
        overhang = next((b["overhang"] for b in first.values() if b["overhang"] is not None), 0.05)
        doc, notes = robocasa_layout(layout, origin_offsets=offsets, overhang=overhang)
        arena, model, data = build(doc, style, seed)
        built, owner, info = measure(arena, model, data, doc)
        cmp_ = compare(twin, built, contact_overlaps(model, data, owner), sinks, tol, built.get("floor"))
        res["robocasa"] = {
            "built": True,
            "robocasa_fixtures": len(arena.fixtures),
            **info,
            "notes": notes,
            "origin_offsets": {k: [round(x, 4) for x in v] for k, v in offsets.items()},
            "overhang": overhang,
            **cmp_,
        }
        if save is not None:
            save.mkdir(parents=True, exist_ok=True)
            (save / f"{room}.robocasa.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))
    except Exception as e:  # noqa: BLE001
        res["robocasa"] = {
            "built": False,
            "error": f"{type(e).__name__}: {e}",
            "notes": notes,
            "errors": [f"RoboCasa could not build it: {type(e).__name__}: {e}"],
        }
    res["ok"] = bool(res["robocasa"]["built"] and not res["robocasa"]["errors"])
    return res


def _print(res: dict, tol: float) -> None:
    print(f"== {res['room']}: twin has {res['twin_fixtures']} fixtures, {res['twin_walls']} walls")
    for part in ("as_written", "robocasa"):
        r = res[part]
        print(f"-- {part}: {'built' if r['built'] else 'NOT BUILT: ' + r.get('error', '')}")
        if not r["built"]:
            continue
        print(
            f"   RoboCasa fixtures (walls and floor included): {r['robocasa_fixtures']}; MuJoCo bodies "
            f"{r['nbody']}, geoms {r['ngeom']}, contacts {r['ncon']}"
        )
        if r["floor"]:
            print(f"   floor: top z {r['floor']['top_z']}, {r['floor']['size_xy'][0]} x {r['floor']['size_xy'][1]} m")
        for f in r["fixtures"]:
            if f.get("class") is None:
                print(f"   {f['name']:40.40s} MISSING")
                continue
            extra = f" yaw_err {f['yaw_err']:.3f}" if "yaw_err" in f else ""
            extra += "" if f.get("z_checked", True) else " (z set by RoboCasa)"
            print(
                f"   {f['name'][:40]:40s} {f['class']:20s} dxy {f['dxy']:.3f} dz {f['dz']:.3f} "
                f"size_err {f.get('size_err', float('nan')):.3f}{extra}"
            )
        print(f"   overlaps > {OVERLAP} m in RoboCasa: {r['overlaps_robocasa'] or 'none'}")
        print(f"   overlaps > {OVERLAP} m already in the twin: {r['overlaps_twin'] or 'none'}")
        for n in r.get("notes", []):
            print("   note:", n)
        for e in r["errors"]:
            print("   ERROR" if part == "robocasa" else "   (as written)", e)
    print(f"   => {'PASS' if res['ok'] else 'FAIL'} (robocasa layout, tol {tol} m)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("twin_dirs", nargs="+", type=Path, help="folders with twin.json and layout.yaml")
    p.add_argument("--style", type=int, default=1, help="RoboCasa style id (textures and fixture models)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=0.05, help="max centre error in metres")
    p.add_argument("--json", type=Path, help="also write the full report here")
    p.add_argument("--save-layouts", type=Path, help="write each <room>.robocasa.yaml that was built here")
    p.add_argument("--require", action="store_true", help="exit 1 instead of skipping when RoboCasa is missing")
    a = p.parse_args(argv)
    try:
        import mujoco  # noqa: F401
        import robocasa  # noqa: F401
        import robosuite
    except ImportError as e:
        print(
            f"SKIP: RoboCasa is not importable here ({e}); nothing was checked. Install robosuite (master) and "
            "robocasa from their repos, or run with their interpreter."
        )
        return 1 if a.require else 0
    import robocasa as rc

    print(
        f"robocasa {getattr(rc, '__version__', '?')} ({Path(rc.__file__).parent}), robosuite "
        f"{robosuite.__version__}, mujoco {mujoco.__version__}, style {a.style}, seed {a.seed}"
    )
    results = [check_twin(d, a.style, a.seed, a.tol, a.save_layouts) for d in a.twin_dirs]
    for res in results:
        _print(res, a.tol)
    if a.json:
        a.json.write_text(json.dumps(results, indent=2))
    ok = all(r["ok"] for r in results)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""What scanned categories and object labels become in the twin and the sim.

Fixtures: RoomPlan / scan category -> RoboCasa fixture type, plus a default size per category.
Objects: label -> catalog id, and the physical spec (shape, nominal size, mass, friction) behind each id.
"""

from __future__ import annotations

import random

# Fixtures

# Every case of Apple's CapturedRoom.Object.Category (iOS 16 to 17), snake-cased, plus our own `counter`.
# RoboCasa has no bed, sofa or bathtub; those stay a plain `box` so the scan is not silently dropped.
FIXTURE_MAP = {
    "counter": "counter",
    "sink": "sink",
    "refrigerator": "fridge_bottom_freezer",
    "stove": "stove",
    "oven": "stove",
    "dishwasher": "dishwasher",
    "storage": "hinge_cabinet",
    "table": "counter",
    "microwave": "microwave",
    "washer_dryer": "box",
    "chair": "box",
    "sofa": "box",
    "bed": "box",
    "bathtub": "box",
    "toilet": "box",
    "television": "box",
    "fireplace": "box",
    "stairs": "box",
}

# Full extents [x, y, z] in metres, Z-up, used when a scan object has no usable dimensions.
DEFAULT_SIZES = {
    "counter": [1.8, 0.65, 0.92],
    "sink": [0.6, 0.5, 0.25],
    "refrigerator": [0.75, 0.7, 1.8],
    "stove": [0.7, 0.65, 0.9],
    "oven": [0.6, 0.6, 0.6],
    "dishwasher": [0.6, 0.6, 0.85],
    "storage": [0.9, 0.5, 2.0],
    "table": [1.4, 0.8, 0.75],
    "microwave": [0.5, 0.4, 0.3],
    "washer_dryer": [0.6, 0.6, 0.85],
    "chair": [0.5, 0.5, 0.9],
    "sofa": [2.0, 0.9, 0.85],
    "bed": [1.6, 2.0, 0.5],
    "bathtub": [1.7, 0.75, 0.55],
    "toilet": [0.4, 0.7, 0.8],
    "television": [1.2, 0.1, 0.7],
    "fireplace": [1.2, 0.4, 1.0],
    "stairs": [1.0, 3.0, 2.7],
}
FALLBACK_SIZE = [0.5, 0.5, 0.5]


def fixture_type(category: str) -> str:
    return FIXTURE_MAP.get(category, "box")


def default_config_name(category: str) -> str | None:
    return "dining" if category == "table" else None


def default_size(category: str) -> list[float]:
    return list(DEFAULT_SIZES.get(category, FALLBACK_SIZE))


# Objects

# shape, nominal size [x, y, z] (m), mass (kg) at the nominal size, sliding friction, rgba
OBJECT_SPECS = {
    "mug": ("cylinder", [0.08, 0.08, 0.10], 0.30, 1.0, "0.85 0.3 0.2 1"),
    "cup": ("cylinder", [0.07, 0.07, 0.09], 0.20, 1.0, "0.3 0.55 0.85 1"),
    "bowl": ("cylinder", [0.14, 0.14, 0.06], 0.25, 1.0, "0.9 0.9 0.8 1"),
    "plate": (
        "cylinder",
        [0.22, 0.22, 0.03],
        0.40,
        0.8,
        "0.92 0.92 0.9 1",
    ),  # a deep plate: the rim is what the gripper holds
    "sponge": ("box", [0.10, 0.06, 0.03], 0.03, 1.2, "0.95 0.85 0.2 1"),
    "apple": ("sphere", [0.076, 0.076, 0.076], 0.18, 0.9, "0.75 0.15 0.12 1"),
    "bottle": ("bottle", [0.066, 0.066, 0.22], 0.55, 0.6, "0.55 0.75 0.9 0.7"),
    "pan": ("pan", [0.24, 0.24, 0.045], 0.90, 0.6, "0.2 0.2 0.22 1"),
    "spatula": ("box", [0.035, 0.22, 0.010], 0.05, 0.7, "0.3 0.3 0.32 1"),
    # office and living room (L9): flat boxes, long side along x, so the pads close across the short side
    "book": ("box", [0.17, 0.075, 0.025], 0.30, 0.6, "0.2 0.35 0.6 1"),
    "remote": ("box", [0.17, 0.045, 0.02], 0.10, 0.6, "0.12 0.12 0.14 1"),
    "laptop": ("box", [0.30, 0.21, 0.02], 1.20, 0.6, "0.62 0.64 0.67 1"),  # closed; too wide to pinch, scenery
}
OBJECT_MAP = {label: label for label in OBJECT_SPECS}
PICKABLE = (
    "mug",
    "cup",
    "sponge",
    "apple",
    "bottle",
    "book",
    "remote",
)  # fit the 10 cm gripper opening and are held by the body
RIM_HELD = ("bowl", "plate")  # hollow, pinched by the rim nearest the robot (MAN-4)
HANDLE_HELD = ("pan",)  # pinched across the handle, close to the body
THIN_HELD = ("spatula",)  # 1 cm flat: pinched across its width with the pad tips just over the counter (L26)
GRASPABLE = PICKABLE + RIM_HELD + HANDLE_HELD + THIN_HELD
WIPE_TOOLS = ("sponge", "spatula")  # what wipe_counter drags along the top; the spatula scrapes it
TABLE_MOVABLE = PICKABLE + ("bowl",)  # what move_on_table carries; the bowl is held by its rim (MAN-4)
# What a scan with no objects gets on its table or counter, by the kind of room the fixtures say it is (L9).
# A kitchen keeps the original five; an office does not hold a plate or a bowl.
ROOM_OBJECTS = {
    "kitchen": ("mug", "bowl", "plate", "cup", "apple"),
    "office": ("laptop", "book", "cup"),
    "living_room": ("remote", "book", "cup"),
    "bedroom": ("book", "cup"),
}
KITCHEN_CATEGORIES = ("counter", "sink", "stove", "oven", "refrigerator", "dishwasher", "microwave")
SIZE_VARIATION = 0.08  # +-8% on the footprint and on the height, drawn separately


def room_kind(fixtures: list[dict]) -> str:
    """ "kitchen", "bedroom", "living_room" or "office", from the scanned categories of `fixtures` (the twin's
    `source_category`): any kitchen fixture makes a kitchen, then a bed a bedroom, a sofa, television or fireplace a
    living room, and a table or chair with none of those an office. A room with none of these (nothing scanned yet)
    is a kitchen, the pipeline's original assumption."""
    cats = {f.get("source_category") for f in fixtures}
    if cats & set(KITCHEN_CATEGORIES):
        return "kitchen"
    if "bed" in cats:
        return "bedroom"
    if cats & {"sofa", "television", "fireplace"}:
        return "living_room"
    if cats & {"table", "chair"}:
        return "office"
    return "kitchen"


def catalog_id(label: str) -> str:
    return OBJECT_MAP.get(label, "box")


def object_spec(cid: str, size: list[float]) -> dict:
    """Shape, mass scaled from the nominal mass by volume, friction and colour for an object of `size`."""
    shape, nominal, mass, friction, rgba = OBJECT_SPECS.get(cid, ("box", list(size), 0.2, 1.0, "0.5 0.5 0.9 1"))
    ratio = (size[0] * size[1] * size[2]) / (nominal[0] * nominal[1] * nominal[2])
    return {"shape": shape, "mass": round(mass * ratio, 4), "friction": friction, "rgba": rgba}


def sample_size(cid: str, rng: random.Random) -> list[float]:
    """Nominal size with seeded variation; round objects stay round."""
    nominal = OBJECT_SPECS[cid][1]
    kxy, kz = (1 + rng.uniform(-SIZE_VARIATION, SIZE_VARIATION) for _ in range(2))
    return [round(nominal[0] * kxy, 4), round(nominal[1] * kxy, 4), round(nominal[2] * kz, 4)]

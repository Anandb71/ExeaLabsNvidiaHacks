"""Dynamic instruction generation with Nemotron Lightning (Token Factory).

Generates diverse natural language robot instructions per task family and object/fixture,
maintaining a strict train vs held-out evaluation split (POL-2).
Results are cached locally so tests and offline runs work without API keys.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from arrival.tasks.propose import WORDINGS

DEFAULT_CACHE = (
    Path("out") / "cache" / "instruction_cache.json"
)  # under the run's out/, never in the package: a cache there
# answered later runs without calling Token Factory and dirtied the checkout

FALLBACK_TEMPLATES = {
    "counter_to_sink": [
        "Pick up the {name} from the counter and place it in the sink.",
        "Put the {name} in the sink.",
        "Move the {name} off the counter into the sink.",
        "Take the {name} and drop it in the sink.",
        "Carefully grab the {name} and transfer it to the sink basin.",
        "Clear the counter by moving the {name} into the kitchen sink.",
        "Set the {name} down inside the sink.",
        "Retrieve the {name} from the countertop and put it into the sink.",
    ],
    "counter_to_counter": [
        "Pick up the {name} from the counter and place it onto the other counter.",
        "Put the {name} onto the other counter.",
        "Move the {name} off the counter onto the other counter.",
        "Take the {name} and drop it onto the other counter.",
        "Transfer the {name} from this counter across to the opposite counter.",
        "Shift the {name} over to the second countertop.",
        "Relocate the {name} onto the other counter space.",
        "Lift the {name} and set it down on the adjoining counter.",
    ],
    "open_drawer": [
        "Open the top drawer.",
        "Pull the top drawer of the {name} open.",
        "Slide open the drawer under the counter.",
        "Pull out the top drawer on the {name}.",
        "Gently slide the {name} drawer open.",
        "Open up the kitchen drawer.",
        "Pull the upper drawer handle to open it.",
        "Access the counter drawer by sliding it outward.",
    ],
    "open_door": [
        "Open the cabinet door.",
        "Swing the {name} cabinet door open.",
        "Pull the cabinet door open.",
        "Swing open the lower cabinet door on the {name}.",
        "Grip the cabinet door handle and pull it outward.",
        "Open up the cupboard door beneath the counter.",
        "Unlatch and swing the cabinet door open.",
        "Pull open the door to reveal the cabinet interior.",
    ],
    # the families below have the sim's own four wordings (arrival.sim.tasks.SIM_WORDINGS), so the last one, the
    # template path's held-out wording, is the eval split here too
    "close_drawer": [
        "Close the drawer.",
        "Push the open drawer of the {name} shut.",
        "Slide the drawer closed.",
        "Push the drawer on the {name} back in.",
    ],
    "close_door": [
        "Close the cabinet door.",
        "Push the {name} cabinet door shut.",
        "Swing the open door closed.",
        "Shut the cabinet door on the {name}.",
    ],
    "put_in_drawer": [
        "Put the {name} in the open drawer.",
        "Place the {name} into the drawer.",
        "Move the {name} from the counter into the drawer.",
        "Store the {name} in the drawer.",
    ],
    "wipe_counter": [
        "Wipe the counter with the {name}.",
        "Clean the counter top with the {name}.",
        "Scrub the counter using the {name}.",
        "Use the {name} to wipe down the counter.",
    ],
    "open_fridge": [
        "Open the fridge.",
        "Pull the {name} door open.",
        "Swing the fridge door open.",
        "Get the fridge open.",
    ],
    "move_on_table": [
        "Move the {name} further along the table.",
        "Pick up the {name} and set it down along the table.",
        "Shift the {name} to another spot on the table.",
        "Reposition the {name} on the table.",
    ],
    "open_oven": [
        "Open the oven.",
        "Pull the oven door down.",
        "Open the oven door of the {name}.",
        "Lower the oven door until it is flat.",
    ],
    "open_dishwasher": [
        "Open the dishwasher.",
        "Pull the dishwasher door open.",
        "Open the {name} door.",
        "Fold the dishwasher door down.",
    ],
    "turn_on_faucet": [
        "Turn on the faucet.",
        "Lift the faucet lever to start the water.",
        "Turn the tap on.",
        "Push the faucet handle up to run the water.",
    ],
    "load_dishwasher": [
        "Put the {name} in the dishwasher.",
        "Load the {name} into the dishwasher rack.",
        "Move the {name} from the counter into the dishwasher.",
        "Set the {name} on the pulled-out dishwasher rack.",
    ],
    "unload_dishwasher": [
        "Take the {name} out of the dishwasher.",
        "Unload the {name} onto the counter.",
        "Move the {name} from the dishwasher rack to the counter.",
        "Lift the {name} off the dishwasher rack and put it on the counter.",
    ],
}


INSTRUCTION_STYLES = ("mixed", "concise", "polite", "descriptive", "urgent", "goal_oriented")

STYLE_PROMPTS = {
    "mixed": "a balanced mix of concise, polite, descriptive, urgent, and goal-oriented imperative phrasing",
    "concise": "short, concise, direct imperative instructions (e.g. 'Put mug in sink', 'Open top drawer')",
    "polite": "polite, courteous phrasing (e.g. 'Could you please place the mug into the sink?', 'Please open the top drawer')",
    "descriptive": "detailed, descriptive phrasing referencing objects, surfaces, and actions (e.g. 'Pick up the mug from the counter and place it inside the sink basin')",
    "urgent": "direct, urgent, time-sensitive phrasing (e.g. 'Immediately move the mug to the sink', 'Quickly pull open the drawer')",
    "goal_oriented": "purpose-driven, outcome-oriented phrasing (e.g. 'Clear the counter by putting the mug in the sink', 'Prepare the drawer for storage by opening it')",
}


def _style_templates(templates: list[str], style: str, name: str) -> list[str]:
    base = [t.format(name=name) for t in templates]
    if style == "mixed":
        return base
    styled = []
    for s in base:
        clean = s.rstrip(".")
        if style == "concise":
            c = re.sub(r"^(Carefully|Gently|Please|Kindly|Now)\s+", "", clean, flags=re.IGNORECASE)
            styled.append(c + ".")
        elif style == "polite":
            styled.append(f"Please {clean[:1].lower() + clean[1:]}.")
        elif style == "urgent":
            styled.append(f"Quickly {clean[:1].lower() + clean[1:]}.")
        elif style == "descriptive":
            styled.append(f"Carefully {clean[:1].lower() + clean[1:]}.")
        elif style == "goal_oriented":
            styled.append(f"To tidy up, {clean[:1].lower() + clean[1:]}.")
        else:
            styled.append(s)
    return styled


def _cache_key(
    family: str,
    object_name: str | None,
    fixture_name: str | None,
    n: int = 8,
    role: str = "lightning",
    eval_ratio: float = 0.25,
    style: str = "mixed",
) -> str:
    key = f"{family}::obj={object_name or 'none'}::fx={fixture_name or 'none'}"
    if (n, role, eval_ratio) != (8, "lightning", 0.25):  # defaults keep the plain key
        key += f"::n={n}::role={role}::ev={eval_ratio}"
    if style != "mixed":
        key += f"::style={style}"
    return key


def load_cache(cache_path: Path | str | None = None) -> dict:
    """The instruction cache, or {} when the file is missing or unreadable (the generator then falls back to templates)."""
    p = Path(cache_path) if cache_path else DEFAULT_CACHE
    if p.is_file():
        try:
            value = json.loads(p.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
    return {}


def save_cache(cache: dict, cache_path: Path | str | None = None) -> None:
    """Write the instruction cache. A failed write is ignored: the cache only saves API calls."""
    p = Path(cache_path) if cache_path else DEFAULT_CACHE
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def generate_instructions(
    family: str,
    object_name: str | None = None,
    fixture_name: str | None = None,
    n: int = 8,
    role: str = "lightning",
    eval_ratio: float = 0.25,
    style: str = "mixed",
    cache_path: Path | str | None = None,
) -> dict[str, list[str]]:
    """Generate diverse instructions split into 'train' and 'eval' sets with stylistic variation.

    Uses Nemotron Lightning via Nebius Token Factory if available, with offline
    caching and deterministic template fallback. Only real model output is cached: a fallback
    (no key, network error, unusable reply) is recomputed so configuring the key later takes effect.
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive whole number")
    if type(eval_ratio) not in (int, float) or not math.isfinite(eval_ratio) or not 0 < eval_ratio < 1:
        raise ValueError("eval_ratio must be finite and between 0 and 1")
    if style not in INSTRUCTION_STYLES:
        raise ValueError(f"style must be one of {INSTRUCTION_STYLES}, got {style!r}")
    key = _cache_key(family, object_name, fixture_name, n, role, eval_ratio, style=style)
    cache = load_cache(cache_path)
    hit = cache.get(key)
    if isinstance(hit, dict) and hit.get("source") != "template_fallback" and hit.get("train"):
        return hit

    instructions: list[str] = []
    source = "token_factory"
    error = None

    from arrival.models.token_factory import TokenFactoryNotConfigured, complete

    name = object_name or fixture_name or "object"
    style_guidance = STYLE_PROMPTS[style]
    prompt = (
        f"Generate {n} distinct, natural imperative instructions commanding a kitchen robot to perform this task:\n"
        f"Task family: {family}\n"
        f"Target/object: {name}\n"
        f"Style: {style_guidance}\n\n"
        f'Output ONLY a valid JSON array of {n} strings, e.g. ["Do X", "Please do Y"].'
    )

    try:
        raw = complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI generating diverse robot instructions. Return only a JSON list of strings.",
                },
                {"role": "user", "content": prompt},
            ],
            role=role,
        )
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                instructions = [x.strip() for x in parsed if x.strip()]
        if not instructions:
            error = "the reply had no JSON list of strings"
    except (TokenFactoryNotConfigured, Exception) as e:
        instructions = []
        error = f"{type(e).__name__}: {e}"

    instructions = list(dict.fromkeys(instructions))[:n]
    if not instructions:
        source = "template_fallback"
        templates = FALLBACK_TEMPLATES.get(family, WORDINGS.get(family, ["Do task {name}"]))
        instructions = _style_templates(templates, style, name)[:n]

    # Split into train and held-out evaluation sets; train is never empty (a lone string is train-only)
    eval_count = min(max(1, int(len(instructions) * eval_ratio)), len(instructions) - 1)
    split = {
        "source": source,
        "style": style,
        "train": instructions[: len(instructions) - eval_count],
        "eval": instructions[len(instructions) - eval_count :],
        **({"error": error} if source == "template_fallback" and error else {}),
    }

    if source != "template_fallback":
        cache[key] = split
        save_cache(cache, cache_path)
    return split

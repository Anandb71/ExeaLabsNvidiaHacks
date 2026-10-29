"""Four-photo input slot. Geometry is NOT measured: copies the hero kitchen and blocks claims."""

from __future__ import annotations

import copy
import re
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from arrival.scan.bundle import load_room, validate


def bundle_from_photos(
    paths: list, room_id: str, scans_dir: str | Path = "scans", template_room: str = "hero_kitchen"
) -> dict:
    """A ScanBundle for a room given only four photos: the photos are copied under scans/<room_id>/photos and the
    geometry is the template room's, flagged four_photo_unmeasured with geometry and realism claims blocked, since
    nothing was measured. Raises ValueError on a bad room id or photo."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", room_id or "") or ".." in room_id:
        raise ValueError(f"invalid room_id: {room_id!r}")
    paths = [Path(p) for p in paths]
    if len(paths) != 4:
        raise ValueError(f"bundle_from_photos needs exactly 4 photos, got {len(paths)}")
    for p in paths:
        if not p.is_file():
            raise ValueError(f"photo not found: {p}")
        try:
            with Image.open(p) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"photo is not a readable image: {p}") from exc
    scans_dir = Path(scans_dir)
    template = load_room(template_room, scans_dir)
    photo_dir = scans_dir / room_id / "photos"
    photo_dir.mkdir(parents=True, exist_ok=True)
    # Stage inputs first: a re-run may be given files already in photo_dir.
    with tempfile.TemporaryDirectory(prefix=f".{room_id}-photos-", dir=scans_dir) as stage:
        staged = []
        for i, p in enumerate(paths):
            staged_path = Path(stage) / f"photo_{i}{p.suffix.lower()}"
            shutil.copy2(p, staged_path)
            staged.append(staged_path)
        for old in photo_dir.glob("photo_*"):  # a re-run with other suffixes must not leave stale photos
            old.unlink()
        for staged_path in staged:
            shutil.copy2(staged_path, photo_dir / staged_path.name)
    bundle = copy.deepcopy(template)
    bundle["room_id"] = room_id
    bundle["geometry_source"] = "four_photo_unmeasured"
    bundle["blocks_claims"] = ["geometry", "realism"]
    return validate(bundle)

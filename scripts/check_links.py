"""Offline link checker for every Markdown file in the repo.

    python scripts/check_links.py [--root DIR]

For each `[text](target)` and `![alt](target)` outside code fences and inline code, checks that a
relative target exists and, when it carries a `#fragment` into a Markdown file, that a heading with
that GitHub-style anchor exists. `http(s):` and `mailto:` links are skipped (nothing touches the
network). Prints one line per problem; exit 0 when clean, 1 otherwise.
scripts/check_docs.py covers commands and code references in the main docs; this covers all links.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "out", "node_modules", ".claude", "build", "dist", "__pycache__"}

FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`[^`\n]*`")
LINK = re.compile(r"!?\[(?:[^\]\\]|\\.)*\]\(\s*(<[^>]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
REF_DEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(\S+)")
HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
EXTERNAL = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.\-]*:|//)")


def markdown_files(root: Path) -> list[Path]:
    """Tracked .md files when `root` is a git checkout, else every .md outside the usual junk dirs."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", "*.md"], capture_output=True, check=True
        ).stdout.decode()
        files = [root / p for p in out.split("\0") if p]
        if files:
            return sorted(f for f in files if f.is_file())
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(p for p in root.rglob("*.md") if not SKIP_DIRS & set(p.relative_to(root).parts))


def slug(heading: str) -> str:
    """GitHub's heading anchor: strip markup, lowercase, drop punctuation, spaces to hyphens."""
    t = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # links keep their text
    t = re.sub(r"[`*~]", "", t)  # underscores stay: GitHub keeps them in anchors
    t = re.sub(r"<[^>]+>", "", t).strip().lower()
    t = re.sub(r"[^\w\- ]", "", t, flags=re.UNICODE)
    return t.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    out: set[str] = set()
    inside = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if FENCE.match(line):
            inside = not inside
            continue
        m = None if inside else HEADING.match(line)
        if not m:
            continue
        s = slug(m.group(2))
        n = seen.get(s, 0)
        seen[s] = n + 1
        out.add(s if n == 0 else f"{s}-{n}")
    return out


def links(text: str) -> list[tuple[int, str]]:
    """(line number, target) for every link outside code."""
    found, inside = [], False
    for n, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            inside = not inside
            continue
        if inside:
            continue
        line = INLINE_CODE.sub("", line)
        for t in LINK.findall(line) + REF_DEF.findall(line):
            found.append((n, t.strip("<>")))
    return found


def check(root: Path = REPO) -> list[str]:
    root = root.resolve()
    probs: list[str] = []
    cache: dict[Path, set[str]] = {}
    for doc in markdown_files(root):
        rel = doc.relative_to(root).as_posix()
        for n, target in links(doc.read_text(encoding="utf-8", errors="replace")):
            if EXTERNAL.match(target):
                continue
            path_part, _, frag = target.partition("#")
            dest = doc if not path_part else (doc.parent / path_part.split("?")[0])
            try:
                dest = dest.resolve()
                dest.relative_to(root)
            except ValueError:
                probs.append(f"{rel}:{n}: link leaves the repo {target}")
                continue
            if not dest.exists():
                probs.append(f"{rel}:{n}: broken link {target}")
                continue
            if frag and dest.is_file() and dest.suffix.lower() == ".md":
                if dest not in cache:
                    cache[dest] = anchors(dest)
                if frag.lower() not in cache[dest]:
                    probs.append(f"{rel}:{n}: no heading anchor #{frag} in {dest.relative_to(root).as_posix()}")
    return probs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(REPO), help="repo copy to check (default: this repo)")
    a = ap.parse_args(argv)
    probs = check(Path(a.root))
    for p in probs:
        print(p)
    if not probs:
        print("check_links: ok", file=sys.stderr)
    return 1 if probs else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Canonical repo paths + the slug -> title map. Pure stdlib."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FRAMES = DATA / "frames"
MANIFEST = DATA / "manifest.json"
PALETTES = DATA / "palettes.json"      # written by pipeline/3, read by renderers
SUBJECTS = DATA / "subjects.json"      # written by pipeline/2, read by pipeline/3


def title_map():
    if not MANIFEST.exists():
        return {}
    m = json.loads(MANIFEST.read_text())
    return {v["slug"]: t for t, v in m.items()
            if isinstance(v, dict) and v.get("slug")}

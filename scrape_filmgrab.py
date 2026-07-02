#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["tqdm"]
# ///
"""Scrape representative frames from FilmGrab (film-grab.com).

For each film title it searches FilmGrab, opens the post, collects the full-size
gallery frames, evenly samples up to MAX_FRAMES of them, and saves them to
data/frames/<slug>/. Writes/updates data/manifest.json.

Usage:
    uv run scrape_filmgrab.py "Blue Velvet" "Stalker" "Ran"
    uv run scrape_filmgrab.py            # uses the default sample list
"""
import json
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

from tqdm import tqdm

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
ROOT = Path(__file__).resolve().parent
FRAMES = ROOT / "data" / "frames"
MANIFEST = ROOT / "data" / "manifest.json"
MAX_FRAMES = 20          # frames kept per film
DELAY = 0.3              # seconds between requests, be polite


def get(url):
    # Route through curl: macOS framework Python ships without a CA bundle, so
    # urllib fails SSL verification while the system curl works fine.
    r = subprocess.run(["curl", "-sSL", "--max-time", "30", "-A", UA, url],
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace")[:200])
    return r.stdout


def find_post_url(title):
    q = urllib.parse.urlencode({"s": title})
    html = get(f"https://film-grab.com/?{q}").decode("utf-8", "replace")
    m = re.search(r'href="(https://film-grab\.com/\d{4}/\d{2}/\d{2}/[^"]+/)"', html)
    return m.group(1) if m else None


def find_frames(post_url):
    html = get(post_url).decode("utf-8", "replace")
    urls = re.findall(
        r'https://film-grab\.com/wp-content/uploads/photo-gallery/[^"\']+?\.(?:jpg|jpeg|png)',
        html)
    seen, out = set(), []
    for u in urls:
        if "/thumb/" in u:                              # skip thumbnails
            continue
        if re.search(r'-\d+x\d+\.(?:jpg|jpeg|png)$', u):  # skip resized variants
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def sample(frames, n):
    if len(frames) <= n:
        return frames
    step = len(frames) / n
    return [frames[int(i * step)] for i in range(n)]


def encode(url):
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(
        (p.scheme, p.netloc, urllib.parse.quote(p.path), p.query, p.fragment))


def main(titles):
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for title in tqdm(titles, desc="films", unit="film"):
        post = find_post_url(title)
        if not post:
            tqdm.write(f"[skip] {title}: no FilmGrab page found")
            manifest[title] = {"found": False}
            continue
        slug = post.rstrip("/").split("/")[-1]
        available = find_frames(post)
        chosen = sample(available, MAX_FRAMES)
        outdir = FRAMES / slug
        outdir.mkdir(parents=True, exist_ok=True)
        saved = []
        for i, fu in enumerate(tqdm(chosen, desc=title[:24], unit="img", leave=False)):
            dest = outdir / f"{i:02d}.jpg"
            if not dest.exists():
                try:
                    dest.write_bytes(get(encode(fu)))
                    time.sleep(DELAY)
                except Exception as e:
                    tqdm.write(f"  ! {fu}: {e}")
                    continue
            saved.append(str(dest.relative_to(ROOT)))
        manifest[title] = {"found": True, "post": post, "slug": slug,
                           "n_available": len(available), "frames": saved}
        tqdm.write(f"[ok] {title}: {len(saved)}/{len(available)} frames -> {slug}/")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    tqdm.write(f"[done] {len(titles)} film(s) -> {MANIFEST}")


if __name__ == "__main__":
    default = ["Blue Velvet", "Stalker", "Ran", "The Handmaiden"]
    main(sys.argv[1:] or default)

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["tqdm"]
# ///
"""Scrape representative frames from FilmGrab (film-grab.com).

For each film title it searches FilmGrab, opens the post, collects the full-size
gallery frames, evenly samples up to MAX_FRAMES of them, and saves them to
data/frames/<slug>/. Writes/updates data/manifest.json.

Films missing from FilmGrab fall back to TMDB backdrops (noisier — promo shots
mixed in — but broad coverage). Requires a free API key from
themoviedb.org/settings/api in the TMDB_API_KEY env var; without it the
fallback is skipped.

Usage:
    uv run pipeline/1_scrape_frames.py "Blue Velvet" "Stalker" "Ran"
    uv run pipeline/1_scrape_frames.py watchlist.txt   # one title per line, # = comment
    uv run pipeline/1_scrape_frames.py                 # ./watchlist.txt if present,
                                              # else the default sample list
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

from tqdm import tqdm

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data" / "frames"
MANIFEST = ROOT / "data" / "manifest.json"
MAX_FRAMES = 20          # frames kept per film
DELAY = 0.3              # seconds between requests, be polite


def load_env():
    """Minimal .env loader (KEY=value lines); env vars already set win."""
    envfile = ROOT / ".env"
    if not envfile.exists():
        return
    for line in envfile.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


load_env()
TMDB_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_IMG = "https://image.tmdb.org/t/p/original"


def get(url, tries=3):
    # Route through curl: macOS framework Python ships without a CA bundle, so
    # urllib fails SSL verification while the system curl works fine.
    for attempt in range(1, tries + 1):
        r = subprocess.run(["curl", "-sSL", "--max-time", "60", "-A", UA, url],
                           capture_output=True)
        if r.returncode == 0:
            return r.stdout
        err = r.stderr.decode("utf-8", "replace")[:200]
        if attempt < tries:
            wait = 10 * attempt
            tqdm.write(f"[retry {attempt}/{tries - 1}] {url}: {err.strip()} "
                       f"— waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(err)


def slugify(title):
    t = re.sub(r"['’]", "", title.lower())  # Schindler's -> schindlers
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")


def split_year(title):
    """'A Brighter Summer Day (1991)' -> ('A Brighter Summer Day', '1991').
    Searching with the parenthetical year returns nothing on either source;
    stripped out, it instead sharpens the match."""
    m = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", title)
    return (m.group(1), m.group(2)) if m else (title, None)


def find_post_url(title, year=None):
    """Best-matching post for a title, plus how it matched. Search returns any
    post that merely MENTIONS the title (searching "Stalker" once returned The
    Green Mile, "Cars" returned Drive My Car), so ONLY slug matches count:
    year-suffixed exact first when the year is known, then exact, then a slug
    prefix (year-suffixed posts like solaris-1972). No slug match = not on
    FilmGrab; the caller falls through to TMDB. Returns (url, match) where
    match is "exact" | "prefix", or (None, "none")."""
    q = urllib.parse.urlencode({"s": title})
    html = get(f"https://film-grab.com/?{q}").decode("utf-8", "replace")
    links = re.findall(
        r'href="(https://film-grab\.com/\d{4}/\d{2}/\d{2}/([^"/]+)/)"', html)
    if not links:
        return None, "none"
    want = slugify(title)
    if year:
        for url, slug in links:
            if slug == f"{want}-{year}":
                return url, "exact"
    for url, slug in links:
        if slug == want:
            return url, "exact"
    for url, slug in links:
        if slug.startswith(want + "-"):
            return url, "prefix"
    tqdm.write(f"[warn] {title}: search hits but no slug match "
               f"(first hit: {links[0][1]}) — treating as not on FilmGrab")
    return None, "none"


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


def tmdb_json(path, **params):
    params["api_key"] = TMDB_KEY
    url = f"https://api.themoviedb.org/3{path}?{urllib.parse.urlencode(params)}"
    return json.loads(get(url))


def find_frames_tmdb(title, year=None):
    """Fallback source: TMDB backdrops for the top search hit. Returns
    (movie_dict, urls) or (None, []). Prefers textless backdrops."""
    if not TMDB_KEY:
        return None, []
    params = {"query": title}
    if year:
        params["primary_release_year"] = year
    results = tmdb_json("/search/movie", **params).get("results") or []
    if not results and year:   # year mismatch on TMDB's side: retry without
        results = tmdb_json("/search/movie", query=title).get("results") or []
    if not results:
        return None, []
    movie = results[0]
    imgs = tmdb_json(f"/movie/{movie['id']}/images")
    backdrops = imgs.get("backdrops") or []
    textless = [b for b in backdrops if not b.get("iso_639_1")]
    urls = [TMDB_IMG + b["file_path"] for b in (textless or backdrops)]
    return movie, urls


def sample(frames, n):
    if len(frames) <= n:
        return frames
    step = len(frames) / n
    return [frames[int(i * step)] for i in range(n)]


def encode(url):
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(
        (p.scheme, p.netloc, urllib.parse.quote(p.path), p.query, p.fragment))


MIN_FRAMES = 10          # below this the palette is undersampled — flag it


def match_report(manifest, titles):
    """Per-film audit of this run: anything not exact-matched with enough
    frames is a line item to eyeball before running extraction."""
    problems, ok = [], 0
    for title in titles:
        e = manifest.get(title, {})
        if not e.get("found"):
            if e.get("error"):
                problems.append(f"  [FAIL]   {title}: {e['error']} "
                                f"(re-run to retry)")
            else:
                problems.append(f"  [MISS]   {title}: no FilmGrab page")
            continue
        n, match = len(e.get("frames", [])), e.get("match", "?")
        if match == "fallback":
            problems.append(f"  [VERIFY] {title}: fallback match '{e['slug']}' "
                            f"(first search hit, not a slug match), {n} frames")
        elif match == "tmdb":
            problems.append(f"  [TMDB]   {title}: {n} TMDB backdrops "
                            f"({e['post']}) — noisier source, eyeball frames")
        elif n < MIN_FRAMES:
            problems.append(f"  [LOW]    {title}: only {n} frames "
                            f"({e['n_available']} available) -> {e['slug']}/")
        else:
            ok += 1
    print(f"\nmatch report: {ok}/{len(titles)} clean "
          f"(exact/prefix slug match, >={MIN_FRAMES} frames)")
    for line in problems:
        print(line)
    if problems:
        print("fix the flagged films (retitle and re-run, or drop) "
              "before extraction.")


def scrape_film(title, manifest, force_tmdb=False):
    clean, year = split_year(title)
    # --tmdb: skip FilmGrab — for films it doesn't have, where its search
    # would return a wrong first-hit fallback (e.g. Cars -> drive-my-car)
    post, match = (None, None) if force_tmdb else find_post_url(clean, year)
    if post:
        slug = post.rstrip("/").split("/")[-1]
        available = find_frames(post)
    else:
        movie, available = find_frames_tmdb(clean, year)
        if not available:
            hint = "" if TMDB_KEY else " (set TMDB_API_KEY to enable fallback)"
            tqdm.write(f"[skip] {title}: not on FilmGrab or TMDB{hint}")
            manifest[title] = {"found": False}
            return
        slug = slugify(clean)
        post = f"https://www.themoviedb.org/movie/{movie['id']}"
        match = "tmdb"
        tqdm.write(f"[tmdb] {title}: not on FilmGrab, using TMDB backdrops of "
                   f"'{movie.get('title')}' "
                   f"({movie.get('release_date', '????')[:4]})")
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
                       "match": match, "n_available": len(available),
                       "frames": saved}
    tqdm.write(f"[ok] {title}: {len(saved)}/{len(available)} frames -> {slug}/")


def main(titles, force_tmdb=False):
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    for title in tqdm(titles, desc="films", unit="film"):
        prev = manifest.get(title, {})
        if prev.get("found") and prev.get("frames"):
            continue  # already scraped; delete its manifest entry to redo
        try:
            scrape_film(title, manifest, force_tmdb)
        except Exception as e:
            tqdm.write(f"[fail] {title}: {e}")
            manifest[title] = {"found": False, "error": str(e)[:200]}
        # save after every film so a crash or ^C loses nothing
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    tqdm.write(f"[done] {len(titles)} film(s) -> {MANIFEST}")
    match_report(manifest, titles)


def read_watchlist(path):
    titles = []
    for line in Path(path).read_text().splitlines():
        t = line.strip()
        if t and not t.startswith("#"):
            titles.append(t)
    return titles


if __name__ == "__main__":
    args = sys.argv[1:]
    force_tmdb = "--tmdb" in args
    args = [a for a in args if a != "--tmdb"]
    watchlist = ROOT / "watchlist.txt"
    if len(args) == 1 and args[0].endswith(".txt"):
        titles = read_watchlist(args[0])
    elif args:
        titles = args
    elif watchlist.exists():
        titles = read_watchlist(watchlist)
    else:
        titles = ["Blue Velvet", "Stalker", "Ran", "The Handmaiden"]
    if not titles:
        sys.exit("watchlist is empty")
    main(titles, force_tmdb)

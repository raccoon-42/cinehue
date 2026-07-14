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
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data" / "frames"
MANIFEST = ROOT / "data" / "manifest.json"
MAX_FRAMES = 20          # frames kept per film

# Politeness: cap requests/sec PER HOST, not with blind fan-out. film-grab.com
# is a one-person WordPress blog — keep it gentle. TMDB is real infra, hit
# harder. Threads overlap network latency; the limiter enforces the spacing.
HOST_RATE = {
    "film-grab.com": 8.0,        # search + post + frame uploads all share this
    "api.themoviedb.org": 15.0,
    "image.tmdb.org": 20.0,
}
DEFAULT_RATE = 5.0               # unknown hosts
FILM_WORKERS = 4                 # films scraped concurrently
FRAME_WORKERS = 8                # frame downloads concurrent within a film


class RateLimiter:
    """Serialize the *spacing* between requests, not the requests. A thread
    claims the next time-slot under the lock, releases it, then sleeps until
    its slot — so curl/network runs unlocked and other threads' slots overlap
    it. Net effect: steady <=rate req/s with latency hidden."""

    def __init__(self, rate):
        self.min_interval = 1.0 / rate
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_time)
            self.next_time = slot + self.min_interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


_LIMITERS = {}
_LIMITERS_LOCK = threading.Lock()


def limiter_for(host):
    with _LIMITERS_LOCK:
        lim = _LIMITERS.get(host)
        if lim is None:
            lim = RateLimiter(HOST_RATE.get(host, DEFAULT_RATE))
            _LIMITERS[host] = lim
        return lim


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
TMDB_IMG = "https://image.tmdb.org/t/p/w1280"  # palette/subject work downscales anyway; original-size multi-MB files just time out


def get(url, tries=3):
    # Route through curl: macOS framework Python ships without a CA bundle, so
    # urllib fails SSL verification while the system curl works fine.
    limiter = limiter_for(urllib.parse.urlsplit(url).netloc)
    for attempt in range(1, tries + 1):
        limiter.wait()          # per-host pacing (thread-safe)
        r = subprocess.run(["curl", "-sSL", "--max-time", "60", "-A", UA, url],
                           capture_output=True)
        if r.returncode == 0:
            return r.stdout
        err = r.stderr.decode("utf-8", "replace")[:200]
        if attempt < tries:
            wait = 10 * attempt
            safe_url = re.sub(r"api_key=[^&]+", "api_key=***", url)
            tqdm.write(f"[retry {attempt}/{tries - 1}] {safe_url}: {err.strip()} "
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


def _download_frame(i, fu, outdir):
    """Fetch one frame -> outdir/NN.jpg. Returns (i, relpath) or None. Pacing
    handled by the per-host rate limiter inside get(), so no sleep here."""
    dest = outdir / f"{i:02d}.jpg"
    if not dest.exists():
        try:
            dest.write_bytes(get(encode(fu)))
        except Exception as e:
            tqdm.write(f"  ! {fu}: {e}")
            return None
    return i, str(dest.relative_to(ROOT))


def scrape_film(title, manifest, lock, position, force_tmdb=False):
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
            with lock:
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
    # Frames download concurrently; the host limiter still bounds actual req/s.
    # Each film worker owns a fixed screen row (position) below the films bar.
    results = {}
    with tqdm(total=len(chosen), desc=title[:24].ljust(24), unit="img",
              position=position, leave=False) as bar, \
         ThreadPoolExecutor(max_workers=FRAME_WORKERS) as ex:
        futs = [ex.submit(_download_frame, i, fu, outdir)
                for i, fu in enumerate(chosen)]
        for fut in as_completed(futs):
            r = fut.result()
            bar.update(1)
            if r:
                results[r[0]] = r[1]
    saved = [results[i] for i in sorted(results)]   # keep sampled order
    with lock:
        manifest[title] = {"found": True, "post": post, "slug": slug,
                           "match": match, "n_available": len(available),
                           "frames": saved}
    tqdm.write(f"[ok] {title}: {len(saved)}/{len(available)} frames -> {slug}/")


def main(titles, force_tmdb=False):
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    # Screen rows 1..FILM_WORKERS for per-film bars; row 0 is the films bar.
    slots = queue.Queue()
    for i in range(FILM_WORKERS):
        slots.put(i + 1)
    films_bar = tqdm(total=len(titles), desc="films", unit="film", position=0)

    def work(title):
        try:
            if manifest.get(title, {}).get("found") and manifest[title].get("frames"):
                return  # already scraped; delete its manifest entry to redo
            pos = slots.get()
            try:
                scrape_film(title, manifest, lock, pos, force_tmdb)
            except Exception as e:
                tqdm.write(f"[fail] {title}: {e}")
                with lock:
                    manifest[title] = {"found": False, "error": str(e)[:200]}
            finally:
                slots.put(pos)
            # save after every film so a crash or ^C loses nothing
            with lock:
                MANIFEST.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False))
        finally:
            films_bar.update(1)

    with ThreadPoolExecutor(max_workers=FILM_WORKERS) as ex:
        list(ex.map(work, titles))
    films_bar.close()
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

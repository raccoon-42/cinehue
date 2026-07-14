#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["tqdm"]   # imported transitively via 1_scrape_frames.py
# ///
"""REAL-network benchmark against film-grab.com: old serial vs new concurrent.

Unlike bench_scrape_concurrency.py (fake latency), this hits the live server —
so it captures true curl-spawn, TLS, download bytes, and server behavior. It
does NOT write files or touch the manifest; it only times fetches.

Method (fairness matters on a live host):
  1. Resolve ONE film to real frame URLs with the actual find_post_url /
     find_frames code.
  2. WARM-UP: fetch every URL once, uncounted — primes server/CDN cache + TLS
     so no run gets an unfair cold/warm advantage.
  3. Time the same URL set three ways:
       - serial (old): raw curl + 0.3s sleep, no threads, no limiter
       - concurrent @ current rate (film-grab.com in HOST_RATE)
       - concurrent @ 8/s (headroom probe)
  4. Watch stderr for [retry .../...] — that's the server pushing back. If 8/s
     produces retries and 4/s doesn't, you've found the ceiling.

    uv run pipeline/bench_scrape_real.py                 # "Stalker", 10 frames
    uv run pipeline/bench_scrape_real.py "Blue Velvet" 12

Politeness: ~4xK requests total, one time. Keep K modest.
"""
import importlib.util
import subprocess
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "scrape_frames", ROOT / "pipeline" / "1_scrape_frames.py")
scrape = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scrape)

TITLE = sys.argv[1] if len(sys.argv) > 1 else "Stalker"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OLD_DELAY = 0.3
PROBE_RATE = 8.0


def get_raw(url):
    """Old-style fetch: raw curl, no limiter, no retry — the pre-change path."""
    r = subprocess.run(["curl", "-sSL", "--max-time", "60", "-A", scrape.UA, url],
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace")[:120])
    return r.stdout


def serial_old(urls):
    for u in urls:
        get_raw(u)
        time.sleep(OLD_DELAY)


def concurrent_new(urls, rate):
    # Force the limiter for this host to `rate`, fresh (resets pacing clock).
    host = urllib.parse.urlsplit(urls[0]).netloc
    scrape._LIMITERS[host] = scrape.RateLimiter(rate)
    with ThreadPoolExecutor(max_workers=scrape.FRAME_WORKERS) as ex:
        list(ex.map(scrape.get, urls))   # real get(): limiter + retry included


def timed(fn, *a):
    t0 = time.monotonic()
    fn(*a)
    return time.monotonic() - t0


def main():
    print(f"resolving real frame URLs for '{TITLE}' ...")
    clean, year = scrape.split_year(TITLE)
    post, match = scrape.find_post_url(clean, year)
    if not post:
        sys.exit(f"'{TITLE}' not found on FilmGrab (match={match}); "
                 f"pick a title that is, or check the name.")
    frames = [scrape.encode(u) for u in scrape.find_frames(post)][:K]
    if len(frames) < 3:
        sys.exit(f"only {len(frames)} frames found — pick a film with more.")
    n = len(frames)
    print(f"post: {post}\nusing {n} real frame URLs\n")

    print("warm-up (uncounted, primes cache + TLS) ...")
    for u in frames:
        get_raw(u)

    print(f"\n{'mode':<22}{'rate':>6}{'wall':>10}{'req/s':>9}{'speedup':>10}")
    print("-" * 57)
    base = timed(serial_old, frames)
    cur_rate = scrape.HOST_RATE.get("film-grab.com", scrape.DEFAULT_RATE)
    runs = [
        ("serial (old)", None, base),
        (f"concurrent @{cur_rate:g}", cur_rate, timed(concurrent_new, frames, cur_rate)),
        (f"concurrent @{PROBE_RATE:g}", PROBE_RATE, timed(concurrent_new, frames, PROBE_RATE)),
    ]
    for label, rate, wall in runs:
        rate_s = "-" if rate is None else f"{rate:g}"
        print(f"{label:<22}{rate_s:>6}{wall:>9.1f}s{n / wall:>9.1f}"
              f"{base / wall:>9.2f}x")

    print("\nany [retry .../...] lines above = server pushback at that rate.\n"
          "no retries + higher req/s at 8 vs current = safe to raise the rate.")


if __name__ == "__main__":
    main()

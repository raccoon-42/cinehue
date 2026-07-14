#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["tqdm"]   # only because we import 1_scrape_frames.py, which needs it
# ///
"""Synthetic benchmark: old serial scraper vs new threaded + rate-limited one.

Why synthetic? The real scraper can't be benchmarked by re-running it — the
manifest skips already-scraped films (second run is instant), and hammering
film-grab.com for timing is both rude and noisy (network variance drowns the
signal). So we replace the network with a FAKE fetch of fixed latency and
measure only what changed: the request-scheduling design.

It imports the REAL RateLimiter + worker constants from 1_scrape_frames.py, so
you're timing the actual shipped machinery, not a reimplementation.

Model per film: 1 search + 1 post + MAX_FRAMES downloads = REQ_PER_FILM requests,
all to one host (film-grab.com) — so they all share one rate bucket, exactly
like production.

    uv run pipeline/bench_scrape_concurrency.py                # defaults
    uv run pipeline/bench_scrape_concurrency.py 8 0.2          # 8 films, 0.2s latency
"""
import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load the real module despite its digit-leading filename (can't `import` it).
# Executing it runs only top-level defs (load_env etc.) — no network, no scrape,
# since __name__ != "__main__".
_spec = importlib.util.spec_from_file_location(
    "scrape_frames", ROOT / "pipeline" / "1_scrape_frames.py")
scrape = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scrape)

# --- workload knobs -------------------------------------------------------
N_FILMS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
FAKE_LATENCY = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15  # sec/request
REQ_PER_FILM = 2 + scrape.MAX_FRAMES        # search + post + frames
OLD_DELAY = 0.3                             # the old fixed inter-request sleep
RATE_SWEEP = [scrape.HOST_RATE["film-grab.com"], 8.0, 12.0, 16.0]
TOTAL_REQ = N_FILMS * REQ_PER_FILM


def fake_fetch():
    """Stand-in for get(): a fixed-latency network round trip, no server."""
    time.sleep(FAKE_LATENCY)


def baseline_serial():
    """Old design: one request at a time, fixed sleep after each."""
    for _ in range(N_FILMS):
        for _ in range(REQ_PER_FILM):
            fake_fetch()
            time.sleep(OLD_DELAY)


def concurrent_run(rate):
    """New design: films parallel, frames parallel, one shared host limiter."""
    limiter = scrape.RateLimiter(rate)

    def one_request():
        limiter.wait()      # real limiter — the thing we're testing
        fake_fetch()

    def run_film(_):
        with ThreadPoolExecutor(max_workers=scrape.FRAME_WORKERS) as ex:
            list(ex.map(lambda _: one_request(), range(REQ_PER_FILM)))

    with ThreadPoolExecutor(max_workers=scrape.FILM_WORKERS) as fex:
        list(fex.map(run_film, range(N_FILMS)))


def timed(fn, *a):
    t0 = time.monotonic()
    fn(*a)
    return time.monotonic() - t0


def row(label, rate, wall, base):
    rate_s = "-" if rate is None else f"{rate:g}"
    print(f"{label:<16}{rate_s:>6}{wall:>9.1f}s{TOTAL_REQ / wall:>9.1f}"
          f"{base / wall:>9.2f}x")


def main():
    print(f"workload: {N_FILMS} films x {REQ_PER_FILM} req = {TOTAL_REQ} reqs, "
          f"fake latency {FAKE_LATENCY:g}s/req")
    print(f"config:   FILM_WORKERS={scrape.FILM_WORKERS} "
          f"FRAME_WORKERS={scrape.FRAME_WORKERS}\n")
    print(f"{'mode':<16}{'rate':>6}{'wall':>10}{'req/s':>9}{'speedup':>10}")
    print("-" * 51)

    base = timed(baseline_serial)
    row("serial (old)", None, base, base)
    for rate in RATE_SWEEP:
        row("concurrent", rate, timed(concurrent_run, rate), base)

    print(f"\nfloor at a given rate ~= {TOTAL_REQ}/rate sec (all reqs share one "
          f"host bucket).\nreal runs add curl spawn (~10-30ms/req) + network "
          f"jitter, not modeled here.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["tqdm", "curl_cffi"]
# ///
"""Turn a Letterboxd account into a titles file the scraper understands —
one "Title (Year)" per line.

Two sources:

scrape (default): the public profile grid pages, fetched with curl_cffi
    impersonating Chrome's TLS fingerprint — plain curl gets Cloudflare's
    "Just a moment" JS challenge on paginated pages, but the fingerprint
    match passes it. If Cloudflare still blocks, the script stops with a
    pointer to --export.

--export: the official data export zip from letterboxd.com/user/exportdata —
    complete, instant, Cloudflare-proof. Reads watched.csv (+ watchlist.csv
    with --watchlist) from the zip.

Usage:
    uv run pipeline/0_letterboxd.py aliozkaya [--watchlist]
    uv run pipeline/0_letterboxd.py aliozkaya --export ~/Downloads/letterboxd-*.zip
"""
import argparse
import csv
import html
import io
import re
import sys
import time
import zipfile
from pathlib import Path

from curl_cffi import requests as cffi_requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DELAY = 1.0              # seconds between page fetches, be polite
ITEM = re.compile(r'data-item-name="([^"]+)"')


def get(url, tries=3):
    for attempt in range(1, tries + 1):
        try:
            r = cffi_requests.get(url, impersonate="chrome", timeout=20)
            if r.status_code == 200:
                return r.text
            err = f"HTTP {r.status_code}"
        except Exception as e:
            err = str(e)[:200]
        if attempt < tries:
            wait = 5 * attempt
            tqdm.write(f"[retry {attempt}/{tries - 1}] {url}: {err} "
                       f"— waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(err)


def scrape_section(user, section):
    """All titles from a paginated poster grid. Letterboxd's Cloudflare
    usually challenges /page/N/ for N >= 2 from non-browser clients — if
    that happens we bail out with whatever we got and say so."""
    titles, page, cooled = [], 1, False
    bar = tqdm(desc=section, unit="page")
    while True:
        suffix = "" if page == 1 else f"page/{page}/"
        body = get(f"https://letterboxd.com/{user}/{section}/{suffix}")
        if "Just a moment" in body:      # Cloudflare JS challenge
            if not cooled:               # transient escalation: one cooldown
                cooled = True
                tqdm.write(f"[cloudflare] challenged on {section} page "
                           f"{page} — cooling down 30s, then one retry")
                time.sleep(30)
                continue
            bar.close()
            raise SystemExit(
                f"Cloudflare blocked {section} page {page} "
                f"({len(titles)} titles fetched so far).\n"
                f"Use the official export instead:\n"
                f"  1. download your zip from "
                f"letterboxd.com/user/exportdata\n"
                f"  2. uv run pipeline/0_letterboxd.py {user} "
                f"--export <the-zip>")
        cooled = False
        found = [html.unescape(t) for t in ITEM.findall(body)]
        if not found:
            break
        titles.extend(found)
        bar.update(1)
        page += 1
        time.sleep(DELAY)
    bar.close()
    return titles


def read_export_csv(zf, name):
    """Titles from one csv inside the export zip (columns include
    Name, Year)."""
    matches = [n for n in zf.namelist() if n.endswith(name)]
    if not matches:
        return None
    rows = csv.DictReader(io.TextIOWrapper(zf.open(matches[0]),
                                           encoding="utf-8"))
    out = []
    for r in rows:
        title, year = (r.get("Name") or "").strip(), \
            (r.get("Year") or "").strip()
        if title:
            out.append(f"{title} ({year})" if year else title)
    return out


def from_export(path, want_watchlist):
    zf = zipfile.ZipFile(path)
    watched = read_export_csv(zf, "watched.csv")
    if watched is None:
        sys.exit(f"{path}: no watched.csv inside — is this the Letterboxd "
                 f"data-export zip?")
    titles = list(watched)
    n_watch = 0
    if want_watchlist:
        wl = read_export_csv(zf, "watchlist.csv") or []
        n_watch = len(wl)
        titles += wl
    return titles, len(watched), n_watch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("username", help="Letterboxd username (used for the "
                                     "output filename and scrape mode)")
    ap.add_argument("--export", help="path to the official data-export zip "
                                     "(letterboxd.com/user/exportdata) "
                                     "— complete and Cloudflare-proof")
    ap.add_argument("--watchlist", action="store_true",
                    help="also include the watchlist")
    ap.add_argument("--out", default=None,
                    help="output file (default letterboxd_<user>.txt)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else \
        ROOT / f"letterboxd_{args.username}.txt"

    if args.export:
        titles, n_films, n_watch = from_export(args.export, args.watchlist)
    else:
        titles = scrape_section(args.username, "films")
        n_films = len(titles)
        n_watch = 0
        if args.watchlist:
            wl = scrape_section(args.username, "watchlist")
            n_watch = len(wl)
            titles += wl
    if not titles:
        sys.exit(f"no films found — is letterboxd.com/{args.username} "
                 f"public and spelled right?")

    seen, unique = set(), []
    for t in titles:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    src = "export" if args.export else "profile scrape"
    lines = [f"# letterboxd.com/{args.username} ({src}) — "
             f"{n_films} watched"
             + (f", {n_watch} watchlist" if args.watchlist else "")]
    lines += unique
    out.write_text("\n".join(lines) + "\n")
    print(f"{len(unique)} unique titles -> {out}")
    print(f"next: uv run pipeline/1_scrape_frames.py {out.name}")


if __name__ == "__main__":
    main()

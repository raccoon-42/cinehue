# cinehue

Visualize film taste as color. The core formula:

**map(movie) and map(user) = render(Σ measures of their movies)**

A film is a measure over OKLab color space — mass 1, split across its
identity stops (accent + subject-chroma) proportionally to chroma. A
collection is the sum of its films' measures. The algebra is linear,
associative, and deterministic; all nonlinearity lives in `render()`.

## Structure

```
lib/          shared core: oklab.py (conversions), measure.py (Step 1 + Σ),
              pixels.py (numpy variants), paths.py (data locations)
pipeline/     data stages, in run order:
              0_letterboxd.py       public profile -> titles txt (optional)
              1_scrape_frames.py    FilmGrab + TMDB stills -> data/frames/
              2_subjects.py         U²-Net subject layer   -> subjects.json
              3_extract_palettes.py OKLab k-means          -> palettes.json
render/       taste_space.py — the canonical renderer, five modes:
              atoms | clustered | gradient | sum | spectrum
experiments/  historical render branches (ideas 1-5), kept runnable
data/         gitignored: frames, json outputs, preview html
```

## Run

The pipeline starts from a titles file — one `Title (Year)` per line, `#`
for comments. Bring your own (`watchlist.txt` is the scraper's default), or
pull it from Letterboxd:

```sh
# scrape a public profile (curl_cffi impersonates Chrome's TLS fingerprint;
# Cloudflare may still rate-limit long crawls):
uv run pipeline/0_letterboxd.py <username> [--watchlist]
# reliable fallback: the official export zip from letterboxd.com/user/exportdata
uv run pipeline/0_letterboxd.py <username> --export <zip> [--watchlist]
uv run pipeline/1_scrape_frames.py letterboxd_<username>.txt
uv run pipeline/2_subjects.py                 # resumes; --redo to recompute
uv run pipeline/3_extract_palettes.py         # resumes; --redo to recompute
uv run render/taste_space.py                  # whole collection
uv run render/taste_space.py --film "Ran (1985)"
uv run render/taste_space.py --mode spectrum --bandwidth 6
```

TMDB fallback needs `TMDB_API_KEY` in `.env` (auto-loaded). Frames come from
FilmGrab stills or TMDB backdrops — never full films.

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md) — free for personal, research,
and nonprofit use. **Commercial use is not permitted.**
Copyright © 2026 Ali Özkaya.

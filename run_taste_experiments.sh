#!/usr/bin/env bash
# Run all taste experiments (ideas 1-4) off data/palettes.json and open
# their previews. Usage: ./run_taste_experiments.sh [--no-open]
set -euo pipefail
cd "$(dirname "$0")"

uv run experiment_taste.py            # idea 1: illuminated wheel
uv run experiment_taste_square.py     # idea 2: square, watched-only
uv run experiment_taste_pixels.py     # idea 3: pixel quilt (ribbon + square)
uv run experiment_taste_ring.py       # idea 4: broken taste ring

if [[ "${1:-}" != "--no-open" ]]; then
  open data/preview_taste.html \
       data/preview_taste_square.html \
       data/preview_taste_pixels.html \
       data/preview_taste_ring.html
fi

#!/usr/bin/env bash
# Run the historical taste experiments (ideas 1-5) off data/palettes.json and
# open their previews. The canonical renderer is render/taste_space.py.
# Usage: ./run_experiments.sh [--no-open]
set -euo pipefail
cd "$(dirname "$0")"

uv run experiments/taste_wheel.py     # idea 1: illuminated wheel
uv run experiments/taste_square.py    # idea 2: square, watched-only
uv run experiments/taste_pixels.py    # idea 3: pixel quilt (ribbon + square)
uv run experiments/taste_ring.py      # idea 4: broken taste ring
uv run experiments/taste_stripes.py   # idea 5: palette stripes

if [[ "${1:-}" != "--no-open" ]]; then
  open data/preview_taste.html \
       data/preview_taste_square.html \
       data/preview_taste_pixels.html \
       data/preview_taste_ring.html \
       data/preview_taste_palette.html
fi

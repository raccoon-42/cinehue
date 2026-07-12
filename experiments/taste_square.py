#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Experiment idea 2: taste as a SQUARE woven only from watched films.

No global map this time — nothing exists on the canvas except the films you
watched. Each film contributes one color (its identity blend, or the primary
signal when the film is genuinely two-toned). Colors are sorted by hue and
laid along a serpentine path through a square grid, then blended per-pixel
in OKLab so each color flows into its neighbors the way hues flow around a
color wheel — but this wheel is only YOUR hues, packed into a square.
Achromatic films sit together at the end of the path as a gray passage.

Outputs data/preview_taste_square.html. Hover shows which film a region is.

Usage:
    uv run experiment_taste_square.py [--size 560] [--soft 0.9]
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PALETTES = ROOT / "data" / "palettes.json"
OUT = ROOT / "data" / "preview_taste_square.html"

C_MIN = 0.02          # below this a color has no meaningful hue
MERGE_DEG = 45        # hue spread within which blending two signals is honest


def hex_to_oklab(hexstr):
    r, g, b = (int(hexstr[i:i + 2], 16) / 255 for i in (1, 3, 5))
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = lin(r), lin(g), lin(b)
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    bb = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return L, a, bb


def oklab_to_hex(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    def enc(c):
        c = min(1.0, max(0.0, c))
        c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        return round(c * 255)
    return f"#{enc(r):02x}{enc(g):02x}{enc(bb):02x}"


def lab_polar(L, a, b):
    return math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def hue_dist(h1, h2):
    return abs((h1 - h2 + 540) % 360 - 180)


def film_color(film):
    """One color per film: OKLab blend of its identity stops when their hues
    agree (or it is achromatic), the strongest stop when they genuinely
    disagree — blending far-apart hues would invent a color the film
    doesn't have."""
    stops = [dict(zip(("L", "a", "b"), hex_to_oklab(s["hex"])), hex=s["hex"])
             for s in film.get("mood", {}).get("wheel", [])]
    if not stops:
        return None
    for s in stops:
        s["C"], s["h"] = lab_polar(s["L"], s["a"], s["b"])
    chrom = [s for s in stops if s["C"] >= C_MIN]
    if chrom:
        spread = max(hue_dist(p["h"], q["h"]) for p in chrom for q in chrom)
        pick = chrom if spread <= MERGE_DEG else \
            [max(chrom, key=lambda s: s["C"])]
    else:
        pick = stops
    tw = sum(max(s["C"], 0.005) for s in pick)
    L = sum(s["L"] * max(s["C"], 0.005) for s in pick) / tw
    a = sum(s["a"] * max(s["C"], 0.005) for s in pick) / tw
    b = sum(s["b"] * max(s["C"], 0.005) for s in pick) / tw
    C, h = lab_polar(L, a, b)
    return {"title": film["title"], "hex": oklab_to_hex(L, a, b),
            "L": round(L, 4), "a": round(a, 4), "b": round(b, 4),
            "C": round(C, 4), "h": round(h, 1)}


PAINTER_JS = """
const W = %(size)d, SOFT = %(soft)f;
const FILMS = %(films_json)s;   // hue-sorted [{title,hex,L,a,b,C,h}]

function oklabToRgb(L, a, b) {
  let l = L + 0.3963377774 * a + 0.2158037573 * b;
  let m = L - 0.1055613458 * a - 0.0638541728 * b;
  let s = L - 0.0894841775 * a - 1.2914855480 * b;
  l = l ** 3; m = m ** 3; s = s ** 3;
  let r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  let g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  let bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
  const enc = c => {
    c = Math.min(1, Math.max(0, c));
    return (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055) * 255;
  };
  return [enc(r), enc(g), enc(bb)];
}

// serpentine grid: hue order snakes through the square so path neighbors
// are also spatial neighbors — that is what makes the transitions wheel-like
const N = FILMS.length;
const COLS = Math.ceil(Math.sqrt(N)), ROWS = Math.ceil(N / COLS);
const CW = W / COLS, CH = W / ROWS;
const POINTS = FILMS.map((f, i) => {
  const row = Math.floor(i / COLS);
  const col = row %% 2 === 0 ? i %% COLS : COLS - 1 - (i %% COLS);
  return { x: (col + 0.5) * CW, y: (row + 0.5) * CH, f };
});

function paint() {
  const cv = document.getElementById('square'), ctx = cv.getContext('2d');
  const img = ctx.createImageData(W, W);
  const sigma = SOFT * Math.min(CW, CH), cut2 = (3 * sigma) ** 2;
  const inv2s2 = 1 / (2 * sigma * sigma);
  for (let y = 0; y < W; y++) {
    for (let x = 0; x < W; x++) {
      let sw = 0, L = 0, a = 0, b = 0, bestD = Infinity, bestF = null;
      for (const p of POINTS) {
        const d2 = (x - p.x) * (x - p.x) + (y - p.y) * (y - p.y);
        if (d2 < bestD) { bestD = d2; bestF = p.f; }
        if (d2 > cut2) continue;
        const w = Math.exp(-d2 * inv2s2);
        sw += w; L += w * p.f.L; a += w * p.f.a; b += w * p.f.b;
      }
      if (sw > 0) { L /= sw; a /= sw; b /= sw; }
      else { L = bestF.L; a = bestF.a; b = bestF.b; }
      const [rr, gg, bb] = oklabToRgb(L, a, b);
      const i = (y * W + x) * 4;
      img.data[i] = rr; img.data[i + 1] = gg; img.data[i + 2] = bb;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function initHover() {
  const cv = document.getElementById('square');
  const tip = document.getElementById('tip');
  cv.addEventListener('mousemove', e => {
    const rc = cv.getBoundingClientRect();
    const mx = e.clientX - rc.left, my = e.clientY - rc.top;
    let best = null, bd = Infinity;
    for (const p of POINTS) {
      const d2 = (mx - p.x) * (mx - p.x) + (my - p.y) * (my - p.y);
      if (d2 < bd) { bd = d2; best = p.f; }
    }
    tip.textContent = `${best.title} — h=${best.h}° C=${best.C}`;
    tip.style.display = 'block';
    tip.style.left = (e.pageX + 14) + 'px';
    tip.style.top = (e.pageY + 10) + 'px';
  });
  cv.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
}

paint(); initHover();
"""


def render(colors, size, soft):
    js = PAINTER_JS % {"size": size, "soft": soft,
                       "films_json": json.dumps(colors, ensure_ascii=False)}
    css = f"""
    body{{background:#0c0c10;color:#cfcfd4;font:14px/1.5 -apple-system,sans-serif;
         margin:2rem}}
    h1{{font-size:1.1rem}} h2{{font-size:1rem;margin-top:2.5rem}}
    canvas{{border-radius:12px}}
    .sw{{display:inline-block;width:20px;height:20px;border-radius:4px;
        vertical-align:middle;margin-right:4px}}
    .film{{display:flex;gap:.6rem;align-items:center;margin:.25rem 0}}
    .dim{{color:#77777f}}
    #tip{{position:absolute;display:none;background:#000d;color:#eee;
         padding:4px 9px;border-radius:6px;font-size:12px;
         pointer-events:none;white-space:nowrap;z-index:9}}
    """
    h = [f"<meta charset='utf-8'><title>cinehue taste square</title>"
         f"<style>{css}</style>",
         f"<h1>taste square <span class='dim'>— only your {len(colors)} "
         f"films exist here: one color each, hue-sorted on a serpentine "
         f"path, flowing into each other. hover to see who is who.</span>"
         f"</h1>",
         f"<canvas id='square' width='{size}' height='{size}'></canvas>",
         "<div id='tip'></div>",
         "<h2>path order</h2>"]
    for c in colors:
        h.append(f"<div class='film'><span class='sw' style='background:"
                 f"{c['hex']}'></span><span class='dim'>h{c['h']:.0f} "
                 f"C{c['C']:.2f}</span> {c['title']}</div>")
    h.append(f"<script>{js}</script>")
    OUT.write_text("\n".join(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=560, help="square side in px")
    ap.add_argument("--soft", type=float, default=0.9,
                    help="blend softness in cell widths (higher = dreamier)")
    args = ap.parse_args()

    films = json.loads(PALETTES.read_text())
    colors = [c for c in (film_color(f) for f in films) if c]
    # hue order for chromatic films; achromatic gather at the end (a gray
    # passage) instead of landing at meaningless hue positions
    chrom = sorted((c for c in colors if c["C"] >= C_MIN),
                   key=lambda c: c["h"])
    gray = sorted((c for c in colors if c["C"] < C_MIN),
                  key=lambda c: c["L"])
    render(chrom + gray, args.size, args.soft)
    print(f"{len(colors)} films ({len(gray)} achromatic) -> {OUT}")


if __name__ == "__main__":
    main()

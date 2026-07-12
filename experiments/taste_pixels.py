#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Experiment idea 3: taste as a pixel quilt — a tiny block per movie.

Every film is one cell of its identity color. Cells are sorted by hue and
packed edge to edge; the display interpolates between cell centers in OKLab
(smoothstep-eased, per pixel), so neighboring films flow into each other
like hues around a wheel. A toggle shows the raw blocks instead.

Two arrangements of the same blocks:
  - ribbon: one row, hue-sorted left to right — your taste as a spectrum
  - square: serpentine grid, hue order snaking through a square
Achromatic films sit at the end as a gray passage. Hover names the film.

Outputs data/preview_taste_pixels.html.

Usage:
    uv run experiment_taste_pixels.py
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PALETTES = ROOT / "data" / "palettes.json"
OUT = ROOT / "data" / "preview_taste_pixels.html"

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
    """One color per film (same honesty rule as ideas 1-2): OKLab blend of
    identity stops when hues agree, strongest stop when they disagree."""
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
const FILMS = %(films_json)s;   // hue-sorted [{title,hex,L,a,b,C,h}]
const N = FILMS.length;
const COLS = Math.ceil(Math.sqrt(N)), ROWS = Math.ceil(N / COLS);
const RIB_W = 1100, RIB_H = 90, SQ_W = 560;

function gridIndex(row, col) {
  return row %% 2 === 0 ? row * COLS + col
                        : row * COLS + (COLS - 1 - col);
}
const clampI = i => Math.max(0, Math.min(i, N - 1));

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

// smooth transitions = true per-pixel OKLab interpolation between block
// centers (smoothstep-eased), painted at display resolution — browser
// bilinear upscaling of a tiny canvas was not smooth enough
const ease = t => t * t * (3 - 2 * t);

function labAtGrid(row, col) {
  const f = FILMS[clampI(gridIndex(
    Math.max(0, Math.min(row, ROWS - 1)),
    Math.max(0, Math.min(col, COLS - 1))))];
  return f;
}

function paintSmooth() {
  const rb = document.getElementById('ribbon');
  rb.width = RIB_W; rb.height = RIB_H;
  const rctx = rb.getContext('2d');
  const rimg = rctx.createImageData(RIB_W, RIB_H);
  for (let x = 0; x < RIB_W; x++) {
    const t = (x / RIB_W) * N - 0.5;
    const i0 = clampI(Math.floor(t)), i1 = clampI(i0 + 1);
    const u = ease(Math.max(0, Math.min(1, t - Math.floor(t))));
    const A = FILMS[i0], B = FILMS[i1];
    const [rr, gg, bb] = oklabToRgb(A.L + (B.L - A.L) * u,
                                    A.a + (B.a - A.a) * u,
                                    A.b + (B.b - A.b) * u);
    for (let y = 0; y < RIB_H; y++) {
      const i = (y * RIB_W + x) * 4;
      rimg.data[i] = rr; rimg.data[i + 1] = gg;
      rimg.data[i + 2] = bb; rimg.data[i + 3] = 255;
    }
  }
  rctx.putImageData(rimg, 0, 0);

  const sq = document.getElementById('grid');
  sq.width = SQ_W; sq.height = SQ_W;
  const sctx = sq.getContext('2d');
  const simg = sctx.createImageData(SQ_W, SQ_W);
  for (let y = 0; y < SQ_W; y++) {
    const gy = (y / SQ_W) * ROWS - 0.5;
    const r0 = Math.floor(gy), v = ease(Math.max(0, Math.min(1, gy - r0)));
    for (let x = 0; x < SQ_W; x++) {
      const gx = (x / SQ_W) * COLS - 0.5;
      const c0 = Math.floor(gx), u = ease(Math.max(0, Math.min(1, gx - c0)));
      const p00 = labAtGrid(r0, c0), p01 = labAtGrid(r0, c0 + 1);
      const p10 = labAtGrid(r0 + 1, c0), p11 = labAtGrid(r0 + 1, c0 + 1);
      const L = (p00.L * (1 - u) + p01.L * u) * (1 - v) +
                (p10.L * (1 - u) + p11.L * u) * v;
      const a = (p00.a * (1 - u) + p01.a * u) * (1 - v) +
                (p10.a * (1 - u) + p11.a * u) * v;
      const b = (p00.b * (1 - u) + p01.b * u) * (1 - v) +
                (p10.b * (1 - u) + p11.b * u) * v;
      const [rr, gg, bb] = oklabToRgb(L, a, b);
      const i = (y * SQ_W + x) * 4;
      simg.data[i] = rr; simg.data[i + 1] = gg;
      simg.data[i + 2] = bb; simg.data[i + 3] = 255;
    }
  }
  sctx.putImageData(simg, 0, 0);
}

function paintBlocks() {
  const rb = document.getElementById('ribbon');
  rb.width = RIB_W; rb.height = RIB_H;
  const rctx = rb.getContext('2d');
  FILMS.forEach((f, i) => {
    rctx.fillStyle = f.hex;
    rctx.fillRect(i * RIB_W / N, 0, RIB_W / N + 1, RIB_H);
  });
  const sq = document.getElementById('grid');
  sq.width = SQ_W; sq.height = SQ_W;
  const sctx = sq.getContext('2d');
  for (let row = 0; row < ROWS; row++)
    for (let col = 0; col < COLS; col++) {
      sctx.fillStyle = FILMS[clampI(gridIndex(row, col))].hex;
      sctx.fillRect(col * SQ_W / COLS, row * SQ_W / ROWS,
                    SQ_W / COLS + 1, SQ_W / ROWS + 1);
    }
}

function hookHover(cv, toIndex) {
  const tip = document.getElementById('tip');
  cv.addEventListener('mousemove', e => {
    const rc = cv.getBoundingClientRect();
    const i = toIndex((e.clientX - rc.left) / rc.width,
                      (e.clientY - rc.top) / rc.height);
    const f = FILMS[Math.max(0, Math.min(i, N - 1))];
    tip.textContent = `${f.title} — h=${f.h}° C=${f.C}`;
    tip.style.display = 'block';
    tip.style.left = (e.pageX + 14) + 'px';
    tip.style.top = (e.pageY + 10) + 'px';
  });
  cv.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
}

paintSmooth();
hookHover(document.getElementById('ribbon'),
          (fx, fy) => Math.floor(fx * N));
hookHover(document.getElementById('grid'), (fx, fy) => {
  const col = Math.min(COLS - 1, Math.floor(fx * COLS));
  const row = Math.min(ROWS - 1, Math.floor(fy * ROWS));
  return gridIndex(row, col);
});

let smooth = true;
document.getElementById('toggle').onclick = () => {
  smooth = !smooth;
  if (smooth) paintSmooth(); else paintBlocks();
  document.getElementById('toggle').textContent =
    smooth ? 'show raw blocks' : 'show transitions';
};
"""


def render(colors):
    js = PAINTER_JS % {"films_json": json.dumps(colors, ensure_ascii=False)}
    css = """
    body{background:#0c0c10;color:#cfcfd4;font:14px/1.5 -apple-system,sans-serif;
         margin:2rem}
    h1{font-size:1.1rem} h2{font-size:1rem;margin-top:2.5rem}
    canvas{border-radius:10px;image-rendering:auto}
    #ribbon{width:100%;max-width:1100px;height:90px;display:block}
    #grid{width:560px;height:560px;display:block;margin-top:1.5rem}
    .sw{display:inline-block;width:20px;height:20px;border-radius:4px;
        vertical-align:middle;margin-right:4px}
    .film{display:flex;gap:.6rem;align-items:center;margin:.25rem 0}
    .dim{color:#77777f}
    #tip{position:absolute;display:none;background:#000d;color:#eee;
         padding:4px 9px;border-radius:6px;font-size:12px;
         pointer-events:none;white-space:nowrap;z-index:9}
    button{background:#222;color:#ccc;border:1px solid #444;border-radius:6px;
           padding:4px 10px;cursor:pointer;margin-top:1rem}
    """
    h = [f"<meta charset='utf-8'><title>cinehue pixel quilt</title>"
         f"<style>{css}</style>",
         f"<h1>pixel quilt <span class='dim'>— {len(colors)} films, one "
         f"cell each, hue-sorted, blended in OKLab between cell centers. "
         f"hover names the film.</span></h1>",
         "<h2>ribbon — taste as a spectrum</h2>",
         "<canvas id='ribbon'></canvas>",
         "<h2>square — hue order snaking through a serpentine grid</h2>",
         "<canvas id='grid'></canvas>",
         "<button id='toggle'>show raw blocks</button>",
         "<div id='tip'></div>",
         "<h2>path order</h2>"]
    for c in colors:
        h.append(f"<div class='film'><span class='sw' style='background:"
                 f"{c['hex']}'></span><span class='dim'>h{c['h']:.0f} "
                 f"C{c['C']:.2f}</span> {c['title']}</div>")
    h.append(f"<script>{js}</script>")
    OUT.write_text("\n".join(h))


def main():
    argparse.ArgumentParser().parse_args()
    films = json.loads(PALETTES.read_text())
    colors = [c for c in (film_color(f) for f in films) if c]
    chrom = sorted((c for c in colors if c["C"] >= C_MIN),
                   key=lambda c: c["h"])
    gray = sorted((c for c in colors if c["C"] < C_MIN),
                  key=lambda c: c["C"])
    render(chrom + gray)
    print(f"{len(colors)} films ({len(gray)} achromatic) -> {OUT}")


if __name__ == "__main__":
    main()

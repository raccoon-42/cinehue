#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Experiment idea 5: the palette square — linear algebra, nonlinear rendering.

The underlying object is a MEASURE over OKLab space. Every film contributes
total mass 1, split across its identity stops (accent + subject-chroma, the
mood-wheel stops) proportionally to chroma. Collections just ADD measures —
associative, commutative, no ceiling. Nothing nonlinear ever touches the data.

The palette you SEE is a rendering of that measure:
  - chromatic atoms are clustered by cutting the hue circle at its largest
    gaps until every cluster spans <= 45 deg (the honesty rule: only blend
    hues that are honestly one color; a cut never invents an intermediate)
  - each cluster becomes one swatch (weighted OKLab mean of its atoms)
  - swatch stripe width in the square is proportional to mass ** gamma:
      gamma 0   -> presence   (every swatch equal; today's ring philosophy)
      gamma 0.5 -> portrait   (dominance compressed, rare stuff visible)
      gamma 1   -> measurement (widths are true mass shares)
    All three squares are rendered side by side from the SAME measure.
  - achromatic mass (C < 0.02) becomes a gray stripe at the right end.

Works at both scales: no --film flag = the whole collection; --film TITLE
(substring, case-insensitive) = that one movie's palette, same pipeline.

The measure code itself (film_atoms, hue_clusters, build_measure) lives in
lib/measure.py — this file is only the stripe render, kept as the record of
the geometry Ali rejected ("not what I imagine"); the encoding survived.

Reads data/palettes.json, writes data/preview_taste_palette.html.

Usage:
    uv run experiments/taste_stripes.py [--gamma 0.5] [--film ran]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.measure import build_measure
from lib.paths import DATA, PALETTES

OUT = DATA / "preview_taste_palette.html"

SQ = 340              # px per square
MIN_W_PX = 2          # never render a swatch thinner than this


def widths(swatches, gamma):
    """Stripe widths in px for one square: width proportional to mass**gamma,
    floored at MIN_W_PX so nothing vanishes entirely."""
    raw = [s["mass"] ** gamma for s in swatches]
    tot = sum(raw)
    w = [max(MIN_W_PX, SQ * r / tot) for r in raw]
    scale = SQ / sum(w)
    return [round(x * scale, 2) for x in w]


def render(swatches, gammas, title, out):
    squares = []
    labels = {0.0: "presence", 0.5: "portrait", 1.0: "measurement"}
    for g in gammas:
        ws = widths(swatches, g)
        stripes = "".join(
            f"<div class='stripe' style='width:{w}px;background:{s['hex']}'"
            f" data-tip=\"{s['hex']} — {s['share']*100:.1f}% of taste, "
            f"{s['n_films']} film(s): "
            f"{', '.join(s['films'])}"
            f"{'…' if s['n_films'] > len(s['films']) else ''}\"></div>"
            for s, w in zip(swatches, ws))
        name = labels.get(g, "")
        squares.append(
            f"<figure><div class='sq'>{stripes}</div>"
            f"<figcaption>&gamma; = {g:g}"
            f"{' — ' + name if name else ''}</figcaption></figure>")
    rows = []
    for s in swatches:
        hue = f"h{s['h']:.0f}" if s["h"] is not None else "gray"
        rows.append(
            f"<div class='film'><span class='sw' style='background:"
            f"{s['hex']}'></span><b>{s['share']*100:5.1f}%</b> "
            f"<span class='dim'>{hue} C{s['C']:.2f} — {s['n_films']} "
            f"film(s):</span> {', '.join(s['films'])}"
            f"{'…' if s['n_films'] > len(s['films']) else ''}</div>")
    css = f"""
    body{{background:#0c0c10;color:#cfcfd4;font:14px/1.5 -apple-system,sans-serif;
         margin:2rem}}
    h1{{font-size:1.1rem}} h2{{font-size:1rem;margin-top:2.5rem}}
    #row{{display:flex;gap:2rem;flex-wrap:wrap}}
    .sq{{display:flex;width:{SQ}px;height:{SQ}px;border-radius:6px;
        overflow:hidden}}
    .stripe{{height:100%}}
    figcaption{{color:#8a8a92;font-size:13px;margin-top:.5rem;
               text-align:center}}
    .sw{{display:inline-block;width:20px;height:20px;border-radius:4px;
        vertical-align:middle;margin-right:6px}}
    .film{{margin:.3rem 0}}
    .dim{{color:#77777f}}
    #tip{{position:absolute;display:none;background:#000d;color:#eee;
         padding:4px 9px;border-radius:6px;font-size:12px;
         pointer-events:none;white-space:nowrap;z-index:9}}
    """
    js = """
    const tip = document.getElementById('tip');
    document.querySelectorAll('.stripe').forEach(el => {
      el.addEventListener('mousemove', e => {
        tip.textContent = el.dataset.tip;
        tip.style.display = 'block';
        tip.style.left = (e.pageX + 14) + 'px';
        tip.style.top = (e.pageY + 10) + 'px';
      });
      el.addEventListener('mouseleave', () => tip.style.display = 'none');
    });
    """
    h = [f"<meta charset='utf-8'><title>cinehue palette square</title>"
         f"<style>{css}</style>",
         f"<h1>palette square <span class='dim'>— {title}. same measure, "
         f"three renderings; hover a stripe.</span></h1>",
         f"<div id='row'>{''.join(squares)}</div>",
         "<div id='tip'></div>",
         "<h2>swatches (true mass shares)</h2>",
         *rows,
         f"<script>{js}</script>"]
    out.write_text("\n".join(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="exponent for the middle square (0=presence, "
                         "1=measurement)")
    ap.add_argument("--film", help="substring match: render ONE film's "
                                   "palette instead of the collection")
    args = ap.parse_args()

    films = json.loads(PALETTES.read_text())
    out = OUT
    if args.film:
        q = args.film.lower()
        exact = [f for f in films if f["title"].lower() == q]
        films = exact or [f for f in films if q in f["title"].lower()]
        if not films:
            raise SystemExit(f"no film matching {args.film!r}")
        if len(films) > 1:
            print(f"note: {args.film!r} matched {len(films)} films: "
                  + ", ".join(f["title"] for f in films))
        title = " + ".join(f["title"] for f in films)
        out = OUT.with_stem(OUT.stem + "_film")
    else:
        title = f"{len(films)} films"

    swatches = build_measure(films)
    if not swatches:
        raise SystemExit("no identity stops found — run the subject "
                         "experiment and extraction first")
    render(swatches, [0.0, args.gamma, 1.0], title, out)
    print(f"{len(swatches)} swatches from {len(films)} film(s) -> {out}")


if __name__ == "__main__":
    main()

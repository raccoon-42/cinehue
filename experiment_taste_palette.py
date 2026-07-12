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

Reads data/palettes.json, writes data/preview_taste_palette.html.

Usage:
    uv run experiment_taste_palette.py [--gamma 0.5] [--film ran]
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PALETTES = ROOT / "data" / "palettes.json"
OUT = ROOT / "data" / "preview_taste_palette.html"

C_MIN = 0.02          # below this a color has no meaningful hue
MERGE_DEG = 45        # max hue span an honest swatch may cover
SQ = 340              # px per square
MIN_W_PX = 2          # never render a swatch thinner than this


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


def film_atoms(film):
    """One film -> atoms of a measure with total mass 1.

    Chromatic stops split the film's mass proportionally to chroma;
    a film with no chromatic stop puts all its mass on the gray axis."""
    stops = []
    for s in film.get("mood", {}).get("wheel", []):
        L, a, b = hex_to_oklab(s["hex"])
        C, h = lab_polar(L, a, b)
        stops.append({"L": L, "a": a, "b": b, "C": C, "h": h,
                      "film": film["title"]})
    if not stops:
        return []
    chrom = [s for s in stops if s["C"] >= C_MIN]
    if not chrom:
        gL = sum(s["L"] for s in stops) / len(stops)
        return [{"L": gL, "a": 0.0, "b": 0.0, "C": 0.0, "h": 0.0,
                 "w": 1.0, "film": film["title"], "gray": True}]
    tot = sum(s["C"] for s in chrom)
    return [dict(s, w=s["C"] / tot, gray=False) for s in chrom]


def hue_clusters(atoms):
    """Cut the hue circle at its largest gaps until every cluster spans
    <= MERGE_DEG. Clusters stay hue-contiguous, so blending inside one is
    honest and no cut ever invents an intermediate color."""
    atoms = sorted(atoms, key=lambda a: a["h"])
    n = len(atoms)
    if n == 1:
        return [atoms]
    gaps = [(atoms[(i + 1) % n]["h"] - atoms[i]["h"]) % 360 for i in range(n)]
    # open the circle at the single largest gap, then work on a line
    start = (max(range(n), key=lambda i: gaps[i]) + 1) % n
    line = [atoms[(start + i) % n] for i in range(n)]
    clusters = [line]
    done = []
    while clusters:
        cl = clusters.pop()
        span = (cl[-1]["h"] - cl[0]["h"]) % 360
        if span <= MERGE_DEG or len(cl) == 1:
            done.append(cl)
            continue
        cut = max(range(len(cl) - 1),
                  key=lambda i: (cl[i + 1]["h"] - cl[i]["h"]) % 360)
        clusters.append(cl[:cut + 1])
        clusters.append(cl[cut + 1:])
    return sorted(done, key=lambda cl: cl[0]["h"])


def swatch(cluster):
    """Weighted OKLab mean of a hue-contiguous cluster (honest blend)."""
    tw = sum(a["w"] for a in cluster)
    L = sum(a["L"] * a["w"] for a in cluster) / tw
    aa = sum(a["a"] * a["w"] for a in cluster) / tw
    bb = sum(a["b"] * a["w"] for a in cluster) / tw
    C, h = lab_polar(L, aa, bb)
    films = {}
    for a in cluster:
        films[a["film"]] = films.get(a["film"], 0.0) + a["w"]
    top = sorted(films.items(), key=lambda kv: -kv[1])
    return {"hex": oklab_to_hex(L, aa, bb), "mass": tw,
            "C": round(C, 4), "h": round(h, 1),
            "n_films": len(films),
            "films": [t for t, _ in top[:8]]}


def build_measure(films):
    atoms = [a for f in films for a in film_atoms(f)]
    chrom = [a for a in atoms if not a["gray"]]
    grays = [a for a in atoms if a["gray"]]
    swatches = [swatch(cl) for cl in hue_clusters(chrom)] if chrom else []
    if grays:
        tw = sum(a["w"] for a in grays)
        gL = sum(a["L"] * a["w"] for a in grays) / tw
        swatches.append({"hex": oklab_to_hex(gL, 0.0, 0.0), "mass": tw,
                         "C": 0.0, "h": None,
                         "n_films": len({a["film"] for a in grays}),
                         "films": sorted({a["film"] for a in grays})[:8]})
    total = sum(s["mass"] for s in swatches)
    for s in swatches:
        s["share"] = s["mass"] / total
    return swatches


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

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Experiment idea 4: the taste ring — the ribbon bent into a broken circle.

Hue is circular, so the hue-sorted ribbon's ends already meet. Each film is
an arc (equal width — arc length measures HOW MUCH of your taste lives in a
hue region), colors flow OKLab-smooth within contiguous runs. Where
consecutive films jump more than GAP_DEG in hue, the ring BREAKS: a dark gap
sized by the missing hue span. Dense glowing arcs where you live, open
silence where you never go — identity artifact and blind-spot instrument in
one object, no background map.

Center: the achromatic films (small gray dots, they have no hue to sit on)
and the collection's darkness stat.

Outputs data/preview_taste_ring.html. Hover names films and gaps.

Usage:
    uv run experiment_taste_ring.py [--gap-deg 25]
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PALETTES = ROOT / "data" / "palettes.json"
OUT = ROOT / "data" / "preview_taste_ring.html"

C_MIN = 0.02          # below this a color has no meaningful hue
MERGE_DEG = 45        # hue spread within which blending two signals is honest
GAP_UNIT = 45.0       # a 45deg hue jump costs one film-width of silence
WHEEL = 620           # px


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
    """One color per film (same honesty rule as ideas 1-3)."""
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


def layout(chrom, gap_deg):
    """Assign ring angles. Films weigh 1 each; a hue jump > gap_deg between
    ring-neighbors becomes a gap weighing jump/GAP_UNIT film-widths. Returns
    (runs, gaps): runs = contiguous film stretches with node angles for
    smooth interpolation; angles in degrees, may exceed 360 on the run that
    wraps past zero."""
    n = len(chrom)
    jumps = [(chrom[(i + 1) % n]["h"] - chrom[i]["h"]) % 360 for i in range(n)]
    gapw = [j / GAP_UNIT if j > gap_deg else 0.0 for j in jumps]
    unit = 360.0 / (n + sum(gapw))

    # anchor the ring so the first film sits near its true hue angle
    cursor = chrom[0]["h"]
    runs, gaps, current = [], [], []
    for i, f in enumerate(chrom):
        current.append(dict(f, ang=round(cursor + unit / 2, 3)))
        cursor += unit
        if gapw[i]:
            runs.append(current)
            current = []
            gaps.append({"start": round(cursor, 3),
                         "end": round(cursor + gapw[i] * unit, 3),
                         "h_from": chrom[i]["h"],
                         "h_to": chrom[(i + 1) % n]["h"],
                         "span": round(jumps[i], 1)})
            cursor += gapw[i] * unit
    if current:                    # no gap after the last film: ring wraps
        if runs:                   # ...into the first run — merge across 0
            first = runs.pop(0)
            runs.append(current + [dict(f, ang=f["ang"] + 360)
                                   for f in first])
        else:                      # unbroken ring: close the loop on itself
            first = current[0]
            runs.append(current + [dict(first, ang=first["ang"] + 360)])
    return runs, gaps, unit


PAINTER_JS = """
const W = %(wheel)d, CX = W / 2, CY = W / 2;
const R_OUT = W / 2 - 8, R_IN = R_OUT * 0.60;
const UNIT = %(unit)f;        // arc degrees per film
const RUNS = %(runs_json)s;   // [[{title,hex,L,a,b,C,h,ang}, ...], ...]
const GAPS = %(gaps_json)s;   // [{start,end,h_from,h_to,span}]
const GROOVE = [24, 24, 28]; // near-black arc where the ring is broken

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

const ease = t => t * t * (3 - 2 * t);

// which segment owns ring angle theta? returns {run} or {gap} or null
function segmentAt(theta) {
  for (const g of GAPS)
    for (const t of [theta, theta + 360])
      if (t >= g.start && t < g.end) return { gap: g };
  const half = UNIT / 2;   // each run owns half a film-arc past its end nodes
  for (const run of RUNS) {
    const a0 = run[0].ang, a1 = run[run.length - 1].ang;
    for (const t of [theta, theta + 360])
      if (t >= a0 - half && t <= a1 + half) return { run, t };
  }
  return null;
}

function colorAt(run, t) {
  if (run.length === 1) return run[0];
  if (t <= run[0].ang) return run[0];
  if (t >= run[run.length - 1].ang) return run[run.length - 1];
  let j = 0;
  while (run[j + 1].ang < t) j++;
  const A = run[j], B = run[j + 1];
  const u = ease((t - A.ang) / (B.ang - A.ang));
  return { L: A.L + (B.L - A.L) * u, a: A.a + (B.a - A.a) * u,
           b: A.b + (B.b - A.b) * u };
}

function paint() {
  const cv = document.getElementById('ring'), ctx = cv.getContext('2d');
  const img = ctx.createImageData(W, W);
  for (let y = 0; y < W; y++) {
    for (let x = 0; x < W; x++) {
      const dx = x - CX, dy = y - CY, r = Math.hypot(dx, dy);
      const i = (y * W + x) * 4;
      if (r > R_OUT + 1 || r < R_IN - 1) { img.data[i + 3] = 0; continue; }
      const theta = ((Math.atan2(-dy, dx) * 180 / Math.PI) + 360) %% 360;
      const seg = segmentAt(theta);
      let rr, gg, bb;
      if (!seg || seg.gap) { [rr, gg, bb] = GROOVE; }
      else {
        const c = colorAt(seg.run, seg.t);
        [rr, gg, bb] = oklabToRgb(c.L, c.a, c.b);
      }
      // radial antialias at both rims
      let alpha = 255;
      if (r > R_OUT - 1) alpha = 255 * (R_OUT + 1 - r) / 2;
      if (r < R_IN + 1) alpha = 255 * (r - (R_IN - 1)) / 2;
      img.data[i] = rr; img.data[i + 1] = gg; img.data[i + 2] = bb;
      img.data[i + 3] = Math.max(0, Math.min(255, alpha));
    }
  }
  ctx.putImageData(img, 0, 0);
}

function initHover() {
  const cv = document.getElementById('ring');
  const tip = document.getElementById('tip');
  cv.addEventListener('mousemove', e => {
    const rc = cv.getBoundingClientRect();
    const dx = e.clientX - rc.left - CX, dy = e.clientY - rc.top - CY;
    const r = Math.hypot(dx, dy);
    if (r > R_OUT || r < R_IN) { tip.style.display = 'none'; return; }
    const theta = ((Math.atan2(-dy, dx) * 180 / Math.PI) + 360) %% 360;
    const seg = segmentAt(theta);
    if (!seg) { tip.style.display = 'none'; return; }
    if (seg.gap)
      tip.textContent = `blind spot — hues ${seg.gap.h_from}°–` +
        `${seg.gap.h_to}° (${seg.gap.span}° of silence)`;
    else {
      let best = seg.run[0], bd = Infinity;
      for (const f of seg.run) {
        const d = Math.min(Math.abs(f.ang - seg.t),
                           Math.abs(f.ang - seg.t - 360),
                           Math.abs(f.ang - seg.t + 360));
        if (d < bd) { bd = d; best = f; }
      }
      tip.textContent = `${best.title} — h=${best.h}° C=${best.C}`;
    }
    tip.style.display = 'block';
    tip.style.left = (e.pageX + 14) + 'px';
    tip.style.top = (e.pageY + 10) + 'px';
  });
  cv.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
}

paint(); initHover();
"""


def render(runs, gaps, unit, gray, med_l, n_total):
    runs_json = json.dumps(runs, ensure_ascii=False)
    gaps_json = json.dumps(gaps, ensure_ascii=False)
    js = PAINTER_JS % {"wheel": WHEEL, "unit": unit, "runs_json": runs_json,
                       "gaps_json": gaps_json}
    center_px = int(WHEEL * 0.60 * 0.5)   # inner hole radius
    css = f"""
    body{{background:#0c0c10;color:#cfcfd4;font:14px/1.5 -apple-system,sans-serif;
         margin:2rem}}
    h1{{font-size:1.1rem}} h2{{font-size:1rem;margin-top:2.5rem}}
    #stage{{position:relative;width:{WHEEL}px;height:{WHEEL}px}}
    #center{{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
            text-align:center;color:#8a8a92;font-size:13px;line-height:1.8}}
    .gdot{{display:inline-block;width:11px;height:11px;border-radius:50%;
          border:1px solid #ffffff55;margin:0 3px;vertical-align:middle}}
    .sw{{display:inline-block;width:20px;height:20px;border-radius:4px;
        vertical-align:middle;margin-right:4px}}
    .film{{display:flex;gap:.6rem;align-items:center;margin:.25rem 0}}
    .dim{{color:#77777f}}
    #tip{{position:absolute;display:none;background:#000d;color:#eee;
         padding:4px 9px;border-radius:6px;font-size:12px;
         pointer-events:none;white-space:nowrap;z-index:9}}
    """
    gray_dots = "".join(
        f"<span class='gdot' title=\"{c['title']}\" "
        f"style='background:{c['hex']}'></span>" for c in gray)
    h = [f"<meta charset='utf-8'><title>cinehue taste ring</title>"
         f"<style>{css}</style>",
         f"<h1>taste ring <span class='dim'>— {n_total} films as arcs, "
         f"hue-sorted, OKLab-smooth; the ring breaks where your taste "
         f"does. hover arcs and gaps.</span></h1>",
         f"<div id='stage'>"
         f"<canvas id='ring' width='{WHEEL}' height='{WHEEL}'></canvas>",
         f"<div id='center'>{gray_dots}"
         + ("<br>" if gray else "")
         + f"{n_total} films<br>median L {med_l:.2f}</div>",
         "</div><div id='tip'></div>",
         "<h2>gaps (blind spots)</h2>"]
    if gaps:
        for g in gaps:
            h.append(f"<div class='film'><span class='sw' style='background:"
                     f"#18181c'></span><span class='dim'>hues "
                     f"{g['h_from']}&deg;&ndash;{g['h_to']}&deg; — "
                     f"{g['span']}&deg; unwatched</span></div>")
    else:
        h.append("<div class='dim'>none — the ring is unbroken</div>")
    h.append("<h2>ring order</h2>")
    for run in runs:
        for f in run:
            h.append(f"<div class='film'><span class='sw' style='background:"
                     f"{f['hex']}'></span><span class='dim'>h{f['h']:.0f} "
                     f"C{f['C']:.2f}</span> {f['title']}</div>")
    h.append(f"<script>{js}</script>")
    OUT.write_text("\n".join(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-deg", type=float, default=25.0,
                    help="hue jump (deg) between ring-neighbors that breaks "
                         "the ring")
    args = ap.parse_args()

    films = json.loads(PALETTES.read_text())
    colors = [c for c in (film_color(f) for f in films) if c]
    chrom = sorted((c for c in colors if c["C"] >= C_MIN),
                   key=lambda c: c["h"])
    gray = sorted((c for c in colors if c["C"] < C_MIN),
                  key=lambda c: c["L"])
    if len(chrom) < 2:
        raise SystemExit("need at least 2 chromatic films for a ring")
    med_l = sorted(f["L_mean"] for f in films)[len(films) // 2]

    runs, gaps, unit = layout(chrom, args.gap_deg)
    render(runs, gaps, unit, gray, med_l, len(colors))
    print(f"{len(colors)} films ({len(gray)} achromatic in the center), "
          f"{len(gaps)} gap(s) -> {OUT}")


if __name__ == "__main__":
    main()

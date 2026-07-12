#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Experiment: taste as an ILLUMINATED color wheel.

The wheel is a fixed map of OKLCH color space (angle = hue, radius = chroma,
gray center) — identical for everyone, drawn DIMMED. Watched films light it
up: each film's identity colors (accent + subject-chroma = its mood-wheel
stops) cast a glow at their true (h, C) position, and overlapping films
accumulate — the more of your cinema lives in a region, the brighter it
burns. Unvisited regions stay dark: blind spots are literally the parts of
color space you have never lit.

Outputs data/preview_taste.html (self-contained canvas painter).

Usage:
    uv run experiment_taste.py [--sigma 18] [--glow 34]
"""
import argparse
import cmath
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PALETTES = ROOT / "data" / "palettes.json"
OUT = ROOT / "data" / "preview_taste.html"

C_MIN = 0.02          # below this a color has no meaningful hue (b&w guard)
BINS = 360
WHEEL = 560           # px
# Illumination = saturation reveal, NOT lightness lift. Raising L toward
# white made lit low-chroma regions read as whitish fog; instead the map sits
# at one mid lightness everywhere — unwatched is a DIMMED but visibly colored
# wheel, watched regions saturate to full (plus a whisper of L so pools glow).
# Display chroma is decoupled from film chroma: real film chroma is 0.02-0.15,
# far too low to survive dimming — the map paints a picker-vivid wheel
# (C_DISPLAY at the rim) and radius only POSITIONS colors, it doesn't mute
# the paint.
L_BASE, L_LIFT = 0.51, 0.11
C_DISPLAY = 0.19               # display chroma at the rim
C_DIM_FRAC = 0.35              # unlit regions keep this fraction of it
# Radius mapping: r/R = sqrt(C / c_edge), where c_edge is fitted to the data
# (just past the most vivid film color). Film chroma lives in 0.02-0.15 —
# a fixed picker-style 0-0.17 linear scale dumps every film into the gray
# inner half and the wheel reads as fog. sqrt + data fit spreads the actual
# range over the full disc.


def hex_to_oklch(hexstr):
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
    return L, math.hypot(a, bb), math.degrees(math.atan2(bb, a)) % 360


def stops_of(film):
    out = []
    for stop in film.get("mood", {}).get("wheel", []):
        L, C, h = hex_to_oklch(stop["hex"])
        out.append({"hex": stop["hex"], "L": round(L, 4), "C": round(C, 4),
                    "h": round(h, 1)})
    return out


def density(votes, sigma):
    dens = [0.0] * BINS
    for v in votes:
        for b in range(BINS):
            d = (b - v["h"] + 180) % 360 - 180
            dens[b] += v["C"] * math.exp(-0.5 * (d / sigma) ** 2)
    return dens


def circular_mean(votes):
    z = sum(v["C"] * cmath.exp(1j * math.radians(v["h"])) for v in votes)
    total = sum(v["C"] for v in votes)
    if not total:
        return None, 0.0
    return math.degrees(cmath.phase(z)) % 360, abs(z) / total


def find_modes(dens, films_votes, k=3, min_sep=40):
    peaks = [b for b in range(BINS)
             if dens[b] >= dens[(b - 1) % BINS] and dens[b] > dens[(b + 1) % BINS]]
    peaks.sort(key=lambda b: -dens[b])
    modes = []
    for p in peaks:
        if any(abs((p - m["h"] + 180) % 360 - 180) < min_sep for m in modes):
            continue
        near = [(t, v) for t, vs in films_votes for v in vs
                if abs((p - v["h"] + 180) % 360 - 180) <= min_sep / 2]
        near.sort(key=lambda tv: -tv[1]["C"])
        modes.append({"h": p, "films": near})
        if len(modes) == k:
            break
    return modes


def blind_spots(dens, thresh_frac=0.06, min_width=30):
    top = max(dens) or 1
    low = [b for b in range(BINS) if dens[b] < top * thresh_frac]
    if not low:
        return []
    spots, run = [], [low[0]]
    for b in low[1:]:
        if b == run[-1] + 1:
            run.append(b)
        else:
            spots.append(run)
            run = [b]
    spots.append(run)
    if len(spots) > 1 and spots[0][0] == 0 and spots[-1][-1] == BINS - 1:
        spots[0] = spots.pop() + spots[0]
    return [(r[0] % 360, r[-1] % 360) for r in spots if len(r) >= min_width]


PAINTER_JS = """
const W = %(wheel)d, R = W / 2 - 4, C_EDGE = %(c_edge)f;
const L_BASE = %(l_base)f, L_LIFT = %(l_lift)f, C_DIM_FRAC = %(c_dim_frac)f;
const C_DISPLAY = %(c_display)f;
const GLOW_SIGMA = %(glow)f;               // px radius of each film's lamp
const FILMS = %(films_json)s;              // [{title, stops:[{hex,h,C}]}]

// radius positions colors: r/R = sqrt(C / C_EDGE) (see python-side comment);
// the paint itself uses display chroma, not the (tiny) film chroma
const rFrac = C => Math.min(Math.sqrt(C / C_EDGE), 1);

function oklchToRgb(L, C, hDeg) {
  const hr = hDeg * Math.PI / 180, a = C * Math.cos(hr), b = C * Math.sin(hr);
  let l = L + 0.3963377774 * a + 0.2158037573 * b;
  let m = L - 0.1055613458 * a - 0.0638541728 * b;
  let s = L - 0.0894841775 * a - 1.2914855480 * b;
  l = l ** 3; m = m ** 3; s = s ** 3;
  let r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  let g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  let bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
  const enc = c => {
    c = Math.min(1, Math.max(0, c));
    return c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;
  };
  return [enc(r) * 255, enc(g) * 255, enc(bb) * 255];
}

function stopXY(stop) {
  const r = rFrac(stop.C) * R, ang = stop.h * Math.PI / 180;
  return [W / 2 + r * Math.cos(ang), W / 2 - r * Math.sin(ang)];
}

function paint() {
  const cv = document.getElementById('wheel'), ctx = cv.getContext('2d');
  const img = ctx.createImageData(W, W);
  const lamps = [];
  for (const f of FILMS)
    for (const s of f.stops) lamps.push(stopXY(s));
  const inv2s2 = 1 / (2 * GLOW_SIGMA * GLOW_SIGMA);
  for (let y = 0; y < W; y++) {
    for (let x = 0; x < W; x++) {
      const dx = x - W / 2, dy = y - W / 2, r = Math.hypot(dx, dy);
      const i = (y * W + x) * 4;
      if (r > R + 1) { img.data[i + 3] = 0; continue; }
      const hue = ((Math.atan2(-dy, dx) * 180 / Math.PI) + 360) %% 360;
      const C_here = C_DISPLAY * (r / R);
      let glow = 0;
      for (const [lx, ly] of lamps) {
        const d2 = (x - lx) * (x - lx) + (y - ly) * (y - ly);
        if (d2 < GLOW_SIGMA * GLOW_SIGMA * 25) glow += Math.exp(-d2 * inv2s2);
      }
      // steeper curve than glow/(glow+k): one lamp saturates its pool hard,
      // unwatched stays gray — readability lives in this contrast
      const g = Math.pow(glow / (glow + 0.45), 1.5);
      const L = L_BASE + L_LIFT * g;
      const C = C_here * (C_DIM_FRAC + (1 - C_DIM_FRAC) * g);
      const [rr, gg, bb] = oklchToRgb(L, C, hue);
      img.data[i] = rr; img.data[i + 1] = gg; img.data[i + 2] = bb;
      img.data[i + 3] = r > R - 1 ? Math.round(255 * (R + 1 - r) / 2) : 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// One mark per movie. A film's two identity signals (accent = environment,
// subject-chroma = costume) merge into a single blended dot when their hues
// are neighbors (blend is honest); when they genuinely disagree (2001: red
// vs blue) the film stays one visual unit — primary dot + tethered satellite.
const MERGE_DEG = 45;

function hueDist(a, b) { return Math.abs((a - b + 540) %% 360 - 180); }

function blendStops(stops) {
  let sw = 0, a = 0, b = 0, L = 0;
  for (const s of stops) {
    const w = s.C, hr = s.h * Math.PI / 180;
    a += w * s.C * Math.cos(hr); b += w * s.C * Math.sin(hr);
    L += w * s.L; sw += w;
  }
  a /= sw; b /= sw; L /= sw;
  return { L, C: Math.hypot(a, b),
           h: ((Math.atan2(b, a) * 180 / Math.PI) + 360) %% 360 };
}

function addDot(holder, tip, x, y, sz, color, label) {
  const d = document.createElement('div');
  d.className = 'dot';
  d.style.cssText = `left:${x - sz / 2}px;top:${y - sz / 2}px;` +
    `width:${sz}px;height:${sz}px;background:${color}`;
  d.onmouseenter = () => { tip.textContent = label; tip.style.display = 'block'; };
  d.onmousemove = e => {
    tip.style.left = (e.pageX + 14) + 'px';
    tip.style.top = (e.pageY + 10) + 'px';
  };
  d.onmouseleave = () => { tip.style.display = 'none'; };
  holder.appendChild(d);
  return d;
}

// Reduce a film to one mark: blended dot, or primary + satellites when
// genuinely two-toned. Returns null for films with no stops at all.
function filmMark(f) {
  const chrom = f.stops.filter(s => s.C >= %(c_min)f);
  if (!chrom.length) {
    if (!f.stops.length) return null;
    const s = f.stops[0], [x, y] = stopXY(s);
    return { x, y, C: s.C, col: s.hex,
             label: `${f.title} (achromatic)`, sats: [] };
  }
  const spread = Math.max(...chrom.map(p =>
    Math.max(...chrom.map(q => hueDist(p.h, q.h)))));
  if (chrom.length === 1 || spread <= MERGE_DEG) {
    const m = chrom.length === 1 ? chrom[0] : blendStops(chrom);
    const [x, y] = stopXY(m);
    let col = m.hex;
    if (!col) {
      const [r, g, b] = oklchToRgb(m.L, m.C, m.h);
      col = `rgb(${r | 0},${g | 0},${b | 0})`;
    }
    return { x, y, C: m.C, col,
             label: `${f.title} — h=${m.h.toFixed(0)}° C=${m.C.toFixed(3)}`,
             sats: [] };
  }
  chrom.sort((p, q) => q.C - p.C);
  const [px, py] = stopXY(chrom[0]);
  return { x: px, y: py, C: chrom[0].C, col: chrom[0].hex,
           label: `${f.title} — h=${chrom[0].h}° C=${chrom[0].C}`,
           sats: chrom.slice(1).map(s => {
             const [sx, sy] = stopXY(s);
             return { x: sx, y: sy, col: s.hex,
                      label: `${f.title} — costume h=${s.h}° C=${s.C}` };
           }) };
}

// At scale, individual marks merge into cluster dots (size ~ film count).
const CLUSTER_AT = 30;      // films; below this every film keeps its own mark
const CLUSTER_PX = 24;

function clusterMarks(marks) {
  const clusters = [];
  for (const m of marks) {
    let best = null, bd = CLUSTER_PX;
    for (const c of clusters) {
      const d = Math.hypot(m.x - c.x, m.y - c.y);
      if (d < bd) { bd = d; best = c; }
    }
    if (best) {
      best.members.push(m);
      best.x += (m.x - best.x) / best.members.length;
      best.y += (m.y - best.y) / best.members.length;
      if (m.C > best.top.C) best.top = m;
    } else {
      clusters.push({ x: m.x, y: m.y, members: [m], top: m });
    }
  }
  return clusters;
}

function placeDots() {
  const holder = document.getElementById('dots');
  const tip = document.getElementById('tip');
  const marks = FILMS.map(filmMark).filter(Boolean);

  if (marks.length > CLUSTER_AT) {
    for (const c of clusterMarks(marks)) {
      const n = c.members.length;
      const sz = Math.min(10 + 5 * Math.sqrt(n - 1), 30);
      const names = c.members.slice(0, 6).map(m => m.label.split(' — ')[0]);
      const label = n === 1 ? c.members[0].label
        : `${n} films: ${names.join(', ')}${n > 6 ? ` +${n - 6} more` : ''}`;
      addDot(holder, tip, c.x, c.y, sz, c.top.col, label);
    }
    return;
  }

  // Two-toned films: ONE ringed dot (core = primary color, ring = costume
  // color) instead of tether lines — lines turned the map into spaghetti.
  // Hovering the dot reveals where the costume color actually lives.
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', W); svg.setAttribute('height', W);
  svg.style.cssText = 'position:absolute;inset:0;pointer-events:none';
  holder.appendChild(svg);
  for (const m of marks) {
    const reveal = [];
    for (const s of m.sats) {
      const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      ln.setAttribute('x1', m.x); ln.setAttribute('y1', m.y);
      ln.setAttribute('x2', s.x); ln.setAttribute('y2', s.y);
      ln.setAttribute('stroke', '#ffffffaa');
      ln.style.display = 'none';
      svg.appendChild(ln);
      const sd = addDot(holder, tip, s.x, s.y, 9, s.col, s.label);
      sd.style.display = 'none';
      sd.style.pointerEvents = 'none';
      reveal.push(ln, sd);
    }
    const label = m.sats.length
      ? m.label + ' (ring = costume color; hover shows where it lives)'
      : m.label;
    const d = addDot(holder, tip, m.x, m.y, 13, m.col, label);
    if (m.sats.length) {
      d.style.border = `3.5px solid ${m.sats[0].col}`;
      const enter = d.onmouseenter, leave = d.onmouseleave;
      d.onmouseenter = () => {
        enter(); reveal.forEach(el => { el.style.display = 'block'; });
      };
      d.onmouseleave = () => {
        leave(); reveal.forEach(el => { el.style.display = 'none'; });
      };
    }
  }
}

// Hover anywhere on the wheel -> list the films living near the cursor.
// This is the scalable way to get names; dots/clusters are just landmarks.
function initFinder() {
  const stage = document.getElementById('stage');
  const box = document.getElementById('nearby');
  const marks = FILMS.map(filmMark).filter(Boolean);
  stage.addEventListener('mousemove', e => {
    const rc = stage.getBoundingClientRect();
    const mx = e.clientX - rc.left, my = e.clientY - rc.top;
    const near = marks
      .map(m => ({ m, d: Math.min(Math.hypot(mx - m.x, my - m.y),
        ...m.sats.map(s => Math.hypot(mx - s.x, my - s.y))) }))
      .filter(o => o.d < 55).sort((a, b) => a.d - b.d).slice(0, 12);
    box.innerHTML = near.length
      ? '<b>near cursor</b><br>' + near.map(o =>
          `<span class='sw' style='background:${o.m.col};width:12px;` +
          `height:12px'></span>${o.m.label.split(' — ')[0]}`).join('<br>')
      : '';
  });
  stage.addEventListener('mouseleave', () => { box.innerHTML = ''; });
}

paint(); placeDots(); initFinder();
document.getElementById('toggle').onclick = () => {
  const h = document.getElementById('dots');
  h.style.display = h.style.display === 'none' ? '' : 'none';
};
"""


def render(films_votes, c_edge, mean, modes, spots, med_L, sigma, glow):
    films_json = json.dumps(
        [{"title": t, "stops": vs} for t, vs in films_votes],
        ensure_ascii=False)
    js = PAINTER_JS % {"wheel": WHEEL, "c_edge": c_edge, "l_base": L_BASE,
                       "l_lift": L_LIFT, "c_dim_frac": C_DIM_FRAC,
                       "c_display": C_DISPLAY, "glow": glow,
                       "films_json": films_json, "c_min": C_MIN}
    css = f"""
    body{{background:#0c0c10;color:#cfcfd4;font:14px/1.5 -apple-system,sans-serif;
         margin:2rem}}
    h1{{font-size:1.1rem}} h2{{font-size:1rem;margin-top:2.5rem}}
    #stage{{position:relative;width:{WHEEL}px;height:{WHEEL}px;flex-shrink:0}}
    #dots{{position:absolute;inset:0}}
    .dot{{position:absolute;border-radius:50%;border:1px solid #ffffff99;
         cursor:default}}
    #tip{{position:absolute;display:none;background:#000d;color:#eee;
         padding:4px 9px;border-radius:6px;font-size:12px;
         pointer-events:none;white-space:nowrap;z-index:9}}
    #nearby{{margin-top:.6rem;min-height:5em;font-size:13px;line-height:1.7;
            max-width:{WHEEL}px}}
    .sw{{display:inline-block;width:20px;height:20px;border-radius:4px;
        vertical-align:middle;margin-right:4px}}
    .layout{{display:flex;gap:2.5rem;align-items:flex-start;flex-wrap:wrap}}
    .film{{display:flex;gap:.6rem;align-items:center;margin:.25rem 0}}
    .dim{{color:#77777f}} .mode{{margin:.4rem 0}}
    button{{background:#222;color:#ccc;border:1px solid #444;border-radius:6px;
           padding:4px 10px;cursor:pointer;margin-top:.6rem}}
    """
    h = [f"<meta charset='utf-8'><title>cinehue taste wheel</title>"
         f"<style>{css}</style>",
         f"<h1>illuminated wheel <span class='dim'>— one gray map for "
         f"everyone; your {len(films_votes)} films bring the color back "
         f"where they live. overlaps saturate harder; gray is what you "
         f"haven't watched.</span></h1>"]
    h.append("<div class='layout'>")
    h.append(f"<div><div id='stage'>"
             f"<canvas id='wheel' width='{WHEEL}' height='{WHEEL}'></canvas>"
             f"<div id='dots'></div></div>"
             f"<button id='toggle'>toggle film dots</button>"
             f"<div id='nearby'></div></div>"
             f"<div id='tip'></div>")

    h.append("<div>")
    mh, R = mean
    if mh is not None:
        warn = (" <span class='dim'>(low — taste is multimodal; a single hue "
                "would be a lie)</span>" if R < 0.5 else "")
        h.append(f"<div class='film'><span class='sw' style='background:"
                 f"oklch({L_BASE:.2f} 0.14 {mh:.0f}deg);width:44px;"
                 f"height:44px;border-radius:8px'></span>circular mean "
                 f"{mh:.0f}&deg;, R={R:.2f}{warn}</div>")
    for i, m in enumerate(modes, 1):
        films = ", ".join(f"{t} <span class='sw' style='background:{v['hex']}'>"
                          f"</span>" for t, v in m["films"][:4])
        h.append(f"<div class='mode'><b>zone {i}</b>: {m['h']}&deg; "
                 f"<span class='sw' style='background:oklch({L_BASE:.2f} 0.14 "
                 f"{m['h']}deg)'></span> &larr; {films}</div>")
    h.append("<div class='dim'>gray sectors (blind spots): " +
             (", ".join(f"{a}&deg;&ndash;{b}&deg;" for a, b in spots)
              if spots else "none") + "</div>")
    h.append(f"<div class='dim'>darkness axis (separate from the map): "
             f"median L_mean = {med_L:.2f}. glow radius {glow}px; "
             f"sigma={sigma}&deg; for zone/blind-spot stats.</div>")
    h.append("</div></div>")

    h.append("<h2>evidence — per-film identity colors</h2>")
    for title, votes in sorted(films_votes,
                               key=lambda tv: tv[1][0]["h"] if tv[1] else 999):
        sws = " ".join(f"<span class='sw' style='background:{v['hex']}' "
                       f"title='h={v['h']:.0f} C={v['C']:.3f}'></span>"
                       f"<span class='dim'>h{v['h']:.0f} C{v['C']:.2f}</span>"
                       for v in votes)
        if not sws:
            sws = "<span class='dim'>achromatic — lights the center</span>"
        h.append(f"<div class='film'>{sws} &nbsp;{title}</div>")
    h.append(f"<script>{js}</script>")
    OUT.write_text("\n".join(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=18.0,
                    help="kernel width (deg) for zone/blind-spot stats")
    ap.add_argument("--glow", type=float, default=26.0,
                    help="px radius of each film's glow on the map")
    args = ap.parse_args()

    films = json.loads(PALETTES.read_text())
    films_votes = [(f["title"], stops_of(f)) for f in films]
    chromatic = [v for _, vs in films_votes for v in vs if v["C"] >= C_MIN]
    med_L = sorted(f["L_mean"] for f in films)[len(films) // 2]
    # rim = just past the most vivid color in the collection
    c_edge = max((v["C"] for v in chromatic), default=0.17) * 1.05

    dens = density(chromatic, args.sigma)
    render(films_votes, c_edge, circular_mean(chromatic),
           find_modes(dens, films_votes), blind_spots(dens), med_L,
           args.sigma, args.glow)

    total = sum(len(vs) for _, vs in films_votes)
    print(f"{len(films)} films, {total} lamps on the map -> {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Experiment idea 6: the color space — Fork B of the pipeline.

Same measure as idea 5 (Step 1: film -> mass-1 measure over identity stops;
Step 2: sum). render() draws the measure ON the OKLab hue plane
(angle = hue, radius = sqrt-chroma, the mapping the illuminated wheel
validated). Four modes:

  atoms (default): every atom of the summed measure as its own gaussian,
      sized by global mass ** gamma. Uncompressed; regions emerge from
      pileup; darkness = unwatched space; hover names films.
  clustered: the quantized swatches (Fork A's compression) — for comparison.
  gradient: full-bleed mesh gradient. The quantized taste colors melt into
      each other across the WHOLE square (uniform kernel size, colors
      weighted by mass ** gamma). No darkness, no dots — pure portrait.
  sum: literally the sum of each movie's single-film rendering: every film's
      stops are sized as if that film were rendered alone (within-film
      normalization), all fields added; brightness = accumulation, so hue
      neighborhoods many films share glow while loners stay faint.
  spectrum: one smooth OKLab gradient (was experiment_taste_spectrum.py) —
      hue axis warped so width is proportional to mass ** gamma, unwatched
      ranges collapse into dark seams, gray band at the end. --bandwidth
      controls hue smoothing.

Achromatic mass is one merged seed at the exact center. Three squares per
run: gamma 0 / 0.5 / 1 (presence / portrait / measurement).

Reads data/palettes.json, writes data/preview_taste_space.html
(--film TITLE -> preview_taste_space_film.html, exact match preferred).

Usage:
    uv run render/taste_space.py [--mode atoms|clustered|gradient|sum|
                                      spectrum] [--gamma 0.5] [--soft 1.0]
                                     [--sharp 1.5] [--knee 0.8]
                                     [--bandwidth 10] [--film ran]
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.measure import build_measure, film_atoms
from lib.oklab import hex_to_oklab, oklab_to_hex
from lib.paths import DATA, PALETTES

OUT = DATA / "preview_taste_space.html"

SQ = 560              # px per square
R_EDGE = SQ / 2 - 10
SIG_MIN = 2.5         # never let a seed vanish entirely
GRAY_SIG = 0.15       # fixed gray-seed size (fraction of R_EDGE), sum mode

# spectrum mode
BINS = 720            # hue resolution (0.5 deg)
GAP_MIN_DEG = 15      # unwatched hue span that earns a visible dark seam
GAP_PX = 8            # width of a blind-spot seam
TOP_FILMS = 4         # films named per hue in the tooltip


def place(C, h, gray, c_edge):
    if gray:
        return SQ / 2, SQ / 2
    r = math.sqrt(C / c_edge) * R_EDGE * 0.92
    th = math.radians(h)
    return SQ / 2 + r * math.cos(th), SQ / 2 - r * math.sin(th)


def atom_seeds(films):
    """Every atom of the summed measure. share = global mass (sums to 1);
    wf = within-film mass (sums to 1 per film, drives sigma in sum mode)."""
    per_film = [film_atoms(f) for f in films]
    per_film = [a for a in per_film if a]
    n = len(per_film)
    atoms = [dict(a, wg=a["w"] / n) for al in per_film for a in al]
    chrom = [a for a in atoms if not a["gray"]]
    grays = [a for a in atoms if a["gray"]]
    c_edge = max(a["C"] for a in chrom) * 1.05 if chrom else 1.0
    seeds = []
    for a in chrom:
        x, y = place(a["C"], a["h"], False, c_edge)
        hexs = oklab_to_hex(a["L"], a["a"], a["b"])
        seeds.append({"x": round(x, 1), "y": round(y, 1),
                      "L": round(a["L"], 4), "a": round(a["a"], 4),
                      "b": round(a["b"], 4), "share": round(a["wg"], 6),
                      "wf": round(a["w"], 5), "film": a["film"],
                      "gray": False,
                      "tip": f"{a['film']} — {hexs}, "
                             f"{a['wg'] * 100:.2f}% of taste"})
    if grays:
        tw = sum(a["wg"] for a in grays)
        gL = sum(a["L"] * a["wg"] for a in grays) / tw
        seeds.append({"x": SQ / 2, "y": SQ / 2,
                      "L": round(gL, 4), "a": 0.0, "b": 0.0,
                      "share": round(tw, 6), "wf": 1.0, "film": None,
                      "gray": True,
                      "tip": f"gray axis — {len(grays)} achromatic film(s), "
                             f"{tw * 100:.1f}% of taste"})
    return seeds


def cluster_seeds(swatches):
    chrom = [s for s in swatches if s["h"] is not None]
    c_edge = max(s["C"] for s in chrom) * 1.05 if chrom else 1.0
    seeds = []
    for s in swatches:
        L, a, b = hex_to_oklab(s["hex"])
        x, y = place(s["C"], s["h"] or 0.0, s["h"] is None, c_edge)
        more = "…" if s["n_films"] > len(s["films"]) else ""
        seeds.append({"x": round(x, 1), "y": round(y, 1),
                      "L": round(L, 4), "a": round(a, 4), "b": round(b, 4),
                      "share": round(s["share"], 5), "wf": 1.0, "film": None,
                      "gray": s["h"] is None,
                      "tip": f"{s['hex']} — {s['share'] * 100:.1f}% of "
                             f"taste, {s['n_films']} film(s): "
                             f"{', '.join(s['films'])}{more}"})
    return seeds


def sigma_amp_tables(seeds, gammas, soft, mode):
    """Per-gamma sigma (px) and amplitude per seed. Sigma carries the size
    encoding; amp carries any extra mass weighting a mode needs."""
    sigs, amps = [], []
    for g in gammas:
        sig, amp = [], []
        if mode in ("atoms", "clustered"):
            w = [s["share"] ** g for s in seeds]
            tot = sum(w)
            sig = [max(SIG_MIN, soft * R_EDGE * math.sqrt(x / tot)) for x in w]
            amp = [1.0] * len(seeds)
        elif mode == "gradient":
            # uniform big kernels; mass lives in the color weighting instead
            sig = [soft * R_EDGE * 0.6] * len(seeds)
            amp = [s["share"] ** g for s in seeds]
        elif mode == "sum":
            # each film normalized as if rendered alone
            norms = {}
            for s in seeds:
                if s["film"]:
                    norms[s["film"]] = norms.get(s["film"], 0.0) \
                        + s["wf"] ** g
            for s in seeds:
                if s["gray"]:
                    sig.append(GRAY_SIG * R_EDGE)
                else:
                    frac = s["wf"] ** g / norms[s["film"]]
                    sig.append(max(SIG_MIN,
                                   soft * R_EDGE * math.sqrt(frac)))
                amp.append(s["share"])   # each film adds 1/N of the light
        sigs.append([round(x, 2) for x in sig])
        amps.append([round(x, 6) for x in amp])
    return sigs, amps


PAINTER_JS = """
const SQ = %(sq)d;
const SEEDS = %(seeds_json)s;
const SIGS = %(sigs_json)s;    // per-square, per-seed sigma (px)
const AMPS = %(amps_json)s;    // per-square, per-seed amplitude
const SHARP = %(sharp)f, KNEE = %(knee)f, FULL = %(full)d;
const BG = [12, 12, 16];

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

function paint(cv, qi) {
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(SQ, SQ);
  const sg = SIGS[qi], am = AMPS[qi];
  const N = SQ * SQ;
  const tot = new Float32Array(N), pw = new Float32Array(N);
  const pL = new Float32Array(N), pa = new Float32Array(N),
        pb = new Float32Array(N);
  for (let k = 0; k < SEEDS.length; k++) {
    const s = SEEDS[k], sig = sg[k], inv = 1 / (2 * sig * sig);
    const reach = Math.min(4.5 * sig, SQ);
    const x0 = Math.max(0, Math.floor(s.x - reach)),
          x1 = Math.min(SQ - 1, Math.ceil(s.x + reach)),
          y0 = Math.max(0, Math.floor(s.y - reach)),
          y1 = Math.min(SQ - 1, Math.ceil(s.y + reach));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const g = am[k] * Math.exp(-((x - s.x) ** 2 + (y - s.y) ** 2) * inv);
        const i = y * SQ + x;
        tot[i] += g;
        const p = Math.pow(g, SHARP);
        pw[i] += p; pL[i] += s.L * p; pa[i] += s.a * p; pb[i] += s.b * p;
      }
    }
  }
  for (let i = 0; i < N; i++) {
    const j = i * 4;
    let rr = BG[0], gg = BG[1], bb = BG[2];
    if (pw[i] > 1e-12) {
      const [cr, cg, cb] = oklabToRgb(pL[i] / pw[i], pa[i] / pw[i],
                                      pb[i] / pw[i]);
      const t = FULL ? 1 : Math.pow(tot[i] / (tot[i] + KNEE), 1.3);
      rr = cr * t + BG[0] * (1 - t);
      gg = cg * t + BG[1] * (1 - t);
      bb = cb * t + BG[2] * (1 - t);
    }
    img.data[j] = rr; img.data[j + 1] = gg;
    img.data[j + 2] = bb; img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function initHover(cv, qi) {
  const tip = document.getElementById('tip');
  const sg = SIGS[qi], am = AMPS[qi];
  cv.addEventListener('mousemove', e => {
    const rc = cv.getBoundingClientRect();
    const x = e.clientX - rc.left, y = e.clientY - rc.top;
    let best = -1, bg = 0;
    for (let k = 0; k < SEEDS.length; k++) {
      const d2 = (x - SEEDS[k].x) ** 2 + (y - SEEDS[k].y) ** 2;
      const g = am[k] * Math.exp(-d2 / (2 * sg[k] * sg[k]));
      if (g > bg) { bg = g; best = k; }
    }
    if (best < 0) { tip.style.display = 'none'; return; }
    tip.textContent = SEEDS[best].tip;
    tip.style.display = 'block';
    tip.style.left = (e.pageX + 14) + 'px';
    tip.style.top = (e.pageY + 10) + 'px';
  });
  cv.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
}

document.querySelectorAll('canvas').forEach((cv, i) => {
  paint(cv, i); initHover(cv, i);
});
"""


def render(seeds_, swatches, gammas, title, out, args, mode):
    sigs, amps = sigma_amp_tables(seeds_, gammas, args.soft, mode)
    js = PAINTER_JS % {"sq": SQ, "seeds_json": json.dumps(
                           [{k: v for k, v in s.items()
                             if k in ("x", "y", "L", "a", "b", "tip")}
                            for s in seeds_]),
                       "sigs_json": json.dumps(sigs),
                       "amps_json": json.dumps(amps),
                       "sharp": args.sharp, "knee": args.knee,
                       "full": 1 if mode == "gradient" else 0}
    labels = {0.0: "presence", 0.5: "portrait", 1.0: "measurement"}
    figs = "".join(
        f"<figure><canvas width='{SQ}' height='{SQ}'></canvas>"
        f"<figcaption>&gamma; = {g:g}"
        f"{' — ' + labels[g] if g in labels else ''}</figcaption></figure>"
        for g in gammas)
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
    canvas{{border-radius:8px}}
    figure{{margin:0}}
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
    h = [f"<meta charset='utf-8'><title>cinehue color space</title>"
         f"<style>{css}</style>",
         f"<h1>color space <span class='dim'>— {title}, mode: {mode}. "
         f"the square is the hue plane (angle = hue, radius = chroma). "
         f"hover the map.</span></h1>",
         f"<div id='row'>{figs}</div>",
         "<div id='tip'></div>",
         "<h2>region summary (compressed legend)</h2>",
         *rows,
         f"<script>{js}</script>"]
    out.write_text("\n".join(h))


# ---------------------------------------------------------------- spectrum
# The sum rendered as one smooth OKLab gradient: hue density warped so width
# is proportional to mass ** gamma; unwatched hue ranges collapse into dark
# seams; achromatic mass is a gray band at the right end.

def spec_gather(films):
    per = [film_atoms(f) for f in films]
    per = [a for a in per if a]
    n = len(per)
    atoms = [dict(a, w=a["w"] / n) for al in per for a in al]
    chrom = [a for a in atoms if not a["gray"]]
    grays = [a for a in atoms if a["gray"]]
    gray_mass = sum(a["w"] for a in grays)
    gray_l = (sum(a["L"] * a["w"] for a in grays) / gray_mass) if grays else 0.5
    return chrom, gray_mass, gray_l, len(grays)


def spec_smooth(chrom, bw):
    """Circular-gaussian KDE of mass over hue + per-bin weighted OKLab color
    and top contributing films."""
    dens = [0.0] * BINS
    sL = [0.0] * BINS
    sa = [0.0] * BINS
    sb = [0.0] * BINS
    contrib = [dict() for _ in range(BINS)]
    reach = max(1, int(3 * bw / 360 * BINS))
    for a in chrom:
        b0 = int(round(a["h"] / 360 * BINS))
        for db in range(-reach, reach + 1):
            b = (b0 + db) % BINS
            ang = ((b + 0.5) * 360 / BINS - a["h"] + 180) % 360 - 180
            k = math.exp(-0.5 * (ang / bw) ** 2) * a["w"]
            if k < 1e-12:
                continue
            dens[b] += k
            sL[b] += k * a["L"]
            sa[b] += k * a["a"]
            sb[b] += k * a["b"]
            c = contrib[b]
            c[a["film"]] = c.get(a["film"], 0.0) + k
    colors = [None] * BINS
    tips = [""] * BINS
    for b in range(BINS):
        if dens[b] <= 0:
            continue
        L, aa, bb = sL[b] / dens[b], sa[b] / dens[b], sb[b] / dens[b]
        colors[b] = (L, aa, bb)
        top = sorted(contrib[b].items(), key=lambda kv: -kv[1])[:TOP_FILMS]
        hue = (b + 0.5) * 360 / BINS
        tips[b] = f"h{hue:.0f}° — " + ", ".join(t for t, _ in top)
    return dens, colors, tips


def spec_segments(dens):
    """Live hue runs and the dead runs between them (circular). The spectrum
    is cut at the longest dead run; dead runs >= GAP_MIN_DEG become seams."""
    peak = max(dens)
    if peak <= 0:
        raise SystemExit("no chromatic mass to render")
    live = [d > peak * 1e-3 for d in dens]
    if all(live):
        return [list(range(BINS))], []
    runs = []
    b = 0
    while b < BINS:
        if not live[b]:
            start = b
            while b < BINS and not live[b]:
                b += 1
            runs.append([start, b - 1])
        else:
            b += 1
    if not live[0] and not live[-1] and len(runs) > 1:   # wraps midnight
        runs[0] = [runs[-1][0], runs[0][1] + BINS]
        runs.pop()
    longest = max(runs, key=lambda r: r[1] - r[0])
    cut = (longest[1] + 1) % BINS
    ordered = [(cut + i) % BINS for i in range(BINS)]
    segs, gaps, cur = [], [], []
    dead = 0
    for b in ordered:
        if live[b]:
            if dead:
                span = dead * 360 / BINS
                if segs and span >= GAP_MIN_DEG:
                    gaps.append({"after_seg": len(segs) - 1,
                                 "span": round(span, 1)})
                dead = 0
            cur.append(b)
        else:
            if cur:
                segs.append(cur)
                cur = []
            dead += 1
    if cur:
        segs.append(cur)
    return segs, gaps


def spec_columns(dens, colors, tips, segs, gaps, gray, gamma):
    """Pixel columns for one square: hue axis warped by dens**gamma, dark
    seams for blind spots, gray band at the end. Colors are OKLab-lerped
    between neighboring bins per column."""
    gray_mass, gray_l, gray_n = gray
    live = [b for s in segs for b in s]
    raw = {b: dens[b] ** gamma for b in live}
    chrom_mass = sum(dens[b] for b in live) * (360 / BINS)
    if gray_mass > 0:
        avg = sum(dens[b] for b in live) / len(live)
        g_bins = max(2, round(gray_mass / max(chrom_mass, 1e-9) * len(live)))
        g_raw = g_bins * (avg ** gamma)
    else:
        g_bins, g_raw = 0, 0.0
    n_seams = len(gaps) + (1 if g_bins else 0)
    px = SQ - GAP_PX * n_seams
    total_raw = sum(raw.values()) + g_raw
    scale = px / total_raw

    cols_hex, cols_tip, tiplist = [], [], []

    def tip_idx(t):
        tiplist.append(t)
        return len(tiplist) - 1

    gap_after = {g["after_seg"]: g for g in gaps}
    for si, seg in enumerate(segs):
        widths = [raw[b] * scale for b in seg]
        seg_px = sum(widths)
        n_cols = max(1, round(seg_px))
        cum, acc = [], 0.0
        for w in widths:
            acc += w
            cum.append(acc)
        ti_cache = {}
        for x in range(n_cols):
            t = (x + 0.5) / n_cols * seg_px
            i = 0
            while cum[i] < t and i < len(seg) - 1:
                i += 1
            lo = cum[i - 1] if i else 0.0
            u = (t - lo) / max(cum[i] - lo, 1e-9)
            b0 = seg[i]
            b1 = seg[min(i + 1, len(seg) - 1)] if u > 0.5 \
                else seg[max(i - 1, 0)]
            v = abs(u - 0.5)
            c0, c1 = colors[b0], colors[b1]
            L = c0[0] + (c1[0] - c0[0]) * v
            aa = c0[1] + (c1[1] - c0[1]) * v
            bb = c0[2] + (c1[2] - c0[2]) * v
            cols_hex.append(oklab_to_hex(L, aa, bb))
            if b0 not in ti_cache:
                ti_cache[b0] = tip_idx(tips[b0])
            cols_tip.append(ti_cache[b0])
        if si in gap_after:
            g = gap_after[si]
            ti = tip_idx(f"blind spot — {g['span']}° of unwatched hue")
            for _ in range(GAP_PX):
                cols_hex.append("#111116")
                cols_tip.append(ti)
    if g_bins:
        ti = tip_idx("blind spot seam")
        for _ in range(GAP_PX):
            cols_hex.append("#111116")
            cols_tip.append(ti)
        n_cols = max(1, round(g_raw * scale))
        gt = tip_idx(f"gray axis — {gray_n} achromatic film(s), "
                     f"{gray_mass * 100:.1f}% of taste")
        ghex = oklab_to_hex(gray_l, 0.0, 0.0)
        for _ in range(n_cols):
            cols_hex.append(ghex)
            cols_tip.append(gt)
    return cols_hex, cols_tip, tiplist


SPECTRUM_JS = """
const SQ = %(sq)d;
const SQUARES = %(squares_json)s;  // [{hex:[], tip:[], tips:[]}, ...]

document.querySelectorAll('canvas').forEach((cv, i) => {
  const ctx = cv.getContext('2d');
  const d = SQUARES[i];
  for (let x = 0; x < d.hex.length && x < SQ; x++) {
    ctx.fillStyle = d.hex[x];
    ctx.fillRect(x, 0, 1, SQ);
  }
  const tip = document.getElementById('tip');
  cv.addEventListener('mousemove', e => {
    const rc = cv.getBoundingClientRect();
    const x = Math.floor(e.clientX - rc.left);
    if (x < 0 || x >= d.tip.length) { tip.style.display = 'none'; return; }
    tip.textContent = d.tips[d.tip[x]];
    tip.style.display = 'block';
    tip.style.left = (e.pageX + 14) + 'px';
    tip.style.top = (e.pageY + 10) + 'px';
  });
  cv.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
});
"""


def render_spectrum(squares, gammas, title, out):
    js = SPECTRUM_JS % {"sq": SQ, "squares_json": json.dumps(
        [{"hex": h, "tip": t, "tips": tl} for h, t, tl in squares])}
    labels = {0.0: "presence", 0.5: "portrait", 1.0: "measurement"}
    figs = "".join(
        f"<figure><canvas width='{SQ}' height='{SQ}'></canvas>"
        f"<figcaption>&gamma; = {g:g}"
        f"{' — ' + labels[g] if g in labels else ''}</figcaption></figure>"
        for g in gammas)
    css = f"""
    body{{background:#0c0c10;color:#cfcfd4;font:14px/1.5 -apple-system,sans-serif;
         margin:2rem}}
    h1{{font-size:1.1rem}}
    #row{{display:flex;gap:2rem;flex-wrap:wrap}}
    canvas{{border-radius:8px}}
    figure{{margin:0}}
    figcaption{{color:#8a8a92;font-size:13px;margin-top:.5rem;
               text-align:center}}
    #tip{{position:absolute;display:none;background:#000d;color:#eee;
         padding:4px 9px;border-radius:6px;font-size:12px;
         pointer-events:none;white-space:nowrap;z-index:9}}
    """
    h = [f"<meta charset='utf-8'><title>cinehue taste spectrum</title>"
         f"<style>{css}</style>",
         f"<h1>taste spectrum <span style='color:#77777f'>— {title}. "
         f"one smooth OKLab gradient; hue width = taste mass^&gamma;; dark "
         f"seams = blind spots; gray band = achromatic mass. hover for "
         f"films.</span></h1>",
         f"<div id='row'>{figs}</div>",
         "<div id='tip'></div>",
         f"<script>{js}</script>"]
    out.write_text("\n".join(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["atoms", "clustered", "gradient",
                                       "sum", "spectrum"], default="atoms")
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="exponent for the middle square")
    ap.add_argument("--soft", type=float, default=None,
                    help="kernel scale (default: 1.0; 0.55 in sum mode)")
    ap.add_argument("--sharp", type=float, default=1.5,
                    help="border crispness (higher = harder edges)")
    ap.add_argument("--knee", type=float, default=None,
                    help="glow threshold (default: 0.8; 0.05 in sum mode)")
    ap.add_argument("--bandwidth", type=float, default=10.0,
                    help="spectrum mode: hue smoothing in degrees")
    ap.add_argument("--film", help="exact title (fallback: substring) — "
                                   "render one film's space")
    args = ap.parse_args()
    if args.soft is None:
        args.soft = 0.55 if args.mode == "sum" else 1.0
    if args.knee is None:
        args.knee = 0.05 if args.mode == "sum" else 0.8

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

    if args.mode == "spectrum":
        chrom, gray_mass, gray_l, gray_n = spec_gather(films)
        if not chrom and not gray_mass:
            raise SystemExit("no identity stops found")
        dens, colors, tips = spec_smooth(chrom, args.bandwidth)
        segs, gaps = spec_segments(dens)
        gammas = [0.0, args.gamma, 1.0]
        squares = [spec_columns(dens, colors, tips, segs, gaps,
                                (gray_mass, gray_l, gray_n), g)
                   for g in gammas]
        render_spectrum(squares, gammas, title, out)
        print(f"[spectrum] {len(segs)} hue run(s), {len(gaps)} blind-spot "
              f"seam(s) from {len(films)} film(s) -> {out}")
        return

    swatches = build_measure(films)   # legend + clustered/gradient seeds
    if not swatches:
        raise SystemExit("no identity stops found — run the subject "
                         "experiment and extraction first")
    if args.mode in ("clustered", "gradient"):
        seeds_ = cluster_seeds(swatches)
        what = f"{len(seeds_)} regions"
    else:
        seeds_ = atom_seeds(films)
        what = f"{len(seeds_)} atoms"
    render(seeds_, swatches, [0.0, args.gamma, 1.0], title, out, args,
           args.mode)
    print(f"[{args.mode}] {what} from {len(films)} film(s) -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "scikit-learn", "tqdm"]
# ///
"""Extract a dominant-color palette per film from its scraped frames.

For each film in data/frames/<slug>/ this pools the frame pixels, converts to
OKLab (perceptually uniform), runs k-means for K dominant colors + weights, and
also computes a single weighted-blend color. Writes data/palettes.json and a
standalone data/preview.html for the eyeball gut-check.

Two honest layers per film: blend/L_mean (mood, by-area) and accent (saturated
identity, top-decile chroma).

DEAD-END (2026-07-02): a third "subject" layer (saliency-weighted pooling via
Achanta distance-from-mean, --saliency-gamma) is kept for reproducibility but
LOST to accent on all 4 test films — it collapses to near-black on dark films
(symmetric distance rewards dark backgrounds) or to the brightest region. Accent
is already the cheap "what stands out" signal. A real subject layer would need
foreground segmentation (depth / rembg), not this proxy.

Usage:
    uv run extract_palettes.py
    uv run extract_palettes.py --k 5 --max-px 40000
    uv run extract_palettes.py --saliency-gamma 2   # sharpen the subject layer
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
FRAMES = ROOT / "data" / "frames"
MANIFEST = ROOT / "data" / "manifest.json"
PALETTES = ROOT / "data" / "palettes.json"
PREVIEW = ROOT / "data" / "preview.html"
THUMB = 160  # longest side each frame is shrunk to before pooling pixels


# --- OKLab <-> sRGB (Bjorn Ottosson) -----------------------------------------
def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    # out-of-gamut OKLab colors land slightly outside [0,1]; clip first so
    # np.power never sees negatives (the NaN lane was discarded by np.where,
    # but numpy still warned)
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def srgb_to_oklab(rgb):  # rgb in [0,1], shape (...,3)
    r, g, b = [srgb_to_linear(rgb[..., i]) for i in range(3)]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([L, a, bb], axis=-1)


def oklab_to_srgb(lab):  # returns [0,1] sRGB, clipped
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return np.clip(linear_to_srgb(np.stack([r, g, bb], axis=-1)), 0, 1)


def hexof(lab):
    rgb = (oklab_to_srgb(np.asarray(lab)) * 255).round().astype(int)
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# --- pipeline ----------------------------------------------------------------
def frame_saliency(im):
    """Frequency-tuned (Achanta 2009) saliency in OKLab: each pixel's distance
    from the frame's mean colour after a light blur, normalised to [0,1] per
    frame. Colour-aware, no new deps. A flat frame yields ~uniform weights, so
    the subject layer degrades gracefully to by-area on subject-less shots."""
    blur = im.filter(ImageFilter.GaussianBlur(radius=2))
    lab = srgb_to_oklab(np.asarray(blur, dtype=np.float32) / 255.0)
    d = np.linalg.norm(lab - lab.reshape(-1, 3).mean(0), axis=-1)
    m = float(d.max())
    return (d / m if m > 0 else np.ones_like(d)).reshape(-1)


def load_pixels(frame_dir, max_px, rng):
    px, sal = [], []
    for f in tqdm(sorted(frame_dir.glob("*.jpg")), desc=frame_dir.name,
                  unit="frame", position=1, leave=False):
        try:
            im = Image.open(f).convert("RGB")
        except Exception:
            continue
        im.thumbnail((THUMB, THUMB))
        sal.append(frame_saliency(im))
        px.append(np.asarray(im, dtype=np.float32).reshape(-1, 3) / 255.0)
    if not px:
        return None, None
    px = np.concatenate(px, axis=0)
    sal = np.concatenate(sal, axis=0)
    if len(px) > max_px:
        idx = rng.choice(len(px), max_px, replace=False)
        px, sal = px[idx], sal[idx]
    return px, sal


def dominant(lab, k):
    """Most populous OKLab cluster center among the given pixels."""
    k = max(1, min(k, len(np.unique(lab.round(3), axis=0))))
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(lab)
    counts = np.bincount(km.labels_, minlength=k)
    return km.cluster_centers_[int(np.argmax(counts))]


def palette_for(px, k, gamma, sal=None, sgamma=1.0):
    lab = srgb_to_oklab(px)
    pix_c = np.hypot(lab[:, 1], lab[:, 2])
    # Main "area" palette. gamma=0 -> pure by-area (honest "what fills the
    # screen"). gamma>0 weights pixels by chroma^gamma so saturated minorities
    # can form their own clusters instead of dissolving into the neutrals.
    sw = None if gamma <= 0 else (pix_c + 1e-4) ** gamma
    kk = min(k, len(np.unique(lab.round(3), axis=0)))
    km = KMeans(n_clusters=kk, random_state=0, n_init=10).fit(lab, sample_weight=sw)
    centers = km.cluster_centers_
    wbase = np.ones(len(lab)) if sw is None else sw
    counts = np.bincount(km.labels_, weights=wbase, minlength=kk)
    weights = counts / counts.sum()
    order = np.argsort(weights)[::-1]
    centers, weights = centers[order], weights[order]
    cc = np.hypot(centers[:, 1], centers[:, 2])
    entries = [{"hex": hexof(c), "weight": round(float(w), 4),
                "L": round(float(c[0]), 4), "C": round(float(ci), 4),
                "h": round(float(np.degrees(np.arctan2(c[2], c[1]))) % 360, 1)}
               for c, w, ci in zip(centers, weights, cc)]
    blend = (weights[:, None] * centers).sum(axis=0)

    # Signature accent = dominant color among the top-decile-chroma pixels —
    # computed directly so a saturated minority (Ran's red) can't be buried.
    hi = lab[pix_c >= np.percentile(pix_c, 90)]
    ac = dominant(hi, 3) if len(hi) >= 10 else (hi.mean(0) if len(hi) else centers[0])
    accent = {"hex": hexof(ac), "L": round(float(ac[0]), 4),
              "C": round(float(np.hypot(ac[1], ac[2])), 4),
              "h": round(float(np.degrees(np.arctan2(ac[2], ac[1]))) % 360, 1)}

    # Subject = dominant colour once pixels are weighted by saliency. DEAD-END
    # (see module docstring): distance-from-mean is symmetric, so dark/large
    # backgrounds dominate and this loses to accent. Kept for reproducibility.
    # sgamma sharpens the weighting (>1 toward the most salient; 0 -> by-area).
    subject = None
    if sal is not None:
        sw2 = (sal + 1e-6) ** sgamma
        kk2 = min(k, len(np.unique(lab.round(3), axis=0)))
        skm = KMeans(n_clusters=kk2, random_state=0, n_init=10).fit(lab, sample_weight=sw2)
        scounts = np.bincount(skm.labels_, weights=sw2, minlength=kk2)
        sc = skm.cluster_centers_[int(np.argmax(scounts))]
        subject = {"hex": hexof(sc), "L": round(float(sc[0]), 4),
                   "C": round(float(np.hypot(sc[1], sc[2])), 4)}

    p50, p95, p99 = (round(float(np.percentile(pix_c, q)), 4) for q in (50, 95, 99))
    return {
        "palette": entries,
        "blend": hexof(blend),
        "blend_lab": [round(float(v), 4) for v in blend],
        "accent": accent,
        "subject": subject,
        "L_mean": round(float((weights * centers[:, 0]).sum()), 4),
        "C_mean": round(float((weights * cc).sum()), 4),
        "chroma_pct": {"p50": p50, "p95": p95, "p99": p99},
    }


SUBJECTS = ROOT / "data" / "subjects.json"  # written by experiment_subject_rembg.py


def with_mood(results):
    """Attach a 'mood' wheel to each film, built from the two IDENTITY signals:
    accent (environmental, always present) and subject-chroma (costume, when
    experiment_subject_rembg.py has written subjects.json and found one).
    Colors sit around the wheel sorted by hue angle, each owning an arc
    proportional to its chroma — the more saturated signal dominates the wheel.
    A film with no costume signal is a pure accent circle."""
    subj = json.loads(SUBJECTS.read_text()) if SUBJECTS.exists() else {}
    for r in results:
        cols = [r["accent"]]
        sc = (subj.get(r["slug"]) or {}).get("chroma")
        if sc:
            cols.append(sc)
        cols.sort(key=lambda c: c["h"])
        total = sum(c["C"] for c in cols) or 1.0
        stops, acc = [], 0.0
        for c in cols:
            stops.append({"hex": c["hex"], "weight": round(c["C"] / total, 4),
                          "pos": round((acc + c["C"] / 2) / total, 4)})
            acc += c["C"]
        r["mood"] = {"h": r["accent"]["h"], "wheel": stops}


def title_map():
    if not MANIFEST.exists():
        return {}
    m = json.loads(MANIFEST.read_text())
    return {v["slug"]: t for t, v in m.items() if isinstance(v, dict) and v.get("slug")}


def render_html(results):
    rows = []
    for r in results:
        sw = "".join(
            f'<span title="{p["hex"]} w={p["weight"]}" '
            f'style="flex:{p["weight"]:.4f} 0 0;min-width:4px;height:38px;'
            f'background:{p["hex"]}"></span>' for p in r["palette"])
        cp = r["chroma_pct"]
        sub = r.get("subject")
        sub_sw = (
            f'<span title="subject {sub["hex"]}" style="display:inline-block;'
            f'width:52px;height:38px;border-radius:6px;outline:2px dashed #999;'
            f'background:{sub["hex"]}"></span>') if sub else ''
        sub_cap = f' · subject {sub["hex"]} (C={sub["C"]})' if sub else ''
        mood = r["mood"]
        grad = ", ".join(f'{s["hex"]} {s["pos"] * 100:.1f}%' for s in mood["wheel"])
        grad += f', {mood["wheel"][0]["hex"]} 100%'  # wrap last hue back to first
        rows.append(
            f'<div style="margin:10px 0;font:13px system-ui">'
            f'<div style="display:flex;align-items:center;gap:12px">'
            f'<span title="mood wheel h={mood["h"]}" '
            f'style="display:inline-block;width:44px;height:44px;border-radius:50%;'
            f'background:conic-gradient(from 0deg, {grad})"></span>'
            f'<span title="blend {r["blend"]}" style="display:inline-block;width:52px;'
            f'height:38px;border-radius:6px;background:{r["blend"]}"></span>'
            f'<span title="accent {r["accent"]["hex"]}" style="display:inline-block;'
            f'width:52px;height:38px;border-radius:6px;outline:2px solid #444;'
            f'background:{r["accent"]["hex"]}"></span>'
            f'{sub_sw}'
            f'<div style="display:flex;width:220px;flex:none">{sw}</div>'
            f'<b>{r["title"]}</b>'
            f'<span style="color:#888">L={r["L_mean"]} C={r["C_mean"]} · '
            f'accent {r["accent"]["hex"]} (C={r["accent"]["C"]}){sub_cap} · '
            f'pixC p95={cp["p95"]} p99={cp["p99"]}</span>'
            f'</div></div>')
    PREVIEW.write_text(
        '<!doctype html><meta charset="utf-8"><title>cinehue preview</title>'
        '<body style="background:#111;color:#eee;padding:24px">'
        '<h2 style="font:600 18px system-ui">cinehue — empirical palettes</h2>'
        '<div style="font:12px system-ui;color:#888;margin-bottom:16px">'
        'swatches: mood wheel (circle — accent + subject-chroma around the wheel, '
        'arc = chroma, blended) · blend (raw) · accent (solid outline) '
        '· subject (dashed outline) · strip = dominant palette (width = weight)</div>'
        + "".join(rows) + "</body>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-px", type=int, default=40000)
    ap.add_argument("--chroma-gamma", type=float, default=0.0,
                    help="0 = area palette; >0 weights palette clusters by chroma^gamma")
    ap.add_argument("--saliency-gamma", type=float, default=1.0,
                    help="subject layer: >1 sharpens toward the most salient pixels; "
                         "0 = uniform (subject collapses to the by-area mood color)")
    ap.add_argument("--redo", action="store_true",
                    help="recompute films already present in palettes.json")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    titles = title_map()
    dirs = sorted(d for d in FRAMES.iterdir() if d.is_dir()) if FRAMES.exists() else []
    if not dirs:
        print(f"no frames found under {FRAMES} — run the scraper first")
        return

    # resume: reuse palettes already computed (films whose frame dir is gone
    # drop out automatically since we only walk existing dirs)
    prev = {r["slug"]: r for r in json.loads(PALETTES.read_text())} \
        if PALETTES.exists() else {}
    slugs = [d.name for d in dirs]

    def save(done_idx, results):
        pending = [prev[s] for s in slugs[done_idx + 1:] if s in prev]
        PALETTES.write_text(json.dumps(results + pending, indent=2,
                                       ensure_ascii=False))

    results, skipped = [], 0
    outer = tqdm(dirs, desc="films", unit="film", position=0)
    for i, d in enumerate(outer):
        if d.name in prev and not args.redo:
            results.append(prev[d.name])
            skipped += 1
            continue
        px, sal = load_pixels(d, args.max_px, rng)
        if px is None:
            tqdm.write(f"[skip] {d.name}: no readable frames")
            outer.refresh()
            continue
        r = palette_for(px, args.k, args.chroma_gamma, sal, args.saliency_gamma)
        r["slug"] = d.name
        r["title"] = titles.get(d.name, d.name)
        with_mood([r])
        results.append(r)
        save(i, results)   # after every film so a crash or ^C loses nothing
        sub = r.get("subject")
        subtxt = f'{sub["hex"]} (C={sub["C"]:.3f})' if sub else "-"
        tqdm.write(f'{r["title"]:24.24} blend {r["blend"]} · '
                   f'accent {r["accent"]["hex"]} (C={r["accent"]["C"]:.3f}) · '
                   f'subject {subtxt}')
        outer.refresh()

    with_mood(results)   # refresh every film's mood from current subjects.json
    results.sort(key=lambda r: (r["mood"]["h"], r["L_mean"]))  # around the wheel
    PALETTES.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    render_html(results)
    print(f"[done] {len(results) - skipped} computed, {skipped} reused "
          f"(--redo to recompute) -> {PALETTES.name}, {PREVIEW.name}")


if __name__ == "__main__":
    main()

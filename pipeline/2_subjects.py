#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["onnxruntime", "numpy", "pillow", "scikit-learn", "tqdm"]
# ///
"""EXPERIMENT: subject layer via real region proposal (U^2-Net salient-object
segmentation), replacing the failed Achanta proxy.

Runs the u2net.onnx model directly through onnxruntime — the rembg wrapper is
uninstallable on Python 3.13 (pymatting -> numba -> llvmlite 0.36 build failure)
and we only need the raw mask anyway. Model auto-downloads to ~/.u2net/ via curl
(same CA-bundle workaround as the scraper).

Per frame: predict a foreground mask, keep pixels with mask > 127. Frames where
the mask covers <2% (segmentation found nothing) or >90% (whole frame = no
distinct subject) contribute no subject pixels. Subject pixels pool across the
film's frames, then:

  subject_area   = dominant OKLab cluster within the mask (by area)
  subject_chroma = dominant cluster among top-decile-chroma pixels within the mask
                   (accent, but region-restricted)
  background     = weighted cluster-blend OUTSIDE the mask (mood/atmosphere,
                   uncontaminated by skin/costume). Subject-less frames
                   (cover < 2%) contribute ALL their pixels to background —
                   a landscape shot is pure mood; >90%-cover frames contribute
                   nothing (mask is meaningless there).
  L_mean         = mean OKLab L over every pixel of every frame (darkness layer)

Compares against the existing blend + accent and writes data/preview_subject.html
showing EVERY frame's proposed RoI (background dimmed) for the eyeball check.

Resume: films already present in subjects.json are skipped (the preview only
shows newly computed films). Delete a film's entry to redo it, or pass
--redo to recompute everything. subjects.json is saved after EVERY film.

Usage:
    uv run pipeline/2_subjects.py [--redo]
"""
import argparse
import base64
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import DATA, FRAMES, PALETTES, SUBJECTS, title_map
from lib.pixels import dominant, hexof, srgb_to_oklab

OUT = DATA / "preview_subject.html"
U2NET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
MODEL = Path.home() / ".u2net" / "u2net.onnx"
THUMB = 320  # u2net's native input side; keep mask and pooled pixels aligned
ALPHA = 127
MIN_COVER, MAX_COVER = 0.02, 0.90


def ensure_model():
    if MODEL.exists() and MODEL.stat().st_size > 100_000_000:
        return
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading u2net.onnx (~176MB) -> {MODEL}")
    subprocess.run(["curl", "-L", "--fail", "-o", str(MODEL), U2NET_URL], check=True)


def u2net_mask(im, sess):
    """Salient-object mask for an RGB PIL image, uint8 HxW at the image's size.
    Preprocessing mirrors rembg: scale by per-image max, ImageNet-normalize."""
    x = np.asarray(im.resize((320, 320), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x / max(float(x.max()), 1e-6) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    y = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    m = Image.fromarray((y * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR)
    return np.asarray(m)


def skin_score(lab):
    """Soft skin likelihood in [0,1]: product of gaussians around the skin
    locus (orange hue ~38deg, chroma ~0.07, mid lightness). A hard gamut box
    was too rigid — it swallowed borderline costume colours (crimson at C~0.12
    sits inside any box wide enough to catch real skin). Soft scoring penalises
    proximity to skin instead of drawing a cliff."""
    L, a, b = lab
    C = float(np.hypot(a, b))
    h = float(np.degrees(np.arctan2(b, a)))
    dh = min(abs(h - 38), 360 - abs(h - 38))
    return float(np.exp(-(dh / 18) ** 2)
                 * np.exp(-((C - 0.07) / 0.045) ** 2)
                 * np.exp(-((L - 0.55) / 0.35) ** 2))


def dominant_nonskin(lab, k):
    """Cluster the top-chroma pixels, score each cluster by
    population x (1 - skin_score), and return the winner. Only clusters that
    are more-skin-than-not (score >= 0.5) are ineligible; if every cluster is,
    return None — the film's subjects genuinely carry no colour signal."""
    kk = max(1, min(k, len(np.unique(lab.round(3), axis=0))))
    km = KMeans(n_clusters=kk, random_state=0, n_init=10).fit(lab)
    counts = np.bincount(km.labels_, minlength=kk).astype(float)
    skin = np.array([skin_score(c) for c in km.cluster_centers_])
    eligible = skin < 0.5
    if not eligible.any():
        return None
    scores = np.where(eligible, counts * (1 - skin), -1.0)
    return km.cluster_centers_[int(np.argmax(scores))]


def subject_colors(px):
    lab = srgb_to_oklab(px)
    area = dominant(lab, 4)  # skin is honest here: it IS what the subject looks like
    c = np.hypot(lab[:, 1], lab[:, 2])
    hi = lab[c >= np.percentile(c, 90)]
    chroma = dominant_nonskin(hi, 4) if len(hi) >= 10 else None
    return area, chroma


def subsample(px, rng, cap=40_000):
    if len(px) <= cap:
        return px
    return px[rng.choice(len(px), cap, replace=False)]


def blend_of(lab, k=5):
    """Weighted blend of the k cluster centers — the mood recipe. Taking the
    single most populous cluster instead collapses to black on dark films
    (area dominance), which is exactly what blend was designed to avoid."""
    kk = min(k, len(np.unique(lab.round(3), axis=0)))
    km = KMeans(n_clusters=kk, random_state=0, n_init=10).fit(lab)
    w = np.bincount(km.labels_, minlength=kk).astype(float)
    return (w[:, None] * km.cluster_centers_).sum(0) / w.sum()


def roi_overlay(im, mask, height=120):
    """Original frame with everything OUTSIDE the proposed region dimmed to 25%,
    so the RoI reads at full brightness in place."""
    keep = (mask > ALPHA)[..., None]
    rgb = np.asarray(im, dtype=np.float32)
    out = Image.fromarray(np.where(keep, rgb, rgb * 0.25).astype(np.uint8))
    out.thumbnail((10_000, height))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", action="store_true",
                    help="recompute films already present in subjects.json")
    args = ap.parse_args()

    ensure_model()
    # CoreML dispatches to the Apple GPU/ANE; unsupported ops fall back to CPU.
    sess = ort.InferenceSession(
        str(MODEL), providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    titles = title_map()
    pal = {r["slug"]: r for r in json.loads(PALETTES.read_text())} \
        if PALETTES.exists() else {}

    rows = []
    subjects = json.loads(SUBJECTS.read_text()) if SUBJECTS.exists() else {}
    skipped = 0
    dirs = sorted(p for p in FRAMES.iterdir() if p.is_dir())
    outer = tqdm(dirs, desc="films", unit="film", position=0)
    for d in outer:
        if d.name in subjects and not args.redo:
            skipped += 1
            continue
        subj_px, bg_px, thumbs, covers, l_sum, l_n = [], [], [], [], 0.0, 0
        for f in tqdm(sorted(d.glob("*.jpg")), desc=d.name, unit="frame",
                      position=1, leave=False):
            try:
                im = Image.open(f).convert("RGB")
            except Exception:
                tqdm.write(f"[bad frame] {d.name}/{f.name}: unreadable, skipped")
                continue
            im.thumbnail((THUMB, THUMB))
            mask = u2net_mask(im, sess)
            rgb = np.asarray(im, dtype=np.float32) / 255.0
            keep = mask > ALPHA
            cover = float(keep.mean())
            covers.append(cover)
            L = srgb_to_oklab(rgb)[..., 0]
            l_sum += float(L.sum()); l_n += L.size
            used_frame = MIN_COVER <= cover <= MAX_COVER
            if used_frame:
                subj_px.append(rgb[keep])
                bg_px.append(rgb[~keep])
            elif cover < MIN_COVER:  # subject-less shot: the whole frame is mood
                bg_px.append(rgb.reshape(-1, 3))
            thumbs.append((roi_overlay(im, mask), cover, used_frame))
        used = len(subj_px)
        if not subj_px:
            tqdm.write(f"[skip] {d.name}: no frame produced a usable mask")
            outer.refresh()
            continue
        area, chroma = subject_colors(subsample(np.concatenate(subj_px), rng))
        bg = blend_of(srgb_to_oklab(subsample(np.concatenate(bg_px), rng)))
        chroma_hex = hexof(chroma) if chroma is not None else None
        subjects[d.name] = {"chroma": None if chroma is None else {
            "hex": chroma_hex,
            "L": round(float(chroma[0]), 4),
            "C": round(float(np.hypot(chroma[1], chroma[2])), 4),
            "h": round(float(np.degrees(np.arctan2(chroma[2], chroma[1]))) % 360, 1)}}
        SUBJECTS.write_text(json.dumps(subjects, indent=2, ensure_ascii=False))
        l_mean = l_sum / l_n
        title = titles.get(d.name, d.name)
        p = pal.get(d.name, {})
        blend, acc = p.get("blend", "-"), p.get("accent", {}).get("hex", "-")
        tqdm.write(
            f"{title:24.24} bg {hexof(bg)} (blend {blend}) · "
            f"subj_area {hexof(area)} · subj_chroma {chroma_hex or 'none (all skin)'} · "
            f"accent {acc} · L_mean {l_mean:.3f} · {used}/{len(covers)} frames, "
            f"median cover {np.median(covers):.2f}")
        sw = "".join(
            f'<span style="display:inline-flex;flex-direction:column;align-items:center;'
            f'gap:2px;margin-right:8px">'
            f'<span title="{lbl} {hx}" style="display:inline-block;width:52px;height:38px;'
            f'border-radius:6px;background:{hx};outline:{ol}"></span>'
            f'<span style="font:10px system-ui;color:#888">{lbl}</span></span>'
            for lbl, hx, ol in [
                ("blend", blend, "none"),
                ("background", hexof(bg), "2px solid #b84"),
                ("accent", acc, "2px solid #444"),
                ("subject-area", hexof(area), "2px dashed #999"),
                ("subject-chroma" if chroma_hex else "chroma: none",
                 chroma_hex or "transparent", "2px dotted #4af"),
            ])
        imgs = "".join(
            f'<figure style="display:inline-block;margin:0 6px 6px 0;text-align:center">'
            f'<img src="data:image/jpeg;base64,{t}" style="height:120px;display:block;'
            f'outline:2px solid {"#2a2" if ok else "#c33"}">'
            f'<figcaption style="font:11px system-ui;color:{"#888" if ok else "#c66"}">'
            f'{cover:.0%}{"" if ok else " rejected"}</figcaption></figure>'
            for t, cover, ok in thumbs)
        rows.append(
            f'<div style="margin:26px 0;font:13px system-ui">'
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">'
            f'{sw}<b>{title}</b><span style="color:#888">blend {blend} · '
            f'bg {hexof(bg)} · accent {acc} · area {hexof(area)} · '
            f'chroma {chroma_hex or "none"} · L_mean {l_mean:.3f} · '
            f'{used}/{len(covers)} frames</span></div>{imgs}</div>')
        outer.refresh()

    OUT.write_text(
        '<!doctype html><meta charset="utf-8"><title>cinehue subject experiment</title>'
        '<body style="background:#111;color:#eee;padding:24px">'
        '<h2 style="font:600 18px system-ui">subject via U&#178;-Net region proposal</h2>'
        '<div style="font:12px system-ui;color:#888;margin-bottom:16px">'
        'swatches: blend (no outline) · background-outside-mask (orange) · '
        'accent (solid grey) · subject by-area-in-mask (dashed) · '
        'subject top-chroma-in-mask (dotted). Every frame shown with its proposed '
        'RoI at full brightness, background dimmed; green = pooled, red = rejected '
        'by the coverage guard (&lt;2% or &gt;90%)</div>'
        + "".join(rows) + "</body>")
    SUBJECTS.write_text(json.dumps(subjects, indent=2, ensure_ascii=False))
    print(f"[done] {len(rows)} computed, {skipped} skipped (already in "
          f"{SUBJECTS.name}; --redo or delete an entry to recompute) "
          f"-> {OUT.name}, {SUBJECTS.name}")


if __name__ == "__main__":
    main()

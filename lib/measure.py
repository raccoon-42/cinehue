"""The measure algebra — Step 1 and the quantizer.

map(movie) and map(user) = render(Σ measures of their movies):
a film is a measure over OKLab (mass 1, split across its identity stops
proportionally to chroma); collections just add. Everything here is linear
and deterministic; nonlinearity belongs in the renderers. Pure stdlib."""
import math

from lib.oklab import hex_to_oklab, lab_polar, oklab_to_hex

C_MIN = 0.02          # below this a color has no meaningful hue
MERGE_DEG = 45        # max hue span an honest swatch may cover


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


def baseline_bins(films, sigma=15.0):
    """A collection as a probability distribution over 361 bins — 360 hue
    degrees (chromatic mass smeared by a wrapped gaussian) + 1 gray bin.
    Dividing another collection's seed masses by these probabilities renders
    what it OVER/UNDER-represents relative to this baseline, instead of the
    shape of cinema itself — which is what any two large collections share.
    Returns (hue_probs[360], gray_prob)."""
    dens = [0.0] * 360
    gray = 0.0
    span = int(3 * sigma)
    kern = [math.exp(-(d * d) / (2 * sigma * sigma))
            for d in range(-span, span + 1)]
    for f in films:
        for a in film_atoms(f):
            if a["gray"]:
                gray += a["w"]
                continue
            h0 = int(a["h"])
            for i, k in enumerate(kern):
                dens[(h0 + i - span) % 360] += a["w"] * k
    tot = sum(dens) + gray
    if tot <= 0:
        return [1.0 / 361] * 360, 1.0 / 361
    return [x / tot for x in dens], gray / tot


def build_measure(films):
    """films (palettes.json records) -> quantized swatches with mass shares.
    The uncompressed atoms are film_atoms(); this is the <=45deg-per-swatch
    compression used for legends and palette renders."""
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

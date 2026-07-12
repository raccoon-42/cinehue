"""Numpy-vectorized OKLab conversions + pixel clustering helpers.
Importing script must declare numpy + scikit-learn in its uv deps."""
import numpy as np
from sklearn.cluster import KMeans


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    # out-of-gamut OKLab colors land slightly outside [0,1]; clip first so
    # np.power never sees negatives (the NaN lane was discarded by np.where,
    # but numpy still warned)
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c,
                    1.055 * np.power(c, 1 / 2.4) - 0.055)


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


def dominant(lab, k):
    """Most populous OKLab cluster center among the given pixels."""
    k = max(1, min(k, len(np.unique(lab.round(3), axis=0))))
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(lab)
    counts = np.bincount(km.labels_, minlength=k)
    return km.cluster_centers_[int(np.argmax(counts))]

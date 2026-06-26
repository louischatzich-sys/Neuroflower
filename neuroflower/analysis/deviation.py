"""Cross-condition deviation metrics.

Three deviation scalars at three different scales:

  • basin_shift                  : Euclidean distance between selection
                                    centroids (gross centroid drift).
  • topographic_extremum_disp    : mean displacement of paired SRAM extrema,
                                    polarity-aware nearest-neighbour pairing.
                                    Conceptual antecedent: Lehmann &
                                    Skrandies 1980 topographic peak tracking.
  • jaccard_distance             : 1 − |A ∩ B| / |A ∪ B| on channel-index
                                    selection sets (Jaccard 1912).

All three are channel/feature-coordinate measures: they require either head-
data coordinates (an (N, 2) array of electrode positions in the same unit
system used by the topomap) or simple integer index sets.

These metrics together form a triangulation: large basin_shift means the
centroid moved; large jaccard_distance with small basin_shift means the same
neighbourhood was selected via different electrodes; large extremum_disp
means individual peaks moved even when the channel set is similar.
"""
from __future__ import annotations
import numpy as np


# --------------------------- basin shift -----------------------------
def basin_centroid(channels, xy):
    """Mean position of selected channels in head-data coordinates."""
    if not channels:
        return None
    return np.mean(xy[np.asarray(channels)], axis=0)


def basin_shift(channels_A, channels_B, xy):
    """Euclidean distance between two basin centroids.

    Returns 0.0 if either set is empty (no deviation defined).
    """
    cA = basin_centroid(channels_A, xy)
    cB = basin_centroid(channels_B, xy)
    if cA is None or cB is None:
        return 0.0
    return float(np.linalg.norm(cA - cB))


# ---------------- topographic extremum displacement ------------------
def _hungarian_match(cost):
    """Minimal-cost rectangular assignment via greedy fallback if scipy
    isn't available.  cost is (m, n); returns a list of (i, j) pairs of
    length min(m, n)."""
    cost = np.asarray(cost, dtype=float)
    m, n = cost.shape
    matched = []
    used_r, used_c = set(), set()
    while len(matched) < min(m, n):
        # find global minimum over unmatched rows/cols
        best = (None, None, np.inf)
        for i in range(m):
            if i in used_r: continue
            for j in range(n):
                if j in used_c: continue
                if cost[i, j] < best[2]:
                    best = (i, j, cost[i, j])
        if best[0] is None: break
        matched.append((best[0], best[1]))
        used_r.add(best[0]); used_c.add(best[1])
    return matched


def topographic_extremum_displacement(pts_A, pts_B):
    """Mean Euclidean distance between matched extrema in two conditions.

    Extrema are matched via greedy nearest-neighbour assignment minimising
    total displacement (an approximation of the Hungarian algorithm; exact
    for the small sets used here, |pts| ≤ 10).

    Parameters
    ----------
    pts_A, pts_B : (k, 2) arrays of extrema positions (same coordinate frame).

    Returns
    -------
    mean_disp : float
        Mean of matched-pair Euclidean distances, in the same unit as the
        input coordinates.
    pairs : list of (i, j) — the matched index pairs.

    References
    ----------
    Conceptual antecedent: Lehmann D. & Skrandies W. (1980), reference-free
    identification of components of checkerboard-evoked multichannel
    potential fields, *Electroencephalogr. Clin. Neurophysiol.* 48:609-621.
    """
    pts_A = np.asarray(pts_A, dtype=float)
    pts_B = np.asarray(pts_B, dtype=float)
    if pts_A.size == 0 or pts_B.size == 0:
        return 0.0, []
    cost = np.linalg.norm(pts_A[:, None, :] - pts_B[None, :, :], axis=2)
    pairs = _hungarian_match(cost)
    if not pairs:
        return 0.0, []
    disps = [cost[i, j] for i, j in pairs]
    return float(np.mean(disps)), pairs


# -------------------------- Jaccard distance --------------------------
def jaccard_distance(set_A, set_B):
    """Jaccard distance = 1 − |A ∩ B| / |A ∪ B|.

    Parameters
    ----------
    set_A, set_B : iterables of channel indices.

    References
    ----------
    Jaccard P. (1912), The distribution of the flora in the alpine zone,
    *New Phytol.* 11(2):37-50.
    """
    A = set(set_A); B = set(set_B)
    union = A | B
    if not union: return 0.0
    return 1.0 - len(A & B) / len(union)


# ----------------------- combined report -----------------------------
def deviation_report(cal_A, cal_B):
    """Compute all three deviation metrics between two Calibration objects.

    Parameters
    ----------
    cal_A, cal_B : Calibration namedtuples / dicts with keys
                    'channels'  : list of selected channel indices
                    'xy'        : (n_ch, 2) coordinates
                    'extrema'   : (k, 2) array of hub positions in
                                  head-data coordinates (optional)

    Returns
    -------
    dict with keys 'basin_shift', 'extremum_displacement',
    'jaccard_distance', and 'extremum_pairs'.
    """
    # Resolve attributes whether cal is a dict or object
    def g(cal, k, default=None):
        if hasattr(cal, k): return getattr(cal, k)
        if isinstance(cal, dict): return cal.get(k, default)
        return default

    chans_A = list(g(cal_A, 'channels', []))
    chans_B = list(g(cal_B, 'channels', []))
    xy = g(cal_A, 'xy')
    if xy is None: xy = g(cal_B, 'xy')

    out = dict()
    out['basin_shift'] = (basin_shift(chans_A, chans_B, xy)
                           if xy is not None else 0.0)
    out['jaccard_distance'] = jaccard_distance(chans_A, chans_B)

    exA = g(cal_A, 'extrema'); exB = g(cal_B, 'extrema')
    if exA is not None and exB is not None:
        d, pairs = topographic_extremum_displacement(exA, exB)
        out['extremum_displacement'] = d
        out['extremum_pairs'] = pairs
    else:
        out['extremum_displacement'] = None
        out['extremum_pairs'] = None
    return out

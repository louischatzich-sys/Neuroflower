"""Multi-view mediator.

The mediator fuses three independent "views" of a topomap into a single
channel-selection set.  Each view captures different information:

  1. *Field view*       — Gaussian splats at hub centres + Bezier pathway tubes
                           between matched +/- hubs, polynomial-modulated.
  2. *Inverse view*     — 1 − normalised SRAM picture (the "absence pattern").
  3. *Transport view*   — Helmholtz |∇φ| from a Poisson solve of source−sink
                           masses; intensifies where charge is moving.

Each view yields local-maximum coordinates that are mapped to channels by
activity-rank matching.  The three selections are unioned with cross-exclusion
(a channel claimed by one view is removed from the eligible pool of the next).

Public API
----------
find_intersections : per-view local-maximum detector
match_unique       : rank-based map from coordinates to channels
mediator           : full three-view union → dict with {field, inv, flow, MOD,
                     spTRIO} channel index lists
"""
from __future__ import annotations
import numpy as np


# ---- local-maximum finder (lifted from the original play_helpers.py) -----
def find_intersections(heat, top_k=12, min_sep=10, threshold_ratio=0.55):
    s = heat.copy()
    for _ in range(3):
        s = (0.5 * s + 0.125 * np.roll(s, 1, 0) + 0.125 * np.roll(s, -1, 0)
                     + 0.125 * np.roll(s, 1, 1) + 0.125 * np.roll(s, -1, 1))
    cand = ((s > np.roll(s,  1, 0)) & (s > np.roll(s, -1, 0))
            & (s > np.roll(s,  1, 1)) & (s > np.roll(s, -1, 1)))
    thr = threshold_ratio * float(np.max(s)) if np.max(s) > 1e-12 else 0.0
    cand &= (s > thr)
    ys, xs = np.where(cand); vals = s[ys, xs]
    order = np.argsort(-vals)
    picks = []; pvals = []
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if all((y - py) ** 2 + (x - px) ** 2 >= min_sep ** 2
               for py, px in picks):
            picks.append((y, x))
            pvals.append(float(vals[idx]))
            if len(picks) >= top_k:
                break
    return np.array(picks, dtype=float), np.array(pvals)


# ---- activity-rank → channel matching --------------------------------
def match_unique(intersections, intersection_values, p_norm, exclude=None):
    """Match each intersection to a channel by closest activity rank.

    Channels already in `exclude` are unavailable; subsequent intersections
    pick from the remaining pool.  Returns a list of channel indices.
    """
    used = set(exclude) if exclude else set()
    matches = []
    for (py, px), hv in zip(intersections, intersection_values):
        diffs = np.abs(p_norm - hv)
        avail = np.ones(len(p_norm), dtype=bool)
        for u in used:
            avail[u] = False
        if not avail.any():
            avail[:] = True
        diffs[~avail] = np.inf
        idx = int(np.argmin(diffs))
        used.add(idx)
        matches.append(idx)
    return matches


# ---- the mediator itself ---------------------------------------------
def mediator(field_view, inverse_view, flow_view, p_norm,
              topk_field=12, topk_inv=12, topk_flow=10,
              sep_field=8, sep_inv=8, sep_flow=10,
              threshold_field=0.20, threshold_inv=0.20, threshold_flow=0.25):
    """Run all three views and combine them.

    Returns a dict with:
        'field'   — channels selected by field view alone
        'inv'     — channels added by inverse view (excluding 'field')
        'flow'    — channels added by transport view (excluding above)
        'MOD'     — union(field, inv)             ("dual selection")
        'spTRIO'  — union(field, inv, flow)       ("trio selection")

    Each component channel set is non-overlapping by construction; spTRIO is
    the recommended default selection.
    """
    iF, vF = find_intersections(field_view, top_k=topk_field,
                                 min_sep=sep_field,
                                 threshold_ratio=threshold_field)
    chan_field = match_unique(iF, vF, p_norm)

    iI, vI = find_intersections(inverse_view, top_k=topk_inv,
                                 min_sep=sep_inv,
                                 threshold_ratio=threshold_inv)
    chan_inv = match_unique(iI, 1 - vI, p_norm, exclude=set(chan_field))

    iL, vL = find_intersections(flow_view, top_k=topk_flow,
                                 min_sep=sep_flow,
                                 threshold_ratio=threshold_flow)
    chan_flow = match_unique(iL, vL, p_norm,
                              exclude=set(chan_field) | set(chan_inv))

    return dict(
        field=chan_field, inv=chan_inv, flow=chan_flow,
        MOD=list(set(chan_field) | set(chan_inv)),
        spTRIO=list(set(chan_field) | set(chan_inv) | set(chan_flow)),
    )

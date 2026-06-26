"""Scaffold and field-construction module.

The scaffold is the geometric structure built on top of the SRAM
reconstruction: discrete hub centres (positive/negative extrema), arc-length-
parameterised Bezier pathways between them, and the Helmholtz transport
field |∇φ| derived from treating the hubs as sources and sinks.

Public functions
----------------
hub_field(...)             : Gaussian splats at hub locations.
pathway_field(...)         : arc-length Bezier tubes between hub pairs.
flow_field(...)            : |∇φ| from a Poisson solve over source-sink masses.
make_modulated_field(...)  : 0.55·hubs**p + 0.45·paths**p, the locked field.
inverse_field(...)         : (1 − normalised picture)**p, smoothed.
ellipse_mask(...)          : default head-shape mask used throughout.
"""
from __future__ import annotations
import numpy as np

LOCKED_P = 2  # global polynomial-modulation power (locked, v1.0.0)
SPAN = 1.15


# -------------------------------- masks --------------------------------
def ellipse_mask(grid, span=SPAN, a=1.0, b=1.05, soft=0.05):
    g = np.linspace(-span, span, grid); gx, gy = np.meshgrid(g, g)
    rr = (gx / a) ** 2 + (gy / b) ** 2
    return np.clip((1 + soft - rr) / soft, 0.0, 1.0)


def pic_to_head(pic, mask):
    p = pic.astype(float) - pic.min()
    if p.max() > 0:
        p /= p.max()
    p = np.flipud(p)
    if p.shape != mask.shape:
        H, W = mask.shape
        ys = np.clip(np.round(np.linspace(0, p.shape[0] - 1, H)).astype(int),
                     0, p.shape[0] - 1)
        xs = np.clip(np.round(np.linspace(0, p.shape[1] - 1, W)).astype(int),
                     0, p.shape[1] - 1)
        p = p[ys][:, xs]
    return p * mask


def signed_picture(pic, mask):
    """Median-centred picture restricted to the head mask."""
    head = pic_to_head(pic, mask)
    inside = mask > 0.5
    ref = float(np.median(head[inside])) if inside.any() else 0.0
    return (head - ref) * mask


# ------------------------------ extrema --------------------------------
def topk_extrema(field, k=5, sign=+1, min_sep_px=18, smooth=3,
                  threshold_ratio=0.4):
    """Find up to k well-separated local extrema."""
    s = (sign * field).copy()
    for _ in range(smooth):
        s = (0.5 * s + 0.125 * np.roll(s, 1, 0) + 0.125 * np.roll(s, -1, 0)
                     + 0.125 * np.roll(s, 1, 1) + 0.125 * np.roll(s, -1, 1))
    cand = ((s > np.roll(s,  1, 0)) & (s > np.roll(s, -1, 0))
            & (s > np.roll(s,  1, 1)) & (s > np.roll(s, -1, 1)))
    if s.max() <= 0:
        return np.zeros((0, 2)), np.zeros(0)
    cand &= (s > threshold_ratio * s.max())
    ys, xs = np.where(cand); vals = s[ys, xs]
    order = np.argsort(-vals); picks = []; pvals = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if all((y - py) ** 2 + (x - px) ** 2 >= min_sep_px ** 2
               for py, px in picks):
            picks.append((y, x))
            pvals.append(float(vals[i]))
            if len(picks) >= k:
                break
    return np.array(picks, dtype=float), np.array(pvals)


# -------------------------- arc-length Bezier --------------------------
def _bezier_arclength(p0, p1, n_samples=160, curvature=0.30):
    p0 = np.array(p0, dtype=float); p1 = np.array(p1, dtype=float)
    mid = (p0 + p1) / 2; v = p1 - p0
    perp = np.array([-v[1], v[0]])
    perp = perp / (np.linalg.norm(perp) + 1e-9)
    ctrl = mid + perp * (curvature * np.linalg.norm(v))
    t_dense = np.linspace(0, 1, 600)[:, None]
    pts = ((1 - t_dense) ** 2) * p0 + 2 * (1 - t_dense) * t_dense * ctrl + (t_dense ** 2) * p1
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_cum = np.concatenate(([0], np.cumsum(seg)))
    L = s_cum[-1] + 1e-12
    s_uniform = np.linspace(0, L, n_samples)
    out = np.zeros((n_samples, 2))
    for d in range(2):
        out[:, d] = np.interp(s_uniform, s_cum, pts[:, d])
    return out, s_uniform, L


def pathway_field(grid_shape, pairs, sigma=8.0, heat='gaussian'):
    """Sum of arc-length Bezier tubes between paired hubs.

    pairs : list of {'pos': (y, x), 'neg': (y, x), 'depth': float}
    heat  : 'gaussian' (depth · gauss((s/L − 0.5)/0.35)) or 'uniform'.
    """
    out = np.zeros(grid_shape, dtype=np.float64)
    yy, xx = np.indices(grid_shape); s2 = 2.0 * sigma ** 2
    for p in pairs:
        pts, su, L = _bezier_arclength(p['pos'], p['neg'])
        if heat == 'gaussian':
            u = (su / L - 0.5) / 0.35
            wts = np.exp(-0.5 * u * u)
        else:
            wts = np.ones_like(su)
        for k, (py, px) in enumerate(pts):
            out += p['depth'] * wts[k] * np.exp(
                -((yy - py) ** 2 + (xx - px) ** 2) / s2)
    return out / max(out.max(), 1e-12)


def hub_field(grid_shape, hub_positions, sigma=10.0):
    """Sum of isotropic Gaussians at each hub position."""
    out = np.zeros(grid_shape, dtype=np.float64)
    yy, xx = np.indices(grid_shape); s2 = 2.0 * sigma ** 2
    for (py, px) in hub_positions:
        out += np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / s2)
    return out / max(out.max(), 1e-12)


# ---------------- Helmholtz transport flow ----------------------------
def _poisson_fft(rhs):
    H, W = rhs.shape
    rhs = rhs - rhs.mean()
    fr = np.fft.fft2(rhs)
    ky = np.fft.fftfreq(H) * 2 * np.pi
    kx = np.fft.fftfreq(W) * 2 * np.pi
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K2 = KX ** 2 + KY ** 2; K2[0, 0] = 1.0
    fphi = -fr / K2; fphi[0, 0] = 0.0
    return np.real(np.fft.ifft2(fphi))


def flow_field(grid_shape, pos_pts, neg_pts, pos_w, neg_w, sigma=10.0,
                mask=None):
    """Source–sink Poisson solve.  Returns (|∇φ|, φ).

    Treats positive hubs as sources, negative hubs as sinks, builds a mass
    distribution m = Σ source · gauss − Σ sink · gauss, solves ∇²φ = −m via
    FFT, and returns the gradient magnitude (the transport-flow magnitude).
    """
    H, W = grid_shape
    yy, xx = np.indices((H, W)); s2 = 2.0 * sigma ** 2
    m = np.zeros((H, W))
    for (py, px), w in zip(pos_pts, pos_w):
        m += w * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / s2)
    for (py, px), w in zip(neg_pts, neg_w):
        m -= w * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / s2)
    if mask is not None:
        m *= (mask > 0.5).astype(float)
        if mask.sum() > 0:
            m -= m[mask > 0.5].mean()
            m *= (mask > 0.5).astype(float)
    phi = _poisson_fft(m)
    gy, gx = np.gradient(phi)
    mag = np.sqrt(gy ** 2 + gx ** 2)
    if mask is not None:
        mag *= (mask > 0.5).astype(float)
    return mag / max(mag.max(), 1e-12), phi


# ---------------- modulated and inverse field combiners ---------------
def make_modulated_field(hubs_f, paths_f, p=LOCKED_P,
                          hub_weight=0.55, path_weight=0.45):
    """Locked global-power modulation.

    Default p=LOCKED_P=2 (symmetric).  Returns the normalised combined field.
    """
    field = hub_weight * (hubs_f ** p) + path_weight * (paths_f ** p)
    return field / max(field.max(), 1e-12)


def inverse_field(picture, mask, p=LOCKED_P, smoothing_passes=3):
    """1 − normalised picture, smoothed and raised to p (locked)."""
    head = pic_to_head(picture, mask)
    inv = (1 - head) * (mask > 0.5).astype(float)
    for _ in range(smoothing_passes):
        inv = (0.5 * inv
                + 0.125 * np.roll(inv, 1, 0) + 0.125 * np.roll(inv, -1, 0)
                + 0.125 * np.roll(inv, 1, 1) + 0.125 * np.roll(inv, -1, 1))
    inv = inv ** p
    return inv / max(inv.max(), 1e-12)


# ---------------- coordinate helpers ----------------------------------
def pixel_to_data(px, py, grid, span=SPAN):
    """Convert (col, row) image pixel to head-data (x, y) coordinates."""
    x = ((px / (grid - 1)) - 0.5) * 2 * span
    y = (0.5 - (py / (grid - 1))) * 2 * span
    return x, y

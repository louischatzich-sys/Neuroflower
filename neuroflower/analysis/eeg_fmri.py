from __future__ import annotations
import numpy as np
from math import gamma as Gamma

def canonical_hrf(tr, duration_s=32.0):
    t = np.arange(0, duration_s, tr)
    g1 = (t**5 * np.exp(-t)) / (1 * Gamma(6))
    g2 = (t**15 * np.exp(-t)) / (1 * Gamma(16))
    h = g1 - g2/6.0
    return h / np.max(np.abs(h) + 1e-12)

def convolve_with_hrf(tc, times_s, tr, n_vol):
    bold_t = np.arange(n_vol) * tr
    reg = np.interp(bold_t, times_s, tc) if len(times_s) >= 2 else np.full(n_vol, float(np.mean(tc)))
    h = canonical_hrf(tr)
    c = np.convolve(reg - reg.mean(), h, mode='full')[:n_vol]
    return c - c.mean()

def voxelwise_correlation(reg, bold_4d, mask=None):
    X,Y,Z,T = bold_4d.shape
    r = reg - reg.mean()
    rs = float(np.sqrt((r**2).sum())) or 1e-12
    out = np.zeros((X,Y,Z))
    flat = bold_4d.reshape(X*Y*Z, T)
    m = mask.reshape(X*Y*Z).astype(bool) if mask is not None else np.ones(X*Y*Z, bool)
    for i in np.where(m)[0]:
        v = flat[i]; v = v - v.mean()
        s = float(np.sqrt((v**2).sum())) or 1e-12
        out.flat[i] = float((r*v).sum() / (rs*s))
    return out

def occipital_mask(shape_xyz):
    X,Y,Z = shape_xyz
    m = np.zeros(shape_xyz, dtype=bool)
    m[:, :max(1, Y//3), :Z//2] = True
    return m

"""Public reconstruction API."""
from __future__ import annotations
import numpy as np

def reconstruct_picture(*, source, seed=None, n_lobes_override=None):
    from ._reconstruction_impl import _reconstruct_picture_impl
    return _reconstruct_picture_impl(source=source, seed=seed, n_lobes_override=n_lobes_override)

def normalize(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.min(x)
    m = float(np.max(x))
    return np.zeros_like(x) if m < 1e-12 else x / m

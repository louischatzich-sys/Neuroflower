"""Welch PSD + canonical band powers (pure numpy)."""
from __future__ import annotations
import numpy as np

BANDS = {
    "delta":(1.0,4.0),"theta":(4.0,7.0),"alpha":(8.0,12.0),
    "spindle":(12.0,16.0),"beta":(13.0,30.0),"gamma":(30.0,45.0),
}

def welch_psd(x, fs, nperseg=None, noverlap=None):
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if nperseg is None: nperseg = min(n, max(256, int(fs*2)))
    if noverlap is None: noverlap = nperseg // 2
    step = nperseg - noverlap
    n_seg = 1 + (n - nperseg) // step if n >= nperseg else 0
    if n_seg <= 0:
        x = np.pad(x, (0, nperseg - n)); n_seg = 1
    win = np.hanning(nperseg)
    win_norm = (win**2).sum() * fs
    acc = np.zeros(nperseg // 2 + 1)
    for i in range(n_seg):
        seg = x[i*step:i*step+nperseg]
        seg = (seg - seg.mean()) * win
        f = np.fft.rfft(seg)
        acc += (np.abs(f)**2) / win_norm
    psd = acc / n_seg
    psd[1:-1] *= 2
    return np.fft.rfftfreq(nperseg, 1.0/fs), psd

def band_power(freqs, psd, lo, hi):
    """Trapezoidal-rule integration of PSD over [lo, hi].

    Hand-implemented rather than calling np.trapz/np.trapezoid so the code
    works across NumPy 1.x and 2.x without an alias shim (np.trapz was
    renamed to np.trapezoid in NumPy 2.0 and
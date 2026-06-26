from __future__ import annotations
import numpy as np
from ..core.psd import welch_psd, band_power, BANDS

def channel_band_powers(signals, fs, band='alpha', nperseg=None):
    lo, hi = BANDS[band] if isinstance(band, str) else band
    if nperseg is None: nperseg = max(256, int(fs*2))
    out = np.zeros(signals.shape[0])
    for i, x in enumerate(signals):
        f, p = welch_psd(x, fs, nperseg=nperseg)
        out[i] = band_power(f, p, lo, hi)
    return out

def windowed_band_powers(signals, fs, band='alpha', window_s=2.0, step_s=None):
    if step_s is None: step_s = window_s
    lo, hi = BANDS[band] if isinstance(band, str) else band
    n_ch, n_samp = signals.shape
    win_n = int(window_s * fs); step_n = int(step_s * fs)
    n_win = max(0, 1 + (n_samp - win_n) // step_n)
    pows = np.zeros((n_ch, n_win)); times = np.zeros(n_win)
    nperseg = min(win_n, max(128, int(fs)))
    for w in range(n_win):
        s = w*step_n; e = s + win_n
        times[w] = (s + e) / 2.0 / fs
        for c in range(n_ch):
            f, p = welch_psd(signals[c, s:e], fs, nperseg=nperseg)
            pows[c, w] = band_power(f, p, lo, hi)
    return times, pows

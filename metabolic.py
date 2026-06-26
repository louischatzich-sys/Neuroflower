"""Leaky-integrator metabolic model.

Implements
    E(t) = bandpass(EEG(t), [lo, hi])
    M(t+1) = α · M(t) + |E(t)|²,  α = exp(−Δt / τ)

Time constant τ is set to mimic neurovascular coupling (Buxton et al. 1998,
balloon model; Magri et al. 2012, LFP→BOLD).  M(t) is interpreted as a slow
metabolic envelope derived from electrical demand.

This module is independent of the spatial pipeline: it produces a per-channel
score that can be used as the topomap input, or correlated against a measured
BOLD timecourse.
"""
from __future__ import annotations
import numpy as np
from ..core.psd import BANDS


def band_filter(signal, fs, lo, hi):
    """Zero-phase FFT band-pass on the last axis."""
    n = signal.shape[-1]
    X = np.fft.rfft(signal, axis=-1)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    H = ((f >= lo) & (f <= hi)).astype(float)
    return np.fft.irfft(X * H, n=n, axis=-1)


def metabolic_signal(signals_2d, fs, band='alpha', tau_s=5.0, burn_in_s=10.0):
    """Apply the leaky-integrator metabolic model to each channel.

    Parameters
    ----------
    signals_2d : (n_channels, n_samples)
    fs : float
    band : str or (lo, hi)
    tau_s : float
        Integration time constant in seconds (default 5 s, matching the
        rise-time of the canonical hemodynamic response).
    burn_in_s : float
        Initial samples to discard from the per-channel score.

    Returns
    -------
    M : (n_channels, n_samples) array of metabolic envelopes
    score : (n_channels,) post-burn-in mean of M
    """
    lo, hi = BANDS[band] if isinstance(band, str) else band
    E = band_filter(signals_2d, fs, lo, hi)
    Esq = E * E
    alpha = float(np.exp(-1.0 / (fs * tau_s)))
    n_ch, n_t = Esq.shape
    M = np.zeros_like(Esq)
    m = np.zeros(n_ch)
    for t in range(n_t):
        m = alpha * m + Esq[:, t]
        M[:, t] = m
    burn = int(burn_in_s * fs)
    score = M[:, burn:].mean(axis=1) if n_t > burn else M.mean(axis=1)
    return M, score


def metabolic_score(signals_2d, fs, band='alpha', tau_s=5.0):
    """Convenience wrapper returning only the per-channel score."""
    return metabolic_signal(signals_2d, fs, band=band, tau_s=tau_s)[1]

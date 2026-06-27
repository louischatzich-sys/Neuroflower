"""comparator_analysis.py

Pre-registered ICA-vs-spTRIO multi-band comparator.
Protocol: see comparator_analysis_protocol.md (timestamped in git history).

Locked specifications (do not change after viewing results):
  - Comparator:   sklearn FastICA, n_components=full,
                  max_iter=1000, tol=1e-5, random_state=42,
                  whiten='unit-variance'
  - IC selection: the single IC whose mixing-column topography has the
                  highest absolute Pearson correlation with the
                  all-channel band-power topomap.
  - K matching:   K = number of channels spTRIO selects for that band
                  on the same subject (recorded at runtime).
  - Bands:        delta theta alpha beta gamma (definitions from
                  neuroflower.core.psd.BANDS).
  - Subject:      S002R01.
  - Metric:       spatial r (Pearson) between subset-topomap and
                  reference-topomap, restricted to head-mask interior.
                  Same recipe as the rest of the paper.

Run:  python comparator_analysis.py
Env:  set S002R01 to the EDF path if it isn't in the current directory.
"""

from __future__ import annotations
import os
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.decomposition import FastICA

import neuroflower as nf
from neuroflower.io import read_edf
from neuroflower.core.topomap import project_to_topomap, get_layout_xy
from neuroflower.analysis.eeg_bands import channel_band_powers

# --------------------------------------------------------------------
# Locked configuration
# --------------------------------------------------------------------
EDF_PATH = os.environ.get("S002R01", "S002R01.edf")
GRID = 180
RBF_SIGMA_REF = 0.13   # truth-topomap kernel (matches paper / pipeline)
RBF_SIGMA_SUB = 0.18   # subset-topomap kernel (matches existing robustness script)
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

ICA_KW = dict(
    max_iter=1000,
    tol=1e-5,
    random_state=42,
    whiten="unit-variance",
)


# --------------------------------------------------------------------
# Helpers (same as the existing robustness script)
# --------------------------------------------------------------------
def _clean(labels):
    return [
        x.replace("EEG ", "").replace("-Ref", "").replace("-REF", "")
        .strip().rstrip(".")
        for x in labels
    ]


def _mat(signal_dict, labels):
    n = max(len(s) for s in signal_dict.values())
    m = np.zeros((len(labels), n))
    for i, l in enumerate(labels):
        m[i, : len(signal_dict[l])] = signal_dict[l]
    return m


def load_subject(edf_path):
    e = read_edf(edf_path)
    labels0 = _clean(e["labels"])
    sigs0 = _mat(e["signals"], e["labels"])
    fs = e["sample_rate"]
    coords, found = get_layout_xy(labels0)
    keep = np.array(found)
    sigs = sigs0[keep]
    lbl = [l for l, k in zip(labels0, keep) if k]
    xy = coords[keep]
    return sigs, fs, lbl, xy


def topomap_from_power(power, labels, sigma):
    return project_to_topomap(power, labels, grid_size=GRID, rbf_sigma=sigma)


def spatial_r(channel_idx, p_norm, labels, topo_ref, inside):
    """Spatial r of a channel-subset topomap against the reference."""
    if len(channel_idx) == 0:
        return 0.0
    sub_labels = [labels[i] for i in channel_idx]
    sub_topo = topomap_from_power(p_norm[channel_idx], sub_labels, RBF_SIGMA_SUB)
    return float(np.corrcoef(topo_ref[inside], sub_topo[inside])[0, 1])


# --------------------------------------------------------------------
# Per-band ICA + comparison
# --------------------------------------------------------------------
def run_band(band, signals, fs, labels):
    # 1) per-channel band power and normalisation (same recipe as paper)
    p = channel_band_powers(signals, fs, band=band)
    p_norm = (p - p.min()) / (p.max() - p.min() + 1e-12)

    # 2) reference topomap from all channels
    topo_ref = topomap_from_power(p_norm, labels, RBF_SIGMA_REF)
    inside = topo_ref != 0

    # 3) spTRIO calibration for this band; K = number of channels it selected
    cal = nf.calibrate(signals, fs, labels, band=band)
    K = len(cal.channels)
    r_sptrio = spatial_r(cal.channels, p_norm, labels, topo_ref, inside)

    # 4) sklearn FastICA on signals × time (samples x channels)
    n_ch = signals.shape[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ica = FastICA(n_components=n_ch, **ICA_KW)
        ica.fit(signals.T)  # rows = samples, cols = channels
    converged = not any("did not converge" in str(w.message) for w in caught)

    # 5) pick the IC whose mixing-column topography correlates best
    #    (in absolute value) with the truth topomap. Locked rule.
    mixing = ica.mixing_  # (n_channels, n_components)
    best_ic = -1
    best_topo_r = 0.0
    for ic in range(mixing.shape[1]):
        ic_topo = topomap_from_power(mixing[:, ic], labels, RBF_SIGMA_REF)
        if not np.any(ic_topo != 0):
            continue
        r = float(np.corrcoef(topo_ref[inside], ic_topo[inside])[0, 1])
        if abs(r) > abs(best_topo_r):
            best_topo_r = r
            best_ic = ic

    # 6) the K highest-magnitude channels in that IC's topography
    chosen = list(map(int, np.argsort(-np.abs(mixing[:, best_ic]))[:K]))
    r_ica = spatial_r(chosen, p_norm, labels, topo_ref, inside)

    # 7) raw top-K-by-power control (same as existing script)
    raw_chosen = list(map(int, np.argsort(-p_norm)[:K]))
    r_raw = spatial_r(raw_chosen, p_norm, labels, topo_ref, inside)

    return {
        "band": band,
        "K": K,
        "sptrio_r": r_sptrio,
        "ica_r": r_ica,
        "raw_r": r_raw,
        "delta_sptrio_minus_ica": r_sptrio - r_ica,
        "best_ic": int(best_ic),
        "best_ic_topo_r": float(best_topo_r),
        "ica_converged": bool(converged),
        "ica_n_iter": int(getattr(ica, "n_iter_", -1)),
    }


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    print(f"Loading {EDF_PATH} ...")
    signals, fs, labels, xy = load_subject(EDF_PATH)
    print(f"  {signals.shape[0]} channels @ {fs:.0f} Hz, "
          f"{signals.shape[1]} samples\n")

    rows = []
    for band in BANDS:
        print(f"  running {band:>5} ...", end=" ", flush=True)
        try:
            row = run_band(band, signals, fs, labels)
            rows.append(row)
            conv = "OK" if row["ica_converged"] else f"NO ({row['ica_n_iter']} it)"
            print(f"K={row['K']:>3}  spTRIO={row['sptrio_r']:+.3f}  "
                  f"ICA={row['ica_r']:+.3f}  raw={row['raw_r']:+.3f}  "
                  f"conv={conv}")
        except Exception as exc:
            print(f"FAILED: {exc!r}")
            rows.append({"band": band, "error": repr(exc)})

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "subject": "S002R01",
        "edf_path": str(EDF_PATH),
        "protocol_version": "1.0",
        "ica_settings": ICA_KW,
        "grid_size": GRID,
        "rbf_sigma_reference": RBF_SIGMA_REF,
        "rbf_sigma_subset": RBF_SIGMA_SUB,
        "results": rows,
    }
    out_path = Path("comparator_results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path.resolve()}")

    # Final table
    print("\n=== Locked-protocol results ===")
    print(f"{'band':<6} {'K':>3}  {'spTRIO':>7} {'ICA':>7} {'raw':>7}  "
          f"{'Δ(sp−ICA)':>10}  {'IC#':>4}  {'conv':>5}")
    for r in rows:
        if "error" in r:
            print(f"{r['band']:<6}  ERROR: {r['error']}")
            continue
        print(f"{r['band']:<6} {r['K']:>3}  "
              f"{r['sptrio_r']:+.3f}  {r['ica_r']:+.3f}  {r['raw_r']:+.3f}  "
              f"{r['delta_sptrio_minus_ica']:>+10.3f}  "
              f"{r['best_ic']:>4}  "
              f"{'yes' if r['ica_converged'] else 'NO':>5}")


if __name__ == "__main__":
    main()

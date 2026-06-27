"""Run this locally to complete the two robustness checks:
  (A) sklearn FastICA comparison (verifies the result isn't an
      implementation artifact of our pure-numpy FastICA)
  (B) Multi-component reconstruction (K_ic = 1, 3, 5, 10, 20)

Requirements: pip install scikit-learn matplotlib numpy
Place this in the neuroflower repo root, then:  python run_locally_ica_robustness.py
"""
import sys, os
from pathlib import Path
import numpy as np

# Adjust this if you're running outside the repo root:
REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))

import neuroflower as nf
from neuroflower.io import read_edf
from neuroflower.core.topomap import project_to_topomap, get_layout_xy
from neuroflower.analysis.eeg_bands import channel_band_powers
from neuroflower.core.psd import welch_psd, band_power, BANDS

# EDF path — adjust to wherever S002R01.edf lives on your machine
EDF_PATH = os.environ.get("S002R01", "S002R01.edf")
GRID = 180

def _clean(L):
    return [x.replace('EEG ','').replace('-Ref','').replace('-REF','').strip().rstrip('.') for x in L]
def _mat(d, labels):
    n = max(len(s) for s in d.values()); m = np.zeros((len(labels), n))
    for i, l in enumerate(labels): m[i,:len(d[l])] = d[l]
    return m

# ---- Load ----
e = read_edf(EDF_PATH)
labels0 = _clean(e['labels']); sigs0 = _mat(e['signals'], e['labels'])
fs = e['sample_rate']
coords, found = get_layout_xy(labels0); keep = np.array(found)
SIGS = sigs0[keep]; LBL = [l for l, k in zip(labels0, keep) if k]; XY = coords[keep]
N_CH = SIGS.shape[0]
print(f"Loaded {N_CH} channels @ {fs:.0f} Hz")

# ---- spTRIO + raw baselines ----
cal = nf.calibrate(SIGS, fs, LBL, band='beta')
K = len(cal.channels)
p_beta = channel_band_powers(SIGS, fs, band='beta')
p_norm = (p_beta - p_beta.min())/(p_beta.max() - p_beta.min() + 1e-12)
topo_ref = project_to_topomap(p_norm, LBL, grid_size=GRID, rbf_sigma=0.13)
inside = topo_ref != 0
def spatial_r(channels):
    if not channels: return 0.0
    ts = project_to_topomap(p_norm[channels], [LBL[i] for i in channels],
                              grid_size=GRID, rbf_sigma=0.18)
    return float(np.corrcoef(topo_ref[inside], ts[inside])[0, 1])

r_sptrio = spatial_r(cal.channels)
r_raw = spatial_r(list(np.argsort(-p_norm)[:K].astype(int)))
print(f"K = {K}")
print(f"spTRIO β r = {r_sptrio:+.4f}")
print(f"Raw β r    = {r_raw:+.4f}")

# ---- (A) sklearn FastICA ----
print("\n=== (A) sklearn FastICA ===")
try:
    from sklearn.decomposition import FastICA
    # Standard defaults: tol=1e-4, max_iter=200, fun='logcosh'
    ica = FastICA(n_components=N_CH, random_state=0, max_iter=200,
                    tol=1e-4, fun='logcosh', whiten='unit-variance')
    # sklearn expects samples-as-rows
    src_sk = ica.fit_transform(SIGS.T).T   # back to (n_ch, n_t)
    mix_sk = ica.mixing_                     # (n_ch, n_ic)
    band_pows = np.array([band_power(*welch_psd(src_sk[i], fs),
                                       *BANDS['beta']) for i in range(N_CH)])
    ic_best = int(np.argmax(band_pows))
    chans = list(np.argsort(-np.abs(mix_sk[:, ic_best]))[:K].astype(int))
    r_sk = spatial_r(chans)
    print(f"  sklearn FastICA β r (single best IC, top-K) = {r_sk:+.4f}")
    print(f"  iterations: {ica.n_iter_}")
except ImportError:
    print("  sklearn not installed; run:  pip install scikit-learn")
    src_sk = None
    band_pows = None

# ---- (B) Multi-component reconstruction (use sklearn ICs if available) ----
print("\n=== (B) Multi-component reconstruction ===")
if src_sk is not None:
    # Use sklearn's ICs for the multi-component test
    print(f"{'K_ic':>4} | {'weighted-sum |topos|':>22} | {'union top-channels':>20}")
    print("-" * 55)
    ic_rank = np.argsort(-band_pows)
    for K_ic in [1, 3, 5, 10, 20]:
        top = ic_rank[:K_ic]
        # Weighted-sum topographies
        w = band_pows[top] / band_pows[top].sum()
        combined = np.zeros(N_CH)
        for i, wi in zip(top, w):
            combined += wi * np.abs(mix_sk[:, i])
        c_chans = list(np.argsort(-combined)[:K].astype(int))
        r_c = spatial_r(c_chans)
        # Round-robin union
        seen, union = set(), []
        per_rank = [list(np.argsort(-np.abs(mix_sk[:, i]))) for i in top]
        pos = [0] * len(top)
        while len(union) < K:
            for j in range(len(top)):
                while pos[j] < N_CH:
                    c = int(per_rank[j][pos[j]]); pos[j] += 1
                    if c not in seen:
                        seen.add(c); union.append(c); break
                if len(union) >= K: break
            if all(p >= N_CH for p in pos): break
        r_u = spatial_r(union[:K])
        print(f"{K_ic:>4} | {r_c:+.4f}                | {r_u:+.4f}")
    print(f"\nReference spTRIO: {r_sptrio:+.4f}")
    print(f"Reference raw:    {r_raw:+.4f}")
else:
    print("  Skipped — needs sklearn for the multi-component reconstruction.")

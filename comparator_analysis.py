# comparator_analysis.py
# Pre-registered: see comparator_analysis_protocol.md (timestamped)

import numpy as np
import mne
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr
import json
from datetime import datetime

# === LOAD ===
# Same data-loading code as your main pipeline. Subject S002R01, 58 channels.

# === BAND DEFINITIONS ===
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45),
}

# === SPTRIO RESULTS (from your existing v1.0.0 runs, copy from Table 1) ===
SPTRIO_R = {'delta': 0.994, 'theta': 0.992, 'alpha': 0.939, 
            'beta': 0.742, 'gamma': 0.952}
SPTRIO_K = {'delta': ..., 'theta': ..., 'alpha': ..., 
            'beta': ..., 'gamma': ...}  # fill in actual K per band

# === COMPARATOR LOOP ===
results = []
for band_name, (fmin, fmax) in BANDS.items():
    # 1. Filter to band, compute per-channel band power
    band_power = compute_band_power(raw_data, fmin, fmax)  # (n_channels,)
    
    # 2. Reference topomap from all 58 channels
    truth_topo = project_to_topomap(band_power, montage)  # (180, 180)
    
    # 3. Run sklearn FastICA on band-filtered data
    ica = FastICA(n_components=58, max_iter=1000, tol=1e-5, 
                  random_state=42, whiten='unit-variance')
    band_filtered = bandpass(raw_data, fmin, fmax)  # (n_samples, n_channels)
    ica.fit(band_filtered)
    
    converged = ica.n_iter_ < 1000
    
    # 4. For each IC, project its topography (mixing column) to topomap
    #    and correlate with truth_topo. Pick the best.
    mixing = ica.mixing_  # (n_channels, n_components)
    best_ic, best_topo_r = None, -np.inf
    for ic_idx in range(58):
        ic_topo_vec = mixing[:, ic_idx]  # weights per channel
        ic_topo_map = project_to_topomap(ic_topo_vec, montage)
        r, _ = pearsonr(ic_topo_map[mask], truth_topo[mask])
        if abs(r) > abs(best_topo_r):
            best_ic, best_topo_r = ic_idx, r
    
    # 5. Get the K highest-magnitude channels from the chosen IC
    K = SPTRIO_K[band_name]
    chosen_channels = np.argsort(np.abs(mixing[:, best_ic]))[-K:]
    
    # 6. Reconstruct topomap from those K channels only, compute spatial r
    ica_topo = project_to_topomap(band_power, montage, 
                                   channel_mask=chosen_channels)
    ica_r, _ = pearsonr(ica_topo[mask], truth_topo[mask])
    
    results.append({
        'band': band_name,
        'K': K,
        'sptrio_r': SPTRIO_R[band_name],
        'ica_r': ica_r,
        'delta': SPTRIO_R[band_name] - ica_r,
        'best_ic': best_ic,
        'best_topo_r': best_topo_r,
        'converged': converged,
        'n_iter': ica.n_iter_,
    })

# === SAVE ===
with open('comparator_results.json', 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'subject': 'S002R01',
        'protocol_version': '1.0',
        'results': results,
    }, f, indent=2)

# === PRINT TABLE ===
print(f"{'band':<6} {'K':<4} {'spTRIO r':<10} {'ICA r':<10} {'Δ':<8} {'IC#':<5} {'conv':<6}")
for r in results:
    print(f"{r['band']:<6} {r['K']:<4} {r['sptrio_r']:<10.3f} "
          f"{r['ica_r']:<10.3f} {r['delta']:<+8.3f} {r['best_ic']:<5} "
          f"{'yes' if r['converged'] else 'NO':<6}")
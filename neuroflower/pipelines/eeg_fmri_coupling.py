from __future__ import annotations
import numpy as np
from ..core.topomap import project_to_topomap
from ..core.reconstruction import reconstruct_picture
from ..analysis.eeg_bands import windowed_band_powers, channel_band_powers
from ..analysis.eeg_fmri import convolve_with_hrf, voxelwise_correlation, occipital_mask
from ..analysis.hub_extraction import summarise

DEFAULT_OCC = ('O1','O2','Oz','POz','PO7','PO8','PO3','PO4','Pz')

def run_eeg_fmri_coupling(*, eeg_labels, eeg_signals, eeg_fs, bold_4d, tr,
                          band='alpha', roi_mask=None, occ_channels=DEFAULT_OCC,
                          window_s=2.0, step_s=None, eeg_start_offset_s=0.0):
    chan_pows = channel_band_powers(eeg_signals, eeg_fs, band=band)
    topo = project_to_topomap(chan_pows, eeg_labels)
    occ_idx = [i for i,l in enumerate(eeg_labels)
               if l.strip().upper() in {c.upper() for c in occ_channels}]
    if not occ_idx: occ_idx = list(range(len(eeg_labels)))
    times, pows = windowed_band_powers(eeg_signals[occ_idx], eeg_fs,
                                       band=band, window_s=window_s, step_s=step_s)
    tc = pows.mean(axis=0)
    times = times + eeg_start_offset_s
    n_vol = bold_4d.shape[-1]
    reg = convolve_with_hrf(tc, times, tr, n_vol)
    if roi_mask is None: roi_mask = occipital_mask(bold_4d.shape[:3])
    coup = voxelwise_correlation(reg, bold_4d, mask=roi_mask)
    roi_corr = float(coup[roi_mask].mean()) if roi_mask.any() else 0.0
    rec = reconstruct_picture(source=topo)
    return dict(topomap=topo, reconstruction=rec, hub_summary=summarise(rec),
                coupling_map=coup, regressor=reg, roi_correlation=roi_corr,
                band_powers=chan_pows, eeg_labels=eeg_labels, band=band, tr=tr,
                eeg_offset_s=eeg_start_offset_s)

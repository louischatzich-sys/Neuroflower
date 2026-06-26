"""Top-level subject-specific calibration pipeline (locked v1.0).

Single entry point:  calibrate(signals, fs, labels, band='alpha') → Calibration

The Calibration object collects everything the downstream analysis cares
about: selected channels (spTRIO and components), basin centroid, extrema in
both pixel and head-data coordinates, and the intermediate fields if the
caller asks for them.

Locked parameters (do not tune per-band):
    p              = 2     polynomial modulation power
    hub_weight     = 0.55  field-combination weights
    path_weight    = 0.45
    k_hub          = 5     number of hubs per polarity
    rbf_sigma      = 0.13  topomap projection kernel
    grid_size      = 180
"""
from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import Optional
import numpy as np

from ..core.topomap import project_to_topomap, get_layout_xy
from ..core.reconstruction import reconstruct_picture
from ..analysis.eeg_bands import channel_band_powers
from ..analysis.scaffold import (
    ellipse_mask, signed_picture, topk_extrema, pixel_to_data,
    pathway_field, hub_field, flow_field,
    make_modulated_field, inverse_field, SPAN, LOCKED_P,
)
from ..analysis.mediator import mediator
from ..analysis.deviation import basin_centroid

DEFAULT_GRID = 180


@dataclass
class Calibration:
    """Result of one calibration run.

    Fields
    ------
    band, p_band       : input band and per-channel band power.
    p_norm             : normalised score used as the topomap input.
    topo               : 2D topomap.
    picture            : SRAM-reconstructed picture (raw).
    signed_picture     : median-centred picture restricted to the head mask.
    labels, xy         : kept-channel labels and 2D positions.
    extrema_pix        : (k+k, 2) array of all hub pixel coordinates (pos+neg).
    extrema_xy         : same extrema in head-data coordinates.
    pos_pix, neg_pix   : positive- and negative-polarity hubs in pixel coords.
    channels           : final spTRIO channel-index selection.
    components         : dict 'field', 'inv', 'flow', 'MOD' channel selections.
    basin              : centroid of the spTRIO channels in head-data coords.
    field, inv, flow   : the three view fields (returned if keep_fields=True).
    metadata           : misc (parameter snapshot, version, etc.).
    """
    band: str
    p_band: np.ndarray
    p_norm: np.ndarray
    topo: np.ndarray
    picture: np.ndarray
    signed_picture: np.ndarray
    labels: list
    xy: np.ndarray
    extrema_pix: np.ndarray
    extrema_xy: np.ndarray
    pos_pix: np.ndarray
    neg_pix: np.ndarray
    channels: list
    components: dict
    basin: Optional[np.ndarray]
    field: Optional[np.ndarray] = None
    inv: Optional[np.ndarray] = None
    flow: Optional[np.ndarray] = None
    phi: Optional[np.ndarray] = None
    metadata: dict = dc_field(default_factory=dict)

    @property
    def extrema(self):
        """Alias for extrema_xy (head-data coords) used by deviation.py."""
        return self.extrema_xy


def calibrate(signals, fs, labels, *, band='alpha', score=None,
              k_hub=5, rbf_sigma=0.13, grid_size=DEFAULT_GRID,
              p=LOCKED_P, hub_weight=0.55, path_weight=0.45,
              hub_sigma=10.0, path_sigma=8.0,
              keep_fields=False):
    """Run the locked calibration pipeline on a single EEG segment.

    Parameters
    ----------
    signals : (n_channels, n_samples) array
    fs : float, sampling rate
    labels : list of channel names (matched to the 10-10 layout if possible)
    band : str or (lo, hi)
        Frequency band used for the default score (ignored if `score` given).
    score : (n_channels,) array, optional
        Per-channel score to use as the topomap input.  If None, the band
        power is computed and used.  Pass e.g. metabolic_score(signals, fs,
        band='beta') here to swap the leaky-integrator score in.
    keep_fields : bool
        If True, attach the three view fields and Helmholtz potential to the
        returned Calibration (~4 × 180² ≈ 130 KB extra per call).

    Returns
    -------
    Calibration

    Notes
    -----
    Parameters p, hub_weight, path_weight, k_hub, rbf_sigma have defaults
    locked at v1.0.0 values.  Reviewers and downstream users should not
    re-tune these without re-running the diagnostic battery; see the methods
    paper for the rationale and pre-registered limitations.
    """
    coords, found = get_layout_xy(labels)
    keep = np.array(found)
    if keep.sum() < 4:
        raise ValueError(f"Need at least 4 channels with known layout "
                          f"positions; got {int(keep.sum())}.")
    sigs = signals[keep]
    lbls = [l for l, k in zip(labels, keep) if k]
    xy = coords[keep]

    if score is None:
        p_band = channel_band_powers(sigs, fs, band=band)
    else:
        p_band = np.asarray(score)[keep]
    if p_band.max() == p_band.min():
        raise ValueError("Per-channel score is degenerate (max == min).")
    p_norm = (p_band - p_band.min()) / (p_band.max() - p_band.min() + 1e-12)

    topo = project_to_topomap(p_norm, lbls, grid_size=grid_size,
                              rbf_sigma=rbf_sigma)
    rec = reconstruct_picture(source=topo)
    picture = rec['picture']
    mask = ellipse_mask(grid_size)
    signed = signed_picture(picture, mask)

    pos_pix, pos_vals = topk_extrema(signed, k=k_hub, sign=+1)
    neg_pix, neg_vals = topk_extrema(signed, k=k_hub, sign=-1)
    if min(len(pos_vals), len(neg_vals)) == 0:
        raise ValueError("No hub-class extrema found in the signed picture.")

    n = min(len(pos_vals), len(neg_vals))
    pairs = [{'pos': tuple(pos_pix[i]),
              'neg': tuple(neg_pix[i]),
              'depth': float((pos_vals[i] + neg_vals[i]) / 2)}
             for i in range(n)]
    H, W = signed.shape
    hubs_xy_pix = list(pos_pix) + list(neg_pix)

    hubs_f = hub_field((H, W), hubs_xy_pix, sigma=hub_sigma)
    paths_f = pathway_field((H, W), pairs, sigma=path_sigma, heat='gaussian')
    field_mod = make_modulated_field(hubs_f, paths_f, p=p,
                                     hub_weight=hub_weight,
                                     path_weight=path_weight)
    inv_mod = inverse_field(picture, mask, p=p)
    flow_mag, phi = flow_field((H, W), pos_pix, neg_pix,
                                pos_vals, neg_vals,
                                sigma=hub_sigma, mask=mask)

    components = mediator(field_mod, inv_mod, flow_mag, p_norm)
    channels = components['spTRIO']

    # Convert extrema pixels to head-data coordinates
    extrema_pix = np.vstack([pos_pix, neg_pix]) if (
        len(pos_pix) and len(neg_pix)) else pos_pix
    extrema_xy = np.array([pixel_to_data(px, py, grid_size, span=SPAN)
                            for (py, px) in extrema_pix])

    basin = basin_centroid(channels, xy)

    metadata = dict(
        version='1.0.0', locked_p=p,
        hub_weight=hub_weight, path_weight=path_weight,
        k_hub=k_hub, rbf_sigma=rbf_sigma, grid_size=grid_size,
        fs=fs, band=band,
        n_channels_kept=int(keep.sum()),
        n_channels_input=len(labels),
    )

    cal = Calibration(
        band=band, p_band=p_band, p_norm=p_norm, topo=topo,
        picture=picture, signed_picture=signed,
        labels=lbls, xy=xy,
        extrema_pix=extrema_pix, extrema_xy=extrema_xy,
        pos_pix=pos_pix, neg_pix=neg_pix,
        channels=channels, components=components,
        basin=basin,
        field=field_mod if keep_fields else None,
        inv=inv_mod if keep_fields else None,
        flow=flow_mag if keep_fields else None,
        phi=phi if keep_fields else None,
        metadata=metadata,
    )
    return cal

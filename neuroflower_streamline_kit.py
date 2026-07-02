"""neuroflower_streamline_kit.py

Drop-in extension for neuroflower v1.0.0:
  - Cap layout with labelled electrodes
  - Per-band SRAM reconstruction (topomap → picture → signed → hubs)
  - Per-band streamline density maps (3-pass coverage)
  - Streamline-anchored channel selection (hybrid)
  - Merged canonical boundary
  - Correlation summary bar chart

USAGE (run from the neuroflower repo root):

    python3 neuroflower_streamline_kit.py YOUR_FILE.edf
    python3 neuroflower_streamline_kit.py FILE_A.edf FILE_B.edf

Output goes to ./figures_<timestamp>/  and includes, per EDF file:
    <name>_cap.png          — electrode cap layout
    <name>_reconstruction.png — SRAM reconstruction stages, per band
    <name>_streamlines.png   — streamline density maps, per band
    <name>_canonical.png     — merged canonical boundary
    <name>_correlations.png  — bar chart of raw / spTRIO / streamline / hybrid
"""
import sys, os
from pathlib import Path
from datetime import datetime
import numpy as np

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path: sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import neuroflower as nf
from neuroflower.io import read_edf
from neuroflower.core.topomap import project_to_topomap, get_layout_xy
from neuroflower.core.reconstruction import reconstruct_picture
from neuroflower.core.psd import BANDS
from neuroflower.analysis.eeg_bands import channel_band_powers
from neuroflower.analysis.scaffold import (
    ellipse_mask, signed_picture, topk_extrema, flow_field, SPAN,
    pixel_to_data, _bezier_arclength)
from neuroflower.viz.topomap_plot import plot_topomap

GRID = 180
BANDS_ALL = ['delta', 'theta', 'alpha', 'beta', 'gamma']
BAND_COLOURS = {'delta':'#9966CC','theta':'#3FB3FF','alpha':'#FFAA22',
                'beta':'#FF5566','gamma':'#22BB55'}
N_RING = 24; RING_R = 12; COV_R = 10

def clean_labels(L):
    return [x.replace('EEG ','').replace('-Ref','').replace('-REF','')
             .strip().rstrip('.') for x in L]
def signals_matrix(sd, labels):
    n = max(len(s) for s in sd.values()); M = np.zeros((len(labels), n))
    for i, l in enumerate(labels): M[i, :len(sd[l])] = sd[l]
    return M
def data_to_pixel(x, y):
    return float((x/SPAN+1)*(GRID-1)/2), float((1-y/SPAN)*(GRID-1)/2)
def load_and_prep(path):
    e = read_edf(str(path))
    L0 = clean_labels(e['labels']); S0 = signals_matrix(e['signals'], e['labels'])
    fs = e['sample_rate']
    coords, found = get_layout_xy(L0); keep = np.array(found)
    return S0[keep], fs, [l for l, k in zip(L0, keep) if k], coords[keep]

def trace(gx, gy, x0, y0, steps=300, step=0.7, mask=None):
    H, W = gx.shape; path=[(x0,y0)]; x,y=float(x0),float(y0)
    for _ in range(steps):
        ix=int(np.clip(round(x),0,W-1)); iy=int(np.clip(round(y),0,H-1))
        if mask is not None and mask[iy,ix]<0.5: break
        dx=gx[iy,ix]; dy=gy[iy,ix]; mag=np.hypot(dx,dy)
        if mag<1e-12: break
        x+=step*dx/mag; y+=step*dy/mag
        if not(0<=x<W and 0<=y<H): break
        path.append((x,y))
    return path

def amplitude_centre(p_norm, XY):
    t = p_norm.sum()+1e-12
    return float((p_norm*XY[:,0]).sum()/t), float((p_norm*XY[:,1]).sum()/t)

def dilate(cov, r):
    H, W = cov.shape; yy,xx = np.indices((2*r+1,2*r+1))
    k = (yy-r)**2 + (xx-r)**2 <= r*r
    out=np.zeros_like(cov)
    for dy in range(-r,r+1):
        for dx in range(-r,r+1):
            if k[dy+r,dx+r]: out |= np.roll(np.roll(cov,dy,0),dx,1)
    return out

def build_3pass_density(sigs, fs, labels, XY, band, mask, chan_pix):
    p = channel_band_powers(sigs, fs, band=band)
    p_norm = (p-p.min())/(p.max()-p.min()+1e-12)
    topo = project_to_topomap(p_norm, labels, grid_size=GRID, rbf_sigma=0.13)
    pic = reconstruct_picture(source=topo)['picture']
    signed = signed_picture(pic, mask)
    pos, pv = topk_extrema(signed, k=5, sign=+1)
    neg, nv = topk_extrema(signed, k=5, sign=-1)
    n = min(len(pv), len(nv))
    if n == 0: return None
    _, phi = flow_field(signed.shape, pos, neg, pv, nv, sigma=10.0, mask=mask)
    gy, gx = np.gradient(phi)
    cx, cy = amplitude_centre(p_norm, XY)
    ax_px, ay_py = data_to_pixel(cx, cy)
    paths = []
    for k in range(N_RING):
        a = 2*np.pi*k/N_RING
        sx, sy = ax_px+RING_R*np.cos(a), ay_py+RING_R*np.sin(a)
        paths.append(trace(+gx,+gy, sx, sy, mask=mask))
        paths.append(trace(-gx,-gy, sx, sy, mask=mask))
    H, W = mask.shape
    cov = np.zeros((H, W), bool)
    for pth in paths:
        for px, py in pth:
            cov[int(np.clip(round(py),0,H-1)), int(np.clip(round(px),0,W-1))] = True
    cov = dilate(cov, COV_R)
    uncov = [ci for ci, (px, py) in enumerate(chan_pix)
              if not cov[int(round(py)), int(round(px))]]
    for ci in uncov:
        px, py = chan_pix[ci]
        paths.append(trace(+gx,+gy, px, py, mask=mask))
        paths.append(trace(-gx,-gy, px, py, mask=mask))
    for i in range(n):
        pts, _, _ = _bezier_arclength(tuple(pos[i]), tuple(neg[i]), n_samples=120)
        paths.append([(pp[1], pp[0]) for pp in pts])
    density = np.zeros((H, W))
    for pth in paths:
        for px, py in pth:
            density[int(np.clip(round(py),0,H-1)), int(np.clip(round(px),0,W-1))] += 1
    for _ in range(2):
        density = (0.5*density + 0.125*np.roll(density,1,0) + 0.125*np.roll(density,-1,0)
                    + 0.125*np.roll(density,1,1) + 0.125*np.roll(density,-1,1))
    density *= (mask>0.5).astype(float)
    return dict(density=density, p_norm=p_norm, topo=topo, picture=pic,
                signed=signed, pos_pix=pos, neg_pix=neg,
                cx=cx, cy=cy, ax_px=ax_px, ay_py=ay_py, n_uncov=len(uncov))

def anchor_select(density, chan_pix, K, p_norm=None, r=6):
    H, W = density.shape; scores = np.zeros(len(chan_pix))
    for ci, (px, py) in enumerate(chan_pix):
        ix, iy = int(round(px)), int(round(py))
        y0, y1 = max(0,iy-r), min(H,iy+r+1); x0, x1 = max(0,ix-r), min(W,ix+r+1)
        scores[ci] = float(density[y0:y1, x0:x1].mean())
    scores /= max(scores.max(), 1e-12)
    if p_norm is not None: scores *= p_norm
    return list(np.argsort(-scores)[:K].astype(int))

def evaluate_spatial_r(p_norm, labels, chans, topo_ref):
    if not chans: return 0.0
    ts = project_to_topomap(p_norm[chans], [labels[i] for i in chans],
                             grid_size=GRID, rbf_sigma=0.18)
    inside = topo_ref != 0
    return float(np.corrcoef(topo_ref[inside], ts[inside])[0,1]) \
        if inside.sum()>0 and ts[inside].std()>0 else 0.0

# ---------------- figures -----------------
def figure_cap_layout(XY, labels, out_path):
    fig, ax = plt.subplots(figsize=(9, 9))
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), 'k-', lw=1.8)
    ax.plot([-0.10, 0, 0.10], [0.99, 1.10, 0.99], 'k-', lw=1.5)
    et = np.linspace(-np.pi/2, np.pi/2, 30)
    ax.plot(1+0.05*np.cos(et), 0.15*np.sin(et), 'k-', lw=1.5)
    ax.plot(-1-0.05*np.cos(et), 0.15*np.sin(et), 'k-', lw=1.5)
    for (x, y), l in zip(XY, labels):
        ax.plot(x, y, 'o', color='gold', mec='black', ms=12, zorder=20)
        ax.annotate(l, xy=(x, y), xytext=(0, 0), textcoords='offset points',
                     fontsize=7, fontweight='bold', ha='center', va='center', zorder=21)
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(f'Electrode layout ({len(labels)} channels)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)

def figure_reconstruction_per_band(band_data, XY, mask, out_path):
    """5 rows × 4 cols: topomap, SRAM picture, signed with hubs, streamlines overlay."""
    fig = plt.figure(figsize=(20, 22))
    gs = fig.add_gridspec(5, 4, hspace=0.24, wspace=0.15)
    th = np.linspace(0, 2*np.pi, 200)
    for row, band in enumerate(BANDS_ALL):
        d = band_data.get(band)
        # Col 0: topomap
        ax = fig.add_subplot(gs[row, 0])
        if d is not None:
            plot_topomap(d['topo'], labels=None, ax=ax, show_labels=False,
                          title=f'{band}  (1) topomap')
            for (x, y) in XY:
                ax.plot(x, y, '.', color='black', ms=1.5, zorder=15)
        else:
            ax.axis('off')
        # Col 1: SRAM picture (raw reconstruction)
        ax = fig.add_subplot(gs[row, 1])
        if d is not None:
            pic = d['picture']
            v = float(np.max(np.abs(pic)))
            ax.imshow(pic, cmap='RdBu_r', vmin=-v, vmax=v,
                       extent=(-1.15, 1.15, -1.15, 1.15), origin='upper')
            ax.plot(np.cos(th), np.sin(th), 'k-', lw=1.5)
            ax.plot([-0.10, 0, 0.10], [0.99, 1.10, 0.99], 'k-', lw=1)
            ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            ax.set_title(f'{band}  (2) SRAM picture')
        else:
            ax.axis('off')
        # Col 2: signed picture with hubs
        ax = fig.add_subplot(gs[row, 2])
        if d is not None:
            signed = d['signed']
            v = float(np.max(np.abs(signed)))
            ax.imshow(signed, cmap='RdBu_r', vmin=-v, vmax=v,
                       extent=(-1.15, 1.15, -1.15, 1.15), origin='upper')
            ax.plot(np.cos(th), np.sin(th), 'k-', lw=1.5)
            ax.plot([-0.10, 0, 0.10], [0.99, 1.10, 0.99], 'k-', lw=1)
            for (py, px) in d['pos_pix']:
                xd, yd = pixel_to_data(px, py, GRID)
                ax.plot(xd, yd, 'P', color='red', mec='black', ms=13, mew=1.2, zorder=20)
            for (py, px) in d['neg_pix']:
                xd, yd = pixel_to_data(px, py, GRID)
                ax.plot(xd, yd, 'X', color='blue', mec='black', ms=13, mew=1.2, zorder=20)
            ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            ax.set_title(f'{band}  (3) signed + hubs (+/-)')
        else:
            ax.axis('off')
        # Col 3: topomap with amplitude centre + electrodes
        ax = fig.add_subplot(gs[row, 3])
        if d is not None:
            plot_topomap(d['topo'], labels=None, ax=ax, show_labels=False,
                          title=f'{band}  (4) amplitude centre (★)')
            ax.plot(d['cx'], d['cy'], '*', color='cyan', mec='black', ms=22, mew=1.5, zorder=25)
            for (x, y) in XY:
                ax.plot(x, y, 'o', color='gold', mec='black', ms=4, zorder=15)
        else:
            ax.axis('off')
    fig.suptitle('SRAM reconstruction per band  ·  (1) topomap  →  (2) picture  →  (3) signed+hubs  →  (4) amplitude centre',
                  fontsize=13, y=0.997, fontweight='bold')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)

def figure_streamlines_per_band(band_data, XY, mask, out_path):
    fig = plt.figure(figsize=(15, 22))
    gs = fig.add_gridspec(5, 3, hspace=0.22, wspace=0.15)
    th = np.linspace(0, 2*np.pi, 200)
    for row, band in enumerate(BANDS_ALL):
        d = band_data.get(band)
        ax = fig.add_subplot(gs[row, 0])
        if d is not None:
            plot_topomap(d['topo'], labels=None, ax=ax, show_labels=False, title=f'{band} topomap')
            ax.plot(d['cx'], d['cy'], '*', color='cyan', mec='black', ms=16, mew=1.2, zorder=21)
        else: ax.axis('off')
        ax = fig.add_subplot(gs[row, 1])
        if d is not None:
            dn = d['density'] / max(d['density'].max(), 1e-12)
            ax.imshow(dn, cmap='inferno', origin='upper', extent=(-1.15, 1.15, -1.15, 1.15))
            ax.plot(np.cos(th), np.sin(th), 'w-', lw=1.2)
            ax.plot([-0.10, 0, 0.10], [0.99, 1.10, 0.99], 'w-', lw=1)
            ax.plot(d['cx'], d['cy'], '*', color='cyan', mec='black', ms=14, mew=1, zorder=21)
            for (x, y) in XY:
                ax.plot(x, y, '.', color='gold', ms=2, zorder=15)
            ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            ax.set_title(f'{band} streamline density (3-pass)')
        else: ax.axis('off')
        ax = fig.add_subplot(gs[row, 2])
        if d is not None:
            dn = d['density'] / max(d['density'].max(), 1e-12)
            inv = (1.0 - dn) * (mask > 0.5).astype(float)
            ax.imshow(inv, cmap='inferno', origin='upper', extent=(-1.15, 1.15, -1.15, 1.15))
            ax.plot(np.cos(th), np.sin(th), 'w-', lw=1.2)
            ax.plot([-0.10, 0, 0.10], [0.99, 1.10, 0.99], 'w-', lw=1)
            ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            ax.set_title(f'{band} inverse density')
        else: ax.axis('off')
    fig.suptitle('Per-band streamline density maps', fontsize=12, y=0.997, fontweight='bold')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)

def figure_correlations(rows, subtitle, out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    bands = [r[0] for r in rows]
    raw = [r[2] for r in rows]; sp = [r[3] for r in rows]
    st = [r[4] for r in rows]; hy = [r[5] for r in rows]
    x = np.arange(len(bands)); w = 0.20
    ax.bar(x - 1.5*w, raw, width=w, color='#3FB3FF', edgecolor='black', label='raw top-K')
    ax.bar(x - 0.5*w, sp,  width=w, color='#FFAA22', edgecolor='black', label='spTRIO v1.0')
    ax.bar(x + 0.5*w, st,  width=w, color='#FF5566', edgecolor='black', label='streamline anchor')
    ax.bar(x + 1.5*w, hy,  width=w, color='#22BB55', edgecolor='black', label='hybrid (density × power)')
    ax.set_xticks(x); ax.set_xticklabels(bands)
    ax.set_ylabel('spatial r'); ax.set_ylim(0, 1.05)
    ax.set_title(f'Streamline-anchored channel selection\n{subtitle}')
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=9, loc='lower right')
    for i in range(len(bands)):
        ax.text(i + 1.5*w, hy[i] + 0.012, f'{hy[i]:+.2f}', ha='center', fontsize=7, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)

def figure_canonical(band_data, XY, mask, out_path):
    H, W = mask.shape
    merged = np.zeros((H, W)); coverage = np.zeros((H, W), dtype=int)
    for band, d in band_data.items():
        if d is None: continue
        m = d['density'] > 0
        merged += d['density'] / max(d['density'].max(), 1e-12)
        coverage[m] += 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    th = np.linspace(0, 2*np.pi, 200)
    ax = axes[0]
    mn = merged / max(merged.max(), 1e-12)
    ax.imshow(mn, cmap='inferno', origin='upper', extent=(-1.15, 1.15, -1.15, 1.15))
    ax.plot(np.cos(th), np.sin(th), 'w-', lw=1.5)
    ax.plot([-0.10, 0, 0.10], [0.99, 1.10, 0.99], 'w-', lw=1.2)
    peaks, _ = topk_extrema(mn, k=15, sign=+1, min_sep_px=12, smooth=2, threshold_ratio=0.3)
    for (py, px) in peaks:
        xd, yd = pixel_to_data(px, py, GRID)
        nb = int(coverage[int(py), int(px)])
        ax.plot(xd, yd, '*', color='cyan', mec='black', ms=8+3*nb, mew=1, zorder=20)
    for (x, y) in XY:
        ax.plot(x, y, '.', color='gold', ms=2.5, zorder=15)
    for band, d in band_data.items():
        if d is None: continue
        ax.plot(d['cx'], d['cy'], 'o', color=BAND_COLOURS[band], mec='black', ms=10, zorder=22)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title('Merged density (all bands)\ncanonical boundary (★ size = # bands)')
    ax = axes[1]
    im = ax.imshow(coverage, cmap='viridis', origin='upper',
                    extent=(-1.15, 1.15, -1.15, 1.15), vmin=0, vmax=5)
    ax.plot(np.cos(th), np.sin(th), 'w-', lw=1.5)
    ax.plot([-0.10, 0, 0.10], [0.99, 1.10, 0.99], 'w-', lw=1.2)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title('Band coverage (0=none, 5=all)')
    plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle('Canonical boundary from merged streamline density',
                  fontsize=12, y=1.0, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)

# ---------------- main -----------------
def run_analysis(edf_path, outdir):
    print(f"\n{'='*70}\nAnalysing: {edf_path.name}\n{'='*70}")
    sigs, fs, labels, XY = load_and_prep(edf_path)
    print(f"  {sigs.shape[0]} channels @ {fs:.0f} Hz, {sigs.shape[1]/fs:.0f}s")
    mask = ellipse_mask(GRID)
    chan_pix = np.array([data_to_pixel(x, y) for (x, y) in XY])
    print("  drawing cap...")
    figure_cap_layout(XY, labels, outdir / f'{edf_path.stem}_cap.png')
    band_data = {}; corr_rows = []
    print(f"\n{'band':<6} | {'K':>3} | {'raw sp':>7} | {'spTRIO':>7} | {'streamline':>10} | {'hybrid':>7}")
    print("-" * 60)
    for band in BANDS_ALL:
        print(f"  computing {band}...", flush=True)
        d = build_3pass_density(sigs, fs, labels, XY, band, mask, chan_pix)
        band_data[band] = d
        if d is None: continue
        cal = nf.calibrate(sigs, fs, labels, band=band); K = len(cal.channels)
        pn = d['p_norm']; topo = d['topo']
        r_sp = evaluate_spatial_r(pn, labels, cal.channels, topo)
        r_raw = evaluate_spatial_r(pn, labels, list(np.argsort(-pn)[:K].astype(int)), topo)
        r_st = evaluate_spatial_r(pn, labels, anchor_select(d['density'], chan_pix, K), topo)
        r_hy = evaluate_spatial_r(pn, labels, anchor_select(d['density'], chan_pix, K, p_norm=pn), topo)
        corr_rows.append((band, K, r_raw, r_sp, r_st, r_hy))
        print(f"{band:<6} | {K:>3} | {r_raw:+.3f} | {r_sp:+.3f} | {r_st:+.3f}     | {r_hy:+.3f}")
    print("\n  rendering reconstruction figure...")
    figure_reconstruction_per_band(band_data, XY, mask, outdir / f'{edf_path.stem}_reconstruction.png')
    print("  rendering streamline figure...")
    figure_streamlines_per_band(band_data, XY, mask, outdir / f'{edf_path.stem}_streamlines.png')
    print("  rendering canonical boundary...")
    figure_canonical(band_data, XY, mask, outdir / f'{edf_path.stem}_canonical.png')
    print("  rendering correlations...")
    figure_correlations(corr_rows, edf_path.name, outdir / f'{edf_path.stem}_correlations.png')
    return band_data, corr_rows

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    files = [Path(p) for p in sys.argv[1:]]
    for f in files:
        if not f.exists(): print(f"ERROR: {f} not found"); sys.exit(1)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(f'figures_{ts}'); outdir.mkdir(exist_ok=True)
    print(f"Output directory: {outdir}")
    for f in files: run_analysis(f, outdir)
    print(f"\nDone. All figures in {outdir.resolve()}")

if __name__ == '__main__':
    main()

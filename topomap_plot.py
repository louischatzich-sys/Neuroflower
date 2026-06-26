from __future__ import annotations
import numpy as np
from ..core.topomap import get_layout_xy

def plot_topomap(field, labels=None, ax=None, cmap='RdBu_r', show_labels=True,
                 title=None, vlim=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5,4.5))
    if vlim is None:
        m = float(np.max(np.abs(field))); vlim = (-m, m) if m>0 else (-1,1)
    span = 1.15
    ax.imshow(field, extent=(-span,span,-span,span), origin='lower',
              cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), 'k-', lw=1.5)
    ax.plot([-0.10, 0, 0.10],[0.99,1.10,0.99],'k-',lw=1.5)
    et = np.linspace(-np.pi/2, np.pi/2, 30)
    ax.plot(1+0.05*np.cos(et), 0.15*np.sin(et),'k-',lw=1.5)
    ax.plot(-1-0.05*np.cos(et), 0.15*np.sin(et),'k-',lw=1.5)
    if labels is not None and show_labels:
        coords, found = get_layout_xy(labels)
        for xy, ok, lbl in zip(coords, found, labels):
            if not ok: continue
            ax.plot(xy[0], xy[1], 'ko', ms=3)
            ax.annotate(lbl, xy=xy, xytext=(2,2), textcoords='offset points', fontsize=6)
    ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    if title: ax.set_title(title)
    return ax

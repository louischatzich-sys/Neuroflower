"""Topographic projection on the 10-10 layout via Gaussian RBF."""
from __future__ import annotations
import numpy as np

STANDARD_1010 = {
    "Nz":(0.00,1.00),"Fpz":(0.00,0.95),"AFz":(0.00,0.71),"Fz":(0.00,0.50),
    "FCz":(0.00,0.27),"Cz":(0.00,0.00),"CPz":(0.00,-0.27),"Pz":(0.00,-0.50),
    "POz":(0.00,-0.71),"Oz":(0.00,-0.95),"Iz":(0.00,-1.00),
    "Fp1":(-0.29,0.91),"Fp2":(0.29,0.91),
    "AF3":(-0.27,0.66),"AF4":(0.27,0.66),"AF7":(-0.55,0.78),"AF8":(0.55,0.78),
    "F1":(-0.18,0.50),"F2":(0.18,0.50),"F3":(-0.36,0.50),"F4":(0.36,0.50),
    "F5":(-0.55,0.50),"F6":(0.55,0.50),"F7":(-0.78,0.55),"F8":(0.78,0.55),
    "FC1":(-0.18,0.27),"FC2":(0.18,0.27),"FC3":(-0.36,0.27),"FC4":(0.36,0.27),
    "FC5":(-0.55,0.27),"FC6":(0.55,0.27),"FT7":(-0.91,0.27),"FT8":(0.91,0.27),
    "C1":(-0.18,0.00),"C2":(0.18,0.00),"C3":(-0.36,0.00),"C4":(0.36,0.00),
    "C5":(-0.55,0.00),"C6":(0.55,0.00),"T7":(-1.00,0.00),"T8":(1.00,0.00),
    "T3":(-1.00,0.00),"T4":(1.00,0.00),
    "CP1":(-0.18,-0.27),"CP2":(0.18,-0.27),"CP3":(-0.36,-0.27),"CP4":(0.36,-0.27),
    "CP5":(-0.55,-0.27),"CP6":(0.55,-0.27),"TP7":(-0.91,-0.27),"TP8":(0.91,-0.27),
    "P1":(-0.18,-0.50),"P2":(0.18,-0.50),"P3":(-0.36,-0.50),"P4":(0.36,-0.50),
    "P5":(-0.55,-0.50),"P6":(0.55,-0.50),"P7":(-0.78,-0.55),"P8":(0.78,-0.55),
    "T5":(-0.78,-0.55),"T6":(0.78,-0.55),
    "PO3":(-0.27,-0.66),"PO4":(0.27,-0.66),"PO7":(-0.55,-0.78),"PO8":(0.55,-0.78),
    "O1":(-0.29,-0.91),"O2":(0.29,-0.91),
    "M1":(-1.05,-0.10),"M2":(1.05,-0.10),"A1":(-1.05,-0.10),"A2":(1.05,-0.10),
}

def get_layout_xy(labels):
    coords = np.full((len(labels),2), np.nan)
    found = []
    for i,l in enumerate(labels):
        k = l.strip().rstrip(".").replace(" ","")
        hit = None
        for c in (k, k.upper(), k.capitalize(),
                  k.replace("FP","Fp"), k.replace("AF","AF")):
            if c in STANDARD_1010: hit = c; break
        if hit is None: found.append(False); continue
        coords[i] = STANDARD_1010[hit]; found.append(True)
    return coords, found

def project_to_topomap(values, labels, grid_size=140, rbf_sigma=0.18, mask_outside_head=True):
    coords, found = get_layout_xy(labels)
    keep = np.array(found) & np.isfinite(values)
    pts = coords[keep]; vals = values[keep]
    if pts.shape[0] == 0:
        return np.zeros((grid_size, grid_size))
    span = 1.15
    g = np.linspace(-span, span, grid_size)
    gx, gy = np.meshgrid(g, g)
    field = np.zeros_like(gx); weight = np.zeros_like(gx)
    s2 = 2.0 * rbf_sigma**2
    for (px, py), v in zip(pts, vals):
        w = np.exp(-((gx-px)**2 + (gy-py)**2) / s2)
        field += w*v; weight += w
    field = field / (weight + 1e-12)
    if mask_outside_head:
        rr = gx**2 + gy**2
        field = np.where(rr <= 1.05**2, field, 0.0)
    return field

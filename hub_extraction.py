from __future__ import annotations
import numpy as np

def extract_hubs(r): return np.asarray(r.get('pts43', np.zeros((0,2))), dtype=float)
def extract_pathways(r): return [np.asarray(c, dtype=float) for c in r.get('curve_list', [])]
def extract_intersection_point(r):
    ep = r.get('ext_final_point')
    if ep is not None: return float(ep[0]), float(ep[1])
    cp = r.get('center43')
    if cp is not None: return float(cp[0]), float(cp[1])
    return None
def summarise(r):
    h = extract_hubs(r)
    return dict(n_lobes=int(r.get('n_lobes',0)), hubs=h, n_hubs=int(h.shape[0]),
                pathways=extract_pathways(r), intersection_point=extract_intersection_point(r))

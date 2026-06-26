"""Pure-numpy EDF/EDF+ reader.

Header field order (per signal):
  label[16] transducer[80] physical_dim[8] physical_min[8] physical_max[8]
  digital_min[8] digital_max[8] prefilter[80] n_samples[8] reserved[32]
"""
from __future__ import annotations
import numpy as np

def read_edf(path):
    with open(path,'rb') as f: raw = f.read()
    def s(o,n): return raw[o:o+n].decode('ascii', errors='replace').strip()
    n_records = int(s(236,8)); record_dur = float(s(244,8))
    ns = int(s(252,4));        hdr_bytes = int(s(184,8))
    sig_hdr = raw[256:hdr_bytes]

    def block(off, sz, ns):
        return [sig_hdr[off + i*sz : off + (i+1)*sz].decode('ascii', errors='replace').strip()
                for i in range(ns)]

    labels   = block(0,                  16, ns)
    o = ns*(16+80+8)
    p_min    = [float(v) for v in block(o,       8, ns)]
    p_max    = [float(v) for v in block(o+ns*8,  8, ns)]
    d_min    = [int(float(v))   for v in block(o+ns*16, 8, ns)]
    d_max    = [int(float(v))   for v in block(o+ns*24, 8, ns)]
    o2 = ns*(16+80+8+8+8+8+8+80)
    n_samp   = [int(v) for v in block(o2, 8, ns)]
    spr = sum(n_samp)

    data_i16 = np.frombuffer(raw[hdr_bytes:], dtype=np.int16)
    data_3d = data_i16[:spr * n_records].reshape(n_records, spr)
    offs = np.cumsum([0] + n_samp)

    signals = {}; rates = []
    for i in range(ns):
        denom = (d_max[i] - d_min[i]) or 1
        scale = (p_max[i] - p_min[i]) / denom
        offp = p_min[i] - d_min[i] * scale
        sig = data_3d[:, offs[i]:offs[i+1]].flatten().astype(np.float64) * scale + offp
        signals[labels[i]] = sig
        rates.append(n_samp[i] / record_dur)
    return dict(labels=labels, signals=signals,
                sample_rate=rates[0] if rates else 0.0,
                sample_rates=rates, duration_s=n_records * record_dur)

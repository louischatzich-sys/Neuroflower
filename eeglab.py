"""Pure-numpy EEGLAB .set/.fdt reader."""
from __future__ import annotations
import struct
from pathlib import Path
import numpy as np

def _tag(d,o):
    f4 = struct.unpack_from('<I',d,o)[0]
    if f4>>16 != 0:
        return f4 & 0xFFFF, f4>>16, o+4, 8
    dt = f4; nb = struct.unpack_from('<I',d,o+4)[0]
    tot = 8+nb
    if tot%8 and dt!=0: tot += 8-(tot%8)
    return dt, nb, o+8, tot

def _read_num(d, dt, o, n):
    fmt = {1:'b',2:'B',3:'h',4:'H',5:'i',6:'I',7:'f',9:'d'}.get(dt)
    if fmt is None: return None
    return struct.unpack_from('<'+fmt*n, d, o)

def _matval(raw, start, end):
    p = start
    dt,nb,po,t = _tag(raw,p); ac = struct.unpack_from('<I',raw,po)[0]&0xFF; p += t
    dt,nb,po,t = _tag(raw,p); nd = nb//4
    dims = struct.unpack_from('<'+'i'*nd, raw, po); p += t
    dt,nb,po,t = _tag(raw,p); p += t  # name
    n_el = int(np.prod(dims)) if dims else 0
    val = None
    if n_el>0 and p<end:
        dtv,nbv,pov,tv = _tag(raw,p)
        if ac == 6:  # double
            v = _read_num(raw, dtv, pov, n_el)
            if v is not None: val = float(v[0]) if n_el==1 else np.array(v)
        elif ac == 4:  # char
            if dtv in (1,2):
                val = raw[pov:pov+nbv].decode('ascii',errors='replace').rstrip('\x00 ')
            elif dtv in (4,16):
                chars = struct.unpack_from('<'+'H'*(nbv//2), raw, pov)
                val = ''.join(chr(c) for c in chars).rstrip('\x00 ')
    return ac, dims, val

def _parse_set(raw):
    dt,nb,po,tot = _tag(raw,128)
    if dt != 14: raise ValueError("not miMATRIX")
    p = po
    _,_,_,t = _tag(raw,p); p += t  # flags
    _,_,_,t = _tag(raw,p); p += t  # dims
    _,_,_,t = _tag(raw,p); p += t  # name
    _,_,po2,t = _tag(raw,p); fn_len = struct.unpack_from('<i',raw,po2)[0]; p += t
    _,nb2,po2,t = _tag(raw,p)
    n_f = nb2 // fn_len
    fields = [raw[po2+i*fn_len:po2+(i+1)*fn_len].decode('ascii',errors='replace').rstrip('\x00')
              for i in range(n_f)]
    p += t
    vals = {}; cl_pos = None
    for fname in fields:
        dt,nb,po2,tv = _tag(raw,p)
        if dt == 14:
            ac, dims, val = _matval(raw, po2, po2+nb)
            vals[fname] = val
            if fname == 'chanlocs':
                cl_pos = (po2, nb)
        p += tv
    return vals, cl_pos

def _parse_chanlocs(raw, off, n_channels):
    p = off
    _,_,po,t = _tag(raw,p); cfn_len = struct.unpack_from('<i',raw,po)[0]; p += t
    _,nb,po,t = _tag(raw,p); n_cf = nb // cfn_len
    cfields = [raw[po+i*cfn_len:po+(i+1)*cfn_len].decode('ascii',errors='replace').rstrip('\x00')
               for i in range(n_cf)]
    p += t
    labels = []; xyz = np.full((n_channels,3), np.nan)
    for ch in range(n_channels):
        per = {}
        for fname in cfields:
            dt,nb,po2,t = _tag(raw,p)
            if dt == 14:
                _ac,_dims,v = _matval(raw, po2, po2+nb)
                per[fname] = v
            p += t
        labels.append(per.get('labels', f'ch{ch}'))
        for i,k in enumerate(('X','Y','Z')):
            v = per.get(k)
            if isinstance(v,(int,float)): xyz[ch,i] = float(v)
    return labels, xyz

def read_eeglab(set_path):
    p = Path(set_path); raw = p.read_bytes()
    meta, cl_loc = _parse_set(raw)
    nb_ch = int(meta.get('nbchan',0) or 0)
    pnts = int(meta.get('pnts',0) or 0)
    sr = float(meta.get('srate',0) or 0)
    labels, xyz = ([], None)
    if cl_loc is not None:
        labels, xyz = _parse_chanlocs(raw, cl_loc[0], nb_ch)
    datfile = meta.get('datfile')
    if not datfile:
        raise NotImplementedError("inline data not supported, need .fdt")
    fdt = p.parent / datfile
    if not fdt.exists():
        # Try same-stem .fdt as the .set (common when files are renamed)
        candidate = p.with_suffix('.fdt')
        if candidate.exists(): fdt = candidate
    sigs = np.fromfile(fdt, dtype=np.float32).reshape((nb_ch, pnts), order='F')
    return dict(labels=labels, signals=sigs, sample_rate=sr, positions_xyz=xyz)

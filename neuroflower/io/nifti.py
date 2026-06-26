"""Pure-numpy NIFTI-1 reader (.nii / .nii.gz)."""
from __future__ import annotations
import gzip, struct
from pathlib import Path
import numpy as np

_DT = {2:('u1',1),4:('i2',2),8:('i4',4),16:('f4',4),32:('c8',8),
       64:('f8',8),256:('i1',1),512:('u2',2),768:('u4',4)}

def read_nifti(path):
    p = Path(path)
    raw = gzip.open(p,'rb').read() if p.suffix == '.gz' else p.read_bytes()
    hdr = raw[:348]
    if struct.unpack_from('<i',hdr,0)[0] != 348:
        raise ValueError("not NIfTI-1")
    dim = struct.unpack_from('<8h',hdr,40); ndim=dim[0]; shape=dim[1:1+ndim]
    dt = struct.unpack_from('<h',hdr,70)[0]
    ds, bp = _DT[dt]
    pixdim = struct.unpack_from('<8f',hdr,76)
    vx,vy,vz = pixdim[1],pixdim[2],pixdim[3]; tr = pixdim[4]
    vox_off = float(struct.unpack_from('<f',hdr,108)[0])
    slope = float(struct.unpack_from('<f',hdr,112)[0])
    inter = float(struct.unpack_from('<f',hdr,116)[0])
    n_vox = int(np.prod(shape))
    data_off = int(vox_off) if vox_off>0 else 352
    flat = np.frombuffer(raw[data_off:data_off+n_vox*bp], dtype=np.dtype('<'+ds))
    data = flat.reshape(shape, order='F')
    if slope != 0.0 and not (slope==1.0 and inter==0.0):
        data = data.astype(np.float32)*slope + inter
    return dict(data=data, voxel_sizes=(vx,vy,vz), tr=tr, shape=tuple(shape))

import neuroflower as nf
from neuroflower.io import read_edf
import numpy as np

# Load an EDF file
e = read_edf('S002R01.edf')

# Convert to (n_channels, n_samples) matrix
labels = [l.replace('EEG ','').replace('.','').strip() for l in e['labels']]
n = max(len(s) for s in e['signals'].values())
sigs = np.zeros((len(labels), n))
for i, l in enumerate(e['labels']):
    sigs[i, :len(e['signals'][l])] = e['signals'][l]

# Run calibration for beta band
cal = nf.calibrate(sigs, fs=e['sample_rate'], labels=labels, band='beta')
print("Selected channels:", [cal.labels[i] for i in cal.channels])
print("Basin centroid:", cal.basin)
print("SRAM extrema:", cal.extrema_xy)

# Compare across conditions (rest vs task)
e2 = read_edf('S002R03.edf')
sigs2 = np.zeros((len(labels), max(len(s) for s in e2['signals'].values())))
for i, l in enumerate(e2['labels']):
    sigs2[i, :len(e2['signals'][l])] = e2['signals'][l]

cal_rest = cal  # from S002R01
cal_task = nf.calibrate(sigs2, fs=e2['sample_rate'], labels=labels, band='beta')

dev = nf.deviation_report(cal_rest, cal_task)
print("Deviation metrics:", dev)

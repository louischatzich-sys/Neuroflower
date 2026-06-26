"""End-to-end example: calibrate two segments and compare them.

Run from the repository root:
    python examples/calibrate_example.py /path/to/subject.edf

Output: prints a calibration summary, then a split-half deviation report.
Replace the EDF path with your own file; or modify to load .set files via
neuroflower.io.read_eeglab.
"""
import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))  # make package importable when running from examples/

import neuroflower as nf
from neuroflower.io import read_edf


def signals_matrix(edf):
    """Convert read_edf's {label: array} dict into (n_ch, n_samples)."""
    labels = edf['labels']
    n = max(len(s) for s in edf['signals'].values())
    M = np.zeros((len(labels), n))
    for i, l in enumerate(labels):
        M[i, :len(edf['signals'][l])] = edf['signals'][l]
    return M


def clean_label(s):
    return (s.replace('EEG ', '').replace('-Ref', '').replace('-REF', '')
             .strip().rstrip('.'))


def main(edf_path, band='alpha'):
    e = read_edf(edf_path)
    sigs = signals_matrix(e)
    labels = [clean_label(l) for l in e['labels']]
    fs = e['sample_rate']
    print(f"Loaded {sigs.shape[0]} channels @ {fs:.0f} Hz, "
          f"{sigs.shape[1]/fs:.1f} s")

    # --- single calibration ---------------------------------------------
    cal = nf.calibrate(sigs, fs, labels, band=band)
    print(f"\n=== {band} calibration ===")
    print(f"  channels kept after layout filter: "
          f"{cal.metadata['n_channels_kept']}/{cal.metadata['n_channels_input']}")
    print(f"  spTRIO selection ({len(cal.channels)} ch):")
    print('    ' + ', '.join(cal.labels[i] for i in cal.channels))
    print(f"  components:")
    for k, v in cal.components.items():
        print(f"    {k:>7}: {len(v)} channels")
    print(f"  basin centroid (head-data): {cal.basin}")

    # --- split-half deviation -------------------------------------------
    half = sigs.shape[1] // 2
    cal_A = nf.calibrate(sigs[:, :half], fs, labels, band=band)
    cal_B = nf.calibrate(sigs[:, half:], fs, labels, band=band)
    dev = nf.deviation_report(cal_A, cal_B)
    print(f"\n=== split-half deviation ({band}) ===")
    print(f"  basin shift:              {dev['basin_shift']:.3f}  "
          f"(head-radius units)")
    print(f"  extremum displacement:    {dev['extremum_displacement']:.3f}")
    print(f"  Jaccard distance:         {dev['jaccard_distance']:.3f}")
    print(f"  matched extremum pairs:   {dev['extremum_pairs']}")


if __name__ == '__main__':
    edf = sys.argv[1] if len(sys.argv) > 1 else None
    if edf is None:
        print("Usage: python calibrate_example.py /path/to/eeg.edf")
        sys.exit(1)
    band = sys.argv[2] if len(sys.argv) > 2 else 'alpha'
    main(edf, band=band)

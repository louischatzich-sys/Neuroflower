"""Minimal smoke tests for the v1.0 calibration pipeline.

Run with:  python -m pytest tests/ -v
(or just:  python tests/test_calibration.py)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import neuroflower as nf


def _synthetic_eeg(n_ch=32, fs=160, dur=10.0, seed=0):
    """Generate a synthetic 32-channel EEG with a posterior alpha source."""
    rng = np.random.default_rng(seed)
    n_t = int(fs * dur)
    t = np.arange(n_t) / fs
    base = rng.normal(0, 1.0, size=(n_ch, n_t))
    # Mock posterior alpha: stronger on O1, Oz, O2
    labels = ['Fp1','Fp2','Fpz','F3','F4','Fz','C3','C4','Cz','P3','P4','Pz',
              'O1','O2','Oz','T7','T8','F7','F8','P7','P8','Af3','Af4','Afz',
              'Po3','Po4','Poz','Cp3','Cp4','Cpz','Fc3','Fc4']
    alpha = np.sin(2 * np.pi * 10.0 * t)
    posterior = {'O1', 'O2', 'Oz', 'Pz', 'Po3', 'Po4', 'Poz', 'P3', 'P4'}
    for i, l in enumerate(labels):
        if l in posterior:
            base[i] += 3.0 * alpha
    return base, fs, labels


def test_calibrate_runs():
    sigs, fs, labels = _synthetic_eeg()
    cal = nf.calibrate(sigs, fs, labels, band='alpha')
    assert cal.band == 'alpha'
    assert len(cal.channels) > 0
    assert cal.basin is not None
    assert cal.metadata['version'] == '1.0.0'
    assert cal.metadata['locked_p'] == 2


def test_basin_is_posterior_for_alpha():
    sigs, fs, labels = _synthetic_eeg()
    cal = nf.calibrate(sigs, fs, labels, band='alpha')
    # Posterior alpha source means basin y-coordinate should be negative
    # (posterior = bottom of head-data coordinate frame).
    assert cal.basin[1] < 0, f"basin y={cal.basin[1]} should be posterior"


def test_deviation_within_same_recording():
    sigs, fs, labels = _synthetic_eeg()
    half = sigs.shape[1] // 2
    cal_A = nf.calibrate(sigs[:, :half], fs, labels, band='alpha')
    cal_B = nf.calibrate(sigs[:, half:], fs, labels, band='alpha')
    dev = nf.deviation_report(cal_A, cal_B)
    assert 0.0 <= dev['jaccard_distance'] <= 1.0
    assert dev['basin_shift'] >= 0.0
    assert dev['extremum_displacement'] >= 0.0


def test_metabolic_score_shape():
    sigs, fs, _ = _synthetic_eeg()
    score = nf.metabolic_score(sigs, fs, band='alpha', tau_s=2.0)
    assert score.shape == (sigs.shape[0],)
    assert (score > 0).all()


def test_metabolic_score_as_calibration_input():
    sigs, fs, labels = _synthetic_eeg()
    score = nf.metabolic_score(sigs, fs, band='alpha', tau_s=2.0)
    cal = nf.calibrate(sigs, fs, labels, band='alpha', score=score)
    assert cal.basin is not None


if __name__ == '__main__':
    test_calibrate_runs()
    test_basin_is_posterior_for_alpha()
    test_deviation_within_same_recording()
    test_metabolic_score_shape()
    test_metabolic_score_as_calibration_input()
    print('All 5 tests passed.')

# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/); this
project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-06-22

First public release of the spTRIO calibration pipeline based on the
Field-Seed Reconstruction Framework (Chatzicharalampous, 2026, DOI
[10.5281/ZENODO.20280107](https://doi.org/10.5281/ZENODO.20280107)).

### Added
- `pipelines/calibration.py`: top-level `calibrate(...)` entry point and
  `Calibration` dataclass returning the spTRIO selection, basin centroid,
  SRAM extrema in head-data coordinates, and the three view-component
  channel sets (field, inv, flow, MOD, spTRIO).
- `analysis/scaffold.py`: arc-length Bezier pathway field, Gaussian hub
  field, Helmholtz transport-flow field (FFT-Poisson solve), polynomial
  modulation, signed-picture median-centring.
- `analysis/mediator.py`: three-view union with cross-exclusion; produces
  the spTRIO (Spatial Three-view Relational) selection.
- `analysis/deviation.py`: three orthogonal cross-condition metrics —
  basin shift, topographic extremum displacement (cf. Lehmann & Skrandies
  1980), Jaccard distance (Jaccard 1912) — plus `deviation_report(...)`.
- `analysis/metabolic.py`: leaky-integrator neurovascular envelope
  M(t+1) = α·M(t) + |E(t)|², α = exp(−Δt/τ), τ = 5 s (Buxton et al. 1998).
- `examples/calibrate_example.py`: end-to-end demo on any EDF file.
- `tests/test_calibration.py`: 5 smoke tests covering single-segment
  calibration, deviation reporting, metabolic-score interchangeability,
  and basin-position plausibility on a synthetic posterior-α source.

### Locked parameters (v1.0)
- Polynomial modulation power `p = 2`.
- Mediator field/path weights 0.55 / 0.45.
- Hub count `k = 5` per polarity (positive and negative extrema).
- Hub minimum separation 18 px (matches inter-electrode spacing).
- RBF interpolation σ = 0.13 (~3 cm on a standard head).
- Grid 180 × 180 (~1 mm/pixel for 64-ch caps).

These were selected by pre-registered diagnostic sweeps on S002R01 before
any held-out validation; see proposal Appendices A–F and methods paper §4.2.

### Validated (PhysioNet EEGMMI corpus)
- Cross-band spatial fidelity, S002R01 rest: δ +0.994, θ +0.992, α +0.939,
  β +0.742, γ +0.952; temporal fidelity > +0.977 for all five bands.
- Cross-subject transfer, S002R01 → S098R01 (α): Δr = −0.066 vs. occipital
  control Δr = −0.123.
- Noise robustness: spatial r ≥ +0.83 at 0 dB SNR.
- Channel dropout: spatial r ≥ +0.88 after dropping 30 of 58 channels.
- Within-subject split-half (S002R01 α): basin shift 0.264, extremum
  displacement 0.334, Jaccard 0.522 (same-condition baseline).
- Motor-execution closing-the-loop (S001R03 β): sensorimotor-strip hub
  hits 0/10 at rest → 3/10 during task (C3, C5, C1).

### Documented limitations (pre-registered)
- β-band ceiling on resting and motor-imagery recordings (r ≈ +0.74).
  Diagnosed as data-bound rather than method-bound (five convergent null
  diagnostics) and architectural for β at rest (centrally mediated
  coordination rhythm rather than localised amplitude phenomenon).
  A coordination view is the planned v1.x extension.
- Cross-cohort transfer untested.
- Single-subject per claim; multi-subject template stability pending.

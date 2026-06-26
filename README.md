# neuroflower

**A research-stage Python toolkit for subject-specific spatial calibration in EEG.**

`neuroflower` v1.0.0 implements the spTRIO calibration pipeline: a multi-view
mediator over an SRAM-reconstructed signed picture that produces per-subject
spatial templates and cross-condition deviation metrics. It applies the
Field-Seed Reconstruction Framework (Chatzicharalampous, 2026) to EEG.

> **Status: research prototype, v1.0.0.** Not a medical device. Use only in
> IRB-approved research settings. Clinical translation is future work,
> gated by multi-subject validation.

[![DOI](https://img.shields.io/badge/SRAM%20framework%20DOI-10.5281%2FZENODO.20280107-blue)](https://doi.org/10.5281/ZENODO.20280107)

---

## What it does

Given a multi-channel EEG segment, `neuroflower` derives a *subject-specific*
spatial calibration template: a non-overlapping set of channels (the spTRIO
selection), a basin centroid, and SRAM extrema positions that together
capture where the brain's currently dominant spatial pattern lives.

Two such templates from the same subject under different conditions can be
compared with three orthogonal deviation metrics that triangulate the type
of change: gross spatial drift, local hub reorganisation, and channel-set
turnover.

## Install

```bash
pip install -e .                  # core (numpy only)
pip install -e ".[viz]"           # +matplotlib
pip install -e ".[dev]"           # +pytest +matplotlib
```

The core has a single hard dependency on NumPy.

## Quick start

```python
import neuroflower as nf
from neuroflower.io import read_edf

eeg = read_edf("subject.edf")
sigs = ...  # (n_channels, n_samples) numpy array; see examples/

# Single calibration
cal = nf.calibrate(sigs, fs=eeg["sample_rate"],
                    labels=eeg["labels"], band="alpha")

print("spTRIO channels:", [cal.labels[i] for i in cal.channels])
print("Basin centroid:", cal.basin)
print("SRAM extrema (head-data coords):", cal.extrema_xy)

# Cross-condition comparison
cal_rest = nf.calibrate(rest_sigs, fs, labels, band="beta")
cal_task = nf.calibrate(task_sigs, fs, labels, band="beta")
dev = nf.deviation_report(cal_rest, cal_task)
# {'basin_shift':            head-radius units,
#  'extremum_displacement':  head-radius units,
#  'jaccard_distance':       dimensionless in [0, 1]}
```

## Pipeline at a glance

```
EEG → per-channel score → topomap → SRAM reconstruction
                                          │
                              signed picture, extrema (hubs)
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  │                       │                       │
            field view              inverse view            transport view
       (0.55·hubs² + 0.45·paths²)   ((1 − picture)²)    (|∇φ|, source–sink Poisson)
                  │                       │                       │
                  └───────── mediator (cross-exclusion) ───────────┘
                                          │
                                  spTRIO selection
                                  + basin centroid
                                  + extrema positions
                                          │
                       cross-condition triangulation:
                       basin shift · extremum displacement · Jaccard distance
```

The six calibration stages and the three deviation metrics are described in
`methods_paper.md` and in the proposal (Chatzicharalampous, 2026, §Methods).

## Locked parameters (v1.0.0)

All parameters were determined by pre-registered diagnostic sweeps on
S002R01 *before* validation on held-out subjects, then locked. No per-band,
per-subject, or per-condition tuning is applied.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Polynomial modulation `p` | 2 | sweep over p ∈ {1,2,3,4} across α/β/γ |
| Hub / path weights | 0.55 / 0.45 | empirical, robust across bands |
| `k_hub` | 5 | hub-class extrema plateau at k=5 |
| `rbf_sigma` | 0.13 | ~3 cm Gaussian on standard head |
| Hub min separation | 18 px | matches inter-electrode spacing |
| Grid size | 180 × 180 | ~1 mm/pixel for 64-ch caps |

## Public API

```
neuroflower.calibrate(signals, fs, labels, band='alpha', score=None,
                       keep_fields=False)
       → Calibration

neuroflower.deviation_report(cal_A, cal_B)
       → {basin_shift, extremum_displacement, jaccard_distance,
          extremum_pairs}

neuroflower.metabolic_score(signals, fs, band, tau_s=5.0)
       → (n_channels,) leaky-integrator envelope (Buxton et al. 1998)

neuroflower.mediator(field, inverse, flow, p_norm)
       → {field, inv, flow, MOD, spTRIO} channel index sets

neuroflower.reconstruct_picture(source=topomap)
       → SRAM reconstruction dict
```

## Layout

```
neuroflower/
├── neuroflower/
│   ├── io/             EDF, BrainVision/EEGLAB (.set/.fdt), NIfTI readers
│   ├── core/           reconstruction.py (SRAM), topomap.py, psd.py
│   ├── analysis/       scaffold, mediator, deviation, metabolic,
│   │                   eeg_bands, eeg_fmri, hub_extraction
│   ├── viz/            topomap_plot
│   └── pipelines/      calibration.py (locked v1.0), eeg_fmri_coupling.py
├── examples/           calibrate_example.py
├── tests/              5 smoke tests covering core API
├── CHANGELOG.md        what's in v1.0.0
├── CITATION.cff        machine-readable citation
└── LICENSE             MIT
```

## Validation summary (S002R01 rest, 58 ch, 61 s)

| Frequency band | Spatial r | Temporal r |
|---|---|---|
| Delta (1–4 Hz) | +0.994 | +1.000 |
| Theta (4–8 Hz) | +0.992 | +0.999 |
| Alpha (8–13 Hz) | +0.939 | +0.989 |
| **Beta (13–30 Hz)** | **+0.742** | +0.989 |
| Gamma (30–45 Hz) | +0.952 | +0.977 |

The β-band ceiling is **diagnosed, not waved away**: β is centrally
mediated rather than locally organised (entropy = 0.986, central
electrodes show r = +0.66 with all other regions while peripheral
regions show r ≈ 0 with one another). A *coordination view* is the
planned v1.x extension. See `methods_paper.md` §4.5 and §4.7 for the
five pre-registered diagnostics, and the proposal §Findings.

## Clinical-pilot caveats

This is a research prototype, not a clinical tool. Before piloting:

1. **Pre-process EEG.** The pipeline assumes the input EEG is
   artefact-clean. For simultaneous EEG-fMRI, you must remove the MR
   gradient artefact (AAS; Allen et al. 2000) and the ballistocardiogram
   (BCG; Allen et al. 1998).
2. **Atlas-align ROIs.** The `roi_mask` in the secondary EEG-fMRI coupling
   pipeline defaults to a posterior placeholder. For real work, provide an
   anatomically defined mask (e.g. Harvard-Oxford cuneus / V1).
3. **Motion-correct fMRI.** The pipeline does not realign or smooth BOLD.
4. **Group statistics live outside the package.** `calibrate(...)` and
   `deviation_report(...)` produce per-subject results; combining across
   subjects requires the usual mixed-effects machinery
   (e.g. FSL FLAME, AFNI 3dLME).

## Citing

If you use this software, please cite both:

- **The SRAM reconstruction framework:** Chatzicharalampous, L. (2026).
  Field-Seed Reconstruction Framework and SRAM System: A Computational
  Model of Emergent Structure with Clinical Applications. Zenodo.
  https://doi.org/10.5281/ZENODO.20280107
- **The neuroflower v1.0 calibration toolkit:** see `CITATION.cff`.

## References

- Akalin Acar, Z. & Makeig, S. (2013). Effects of forward model errors on EEG source localization. *Brain Topography* 26: 378–396.
- Allen, P. J. et al. (1998). Identification of EEG events in the MR scanner: the problem of pulse artifact. *NeuroImage* 8: 229–239.
- Allen, P. J. et al. (2000). A method for removing imaging artefact from continuous EEG. *NeuroImage* 12: 230–239.
- Burle, B. et al. (2015). Spatial and temporal resolutions of EEG. *Int. J. Psychophysiology* 97: 210–220.
- Buxton, R. B., Wong, E. C. & Frank, L. R. (1998). The balloon model. *Magn. Reson. Med.* 39: 855–864.
- Chatzicharalampous, L. (2026). Field-Seed Reconstruction Framework and SRAM System. *Zenodo* https://doi.org/10.5281/ZENODO.20280107
- Jaccard, P. (1912). The distribution of the flora in the alpine zone. *New Phytologist* 11: 37–50.
- Khanna, A. et al. (2015). Microstates in resting-state EEG. *Neurosci. Biobehav. Rev.* 49: 105–113.
- Lehmann, D. & Skrandies, W. (1980). Reference-free identification of components of checkerboard-evoked multichannel potential fields. *EEG Clin. Neurophysiol.* 48: 609–621.
- Lotte, F. et al. (2018). A review of classification algorithms for EEG-based BCIs. *J. Neural Eng.* 15: 031005.
- Michel, C. M. & Brunet, D. (2019). EEG source imaging: a practical review. *Front. Neurol.* 10: 325.
- Onton, J. et al. (2006). Imaging human EEG dynamics using independent component analysis. *Neurosci. Biobehav. Rev.* 30: 808–822.
- Pfurtscheller, G. & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization. *Clin. Neurophysiol.* 110: 1842–1857.
- Schalk, G. et al. (2004). BCI2000. *IEEE Trans. Biomed. Eng.* 51: 1034–1043.

## License

MIT — see [`LICENSE`](LICENSE).

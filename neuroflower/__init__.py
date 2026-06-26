"""neuroflower — subject-specific EEG spatial calibration via multi-view
mediation over an SRAM-reconstructed signed picture.

Public API
----------
calibrate(signals, fs, labels, band='alpha') → Calibration
mediator(...)                                 — three-view union
deviation_report(cal_A, cal_B)                — basin, extremum, Jaccard
metabolic_score(...)                          — leaky-integrator envelope
reconstruct_picture(source=topomap)           — SRAM reconstruction

See README.md and the published methods paper for details.
"""
__version__ = "1.0.0"

# Top-level convenience imports
from .core.reconstruction import reconstruct_picture
from .pipelines.calibration import calibrate, Calibration
from .analysis.mediator import mediator
from .analysis.deviation import (
    deviation_report, basin_shift,
    topographic_extremum_displacement, jaccard_distance,
)
from .analysis.metabolic import metabolic_signal, metabolic_score

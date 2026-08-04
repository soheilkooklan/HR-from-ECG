"""
HR from ECG: Quality-Aware, Uncertainty-Calibrated heart-Rate and HRV estimation
from noisy single-lead ECG.

Version 2 of the HR-from-ECG project. Where version 1 reported a single
heart-rate number, this framework reports an interval with a distribution-free
coverage guarantee, together with a learned signal-quality score and an
explicit option to abstain.

Layers
------
hrecg.simulation : ground-truth ECG and artifact generation
hrecg.metrics    : detection, heart-rate agreement, HRV, uncertainty
hrecg.data       : PhysioNet loaders and windowing
hrecg.baselines  : classical reference detectors
"""

__version__ = "2.0.0-dev"

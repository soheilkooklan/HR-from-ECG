# HR from ECG

Quality-aware, uncertainty-calibrated heart-rate and HRV estimation from single-lead ECG.

**Version 2.** Version 1 loaded a CSV, called `scipy.find_peaks`, and printed a number — for any input, including a flat line. This version reports **every quantity as an interval with a distribution-free coverage guarantee**, learns a signal-quality score defined by downstream error rather than by waveform appearance, and **abstains** where the signal cannot support an answer. Parts of this project's UI and documentation were developed with AI assistance; see [NOTICE](https://github.com/soheilkooklan/HR-from-ECG/blob/5827c54f4302fe87e070e41cc0a0bee0f4587f7c/NOTICE.md) for details.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21788946.svg)](https://doi.org/10.5281/zenodo.21788946)
[![License: PolyForm NC](https://img.shields.io/badge/License-PolyForm%20Noncommercial-orange.svg)](LICENSE)
[![Research use only](https://img.shields.io/badge/use-research%20only%20%C2%B7%20not%20a%20medical%20device-red.svg)](NOTICE.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-38%20passing-brightgreen.svg)](tests/)

![analyzer](docs/screenshots/01_noisy_sinus.png)

---

## The problem this addresses

R-peak detection on MIT-BIH is a solved problem: published detectors report F1 around 0.998, and the remaining headroom is smaller than the annotation scatter of the reference database itself. Building a marginally better detector is not a contribution.

The infrastructure study in this repository reframes what is still open. Under increasing noise, the classical detector keeps sensitivity above 0.96 down to −6 dB while precision collapses to 0.71 — **it does not stop finding beats, it starts inventing them** — and heart-rate error crosses the ANSI/AAMI EC13 tolerance of 5 bpm at around 3 dB SNR, well inside the operating range of any wearable.

So the open problem is not *detect better*. It is **know when you cannot detect**, and say so.

## Contributions

| | |
|---|---|
| **C1 · uSQI** | Signal quality defined as `exp(-abs(HR_hat - HR_true) / tau)` with tau = 5 bpm — by how much a segment degrades the estimate, not by how the waveform looks. Labels are generated automatically by controlled corruption of records with known truth. |
| **C2 · HR-from-ECG v2** | CNN U-Net with a bidirectional selective state-space bottleneck (linear in sequence length), a sub-sample offset head, and a differentiable pseudo-periodicity prior acting on the beat probability map. 2.28 M parameters. |
| **C3 · Mondrian conformal intervals** | Heart-rate intervals with coverage guaranteed *within each predicted-quality stratum*, plus conformalised Monte-Carlo propagation of beat-level uncertainty into RMSSD, SDNN, pNN50 and the spectral indices. |
| **C4 · Benchmark and toolbox** | ECG and artifact simulator, cross-tolerance and noise-stratified evaluation on MIT-BIH with real NSTDB noise, 38 tests, and a desktop application. |

Every design decision is defended, with evidence, in **[docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md)**.

---

## Results

All numbers below are produced by `scripts/evaluate_mitdb.py` on 20 records of the MIT-BIH Arrhythmia Database (paced records 102, 104, 107, 217 excluded per AAMI EC57), with real noise from the MIT-BIH Noise Stress Test Database. Calibration and test sets are **split by record, never by window**.

### Detection — and why the usual metric hides the difference

The model was trained **exclusively on simulated ECG** and evaluated on MIT-BIH without any fine-tuning.

| | HR-from-ECG v2 | Pan-Tompkins |
|---|---|---|
| F1 @ ±150 ms | 0.974 | 0.972 |
| F1 @ ±50 ms | **0.974** | 0.921 |
| F1 @ ±25 ms | **0.924** | 0.895 |
| Sensitivity | **0.986** | 0.942 |
| PPV | **0.967** | 0.908 |
| Localisation jitter | **3.06 ms** | 4.75 ms |

![detection](docs/figures/fig1_detection.png)

At the conventional ±150 ms tolerance the two methods are indistinguishable (0.974 vs 0.972). The difference only appears under strict tolerance and in timing jitter — which is precisely why this project reports F1 and jitter as separate columns. A detector can reach F1 = 0.998 and still be useless for HRV, because 5 ms of jitter is a substantial fraction of a 25 ms RMSSD.

That a model trained only on a physics simulator transfers to real patient recordings without adaptation is, in itself, the most surprising result here.

### Noise stress with real NSTDB artifact

![noise](docs/figures/fig2_noise_stress.png)

| artifact | SNR | HR-from-ECG v2 F1 | PT F1 | HR-from-ECG v2 HR MAE | PT HR MAE |
|---|---|---|---|---|---|
| baseline wander | −6 dB | **0.929** | 0.836 | **4.66** | 9.38 bpm |
| electrode motion | −6 dB | **0.711** | 0.552 | **26.2** | 85.0 bpm |
| electrode motion | 0 dB | **0.856** | 0.732 | **14.8** | 36.4 bpm |
| electrode motion | +6 dB | **0.932** | 0.826 | **6.42** | 12.6 bpm |
| muscle artifact | −6 dB | **0.745** | 0.612 | **16.8** | 60.9 bpm |
| muscle artifact | +6 dB | **0.932** | 0.840 | **6.55** | 10.1 bpm |

Heart-rate error under electrode motion at −6 dB falls from 85.0 to 26.2 bpm — a 69 % reduction. But 26 bpm is still a clinically meaningless number, which is exactly the point: **no detector should report a heart rate in this regime**, and that is what the abstention layer is for.

### Conformal intervals — the central result

![conformal](docs/figures/fig3_conformal.png)

| | split conformal | Mondrian (quality-conditional) |
|---|---|---|
| Marginal coverage (target 0.90) | 0.907 ✓ | 0.930 ✓ |
| **Worst-stratum coverage** | **0.564** ✗ | **0.858** |
| Mean interval width | 28.8 bpm | 34.7 bpm |
| Winkler score (lower better) | 50.6 | **40.4** |

Split conformal is marginally valid and **still fails catastrophically where it matters**: in the worst quality stratum it covers 56 % of the time against a 90 % target. Conditioning on predicted quality raises that to 86 % — a 52 % relative improvement — and improves the Winkler score, so the gain is not bought by simply widening the interval everywhere.

This table validates C1 and C3 simultaneously. If uSQI carried no information about error, the strata would share a quantile and Mondrian would collapse back to split conformal. It does not.

### Selective prediction — what abstention buys

![selective](docs/figures/fig4_selective.png)

| ranking score | AURC | MAE, all windows | 90 % kept | 75 % kept |
|---|---|---|---|---|
| **uSQI (learned)** | **1.46** | 9.27 bpm | **5.53** | **1.75 bpm** |
| kSQI (classical) | 3.33 | 9.27 bpm | 7.00 | 5.17 bpm |

Withholding the worst 25 % of windows takes heart-rate error from 9.27 to **1.75 bpm** when ranked by uSQI, against 5.17 bpm when ranked by the classical kurtosis index — nearly a threefold difference in what the same abstention budget can achieve. The risk controller, asked for a 3 bpm target, keeps **80 % of windows at 2.41 bpm**.

### HRV intervals

Conformalised Monte-Carlo intervals on five-minute segments, nominal 90 %:

| index | empirical coverage | mean width |
|---|---|---|
| mean HR | 0.90 | 1.90 bpm |
| SDNN | 0.80 | 59.7 ms |
| RMSSD | 0.80 | 35.4 ms |
| pNN50 | 0.90 | 19.5 % |

Coverage is at or slightly below nominal on a small test set (n = 10 segments); the intervals are honest but the estimate of their coverage is not yet tight. This is the least mature result in the repository and is flagged as such.

### Why a learned quality index was necessary

From `scripts/validate_infrastructure.py`, rank correlation of each index with true heart-rate error:

| index | rho with HR error | rho with SNR |
|---|---|---|
| true SNR (oracle) | 0.632 | 1.000 |
| sSQI | 0.586 | 0.917 |
| kSQI | 0.574 | 0.883 |
| baSQI | 0.470 | 0.840 |
| CQI (cepstral) | 0.207 | 0.176 |
| pSQI | 0.133 | 0.424 |

Classical indices track SNR almost perfectly and heart-rate error only moderately. Even an oracle knowing the true SNR reaches only rho = 0.63. Signal quality in the conventional sense is simply a different quantity from usefulness for heart-rate estimation.

---

## The application

```bash
python -m app.gui
```

| clean signal | atrial fibrillation |
|---|---|
| ![clean](docs/screenshots/05_clean_high_confidence.png) | ![af](docs/screenshots/02_atrial_fibrillation.png) |

| frequent PVCs | electrode failure — the system abstains |
|---|---|
| ![pvc](docs/screenshots/03_frequent_pvcs.png) | ![fail](docs/screenshots/04_electrode_failure.png) |

The electrode-failure case is the one to look at. During the dropout the quality trace collapses, the affected span is shaded, the heart-rate trace breaks rather than interpolating across the gap, and the report states that 20 % of windows were withheld. Removing the unusable segment also *narrows* the RMSSD interval, from [16.5, 51.7] ms when the corrupt beats are included to [32.0, 36.3] ms when they are not — abstention buys precision, it does not merely express doubt.

The application runs from a fresh clone with only NumPy and SciPy: without PyTorch or a checkpoint it falls back to the classical engine and says so, rather than failing to start.

---

## Install and run

```bash
git clone https://github.com/soheilkooklan/HR-from-ECG.git
cd HR-from-ECG
pip install -e ".[data,dev]"

pytest -q                                    # 38 tests
python scripts/validate_infrastructure.py    # simulator + metric validation
python -m app.gui                            # the analyzer
```

Reproducing the benchmark requires the two PhysioNet databases:

```bash
# https://physionet.org/content/mitdb/1.0.0/  and  .../nstdb/1.0.0/
python scripts/train.py --steps 20000 --batch 64 --device cuda
python scripts/evaluate_mitdb.py --data path/to/mitdb --nstdb path/to/nstdb
python scripts/make_figures.py
```

A CUDA GPU is not required to run anything, and PyTorch on Windows with CUDA works fine — no Linux needed. The model in this repository was trained on a single CPU core in about 20 minutes; a longer GPU run will do better.

## Layout

```
hrecg/
  simulation/    rhythm.py     RR generation: HRV spectrum, AF, PVC/APB, bigeminy
                 ecgsyn.py     McSharry phase-domain synthesis with rate-dependent
                               QRS width and Bazett QT scaling
                 noise.py      baseline wander, EMG, electrode motion, mains,
                               lead-off; SNR-calibrated time-varying corruption
  models/        ssm.py        selective SSM via associative scan, no CUDA kernel
                 net.py        HR-from-ECG v2: beat / offset / quality / HR-quantile heads
                 losses.py     focal + soft-Dice + rhythm prior + pinball
                 decode.py     sub-sample decoding, overlap-add inference
  conformal/     hr_interval.py     split and Mondrian conformal, risk controller
                 hrv_uncertainty.py conformalised Monte-Carlo HRV intervals
  metrics/       detection, hr, hrv, quality, uncertainty
  data/          physionet.py  wfdb loaders (online or offline)
                 windows.py    label construction, on-the-fly synthetic dataset
  baselines/     pan_tompkins.py
  pipeline.py    end-to-end analysis
app/gui.py       desktop analyzer
scripts/         train · evaluate_mitdb · validate_infrastructure · make_figures ·
                 make_screenshots
legacy/          version 1, archived
```

## Implementation notes worth knowing

- **No `mamba-ssm` dependency.** It needs a CUDA-compiled kernel and will not run on CPU. The recurrence is implemented with a Hillis–Steele associative scan in log2(L) parallel steps; agreement with the sequential recurrence is 5e-6, verified in the test suite.
- **The rhythm prior acts on the probability map, not on RR intervals**, because peak-picking has no gradient. It rewards a sharp autocorrelation peak in the physiological lag band — differentiable, label-free, and strongest exactly where supervision is weakest.
- **Lomb–Scargle, not interpolate-then-Welch**, for spectral HRV. Interpolation injects HF power and biases LF/HF precisely when beats are missing.
- **Stratification and gating use different statistics of the same quality trace.** Interval width is conditioned on mean window quality; the decision to report at all uses the 10th percentile, because a two-second dropout destroys the RR sequence however good the other eight seconds were.

## Known limitations

Stated plainly, and expanded in [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) §11:

- No validation on genuinely ambulatory wearable recordings (DaLiA, WESAD, Simband).
- uSQI is defined relative to a reference detector, so it inherits that detector's blind spots.
- Mondrian conformal gives stratum-conditional, not fully conditional, coverage.
- The Malik ectopic-correction rule misbehaves in AF, flagging most of the tachogram; use `correction="none"` for AF until rhythm-adaptive selection is implemented.
- Single-lead only.
- The shipped checkpoint is a 700-step CPU training run — a demonstration, not a tuned model.

## Citation

If you use this software, please cite it via [`CITATION.cff`](CITATION.cff), or the archived Zenodo record once released.

## Acknowledgements

Data: MIT-BIH Arrhythmia Database and MIT-BIH Noise Stress Test Database, PhysioNet (Moody & Mark 2001; Goldberger et al. 2000).

## Author

**Soheil Kooklan** — MSc, Biomedical Engineering (Bioelectric), Science and Research Branch, Islamic Azad University (SRBIAU), Tehran, Iran  
ORCID [0009-0003-5035-7833](https://orcid.org/0009-0003-5035-7833)

## License

**PolyForm Noncommercial 1.0.0** — free for research, teaching, personal projects
and any other noncommercial purpose. Commercial use, including incorporation into
a product or medical device, resale, or use in a service offered for a fee,
requires a separate written licence from the author.

See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

**Not a medical device.** This software is a research tool. It is not certified,

# References

Every method, model and dataset this project builds on, organised by the part
of the codebase that uses it. This list exists so that a claim in the code or
in `DESIGN_NOTES.md` can always be traced back to where it came from.

Entries are grouped to match the package layout (`hrecg/simulation`,
`hrecg/models`, `hrecg/conformal`, `hrecg/metrics`, data). Within each group,
entries are in the order they are first relevant.

---

## Datasets

**MIT-BIH Arrhythmia Database.**
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
*IEEE Engineering in Medicine and Biology Magazine*. 2001;20(3):45–50.
DOI: [10.1109/51.932724](https://doi.org/10.1109/51.932724)
Data: [physionet.org/content/mitdb/1.0.0](https://physionet.org/content/mitdb/1.0.0/)

**MIT-BIH Noise Stress Test Database.**
Moody GB, Muldrow WE, Mark RG. A noise stress test for arrhythmia detectors.
*Computers in Cardiology*. 1984;11:381–384.
Data: [physionet.org/content/nstdb/1.0.0](https://physionet.org/content/nstdb/1.0.0/)

**PhysioNet / PhysioBank / PhysioToolkit** (the hosting platform and `wfdb`
tooling used to read both databases above).
Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE,
Moody GB, Peng CK, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet:
components of a new research resource for complex physiologic signals.
*Circulation*. 2000;101(23):e215–e220.
DOI: [10.1161/01.CIR.101.23.e215](https://doi.org/10.1161/01.CIR.101.23.e215)

---

## `hrecg/simulation` — ECG and artefact synthesis

**Phase-domain dynamical ECG model** (`ecgsyn.py`, `rhythm.py`). The basis of
the waveform synthesiser; the rate-dependent QRS width and Bazett QT scaling
in this repository are extensions of the original model.
McSharry PE, Clifford GD, Tarassenko L, Smith LA. A dynamical model for
generating synthetic electrocardiogram signals.
*IEEE Transactions on Biomedical Engineering*. 2003;50(3):289–294.
DOI: [10.1109/TBME.2003.808805](https://doi.org/10.1109/TBME.2003.808805)

**Bazett's QT correction** (used in `ecgsyn.py` to scale T-wave position with
the square root of RR).
Bazett HC. An analysis of the time-relations of electrocardiograms.
*Heart*. 1920;7:353–370. (Reprinted with commentary in
*Annals of Noninvasive Electrocardiology*. 1997;2(2):177–194,
DOI: [10.1111/j.1542-474X.1997.tb00325.x](https://doi.org/10.1111/j.1542-474X.1997.tb00325.x))

**Noise Stress Test methodology** (`noise.py`, SNR-calibrated corruption).
Moody, Muldrow & Mark 1984 — see Datasets above.

---

## `hrecg/models` — network architecture and training

**Selective state space models / Mamba** (`ssm.py`, the bidirectional
bottleneck of the network). This repository re-implements the recurrence with
a portable associative scan rather than depending on the authors' CUDA kernel;
see `DESIGN_NOTES.md` §3 for why.
Gu A, Dao T. Mamba: linear-time sequence modeling with selective state spaces.
Presented at COLM 2024.
arXiv: [2312.00752](https://arxiv.org/abs/2312.00752)

**Structured state space models (S4/S5), the line of work Mamba extends**
(background for `ssm.py`).
Smith JTH, Warrington A, Linderman SW. Simplified state space layers for
sequence modeling. *International Conference on Learning Representations
(ICLR)* 2023.
arXiv: [2208.04933](https://arxiv.org/abs/2208.04933)

**Parallel prefix-sum / associative scan algorithm** (`associative_scan` in
`ssm.py`, the Hillis–Steele formulation used to parallelise the recurrence).
Blelloch GE. Prefix sums and their applications. Technical Report
CMU-CS-90-190, School of Computer Science, Carnegie Mellon University, 1990.
[Report PDF](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)

---

## `hrecg/baselines` — the classical reference detector

**Pan–Tompkins QRS detector** (`pan_tompkins.py`), re-implemented in full
rather than imported, so the baseline comparison does not depend on an
unspecified third-party variant.
Pan J, Tompkins WJ. A real-time QRS detection algorithm.
*IEEE Transactions on Biomedical Engineering*. 1985;32(3):230–236.
DOI: [10.1109/TBME.1985.325532](https://doi.org/10.1109/TBME.1985.325532)

---

## `hrecg/conformal` — distribution-free uncertainty

**Conformal prediction, foundational theory** (`hr_interval.py`, the
finite-sample quantile correction and the split/Mondrian constructions).
Vovk V, Gammerman A, Shafer G. *Algorithmic Learning in a Random World*.
Springer, 2005. DOI: [10.1007/b106715](https://doi.org/10.1007/b106715)

**Distribution-free predictive inference / split conformal regression**
(the split-conformal baseline `ConformalHR` is compared against).
Lei J, G'Sell M, Rinaldo A, Tibshirani RJ, Wasserman L. Distribution-free
predictive inference for regression.
*Journal of the American Statistical Association*. 2018;113(523):1094–1111.
DOI: [10.1080/01621459.2017.1307116](https://doi.org/10.1080/01621459.2017.1307116)

**Conformalised quantile regression** (motivates using the model's own
predicted quantiles as the nonconformity score, rather than a raw residual).
Romano Y, Patterson E, Candès E. Conformalized quantile regression.
*Advances in Neural Information Processing Systems (NeurIPS)* 2019;32.
arXiv: [1905.03222](https://arxiv.org/abs/1905.03222)

**A tutorial introduction to conformal prediction** (general reference for
the exposition in `DESIGN_NOTES.md`).
Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and
distribution-free uncertainty quantification. 2021.
arXiv: [2107.07511](https://arxiv.org/abs/2107.07511)

**Conformal risk control** (theoretical basis for `SelectiveController`, the
abstention mechanism with a guaranteed risk bound).
Angelopoulos AN, Bates S, Candès EJ, Jordan MI, Lei L. Conformal risk control.
2021. arXiv: [2110.01052](https://arxiv.org/abs/2110.01052)

**Selective prediction / the reject option, and the risk–coverage curve**
(`risk_coverage_curve`, `aurc` in `metrics/uncertainty.py`).
El-Yaniv R, Wiener Y. On the foundations of noise-free selective
classification. *Journal of Machine Learning Research*. 2010;11:1605–1641.
[Paper PDF](https://www.jmlr.org/papers/volume11/el-yaniv10a/el-yaniv10a.pdf)

**The Winkler interval score** (`winkler_score` in `metrics/uncertainty.py`,
the proper scoring rule used to compare interval quality without being gamed
by interval width alone).
Winkler RL. A decision-theoretic approach to interval estimation.
*Journal of the American Statistical Association*. 1972;67(337):187–191.
DOI: [10.1080/01621459.1972.10481224](https://doi.org/10.1080/01621459.1972.10481224)

---

## `hrecg/metrics` — heart rate, HRV and signal quality

**Heart rate variability standards** (`hrv.py`, the time- and frequency-domain
indices and the LF/HF band definitions).
Task Force of the European Society of Cardiology and the North American
Society of Pacing and Electrophysiology. Heart rate variability: standards of
measurement, physiological interpretation, and clinical use.
*Circulation*. 1996;93(5):1043–1065.
DOI: [10.1161/01.CIR.93.5.1043](https://doi.org/10.1161/01.CIR.93.5.1043)

**Lomb–Scargle periodogram** (`frequency_domain` in `hrv.py`, spectral
analysis of the unevenly-sampled RR series without resampling).
Lomb NR. Least-squares frequency analysis of unequally spaced data.
*Astrophysics and Space Science*. 1976;39(2):447–462.
DOI: [10.1007/BF00648343](https://doi.org/10.1007/BF00648343)

Scargle JD. Studies in astronomical time series analysis. II. Statistical
aspects of spectral analysis of unevenly spaced data.
*The Astrophysical Journal*. 1982;263:835–853.
DOI: [10.1086/160554](https://doi.org/10.1086/160554)

**Poincaré plot analysis** (`poincare` in `hrv.py`, the SD1/SD2 descriptors
used to characterise atrial fibrillation).
Brennan M, Palaniswami M, Kamen P. Do existing measures of Poincaré plot
geometry reflect nonlinear features of heart rate variability?
*IEEE Transactions on Biomedical Engineering*. 2001;48(11):1342–1347.
DOI: [10.1109/10.959330](https://doi.org/10.1109/10.959330)

**Bland–Altman method comparison** (`bland_altman` in `metrics/hr.py`).
Bland JM, Altman DG. Statistical methods for assessing agreement between two
methods of clinical measurement. *The Lancet*. 1986;327(8476):307–310.
DOI: [10.1016/S0140-6736(86)90837-8](https://doi.org/10.1016/S0140-6736%2886%2990837-8)

**Intraclass correlation coefficient** (`icc21` in `metrics/hr.py`, the
ICC(2,1) two-way random-effects form).
Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater
reliability. *Psychological Bulletin*. 1979;86(2):420–428.
DOI: [10.1037/0033-2909.86.2.420](https://doi.org/10.1037/0033-2909.86.2.420)

**Signal quality indices** (`quality.py`, the classical kSQI, sSQI, pSQI,
baSQI baseline indices that `uSQI` is compared against).
Li Q, Mark RG, Clifford GD. Robust heart rate estimation from multiple
asynchronous noisy sources using signal quality indices and a Kalman filter.
*Physiological Measurement*. 2008;29(1):15–32.
DOI: [10.1088/0967-3334/29/1/002](https://doi.org/10.1088/0967-3334/29/1/002)

Behar J, Oster J, Li Q, Clifford GD. ECG signal quality during arrhythmia and
its application to false alarm reduction.
*IEEE Transactions on Biomedical Engineering*. 2013;60(6):1660–1666.
DOI: [10.1109/TBME.2013.2240452](https://doi.org/10.1109/TBME.2013.2240452)

**AAMI EC13 / ANSI heart-rate accuracy standard** (the 10%-or-5-bpm tolerance
used throughout `metrics/hr.py` and the evaluation scripts).
Association for the Advancement of Medical Instrumentation.
ANSI/AAMI EC13:2002 — Cardiac monitors, heart rate meters, and alarms.
Arlington, VA: AAMI, 2002.

**AAMI EC57 / recommended practice for arrhythmia-detector testing** (basis
for excluding paced records and for the record-disjoint calibration/test
split used in `scripts/evaluate_mitdb.py`).
Association for the Advancement of Medical Instrumentation.
ANSI/AAMI EC57:2012 — Testing and reporting performance results of cardiac
rhythm and ST-segment measurement algorithms. Arlington, VA: AAMI, 2012.

---

## Software this project depends on

NumPy — Harris CR, et al. Array programming with NumPy. *Nature*.
2020;585:357–362. DOI: [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)

SciPy — Virtanen P, et al. SciPy 1.0: fundamental algorithms for scientific
computing in Python. *Nature Methods*. 2020;17:261–272.
DOI: [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)

PyTorch — Paszke A, et al. PyTorch: an imperative style, high-performance deep
learning library. *NeurIPS* 2019;32. arXiv: [1912.01703](https://arxiv.org/abs/1912.01703)

`wfdb-python` — Xie et al. Waveform Database Software Package (WFDB) for
Python. [github.com/MIT-LCP/wfdb-python](https://github.com/MIT-LCP/wfdb-python)

Matplotlib — Hunter JD. Matplotlib: a 2D graphics environment.
*Computing in Science & Engineering*. 2007;9(3):90–95.
DOI: [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)

cleared or approved by any regulatory authority, and must not be used to diagnose,
treat, or make clinical decisions about any person.

Version 1 remains available under its original MIT licence in
[`legacy/`](legacy/); that grant cannot be withdrawn retroactively and is
unaffected by the licence change for version 2.

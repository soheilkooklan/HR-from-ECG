# Design notes

Why the project is built the way it is. Every entry here corresponds to a
decision that could reasonably have gone the other way, and each records the
evidence or reasoning that settled it. If you are reviewing this repository,
this file is the argument; the code is the implementation of it.

---

## 1. Why not simply build a more accurate R-peak detector

Because that problem is closed. Recent published detectors report F1 around
0.998 on MIT-BIH and above 0.99 cross-database. Between the annotation scatter
of the reference database itself and the ceiling imposed by ectopic and paced
morphology, the remaining headroom is smaller than the noise in the evaluation.
An improvement of 0.05 % would be neither measurable nor meaningful.

The infrastructure study (`scripts/validate_infrastructure.py`) also showed
something that reframes the problem. Under increasing noise, Pan-Tompkins keeps
sensitivity above 0.96 down to −6 dB while precision collapses to 0.71. The
detector does not stop finding beats; it starts inventing them. And crucially,
heart-rate error crosses the ANSI/AAMI EC13 tolerance of 5 bpm somewhere around
3 dB — well inside the operating range of any wearable. So the open problem is
not *detect better*, it is *know when you cannot detect*.

## 2. Why quality is defined by downstream error rather than by appearance

Every established signal-quality index — kSQI, sSQI, pSQI, baSQI, the cepstral
CQI, bSQI — is validated against human "acceptable / unacceptable" labels or
against SNR. The measured consequence, from `results/validation.json`:

| index | Spearman ρ with true HR error | ρ with SNR |
|---|---|---|
| true SNR (oracle) | 0.632 | 1.000 |
| sSQI | 0.586 | 0.917 |
| kSQI | 0.574 | 0.883 |
| baSQI | 0.470 | 0.840 |
| CQI | 0.207 | 0.176 |
| pSQI | 0.133 | 0.424 |

The classical indices track SNR extremely well and heart-rate error only
moderately. More striking, an oracle with perfect knowledge of the true SNR
reaches only ρ = 0.63. Signal quality in the conventional sense is simply a
different quantity from usefulness for heart-rate estimation — a segment can
look terrible while its R peaks survive intact, and can look clean while
electrode motion has inserted a plausible false beat.

Hence uSQI:

```
uSQI(w) = exp( -|HR_hat(w) - HR_true(w)| / tau ),   tau = 5 bpm
```

with tau matched to the AAMI tolerance. The label is produced automatically by
controlled corruption of records with known truth, so no manual annotation is
needed and the definition is unambiguous.

**The objection to expect.** "uSQI depends on which reference estimator you
use, so it is not a property of the signal." Partly fair. The mitigation is
that the reference is a standard, well-characterised detector, and the quantity
being measured is how much a segment degrades *that* estimator — which is what
transfers. A stronger version would average over an ensemble of reference
detectors; that is a natural extension and is noted in the roadmap.

## 3. Why a selective state-space bottleneck rather than a Transformer

Attention is quadratic in sequence length. A 24-hour Holter record at 250 Hz is
2.16 × 10⁷ samples; even at the U-Net bottleneck this is far beyond what
quadratic attention can hold. A selective SSM is linear in length and keeps a
learned, input-dependent decay, which is exactly the mechanism needed to *not*
integrate an artefact burst into the running state.

**Why not the `mamba-ssm` package.** It requires a CUDA-compiled kernel, does
not run on CPU at all, and pins narrow PyTorch versions. Anyone trying to
reproduce or audit this work on a laptop would be blocked. The recurrence is
therefore implemented directly with a Hillis–Steele associative scan, in
log₂(L) fully parallel steps. `tests/test_model_and_conformal.py` checks it
against the sequential recurrence; agreement is to 5 × 10⁻⁶. The cost is
roughly a factor of two in throughput against the fused kernel, which is a
reasonable price for a project whose subject is trustworthiness.

## 4. Why the rhythm prior acts on the probability map, not on RR intervals

The obvious formulation — penalise physiologically impossible RR sequences —
is not differentiable, because RR intervals are computed from discrete peak
positions and peak-picking has no useful gradient.

The constraint is therefore imposed one level earlier, on the continuous beat
probability map. A beat train is pseudo-periodic, so the normalised
autocorrelation of the probability map must have a sharp peak at the lag
corresponding to the mean RR interval:

```
L_rhythm = 1 - softmax_lag  rho(lag),   lag in [60/220, 60/30] seconds
```

This is fully differentiable, needs no labels (so it doubles as a
semi-supervised term on unannotated records), and bites hardest exactly where
the supervised term is weakest. Under heavy artefact, false detections are
scattered randomly in time and destroy periodicity, whereas true beats preserve
it — so the prior penalises precisely the failure mode identified in §1.

The soft maximum (log-sum-exp) rather than a hard max is deliberate: early in
training the argmax lag is essentially arbitrary, and a hard max would send
gradient to only one lag.

## 5. Why there is a sub-sample offset head

Localisation jitter measured on the baseline is 3.1 ms at 6 dB SNR, and the
sampling grid alone contributes ±2 ms at 250 Hz. RMSSD in a healthy adult is
often 25–40 ms, so detector jitter is not negligible relative to the quantity
being measured. A segmentation map quantised to the sample grid cannot do
better than ±1 sample; regressing a continuous offset can.

This is also why `detection.py` reports F1 and jitter as separate columns. A
detector can reach F1 = 0.998 and still be unusable for HRV. Reporting only F1
hides that.

## 6. Why Mondrian conformal rather than plain split conformal

Split conformal gives *marginal* coverage: P(HR_true ∈ C) ≥ 1 − α averaged over
everything. In this application that guarantee is close to useless. A predictor
can achieve 90 % marginal coverage by covering 99 % of clean windows and 40 %
of noisy ones — the exact opposite of what a monitor needs, since the interval
only matters when the signal is poor.

Mondrian (group-conditional) conformal computes calibration quantiles
separately within strata defined by predicted uSQI, giving

```
P( HR_true in C(x) | uSQI(x) in bin_k ) >= 1 - alpha    for every k
```

This is the point where contributions C1 and C3 stop being two ideas. uSQI is
not a diagnostic read-out bolted on the side; it is the conditioning variable
that makes the guarantee meaningful. Note the corollary, which is a genuine
test rather than a rhetorical flourish: if uSQI carried no information about
error, the strata would have identical quantiles and Mondrian would collapse
back to split conformal. The conditional-coverage table therefore validates the
quality index and the interval construction simultaneously.

**Implementation details that matter.**
- The finite-sample correction ⌈(n+1)(1−α)⌉/n is applied. Omitting it is the
  most common error in applied conformal work and turns a guarantee into an
  asymptotic hope.
- Strata are defined by *quantiles* of calibration uSQI, not fixed cut-points,
  so no stratum is starved of calibration points.
- A stratum with fewer than `min_per_bin` points falls back to the pooled
  quantile. Without this guard a sparse stratum returns an infinite interval:
  technically valid, practically useless.
- The nonconformity score is the *normalised* residual |y − q₅₀| / (q₉₅ − q₀₅).
  Raw-residual conformal produces constant-width intervals everywhere, which is
  precisely the uninformative behaviour this project exists to avoid.

## 7. Why HRV gets intervals, and how

RMSSD is a function of successive differences, so one missed beat merges two
intervals and one false beat splits one — and the resulting squared difference
enters the mean at full weight. In a 5-minute record at 70 bpm (~350
intervals), a single bad beat can move RMSSD by several milliseconds, the same
order as effect sizes routinely reported as significant in the autonomic
literature. Reporting HRV as a bare point value is therefore hard to defend,
yet it is universal.

The propagation is Monte-Carlo: each detected beat is kept with its predicted
probability and jittered by a Gaussian whose width is inflated where quality is
low; every draw yields a full HRV feature vector; the percentile interval is
then *conformalised* against ground truth on a calibration split. That last
step is what makes it a measurement rather than a modelling assumption — a raw
ensemble interval is only as good as its noise model, whereas a conformalised
one is corrected by however wrong that model was on held-out data.

Non-negative indices are clipped at their physiological bounds. Symmetric
conformal widening would otherwise hand back a negative lower bound for SDNN,
which is a valid coverage statement and a nonsensical measurement.

## 8. Why the evaluation splits by record and not by window

Windows from the same patient share morphology, baseline and noise
characteristics. Splitting by window puts near-duplicates on both sides of the
calibration/test boundary, which inflates coverage and makes the guarantee
vacuous. `scripts/evaluate_mitdb.py` splits by record identifier.

The four paced records (102, 104, 107, 217) are excluded following AAMI EC57.
Paced beats are a different detection problem and including them moves results
in either direction depending on how the detector handles pacing spikes.

## 9. Why the simulator is a core component, not a convenience

uSQI labels require the true heart rate of a *contaminated* segment. That is
available in exactly two situations: simulated data, or annotated data with
synthetic contamination applied on top. Both paths are supported; the simulator
is what makes the first one trustworthy.

Its fidelity is therefore tested rather than assumed. Asked for a target heart
rate, SDNN and LF/HF ratio, the generator returns them to within 0.03 bpm, 2 %
and 11 % respectively, measured after a full round trip through waveform
synthesis and Lomb–Scargle analysis.

Two corrections were added that plain ECGSYN omits and that matter for detector
benchmarking: QRS *duration* is rate-independent in reality, so its angular
width must shrink as RR grows; and the QT interval follows Bazett, so the
T-wave phase position scales with √RR. Without these the model produces QRS
complexes that widen at low heart rates, which is both unphysiological and
quietly favourable to fixed-window detectors.

## 10. Why Lomb–Scargle instead of interpolation plus Welch

The RR tachogram is unevenly sampled by construction. The usual remedy —
cubic-spline interpolation onto a uniform grid, then Welch — injects power into
the HF band and biases LF/HF, and the bias grows exactly where this project
operates, namely when beats are missing. Lomb–Scargle handles gaps natively.

## 11. Known limitations

- Training reported in this repository used simulated data plus real MIT-BIH
  waveforms with synthetic contamination. Validation on genuinely ambulatory
  wearable recordings (DaLiA, WESAD, Simband) has not been done.
- uSQI is defined relative to a reference detector (see §2).
- Mondrian conformal gives coverage conditional on the quality stratum, not
  fully conditional coverage, which is known to be unattainable
  distribution-free without further assumptions.
- The AF rhythm model is a shifted-gamma renewal process. It reproduces the
  Poincaré signature of AF but not the fine structure of AV-node concealed
  conduction.
- Single-lead only. Multi-lead fusion would very likely improve both detection
  and quality estimation and is not attempted here.
- **Ectopic correction interacts badly with AF, and the default does not yet
  adapt.** The Malik 20 % rule flags any interval differing from its
  predecessor by more than a fifth. In sinus rhythm that isolates ectopic
  beats; in atrial fibrillation it flags most of the tachogram, and
  interpolating over them smooths away exactly the irregularity being
  measured. The visible symptom is SD1/SD2 around 0.34 on simulated AF where
  the uncorrected series gives close to 1. The correct fix is to select the
  correction from a detected rhythm class rather than applying one rule
  everywhere; until then, pass `correction="none"` when analysing AF. This is
  not specific to this implementation -- it is a general and under-reported
  hazard in HRV pipelines.

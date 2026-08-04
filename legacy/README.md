# Version 1 (archived)

The original `HR-from-ECG`, released in 2023. Kept here unchanged so that the
two versions can be compared directly; it is no longer maintained.

## What it did

A single-file Tkinter tool: load a CSV, run `scipy.signal.find_peaks`, print an
average heart rate.

## Why it was replaced

Reviewing it before starting version 2 turned up several problems worth
recording, because each of them motivated a specific part of the new design.

| Issue | Detail |
|---|---|
| No filtering | The README claimed "simple noise reduction"; the code contains none. The raw signal goes straight into `find_peaks`. |
| Hard-coded sampling rate | `sampling_rate = 500` regardless of the file. On a 360 Hz MIT-BIH record every reported heart rate is wrong by 39 %. |
| No adaptive threshold | A single fixed `distance` parameter and no amplitude threshold, so the detector fails on any amplitude drift. |
| No artefact handling | No refractory logic, no ectopic-beat handling, no signal-quality check. |
| No uncertainty | A number is printed for any input, including a flat line. |
| No validation | No dataset, no metrics, no comparison against any reference. |

The last two are the ones that shaped version 2. An estimator that reports a
confident number for an unusable signal is not merely inaccurate — it is
misleading in a way the user cannot detect.

## Running it

```bash
python legacy/hr_from_ecg_v1.py
```

Requires `numpy`, `scipy`, `matplotlib` and `tkinter`. Expects a CSV with a
column named `ECG` sampled at 500 Hz.

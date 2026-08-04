"""
PhysioNet loaders.

Requires `wfdb` and network access to physionet.org. Records are cached on
disk; the first call over a full database is slow, subsequent calls are not.

Database roles in the experimental design
-----------------------------------------
mitdb   : primary development set. 48 records, 30 min, 2 leads, 360 Hz.
nstdb   : noise-stress records + the three raw noise signals (bw, em, ma).
incart  : external validation, 75 records, 12 leads, 257 Hz.
qtdb    : external validation with beat-level expert annotations.
afdb    : atrial fibrillation -- the stratum where HRV pipelines fail.
cpsc2020: long-term single-lead with dense ectopy, the closest public proxy
          for the wearable setting.

Only beat annotations belonging to the AAMI beat classes are retained; rhythm
markers, signal-quality markers and paced-beat annotations are handled
explicitly rather than being silently swept into the reference set, which is a
common and consequential preprocessing error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal as sps

__all__ = ["Record", "load_record", "list_records", "DB_INFO", "BEAT_SYMBOLS"]

# AAMI EC57 beat annotation symbols
BEAT_SYMBOLS = set("NLRejAaJSVEFP/fQ")
NON_BEAT_SYMBOLS = set('[!]x()pt"=@~+|s^')

DB_INFO = {
    "mitdb": dict(fs=360, leads=2, name="MIT-BIH Arrhythmia"),
    "nstdb": dict(fs=360, leads=2, name="MIT-BIH Noise Stress Test"),
    "incartdb": dict(fs=257, leads=12, name="St Petersburg INCART"),
    "qtdb": dict(fs=250, leads=2, name="QT Database"),
    "afdb": dict(fs=250, leads=2, name="MIT-BIH Atrial Fibrillation"),
    "ltafdb": dict(fs=128, leads=2, name="Long Term AF"),
    "svdb": dict(fs=128, leads=2, name="MIT-BIH Supraventricular Arrhythmia"),
}


@dataclass
class Record:
    signal: np.ndarray            # (n_samples,) single lead, millivolts
    fs: float
    r_peaks: np.ndarray           # sample indices of annotated beats
    beat_types: list[str]
    record_id: str
    database: str
    lead_name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return len(self.signal) / self.fs


def list_records(database: str = "mitdb", local_dir: str | Path | None = None) -> list[str]:
    """
    Return the record identifiers of a database.

    With `local_dir` the listing is taken from the header files on disk, so an
    offline copy of a database behaves identically to the online one.
    """
    if local_dir is not None:
        d = Path(local_dir)
        return sorted(f.stem for f in d.glob("*.hea"))

    import wfdb

    return list(wfdb.get_record_list(database))


def load_record(
    record_id: str,
    database: str = "mitdb",
    lead: int | str = 0,
    fs_target: float | None = 500.0,
    cache_dir: str | Path | None = None,
    keep_paced: bool = False,
    local_dir: str | Path | None = None,
) -> Record:
    """
    Load one record with its beat annotations, optionally resampled.

    Resampling is done with a polyphase filter and the annotation indices are
    rescaled accordingly. Note that this introduces a sub-sample annotation
    error of up to 0.5/fs_original seconds -- at 360 Hz that is 1.4 ms, which
    is below the localisation tolerance but is recorded in `meta` so that it
    can be accounted for when reporting jitter.
    """
    import wfdb

    if local_dir is not None:
        base = str(Path(local_dir) / record_id)
        rec = wfdb.rdrecord(base)
        ann = wfdb.rdann(base, "atr")
    else:
        cache = str(cache_dir) if cache_dir else None
        rec = wfdb.rdrecord(record_id, pn_dir=database, pn_dir_cache=cache) \
            if cache else wfdb.rdrecord(record_id, pn_dir=database)
        ann = wfdb.rdann(record_id, "atr", pn_dir=database)

    if isinstance(lead, str):
        idx = list(rec.sig_name).index(lead)
    else:
        idx = int(lead)
    x = np.asarray(rec.p_signal[:, idx], dtype=float)
    x = np.nan_to_num(x, nan=0.0)
    fs = float(rec.fs)

    symbols = np.asarray(ann.symbol)
    samples = np.asarray(ann.sample, dtype=int)
    keep = np.array([s in BEAT_SYMBOLS for s in symbols])
    if not keep_paced:
        keep &= np.array([s not in ("/", "f", "Q") for s in symbols])
    peaks, btypes = samples[keep], list(symbols[keep])

    resample_err_ms = 0.0
    if fs_target is not None and abs(fs - fs_target) > 1e-6:
        up, down = int(round(fs_target)), int(round(fs))
        g = np.gcd(up, down)
        x = sps.resample_poly(x, up // g, down // g)
        peaks = np.round(peaks * (fs_target / fs)).astype(int)
        resample_err_ms = 0.5 / fs * 1000
        fs = float(fs_target)

    peaks = peaks[(peaks >= 0) & (peaks < len(x))]
    btypes = btypes[: len(peaks)]

    return Record(
        signal=x, fs=fs, r_peaks=peaks, beat_types=btypes,
        record_id=record_id, database=database,
        lead_name=str(rec.sig_name[idx]),
        meta=dict(original_fs=float(rec.fs), n_annotations=int(keep.sum()),
                  resample_jitter_ms=resample_err_ms,
                  units=str(rec.units[idx]) if rec.units else "mV"),
    )

"""Vocal preservation: analyse a vocal file (tempo, key) so the accompaniment fits it,
and keep the vocal itself bit-for-bit untouched (only gain-matched at mix time)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_io import read_audio, resample
from .theory import NOTE_NAMES

# Krumhansl-Schmuckler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class VocalInfo:
    path: str
    audio: np.ndarray  # (n, 2) float32, original samplerate
    sr: int
    seconds: float
    bpm: float | None
    root_pc: int | None
    scale: str | None  # 'major' | 'minor'
    key_confidence: float
    start_offset_s: float  # time before the vocal becomes audible
    grid_offset_s: float = 0.0  # where the first bar line falls (from onset analysis)

    @property
    def key_name(self) -> str:
        if self.root_pc is None:
            return "unknown"
        return f"{NOTE_NAMES[self.root_pc]} {self.scale}"


def _mono(a: np.ndarray) -> np.ndarray:
    return a.mean(axis=1)


def estimate_tempo(mono: np.ndarray, sr: int, lo: float = 60, hi: float = 180) -> float | None:
    """Onset-envelope autocorrelation tempo estimate. Returns None if the vocal is too arrhythmic."""
    target = 11025
    step = max(1, sr // target)
    x = mono[::step]
    sr2 = sr / step
    hop = 256
    frames = len(x) // hop
    if frames < 64:
        return None
    mag = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2)) for i in range(frames)])
    onset = np.maximum(0, np.diff(np.log1p(mag * 50)))
    onset -= onset.mean()
    if not np.any(onset):
        return None
    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    fps = sr2 / hop
    lags = np.arange(len(ac))
    bpms = 60.0 * fps / np.maximum(lags, 1)
    mask = (bpms >= lo) & (bpms <= hi)
    if not mask.any():
        return None
    # mild prior favouring ~110 BPM so we avoid half/double-time errors
    prior = np.exp(-0.5 * (np.log2(bpms[mask] / 110.0) / 0.9) ** 2)
    score = ac[mask] * prior
    best = int(np.argmax(score))
    if ac[mask][best] <= 0:
        return None
    return float(round(bpms[mask][best]))


def estimate_key(mono: np.ndarray, sr: int) -> tuple[int | None, str | None, float]:
    """Chroma-based key estimate (Krumhansl-Schmuckler). Returns (root_pc, scale, confidence 0..1)."""
    n = 8192
    hop = 4096
    if len(mono) < n:
        return None, None, 0.0
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    valid = (freqs > 80) & (freqs < 2000)
    pcs = (np.round(12 * np.log2(freqs[valid] / 440.0)) + 69) % 12
    pcs = pcs.astype(int)
    chroma = np.zeros(12)
    for i in range(0, len(mono) - n, hop):
        seg = mono[i:i + n] * win
        mag = np.abs(np.fft.rfft(seg))[valid]
        if mag.max() < 1e-4:
            continue
        np.add.at(chroma, pcs, mag ** 2)
    if chroma.sum() <= 0:
        return None, None, 0.0
    chroma /= chroma.sum()
    best = (-2.0, 0, "major")
    scores = []
    for root in range(12):
        for scale, prof in (("major", _MAJOR), ("minor", _MINOR)):
            c = np.corrcoef(chroma, np.roll(prof, root))[0, 1]
            scores.append(c)
            if c > best[0]:
                best = (c, root, scale)
    scores = sorted(scores, reverse=True)
    conf = float(np.clip((scores[0] - scores[1]) * 8 + max(0.0, scores[0]) * 0.5, 0, 1))
    return best[1], best[2], conf


def first_audible_time(mono: np.ndarray, sr: int, thresh_db: float = -40.0) -> float:
    win = int(0.02 * sr)
    thr = 10 ** (thresh_db / 20)
    for i in range(0, len(mono) - win, win):
        if np.sqrt(np.mean(mono[i:i + win] ** 2)) > thr:
            return i / sr
    return 0.0


def load_vocal(path: str) -> VocalInfo:
    """Load a vocal file untouched and analyse it. The audio itself is never modified."""
    audio, sr = read_audio(path)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    audio = audio.astype(np.float32)
    mono = _mono(audio)
    bpm = estimate_tempo(mono, sr)
    bpm, grid = refine_tempo_and_grid(mono, sr, bpm)
    root, scale, conf = estimate_key(mono, sr)
    return VocalInfo(
        path=str(path), audio=audio, sr=sr, seconds=len(audio) / sr, bpm=bpm,
        root_pc=root, scale=scale, key_confidence=conf,
        start_offset_s=first_audible_time(mono, sr), grid_offset_s=grid,
    )


# ---------------------------------------------------------------------------
# Fitting the accompaniment to the vocal
# ---------------------------------------------------------------------------

def _onset_envelope(mono: np.ndarray, sr: int, hop: int = 256) -> tuple[np.ndarray, float]:
    target = 11025
    step = max(1, sr // target)
    x = mono[::step]
    sr2 = sr / step
    frames = len(x) // hop
    if frames < 32:
        return np.zeros(0), 1.0
    mag = np.sqrt(np.mean(x[: frames * hop].reshape(frames, hop) ** 2, axis=1))
    onset = np.maximum(0, np.diff(np.log1p(mag * 50), prepend=0.0))
    onset = np.convolve(onset, np.exp(-0.5 * (np.arange(-3, 4) / 1.2) ** 2), mode="same")
    return onset, sr2 / hop


def refine_tempo_and_grid(mono: np.ndarray, sr: int, bpm_guess: float | None,
                          beats_per_bar: int = 4, search: float = 4.0) -> tuple[float | None, float]:
    """Refine BPM to ~0.05 and find where the first bar line falls (seconds).

    Returns (bpm, grid_offset_s) with 0 <= grid_offset_s < one bar.
    """
    onset, fps = _onset_envelope(mono, sr)
    if bpm_guess is None or len(onset) == 0:
        return bpm_guess, 0.0
    best = (-1.0, bpm_guess, 0.0, 0)
    n = len(onset)
    cands = [bpm_guess] if search <= 0 else np.arange(bpm_guess - search, bpm_guess + search + 0.001, 0.1)
    for bpm in cands:
        period = 60.0 * fps / bpm
        n_beats = int((n - 1) / period)
        if n_beats < 8:
            continue
        k = np.arange(n_beats)
        for phi in np.linspace(0, period, 16, endpoint=False):
            idx = k * period + phi
            idx = idx[idx < n - 1]
            vals = np.interp(idx, np.arange(n), onset)
            sc = float(vals.mean())
            if sc > best[0]:
                best = (sc, float(bpm), phi / fps, len(vals))
    _, bpm, phase_s, _ = best
    spb = 60.0 / bpm
    # which of the 4 beats carries the bar line? pick the one with strongest onsets
    t = np.arange(0, len(mono) / sr, spb) + phase_s
    idx = np.clip(t * fps, 0, n - 1)
    vals = np.interp(idx, np.arange(n), onset)
    scores = [vals[m::beats_per_bar].sum() for m in range(beats_per_bar)]
    m = int(np.argmax(scores))
    return round(bpm, 2), float((phase_s + m * spb) % (beats_per_bar * spb))


def _chroma_of(mono: np.ndarray, sr: int) -> np.ndarray | None:
    n = 4096
    if len(mono) < n:
        return None
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    valid = (freqs > 80) & (freqs < 2000)
    pcs = ((np.round(12 * np.log2(freqs[valid] / 440.0)) + 69) % 12).astype(int)
    chroma = np.zeros(12)
    for i in range(0, len(mono) - n, n // 2):
        mag = np.abs(np.fft.rfft(mono[i:i + n] * win))[valid]
        if mag.max() < 1e-4:
            continue
        np.add.at(chroma, pcs, mag ** 2)
    if chroma.sum() <= 0:
        return None
    return chroma / chroma.sum()


@dataclass
class VocalGuide:
    """What the composer needs to know about the vocal, on the song's bar / 8th-note grid."""
    offset_s: float            # time of the first bar line
    activity: np.ndarray       # bool, one entry per 8th note (True = vocal is singing)
    bar_chroma: np.ndarray     # (bars, 12) pitch-class energy of the vocal in each bar
    energy: np.ndarray         # per bar 0..1 vocal loudness


def build_guide(v: VocalInfo, bpm: float, bars: int, offset_s: float) -> VocalGuide:
    mono = v.audio.mean(axis=1)
    sr = v.sr
    spb = 60.0 / bpm
    slot = spb / 2.0
    n_slots = bars * 8
    rms = np.zeros(n_slots)
    for i in range(n_slots):
        a = int((offset_s + i * slot) * sr)
        b = int((offset_s + (i + 1) * slot) * sr)
        if a >= len(mono):
            break
        seg = mono[max(0, a):max(a + 1, min(b, len(mono)))]
        rms[i] = np.sqrt(np.mean(seg ** 2)) if len(seg) else 0.0
    ref = np.percentile(rms[rms > 1e-4], 90) if np.any(rms > 1e-4) else 1.0
    activity = rms > max(0.12 * ref, 1e-3)
    # fill tiny gaps (< 1 slot) so breaths between words don't count as free space
    for i in range(1, n_slots - 1):
        if not activity[i] and activity[i - 1] and activity[i + 1]:
            activity[i] = True
    bar_len = 4 * spb
    chroma = np.zeros((bars, 12))
    energy = np.zeros(bars)
    for b in range(bars):
        a = int((offset_s + b * bar_len) * sr)
        e = int((offset_s + (b + 1) * bar_len) * sr)
        if a >= len(mono):
            break
        seg = mono[max(0, a):e]
        c = _chroma_of(seg, sr)
        if c is not None:
            chroma[b] = c
        energy[b] = min(1.0, float(np.sqrt(np.mean(seg ** 2))) / max(ref, 1e-6)) if len(seg) else 0.0
    return VocalGuide(offset_s, activity, chroma, energy)

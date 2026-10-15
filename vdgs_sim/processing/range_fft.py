"""Reference range processing: windowed fast-time FFT and peak estimate."""
from __future__ import annotations

import numpy as np

from ..radar.config import RadarConfig, C


def range_axis(cfg: RadarConfig, n_fft: int | None = None) -> np.ndarray:
    """Range in metres for each FFT bin."""
    n_fft = n_fft or cfg.n_samples
    f = np.arange(n_fft) * cfg.fs / n_fft
    return f * C / (2 * cfg.slope)


def range_profile(cfg: RadarConfig, frame: np.ndarray, n_fft: int | None = None) -> np.ndarray:
    """Non-coherent power range profile (linear), averaged over chirps."""
    n_fft = n_fft or cfg.n_samples
    win = np.hanning(cfg.n_samples)
    spec = np.fft.fft(frame * win[None, :], n=n_fft, axis=1)
    return np.mean(np.abs(spec) ** 2, axis=0)


def estimate_range(cfg: RadarConfig, frame: np.ndarray, min_range: float = 1.0) -> float:
    """Strongest target range with parabolic interpolation around the peak bin."""
    p = range_profile(cfg, frame)
    r = range_axis(cfg)
    p = np.where(r < min_range, 0.0, p)         # ignore DC / very near clutter
    i = int(np.argmax(p))
    if 0 < i < len(p) - 1:
        a, b, g = np.log(p[i - 1] + 1e-30), np.log(p[i] + 1e-30), np.log(p[i + 1] + 1e-30)
        delta = 0.5 * (a - g) / (a - 2 * b + g)
    else:
        delta = 0.0
    bin_width_m = (cfg.fs / cfg.n_samples) * C / (2 * cfg.slope)   # metres per FFT bin
    return (i + delta) * bin_width_m

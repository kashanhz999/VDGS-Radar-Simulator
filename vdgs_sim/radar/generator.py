"""Physics-based generation of synthetic FMCW baseband (ADC) samples.

A point target at range R with radial speed v produces, after dechirping,
a complex tone at the beat frequency f_b = 2 S R / c. Its phase advances
from chirp to chirp by 4 pi v Tc / lambda (Doppler). Noise is circular
complex Gaussian, set by a per-sample signal-to-noise ratio.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import RadarConfig, C


@dataclass(frozen=True)
class PointTarget:
    range_m: float
    velocity_mps: float = 0.0
    amplitude: float = 1.0


def generate_frame(
    cfg: RadarConfig,
    targets: list[PointTarget],
    snr_db: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return one frame of complex samples, shape (n_chirps, n_samples).

    snr_db is the per-sample SNR of a unit-amplitude target. None = no noise.
    """
    t = np.arange(cfg.n_samples) / cfg.fs                 # fast time
    k = np.arange(cfg.n_chirps)[:, None]                  # chirp index
    frame = np.zeros((cfg.n_chirps, cfg.n_samples), dtype=np.complex128)

    for tg in targets:
        fb = cfg.beat_frequency(tg.range_m)
        phase0 = 4 * np.pi * tg.range_m / cfg.wavelength          # carrier round-trip phase
        dphi = 4 * np.pi * tg.velocity_mps * cfg.t_chirp / cfg.wavelength
        frame += tg.amplitude * np.exp(1j * (2 * np.pi * fb * t[None, :] + phase0 + dphi * k))

    if snr_db is not None:
        rng = rng or np.random.default_rng()
        sigma = 10 ** (-snr_db / 20) / np.sqrt(2)
        frame += sigma * (rng.standard_normal(frame.shape) + 1j * rng.standard_normal(frame.shape))

    return frame

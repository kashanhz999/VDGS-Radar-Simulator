"""Reference range-Doppler processing with 2D cell-averaging CFAR.

Pipeline per frame: Hann windows in fast and slow time -> 2D FFT ->
power map -> CA-CFAR -> connected-component clustering -> one detection
per cluster at its strongest cell, refined by parabolic interpolation in
range.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage

from ..radar.config import RadarConfig, C


@dataclass(frozen=True)
class Detection:
    range_m: float
    velocity_mps: float
    snr_db: float          # peak power over local CFAR noise estimate


def range_doppler_map(cfg: RadarConfig, frame: np.ndarray) -> np.ndarray:
    """Power map, shape (n_chirps, n_samples); Doppler axis fft-shifted."""
    wr = np.hanning(cfg.n_samples)[None, :]
    wd = np.hanning(cfg.n_chirps)[:, None]
    rfft = np.fft.fft(frame * wr, axis=1)
    rd = np.fft.fftshift(np.fft.fft(rfft * wd, axis=0), axes=0)
    return np.abs(rd) ** 2


def velocity_axis(cfg: RadarConfig) -> np.ndarray:
    m = np.arange(cfg.n_chirps) - cfg.n_chirps // 2
    return m * cfg.wavelength / (2 * cfg.n_chirps * cfg.t_chirp)


def bin_to_range(cfg: RadarConfig, b: float) -> float:
    return b * (cfg.fs / cfg.n_samples) * C / (2 * cfg.slope)


def ca_cfar_2d(power: np.ndarray, guard=(2, 2), train=(4, 8), pfa=1e-6):
    """Return (mask, noise) for cell-averaging CFAR on a square-law map.

    guard/train are half-widths in (Doppler, range) bins. The Doppler axis
    wraps (it is periodic); the range axis is reflected at its edges.
    """
    gd, gr = guard
    td, tr = train
    kd, kr = 2 * (gd + td) + 1, 2 * (gr + tr) + 1
    kernel = np.ones((kd, kr))
    kernel[td:td + 2 * gd + 1, tr:tr + 2 * gr + 1] = 0.0
    n_train = kernel.sum()
    padded = np.pad(power, ((gd + td, gd + td), (0, 0)), mode="wrap")
    padded = np.pad(padded, ((0, 0), (gr + tr, gr + tr)), mode="reflect")
    noise = ndimage.correlate(padded, kernel, mode="constant")[gd + td:-(gd + td), gr + tr:-(gr + tr)] / n_train
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    return power > alpha * noise, noise


def detect(cfg: RadarConfig, frame: np.ndarray, min_range: float = 1.0, pfa: float = 1e-6,
           min_snr_db: float = 13.0) -> list[Detection]:
    """CFAR detections, one per cluster. min_snr_db rejects window sidelobes
    of strong targets, which a cell-averaging CFAR alone can pass."""
    p = range_doppler_map(cfg, frame)
    mask, noise = ca_cfar_2d(p, pfa=pfa)
    r_of_bin = np.array([bin_to_range(cfg, b) for b in range(cfg.n_samples)])
    mask[:, r_of_bin < min_range] = False
    labels, n = ndimage.label(mask)
    v_axis = velocity_axis(cfg)
    dets = []
    for lab in range(1, n + 1):
        cells = np.argwhere(labels == lab)
        vals = p[cells[:, 0], cells[:, 1]]
        d, r = cells[np.argmax(vals)]
        snr = 10 * np.log10(p[d, r] / noise[d, r])
        if snr < min_snr_db:
            continue
        # parabolic refinement in range on log power
        if 0 < r < cfg.n_samples - 1:
            a, b, g = np.log(p[d, r - 1] + 1e-30), np.log(p[d, r] + 1e-30), np.log(p[d, r + 1] + 1e-30)
            denom = a - 2 * b + g
            delta = 0.5 * (a - g) / denom if denom != 0 else 0.0
        else:
            delta = 0.0
        # parabolic refinement in Doppler (axis is periodic)
        dm, dp = (d - 1) % cfg.n_chirps, (d + 1) % cfg.n_chirps
        a, b, g = np.log(p[dm, r] + 1e-30), np.log(p[d, r] + 1e-30), np.log(p[dp, r] + 1e-30)
        denom = a - 2 * b + g
        dd = 0.5 * (a - g) / denom if denom != 0 else 0.0
        dv = v_axis[1] - v_axis[0]
        dets.append(Detection(bin_to_range(cfg, r + delta), float(v_axis[d] + dd * dv), float(snr)))
    dets.sort(key=lambda x: x.range_m)
    return dets

"""Scene description: scatterers with radar cross-section (RCS) and the
two-way path loss of the radar equation.

Amplitude convention: a 1 m^2 scatterer at the reference range R_REF has
unit amplitude, so `snr_ref_db` passed to the generator is the per-sample
SNR of that reference scatterer. Received power falls as RCS / R^4, so
amplitude scales as sqrt(RCS) * (R_REF / R)^2.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .generator import PointTarget

R_REF = 20.0  # m


@dataclass(frozen=True)
class Scatterer:
    range_m: float
    velocity_mps: float
    rcs_m2: float
    label: str = ""


def to_point_targets(scatterers: list[Scatterer], r_min: float = 0.5) -> list[PointTarget]:
    out = []
    for s in scatterers:
        r = max(s.range_m, r_min)
        amp = np.sqrt(s.rcs_m2) * (R_REF / r) ** 2
        out.append(PointTarget(range_m=s.range_m, velocity_mps=s.velocity_mps, amplitude=float(amp)))
    return out


@dataclass(frozen=True)
class AircraftProfile:
    """Illustrative nose-on scattering centres of a narrow-body aircraft.

    Offsets are metres behind the nose along the approach axis. Values are
    illustrative simulation inputs, not measured signatures.
    """
    name: str = "narrow-body (illustrative)"
    offsets_m: tuple[float, ...] = (0.0, 5.0, 11.5)
    rcs_m2: tuple[float, ...] = (10.0, 3.0, 30.0)
    labels: tuple[str, ...] = ("nose", "nose gear", "engine inlets")

    def scatterers(self, nose_range: float, velocity: float) -> list[Scatterer]:
        return [Scatterer(nose_range + o, velocity, s, l)
                for o, s, l in zip(self.offsets_m, self.rcs_m2, self.labels)]

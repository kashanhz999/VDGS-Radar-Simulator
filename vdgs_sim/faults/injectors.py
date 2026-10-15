"""Fault injectors.

Radar-side faults change what the sensor produces; stream-side faults change
how detection messages reach the VDGS. Stationary ghost echoes and reduced
SNR are scenario properties (see scenario.model)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrozenOutput:
    """Sensor repeats its last good frame between t0 and t1 (radar side)."""
    t0: float
    t1: float

    def active(self, t: float) -> bool:
        return self.t0 <= t <= self.t1


@dataclass(frozen=True)
class StreamLatency:
    """Extra delivery delay on every message from t0 on (stream side)."""
    delay_s: float
    t0: float = 0.0

    def delay(self, t: float) -> float:
        return self.delay_s if t >= self.t0 else 0.0


@dataclass(frozen=True)
class StreamDropout:
    """Messages with sensor time in [t0, t1] are lost (stream side)."""
    t0: float
    t1: float

    def dropped(self, t: float) -> bool:
        return self.t0 <= t <= self.t1


@dataclass(frozen=True)
class Faults:
    frozen: FrozenOutput | None = None
    latency: StreamLatency | None = None
    dropout: StreamDropout | None = None

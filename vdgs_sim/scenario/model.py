"""Docking scenarios: aircraft kinematics, other actors, ground truth.

Geometry: the radar sits on the terminal wall on the stand centreline.
Range is measured along the centreline; lateral actors (vehicles, people)
cross at a perpendicular distance d from the wall, so their range is
sqrt(d^2 + x^2) and their radial speed x * vx / range. Actors outside the
antenna field of view are not seen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..radar.scene import AircraftProfile, Scatterer


@dataclass
class DockingTrajectory:
    start_range: float = 60.0      # nose range when the run starts, m
    entry_speed: float = 1.5       # m/s, constant until braking
    decel: float = 0.3             # m/s^2
    stop_line: float = 22.0        # nose range of the correct stop position, m
    stop_error: float = 0.0        # actual stop = stop_line + stop_error (negative = past the line)
    pushback_after: float | None = None   # s after stopping; None = no pushback
    pushback_speed: float = 0.5    # m/s, away from the wall
    pushback_accel: float = 0.2    # m/s^2

    def __post_init__(self):
        self.stop_range = self.stop_line + self.stop_error
        d_brake = self.entry_speed ** 2 / (2 * self.decel)
        d_total = self.start_range - self.stop_range
        if d_brake > d_total:
            raise ValueError("start range too short for this speed and deceleration")
        self.t_brake = (d_total - d_brake) / self.entry_speed
        self.t_stop = self.t_brake + self.entry_speed / self.decel
        self.t_pushback = None if self.pushback_after is None else self.t_stop + self.pushback_after

    def state(self, t: float) -> tuple[float, float]:
        """(nose range, radial velocity) at time t; velocity < 0 = approaching."""
        v0, a = self.entry_speed, self.decel
        if t <= self.t_brake:
            return self.start_range - v0 * t, -v0
        if t <= self.t_stop:
            tb = t - self.t_brake
            r_b = self.start_range - v0 * self.t_brake
            return r_b - (v0 * tb - 0.5 * a * tb ** 2), -(v0 - a * tb)
        if self.t_pushback is None or t <= self.t_pushback:
            return self.stop_range, 0.0
        tp = t - self.t_pushback
        t_acc = self.pushback_speed / self.pushback_accel
        if tp <= t_acc:
            return self.stop_range + 0.5 * self.pushback_accel * tp ** 2, self.pushback_accel * tp
        return (self.stop_range + 0.5 * self.pushback_speed * t_acc
                + self.pushback_speed * (tp - t_acc)), self.pushback_speed


@dataclass
class LateralActor:
    """A vehicle or person crossing the stand at perpendicular distance d."""
    label: str
    d: float              # m from the wall
    x0: float             # lateral start position, m (0 = centreline)
    vx: float             # lateral speed, m/s
    t0: float             # s, start time
    t1: float             # s, end time (actor leaves the scene)
    rcs_m2: float

    def state(self, t: float):
        if not (self.t0 <= t <= self.t1):
            return None
        x = self.x0 + self.vx * (t - self.t0)
        r = float(np.hypot(self.d, x))
        return r, x * self.vx / r, x


@dataclass
class StaticReflector:
    """A stationary echo at a fixed range (e.g. a multipath ghost)."""
    label: str
    range_m: float
    rcs_m2: float
    t0: float
    t1: float


@dataclass
class Scenario:
    sid: str
    title: str
    trajectory: DockingTrajectory
    duration: float
    aircraft: AircraftProfile = field(default_factory=AircraftProfile)
    actors: list[LateralActor] = field(default_factory=list)
    reflectors: list[StaticReflector] = field(default_factory=list)
    snr_ref_db: float = -15.0
    fov_deg: float = 60.0
    faults: object = None          # vdgs_sim.faults.Faults, set by the loader

    def scatterers(self, t: float) -> list[Scatterer]:
        r, v = self.trajectory.state(t)
        out = self.aircraft.scatterers(r, v)
        for a in self.actors:
            s = a.state(t)
            if s is None:
                continue
            ra, va, x = s
            if abs(np.degrees(np.arctan2(x, a.d))) <= self.fov_deg / 2:
                out.append(Scatterer(ra, va, a.rcs_m2, a.label))
        for g in self.reflectors:
            if g.t0 <= t <= g.t1:
                out.append(Scatterer(g.range_m, 0.0, g.rcs_m2, g.label))
        return out

    def actor_visible(self, t: float) -> bool:
        for a in self.actors:
            s = a.state(t)
            if s is not None and abs(np.degrees(np.arctan2(s[2], a.d))) <= self.fov_deg / 2:
                return True
        return False

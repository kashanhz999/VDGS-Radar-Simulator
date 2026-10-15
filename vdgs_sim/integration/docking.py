"""Reference VDGS docking logic (the system under test in the experiments).

It consumes detection messages, keeps a constant-velocity Kalman track of
the aircraft nose, and raises the events that leave the VDGS: capture,
overshoot, track loss, obstacle, on-block (AIBT) and off-block (AOBT).
It is an open stand-in for vendor logic, written so that every rule is
visible and testable; it is not a reproduction of any product.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..processing.range_doppler import Detection


@dataclass
class Event:
    kind: str
    t: float              # stamped time (policy-dependent)
    t_recv: float         # receive time of the message that triggered it
    frame: int
    info: dict = field(default_factory=dict)


@dataclass
class DockingConfig:
    stop_line: float = 22.0           # m
    capture_max: float = 75.0         # m
    confirm_frames: int = 3           # detections needed to start a track
    gate_r: float = 1.5               # m
    gate_v: float = 1.0               # m/s
    sigma_a: float = 0.3              # m/s^2, process noise
    meas_sigma_r: float = 0.05        # m
    meas_sigma_v: float = 0.15        # m/s
    stop_speed: float = 0.1           # m/s
    stop_window: int = 5              # frames (1 s at 5 Hz)
    stop_drift: float = 0.1           # m allowed over the window
    overshoot: float = 0.5            # m past the stop line
    lost_frames: int = 5              # missed associations before track loss
    msg_timeout: float = 1.0          # s without messages before track loss
    away_speed: float = 0.2           # m/s
    away_dist: float = 0.5            # m beyond stop position
    obstacle_margin: float = 1.0      # m in front of the aircraft
    stamp: str = "receive"            # "receive" or "sensor" timestamps on events
    recapture: bool = False           # v2 rule: re-acquire a slow/stationary aircraft after loss
    recapture_window: float = 10.0    # s after the loss during which re-acquisition is tried
    recapture_gate: float = 2.0       # m around the last tracked range
    nose_reselect: bool = False       # v2 rule: move the track to a co-moving echo in front of it
    reselect_frames: int = 3
    reselect_dv: float = 0.3          # m/s
    reselect_min_speed: float = 0.3   # m/s; only while the aircraft is clearly moving
    aobt_backdate: bool = False       # v2 rule: stamp AOBT at the last frame still at the stop position
    backdate_tol: float = 0.05        # m


class DockingLogic:
    def __init__(self, cfg: DockingConfig | None = None):
        self.c = cfg or DockingConfig()
        self.x = None              # [range, velocity]
        self.P = None
        self.t_last_sensor = None
        self.t_last_recv = None
        self.candidates: list[Detection] = []
        self.missed = 0
        self.state = "WAIT"
        self.events: list[Event] = []
        self.history = []          # (frame, t_sensor, t_recv, r_filt, v_filt, state, obstacle)
        self.stop_buf = []         # (frame, t_sensor, t_recv, r)
        self.stop_pos = None
        self.obstacle_active = False
        self.obstacle_frames = 0
        self.lost_at = None        # (t_sensor, last range) of the most recent loss
        self.reselect_buf: list[Detection] = []
        self.parked_buf = []       # (t_sensor, t_recv, frame, r) while parked

    # -- helpers --------------------------------------------------------
    def _emit(self, kind, t_sensor, t_recv, frame, **info):
        t = t_recv if self.c.stamp == "receive" else t_sensor
        self.events.append(Event(kind, t, t_recv, frame, info))

    def _predict(self, dt):
        F = np.array([[1, dt], [0, 1]])
        q = self.c.sigma_a ** 2
        Q = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def _update(self, d: Detection):
        z = np.array([d.range_m, d.velocity_mps])
        R = np.diag([self.c.meas_sigma_r ** 2, self.c.meas_sigma_v ** 2])
        S = self.P + R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.x)
        self.P = (np.eye(2) - K) @ self.P

    def _lose_track(self, t_sensor, t_recv, frame, reason):
        if self.state in ("TRACKING", "OVERSHOOT"):
            self._emit("TRACK_LOST", t_sensor, t_recv, frame, reason=reason)
            self.lost_at = (t_sensor, float(self.x[0]))
        if self.state != "PARKED":
            self.x, self.P = None, None
            self.state = "WAIT"
            self.candidates, self.stop_buf = [], []

    # -- main entry -----------------------------------------------------
    def process(self, frame: int, t_sensor: float, t_recv: float, dets: list[Detection]):
        c = self.c
        if self.t_last_recv is not None and t_recv - self.t_last_recv > c.msg_timeout and self.x is not None:
            t_to = self.t_last_recv + c.msg_timeout
            self._lose_track(self.t_last_sensor + c.msg_timeout, t_to, frame, "message timeout")

        if self.x is None:
            self.obstacle_active = False
            self._try_capture(frame, t_sensor, t_recv, dets)
        else:
            self._predict(t_sensor - self.t_last_sensor)
            assoc = self._associate(dets)
            if assoc is not None:
                self._update(assoc)
                self.missed = 0
            else:
                self.missed += 1
                if self.missed >= c.lost_frames:
                    self._lose_track(t_sensor, t_recv, frame, "no association")
            if self.x is not None:
                self._docking_rules(frame, t_sensor, t_recv, dets, assoc)

        self.t_last_sensor, self.t_last_recv = t_sensor, t_recv
        r = None if self.x is None else float(self.x[0])
        v = None if self.x is None else float(self.x[1])
        self.history.append((frame, t_sensor, t_recv, r, v, self.state, self.obstacle_active))

    def _try_capture(self, frame, t_sensor, t_recv, dets):
        c = self.c
        if c.recapture and self.lost_at is not None and t_sensor - self.lost_at[0] <= c.recapture_window:
            near = [d for d in dets if abs(d.range_m - self.lost_at[1]) < c.recapture_gate]
            if near:
                d = min(near, key=lambda d: abs(d.range_m - self.lost_at[1]))
                if self.candidates and abs(d.range_m - self.candidates[-1].range_m) > 0.5:
                    self.candidates = []
                self.candidates.append(d)
                if len(self.candidates) >= c.confirm_frames:
                    self.x = np.array([d.range_m, d.velocity_mps])
                    self.P = np.diag([c.meas_sigma_r ** 2, c.meas_sigma_v ** 2]) * 4
                    self.state, self.missed, self.lost_at = "TRACKING", 0, None
                    self._emit("REACQUIRE", t_sensor, t_recv, frame, range=d.range_m)
                return
        cands = [d for d in dets if c.stop_line + 2 < d.range_m < c.capture_max and d.velocity_mps < -0.2]
        if not cands:
            self.candidates = []
            return
        d = cands[0]  # nearest approaching object
        if self.candidates and abs(d.range_m - self.candidates[-1].range_m) > 1.0:
            self.candidates = []
        self.candidates.append(d)
        if len(self.candidates) >= c.confirm_frames:
            self.x = np.array([d.range_m, d.velocity_mps])
            self.P = np.diag([c.meas_sigma_r ** 2, c.meas_sigma_v ** 2]) * 4
            self.state = "TRACKING"
            self.missed = 0
            self._emit("CAPTURE", t_sensor, t_recv, frame, range=d.range_m)

    def _associate(self, dets):
        c = self.c
        best, best_cost = None, np.inf
        for d in dets:
            dr, dv = abs(d.range_m - self.x[0]), abs(d.velocity_mps - self.x[1])
            if dr < c.gate_r and dv < c.gate_v:
                cost = dr / c.gate_r + dv / c.gate_v
                if cost < best_cost:
                    best, best_cost = d, cost
        return best

    def _docking_rules(self, frame, t_sensor, t_recv, dets, assoc):
        c = self.c
        r, v = float(self.x[0]), float(self.x[1])

        # v2: the nose is the nearest part moving with the aircraft
        if c.nose_reselect and self.state == "TRACKING" and abs(v) > c.reselect_min_speed:
            front = [d for d in dets if d is not assoc and 2.0 < d.range_m < r - 1.0
                     and abs(d.velocity_mps - v) < c.reselect_dv]
            if front:
                d = front[0]
                if self.reselect_buf and abs(d.range_m - self.reselect_buf[-1].range_m) > 1.0:
                    self.reselect_buf = []
                self.reselect_buf.append(d)
                if len(self.reselect_buf) >= c.reselect_frames:
                    self.x = np.array([d.range_m, d.velocity_mps])
                    self.P = np.diag([c.meas_sigma_r ** 2, c.meas_sigma_v ** 2]) * 4
                    self.reselect_buf, self.stop_buf = [], []
                    self._emit("NOSE_RESELECT", t_sensor, t_recv, frame, range=d.range_m, previous=r)
                    r, v, assoc = d.range_m, d.velocity_mps, d
            else:
                self.reselect_buf = []

        # obstacle: an unassociated detection in front of the aircraft
        obstacle = any(d is not assoc and 2.0 < d.range_m < r - c.obstacle_margin for d in dets)
        if obstacle:
            self.obstacle_frames += 1
            if not self.obstacle_active:
                self._emit("OBSTACLE", t_sensor, t_recv, frame)
        self.obstacle_active = obstacle

        if self.state in ("TRACKING", "OVERSHOOT"):
            if r < c.stop_line - c.overshoot and self.state != "OVERSHOOT":
                self.state = "OVERSHOOT"
                self._emit("OVERSHOOT", t_sensor, t_recv, frame, range=r)
            self.stop_buf.append((frame, t_sensor, t_recv, r, v))
            self.stop_buf = self.stop_buf[-c.stop_window:]
            if len(self.stop_buf) == c.stop_window:
                rs = [b[3] for b in self.stop_buf]
                vs = [abs(b[4]) for b in self.stop_buf]
                if max(vs) < c.stop_speed and max(rs) - min(rs) < c.stop_drift:
                    f0, ts0, tr0, _, _ = self.stop_buf[0]
                    self.stop_pos = float(np.mean(rs))
                    self.state = "PARKED"
                    t_evt = tr0 if c.stamp == "receive" else ts0
                    self.events.append(Event("AIBT", t_evt, tr0, f0,
                                             {"stop_range": self.stop_pos,
                                              "dist_to_line": self.stop_pos - c.stop_line}))
        elif self.state == "PARKED":
            self.parked_buf.append((t_sensor, t_recv, frame, r))
            if v > c.away_speed and r > self.stop_pos + c.away_dist:
                self.state = "DEPARTING"
                if c.aobt_backdate:
                    still = [b for b in self.parked_buf if b[3] - self.stop_pos <= c.backdate_tol]
                    ts0, tr0, f0, _ = still[-1] if still else (t_sensor, t_recv, frame, r)
                    t_evt = tr0 if c.stamp == "receive" else ts0
                    self.events.append(Event("AOBT", t_evt, t_recv, f0, {"range": r, "backdated": True}))
                else:
                    self._emit("AOBT", t_sensor, t_recv, frame, range=r)

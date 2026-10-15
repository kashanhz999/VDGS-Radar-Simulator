"""End-to-end run of one scenario through all five layers.

scenario (truth at 5 Hz) -> radar data (Eq. signal) -> reference processing
(range-Doppler + CFAR) -> JSON over loopback UDP -> VDGS docking logic ->
mock AODB / A-CDM. Faults are injected on the radar side or on the stream.
The simulation clock is the sensor frame time; the VDGS sees each message
at its receive time = sensor time + injected delay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .radar import RadarConfig, generate_frame
from .radar.scene import to_point_targets
from .processing.range_doppler import detect, Detection
from .interface import encode, decode, UdpLink
from .integration import DockingLogic, DockingConfig, MockAodb
from .faults import Faults
from .scenario import Scenario


@dataclass
class FrameLog:
    frame: int
    t: float
    r_true: float
    v_true: float
    dets: list[Detection]
    sensor_check_ok: bool       # a detection within one range bin of the true nose
    nose_err: float | None      # nearest detection minus true nose range
    delivered: bool
    t_recv: float | None
    actor_visible: bool


@dataclass
class RunResult:
    scenario: Scenario
    frames: list[FrameLog]
    logic: DockingLogic
    aodb: MockAodb
    udp_latency_ms: list[float] = field(default_factory=list)
    msg_bytes: list[int] = field(default_factory=list)


def run(scn: Scenario, faults: Faults | None = None, cfg: RadarConfig | None = None,
        dock: DockingConfig | None = None, seed: int = 2026, udp: bool = True) -> RunResult:
    faults = faults or scn.faults or Faults()
    cfg = cfg or RadarConfig()
    dock = dock or DockingConfig(stop_line=scn.trajectory.stop_line)
    rng = np.random.default_rng(seed)
    logic, aodb = DockingLogic(dock), MockAodb()
    link = UdpLink() if udp else None
    dt = 1.0 / cfg.frame_rate
    n = int(round(scn.duration * cfg.frame_rate)) + 1
    last_good = None
    frames, inbox = [], []

    for k in range(n):
        t = k * dt
        r_true, v_true = scn.trajectory.state(t)
        if faults.frozen is not None and faults.frozen.active(t) and last_good is not None:
            samples = last_good
        else:
            samples = generate_frame(cfg, to_point_targets(scn.scatterers(t)), snr_db=scn.snr_ref_db, rng=rng)
            last_good = samples
        dets = detect(cfg, samples)

        errs = [d.range_m - r_true for d in dets]
        nose_err = min(errs, key=abs) if errs else None
        ok = nose_err is not None and abs(nose_err) < cfg.range_resolution

        delivered = not (faults.dropout is not None and faults.dropout.dropped(t))
        t_recv = None
        if delivered:
            payload = encode(k, t, dets)
            if link is not None:
                payload = link.roundtrip(payload)
            delay = faults.latency.delay(t) if faults.latency is not None else 0.0
            t_recv = t + delay
            inbox.append((t_recv, payload))
        frames.append(FrameLog(k, t, r_true, v_true, dets, ok, nose_err, delivered, t_recv,
                               scn.actor_visible(t)))

    for t_recv, payload in sorted(inbox, key=lambda x: x[0]):
        fid, ts, dets = decode(payload)
        logic.process(fid, ts, t_recv, dets)
    for e in logic.events:
        aodb.receive(e.kind, e.t)

    res = RunResult(scn, frames, logic, aodb)
    if link is not None:
        res.udp_latency_ms, res.msg_bytes = link.latencies_ms, link.sizes
        link.close()
    return res

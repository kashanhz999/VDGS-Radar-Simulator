"""Tests for M2-M6: detection, scenarios, interface, docking logic, faults.

The scenario tests are the regression suite reported in the paper: each
encodes one acceptance criterion from the scenario catalogue.
"""
from pathlib import Path

import numpy as np
import pytest

from vdgs_sim.radar import RadarConfig, generate_frame
from vdgs_sim.radar.scene import AircraftProfile, to_point_targets
from vdgs_sim.processing.range_doppler import detect, Detection
from vdgs_sim.interface import encode, decode
from vdgs_sim.scenario import load_all, DockingTrajectory
from vdgs_sim.pipeline import run
from vdgs_sim.integration import DockingConfig
from vdgs_sim.integration.acdm import MockAodb

CFG = RadarConfig()
SCN = load_all(Path(__file__).resolve().parents[1] / "scenarios")
V2 = dict(recapture=True, nose_reselect=True, aobt_backdate=True)


def test_aircraft_scatterers_detected_with_speed():
    rng = np.random.default_rng(0)
    sc = AircraftProfile().scatterers(30.0, -1.0)
    dets = detect(CFG, generate_frame(CFG, to_point_targets(sc), snr_db=-15, rng=rng))
    assert len(dets) == 3
    for d, s in zip(dets, sc):
        assert abs(d.range_m - s.range_m) < CFG.range_resolution
        assert abs(d.velocity_mps - s.velocity_mps) < 0.1


def test_trajectory_stops_at_line():
    tr = DockingTrajectory()
    r, v = tr.state(tr.t_stop + 5)
    assert r == pytest.approx(22.0) and v == 0.0
    r0, _ = tr.state(tr.t_stop - 1e-6)
    assert r0 == pytest.approx(22.0, abs=1e-3)


def test_json_roundtrip():
    dets = [Detection(21.0123, -0.75, 31.2), Detection(32.5, -0.75, 28.0)]
    fid, ts, out = decode(encode(7, 1.4, dets))
    assert fid == 7 and ts == 1.4
    assert [d.range_m for d in out] == [21.0123, 32.5]


def test_tobt_rule():
    aodb = MockAodb()
    assert aodb.tobt(30.0) == 2760.0           # 30 s + 45 min -> next whole minute
    assert aodb.tobt(-1000.0) == aodb.cfg.sobt_s  # SOBT binds when AIBT + MTT is earlier


def _aibt(res):
    return [e for e in res.logic.events if e.kind == "AIBT"]


def _ok(res, scn, tol_t=1.0, tol_r=0.10):
    a = _aibt(res)
    return (len(a) == 1 and abs(a[0].t - scn.trajectory.t_stop) <= tol_t
            and abs(a[0].info["stop_range"] - scn.trajectory.stop_range) <= tol_r)


@pytest.mark.parametrize("sid", ["S3", "S4", "S8"])
def test_nominal_aibt(sid):
    scn = SCN[sid]
    assert _ok(run(scn, dock=DockingConfig(stop_line=22.0, **V2), udp=False), scn)


def test_overshoot_raised():
    scn = SCN["S5"]
    res = run(scn, dock=DockingConfig(stop_line=22.0, **V2), udp=False)
    assert any(e.kind == "OVERSHOOT" for e in res.logic.events) and _ok(res, scn)


@pytest.mark.parametrize("sid", ["S6", "S7"])
def test_obstacle_raised(sid):
    res = run(SCN[sid], dock=DockingConfig(stop_line=22.0, **V2), udp=False)
    assert any(e.kind == "OBSTACLE" for e in res.logic.events)


def test_dropout_v1_loses_aibt_v2_recovers():
    scn = SCN["S9"]
    assert _aibt(run(scn, dock=DockingConfig(stop_line=22.0), udp=False)) == []
    assert _ok(run(scn, dock=DockingConfig(stop_line=22.0, **V2), udp=False), scn)


def test_latency_policy():
    scn = SCN["S10"]
    a_recv = _aibt(run(scn, dock=DockingConfig(stop_line=22.0, stamp="receive"), udp=False))[0].t
    a_sens = _aibt(run(scn, dock=DockingConfig(stop_line=22.0, stamp="sensor"), udp=False))[0].t
    assert a_recv - a_sens == pytest.approx(0.5)


def test_pushback_aobt_v2():
    scn = SCN["S12"]
    res = run(scn, dock=DockingConfig(stop_line=22.0, **V2), udp=False)
    aobt = [e for e in res.logic.events if e.kind == "AOBT"]
    assert len(aobt) == 1 and abs(aobt[0].t - scn.trajectory.t_pushback) <= 1.0


@pytest.mark.xfail(strict=True, reason="open issue: stationary ghost at the stop line (Sec. VII)")
def test_ghost_echo():
    scn = SCN["S11"]
    assert _ok(run(scn, dock=DockingConfig(stop_line=22.0, **V2), udp=False), scn)

"""All end-to-end experiments (RQ2, RQ3). Every number and figure in
Sections VI-C to VI-F of the paper comes from this script.

Run:  python -m experiments.run_all
Out:  experiments/results_all.json, figures/*.pdf (via make_figures.py)
"""
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vdgs_sim.radar import RadarConfig, generate_frame
from vdgs_sim.radar.scene import to_point_targets
from vdgs_sim.processing.range_doppler import range_doppler_map, velocity_axis, detect
from vdgs_sim.processing.range_fft import range_axis
from vdgs_sim.scenario import load_all
from vdgs_sim.pipeline import run
from vdgs_sim.integration import DockingConfig
from vdgs_sim.integration.acdm import MockAodb
from vdgs_sim.faults import Faults, StreamLatency

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
CFG = RadarConfig()
SEED = 2026
SCN = load_all(ROOT / "scenarios")

INK, MUTED, GRID = "#1f2328", "#57606a", "#d0d7de"
C1, C2, C3 = "#2a5bd7", "#c4561d", "#6e7781"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.bbox": "tight",
})
COL_W = 3.5


V2 = {"recapture": True, "nose_reselect": True, "aobt_backdate": True}


def dock_cfg(scn, **kw):
    return DockingConfig(stop_line=scn.trajectory.stop_line, **kw)


def metrics(res):
    scn, tr = res.scenario, res.scenario.trajectory
    ev = res.logic.events
    out = {"events": [{"kind": e.kind, "t": round(e.t, 3), **{k: (round(v, 4) if isinstance(v, float) else v)
                                                              for k, v in e.info.items()}} for e in ev]}
    out["sensor_check_pct"] = 100 * float(np.mean([f.sensor_check_ok for f in res.frames]))
    out["frames"] = len(res.frames)
    out["delivered"] = int(sum(f.delivered for f in res.frames))

    # track accuracy against truth
    er, ev_ = [], []
    for (fid, ts, trc, r, v, st, obs) in res.logic.history:
        if r is None or st not in ("TRACKING", "OVERSHOOT", "PARKED", "DEPARTING"):
            continue
        rt, vt = tr.state(ts)
        er.append(r - rt)
        ev_.append(v - vt)
    out["track_range_rmse_cm"] = float(np.sqrt(np.mean(np.square(er))) * 100) if er else None
    out["track_speed_rmse_mps"] = float(np.sqrt(np.mean(np.square(ev_)))) if ev_ else None

    caps = [e for e in ev if e.kind == "CAPTURE"]
    if caps:
        out["capture_t"] = caps[0].t
        out["capture_range_true"] = float(tr.state(caps[0].t)[0]) if res.scenario.faults is None or True else None

    aibt = [e for e in ev if e.kind == "AIBT"]
    out["aibt_count"] = len(aibt)
    out["aibt_true"] = tr.t_stop
    if aibt:
        a = aibt[0]
        out["aibt_err_s"] = a.t - tr.t_stop
        out["stop_pos_err_m"] = a.info["stop_range"] - tr.stop_range
        out["reported_dist_to_line_m"] = a.info["dist_to_line"]
        aodb = MockAodb()
        out["tobt_reported_s"] = aodb.tobt(a.t)
        out["tobt_true_s"] = aodb.tobt(tr.t_stop)
        out["tobt_change_min"] = (out["tobt_reported_s"] - out["tobt_true_s"]) / 60
    out["overshoot_raised"] = any(e.kind == "OVERSHOOT" for e in ev)
    out["track_lost"] = sum(e.kind == "TRACK_LOST" for e in ev)
    aobt = [e for e in ev if e.kind == "AOBT"]
    if aobt and tr.t_pushback is not None:
        out["aobt_err_s"] = aobt[0].t - tr.t_pushback

    # obstacle alarms vs actor visibility (matched by frame)
    vis = {f.frame: f.actor_visible for f in res.frames}
    hist = {h[0]: h[6] for h in res.logic.history}
    vis_frames = [k for k, v in vis.items() if v]
    if vis_frames:
        flagged = [hist.get(k, False) for k in vis_frames]
        out["obstacle_detect_pct"] = 100 * float(np.mean(flagged))
        first_vis = min(vis_frames) / CFG.frame_rate
        obs_ev = [e for e in ev if e.kind == "OBSTACLE"]
        out["obstacle_first_delay_s"] = (obs_ev[0].t - first_vis) if obs_ev else None
    false_obs = [k for k, v in hist.items() if v and not vis.get(k, False)]
    out["obstacle_false_frames"] = len(false_obs)
    if res.udp_latency_ms:
        out["udp_latency_ms_median"] = float(np.median(res.udp_latency_ms))
        out["udp_latency_ms_p99"] = float(np.percentile(res.udp_latency_ms, 99))
        out["msg_bytes_median"] = float(np.median(res.msg_bytes))
    return out


def passes(sid, m):
    """Acceptance criteria per scenario, fixed before running."""
    one_aibt = m["aibt_count"] == 1 and abs(m.get("aibt_err_s", 99)) <= 1.0 and abs(m.get("stop_pos_err_m", 99)) <= 0.10
    if sid in ("S3", "S4", "S8", "S9", "S10", "S11", "S13"):
        return one_aibt and not m["overshoot_raised"]
    if sid == "S5":
        return one_aibt and m["overshoot_raised"]
    if sid in ("S6", "S7"):
        return one_aibt and (m.get("obstacle_first_delay_s") is not None and m["obstacle_first_delay_s"] <= 1.0)
    if sid == "S12":
        return one_aibt and m.get("aobt_err_s") is not None and abs(m["aobt_err_s"]) <= 1.0
    return False


def main():
    results = {"seed": SEED, "scenarios": {}, "titles": {k: v.title for k, v in SCN.items()}}
    wall = []
    keep = {}
    for sid, scn in SCN.items():
        row = {}
        for ver, kw in (("v1", {}), ("v2", V2)):
            t0 = time.perf_counter()
            res = run(scn, dock=dock_cfg(scn, **kw), seed=SEED, udp=(ver == "v1"))
            wall.append((time.perf_counter() - t0) / len(res.frames) * 1000)
            m = metrics(res)
            m["pass"] = passes(sid, m)
            row[ver] = m
            keep[(sid, ver)] = res
        results["scenarios"][sid] = row
        print(sid, "v1", row["v1"]["pass"], "v2", row["v2"]["pass"])
    results["pipeline_ms_per_frame_mean"] = float(np.mean(wall))

    # --- timestamp policy under latency (S10 and a sweep) --------------------
    lat = {}
    for delay in (0.0, 0.2, 0.5, 1.0, 2.0):
        for stamp in ("receive", "sensor"):
            scn = SCN["S3"]
            res = run(scn, faults=Faults(latency=StreamLatency(delay)), dock=dock_cfg(scn, stamp=stamp), seed=SEED, udp=False)
            m = metrics(res)
            lat[f"{delay}_{stamp}"] = {"delay_s": delay, "stamp": stamp, "aibt_err_s": m.get("aibt_err_s"),
                                     "tobt_change_min": m.get("tobt_change_min")}
    results["latency_sweep"] = lat

    # --- Monte Carlo on nominal approaches ------------------------------------
    mc = {}
    for sid in ("S3", "S4"):
        scn = SCN[sid]
        a, p, rr, vv = [], [], [], []
        for s in range(20):
            m = metrics(run(scn, dock=dock_cfg(scn), seed=1000 + s, udp=False))
            a.append(m["aibt_err_s"]); p.append(m["stop_pos_err_m"])
            rr.append(m["track_range_rmse_cm"]); vv.append(m["track_speed_rmse_mps"])
        mc[sid] = {"runs": 20,
                   "aibt_err_mean_s": float(np.mean(a)), "aibt_err_std_s": float(np.std(a)),
                   "aibt_err_maxabs_s": float(np.max(np.abs(a))),
                   "stop_pos_err_mean_cm": float(np.mean(p) * 100), "stop_pos_err_maxabs_cm": float(np.max(np.abs(p)) * 100),
                   "track_range_rmse_cm_mean": float(np.mean(rr)), "track_speed_rmse_mps_mean": float(np.mean(vv))}
    results["monte_carlo"] = mc

    # --- SNR sweep on the nominal approach --------------------------------------
    sweep = []
    for snr in (-15, -20, -25, -30, -33, -36, -39):
        caps, ok, aerr = [], 0, []
        ok_v1 = 0
        for s in range(10):
            scn1 = load_all(ROOT / "scenarios")["S3"]
            scn1.snr_ref_db = float(snr)
            m1 = metrics(run(scn1, dock=dock_cfg(scn1), seed=3000 + s, udp=False))
            if m1["aibt_count"] == 1 and abs(m1.get("aibt_err_s", 99)) <= 1 and abs(m1.get("stop_pos_err_m", 99)) <= 0.1:
                ok_v1 += 1
            scn = load_all(ROOT / "scenarios")["S3"]
            scn.snr_ref_db = float(snr)
            res = run(scn, dock=dock_cfg(scn, **V2), seed=3000 + s, udp=False)
            m = metrics(res)
            if "capture_t" in m:
                caps.append(scn.trajectory.state(m["capture_t"])[0])
            if m["aibt_count"] == 1 and abs(m.get("aibt_err_s", 99)) <= 1 and abs(m.get("stop_pos_err_m", 99)) <= 0.1:
                ok += 1
                aerr.append(m["aibt_err_s"])
        sweep.append({"snr_ref_db": snr, "runs": 10, "captured": len(caps),
                      "capture_range_median": float(np.median(caps)) if caps else None,
                      "capture_range_min": float(np.min(caps)) if caps else None,
                      "capture_range_max": float(np.max(caps)) if caps else None,
                      "aibt_ok": ok, "aibt_ok_v1": ok_v1})
        print("snr", snr, sweep[-1])
    results["snr_sweep"] = sweep

    (ROOT / "experiments" / "results_all.json").write_text(json.dumps(results, indent=2, default=float))

    from experiments.make_figures import main as make_figures
    make_figures()

    print(json.dumps({k: results[k] for k in ("pipeline_ms_per_frame_mean", "monte_carlo", "latency_sweep")}, indent=1))


if __name__ == "__main__":
    main()

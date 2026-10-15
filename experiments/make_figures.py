"""Figures for Sections VI-C to VI-F. Reads experiments/results_all.json and
re-runs the three cheap cases it plots (same seed as run_all.py).

Run: python -m experiments.make_figures
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vdgs_sim.radar import RadarConfig, generate_frame
from vdgs_sim.radar.scene import to_point_targets
from vdgs_sim.processing.range_doppler import range_doppler_map, velocity_axis
from vdgs_sim.processing.range_fft import range_axis
from vdgs_sim.scenario import load_all
from vdgs_sim.pipeline import run
from vdgs_sim.integration import DockingConfig

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
CFG = RadarConfig()
SEED = 2026
V2 = {"recapture": True, "nose_reselect": True, "aobt_backdate": True}

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


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", dpi=300)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_rd_map(scn):
    t = 11.2
    rng = np.random.default_rng(SEED)
    fr = generate_frame(CFG, to_point_targets(scn.scatterers(t)), snr_db=scn.snr_ref_db, rng=rng)
    p = 10 * np.log10(range_doppler_map(CFG, fr))
    p -= p.max()
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    im = ax.pcolormesh(range_axis(CFG), velocity_axis(CFG), p, cmap="Blues", vmin=-50, vmax=0,
                       shading="auto", rasterized=True)
    ax.grid(False)
    ax.set_xlim(5, 60); ax.set_ylim(-4, 4)
    ax.set_xlabel("Range (m)"); ax.set_ylabel("Radial speed (m/s)")
    cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("Power (dB)"); cb.outline.set_visible(False)
    r_ac, v_ac = scn.trajectory.state(t)
    for off, lab, below in ((0, "nose", True), (5, "gear", False), (11.5, "engines", True)):
        y_mark = v_ac - 0.35 if below else v_ac + 0.35
        y_txt = -3.5 if below else -0.2
        ax.annotate(lab, xy=(r_ac + off, y_mark), xytext=(r_ac + off, y_txt), ha="center", fontsize=6.5,
                    color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    rv, vv, _ = scn.actors[0].state(t)
    ax.annotate("vehicle", xy=(rv + 0.4, vv), xytext=(rv + 4, 2.9), fontsize=6.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    save(fig, "rd_map_s6")


def fig_tracks(scn3, scn9):
    runs = {
        ("S3", "v1"): run(scn3, dock=DockingConfig(stop_line=22.0), seed=SEED, udp=False),
        ("S9", "v1"): run(scn9, dock=DockingConfig(stop_line=22.0), seed=SEED, udp=False),
        ("S9", "v2"): run(scn9, dock=DockingConfig(stop_line=22.0, **V2), seed=SEED, udp=False),
    }
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9), sharey=True)
    tt = np.linspace(0, 40, 400)

    def track(res):
        h = [(x[1], x[3]) for x in res.logic.history if x[3] is not None]
        return [q[0] for q in h], [q[1] for q in h]

    def aibt(res):
        a = [e.t for e in res.logic.events if e.kind == "AIBT"]
        return a[0] if a else None

    for ax, sid in zip(axes, ("S3", "S9")):
        tr = (scn3 if sid == "S3" else scn9).trajectory
        ax.plot(tt, [tr.state(x)[0] for x in tt], color=C3, lw=2.2, alpha=0.35, label="truth")
        ax.axhline(tr.stop_line, color=MUTED, lw=0.5, ls=":")
        ax.set_xlim(0, 40); ax.set_ylim(15, 62); ax.set_xlabel("Time (s)")

    # (a) nominal
    x, y = track(runs[("S3", "v1")])
    axes[0].plot(x, y, ls="none", marker="o", ms=1.3, color=C1, label="track")
    axes[0].axvline(aibt(runs[("S3", "v1")]), color=C1, lw=0.7, ls="--")
    axes[0].text(aibt(runs[("S3", "v1")]) - 0.8, 50, "AIBT", ha="right", fontsize=6.5, color=C1)
    axes[0].set_title("(a) S3 nominal", fontsize=7.5, color=INK)
    axes[0].legend(fontsize=6, loc="lower left", markerscale=3, handlelength=1.2)
    axes[0].set_ylabel("Nose range (m)")

    # (b) dropout: v2 underneath, v1 on top so its end is visible
    ax = axes[1]
    ax.axvspan(25, 27, color=GRID, alpha=0.7, lw=0)
    ax.text(26, 56, "stream\nlost", ha="center", va="top", fontsize=6, color=MUTED)
    x2, y2 = track(runs[("S9", "v2")])
    x1, y1 = track(runs[("S9", "v1")])
    ax.plot(x2, y2, ls="none", marker="s", ms=1.6, color=C2, label="v2 track")
    ax.plot(x1, y1, ls="none", marker="o", ms=1.0, color=C1, label="v1 track")
    a2 = aibt(runs[("S9", "v2")])
    ax.axvline(a2, color=C2, lw=0.7, ls="--")
    ax.text(a2 + 0.8, 40, "AIBT (v2)", fontsize=6.5, color=C2)
    ax.text(a2 + 0.8, 34, "v1: none", fontsize=6.5, color=C1)
    ax.set_title("(b) S9 stream dropout", fontsize=7.5, color=INK)
    ax.legend(fontsize=6, loc="lower left", markerscale=3, handlelength=1.2)
    save(fig, "docking_tracks")


def fig_capture(sweep):
    pts = [s for s in sweep if s["capture_range_median"] is not None]
    xs = [s["snr_ref_db"] for s in pts]
    med = [s["capture_range_median"] for s in pts]
    lo = [s["capture_range_median"] - s["capture_range_min"] for s in pts]
    hi = [s["capture_range_max"] - s["capture_range_median"] for s in pts]
    fig, ax = plt.subplots(figsize=(COL_W, 1.85))
    ax.errorbar(xs, med, yerr=[lo, hi], color=C1, lw=1.0, marker="o", ms=3, capsize=2, elinewidth=0.7)
    ax.axhline(22.0, color=MUTED, lw=0.6, ls=":")
    ax.text(-14.6, 23, "stop line", ha="right", fontsize=6.5, color=MUTED)
    for s in sweep:
        if 0 < s["captured"] < s["runs"]:
            ax.annotate(f"{s['captured']}/{s['runs']} runs captured", xy=(s["snr_ref_db"], s["capture_range_median"]),
                        xytext=(s["snr_ref_db"] + 0.3, 36), fontsize=6.5, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
        elif s["captured"] == 0:
            ax.text(s["snr_ref_db"] - 1.2, 52, f"\u2212{abs(s['snr_ref_db'])} dB:\nno capture\nin {s['runs']} runs",
                    ha="left", va="top", fontsize=6.5, color=INK)
    ax.set_xlim(-40.5, -14)
    ax.set_xlabel("Reference SNR (dB per sample, 1 m$^2$ at 20 m)")
    ax.set_ylabel("Capture range (m)")
    ax.set_ylim(15, 65)
    save(fig, "capture_vs_snr")


def main():
    res = json.loads((ROOT / "experiments" / "results_all.json").read_text())
    scn = load_all(ROOT / "scenarios")
    fig_rd_map(scn["S6"])
    fig_tracks(scn["S3"], scn["S9"])
    fig_capture(res["snr_sweep"])


if __name__ == "__main__":
    main()

"""M1 experiments: every number and figure in the paper's range-accuracy
subsection comes from this script. Run: python -m experiments.run_m1
Outputs: figures/range_profile_21m.pdf, figures/range_rmse_vs_snr.pdf,
         experiments/results_m1.json
"""
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vdgs_sim.radar import RadarConfig, PointTarget, generate_frame
from vdgs_sim.processing import range_axis, range_profile, estimate_range

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
CFG = RadarConfig()
SEED = 2026

INK, MUTED, GRID = "#1f2328", "#57606a", "#d0d7de"
C1, C2 = "#2a5bd7", "#c4561d"          # two series, distinct in grayscale too

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.bbox": "tight",
})
COL_W = 3.5  # IEEE single-column width, inches


def stats(err):
    err = np.asarray(err)
    return {
        "mae_cm": float(np.mean(np.abs(err)) * 100),
        "rmse_cm": float(np.sqrt(np.mean(err ** 2)) * 100),
        "max_cm": float(np.max(np.abs(err)) * 100),
        "within_bin_pct": float(np.mean(np.abs(err) < CFG.range_resolution) * 100),
    }


def main():
    rng = np.random.default_rng(SEED)
    results = {"config": {
        "f_carrier_GHz": CFG.f_carrier / 1e9, "bandwidth_MHz": CFG.bandwidth / 1e6,
        "t_chirp_us": CFG.t_chirp * 1e6, "n_samples": CFG.n_samples,
        "n_chirps": CFG.n_chirps, "frame_rate_Hz": CFG.frame_rate,
        "slope_MHz_per_us": CFG.slope / 1e12, "fs_MHz": CFG.fs / 1e6,
        "range_resolution_m": CFG.range_resolution, "max_range_m": CFG.max_range,
        "max_velocity_mps": CFG.max_velocity, "velocity_resolution_mps": CFG.velocity_resolution,
        "seed": SEED,
    }}

    # --- S2: 21 m, one generation -----------------------------------------
    f21 = generate_frame(CFG, [PointTarget(21.0)], snr_db=0.0, rng=rng)
    results["s2_21m"] = {"commanded_m": 21.0, "estimated_m": estimate_range(CFG, f21), "attempts": 1}

    # --- S1: dense sweep 5-60 m, clean and noisy ----------------------------
    ranges = np.round(np.arange(5.0, 60.0001, 0.05), 4)
    err_clean = np.array([estimate_range(CFG, generate_frame(CFG, [PointTarget(r)])) - r for r in ranges])
    results["s1_clean"] = {"n": int(len(ranges)), **stats(err_clean)}

    snr_list = [10, 0, -10, -20, -25, -30]
    trials = 10
    sweep_coarse = np.arange(5.0, 60.0001, 1.0)
    results["s1_snr"] = {}
    err_0db = None
    for snr in snr_list:
        errs = []
        for r in sweep_coarse:
            for _ in range(trials):
                fr = generate_frame(CFG, [PointTarget(r)], snr_db=snr, rng=rng)
                errs.append(estimate_range(CFG, fr) - r)
        results["s1_snr"][str(snr)] = {"n": len(errs), **stats(errs)}
    err_0db = np.array([estimate_range(CFG, generate_frame(CFG, [PointTarget(r)], snr_db=0.0, rng=rng)) - r
                        for r in ranges])
    results["s1_0db_dense"] = {"n": int(len(ranges)), **stats(err_0db)}

    # --- SNR breakdown curve (Figure 2) -------------------------------------
    snr_grid = np.arange(-34, 10.1, 2)
    curve = []
    for snr in snr_grid:
        errs = []
        for r in sweep_coarse:
            for _ in range(5):
                fr = generate_frame(CFG, [PointTarget(r)], snr_db=float(snr), rng=rng)
                errs.append(estimate_range(CFG, fr) - r)
        curve.append({"snr_db": float(snr), **stats(errs)})
    results["snr_curve"] = curve

    # --- Timing: generate + process one full frame vs 200 ms budget ---------
    n_rep = 200
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fr = generate_frame(CFG, [PointTarget(30.0)], snr_db=0.0, rng=rng)
        estimate_range(CFG, fr)
    per_frame_ms = (time.perf_counter() - t0) / n_rep * 1000
    results["timing"] = {"per_frame_ms": per_frame_ms, "budget_ms": 1000 / CFG.frame_rate}

    (ROOT / "experiments" / "results_m1.json").write_text(json.dumps(results, indent=2))

    # --- Figure 1: range profile at 21 m ------------------------------------
    r_ax = range_axis(CFG)
    p = range_profile(CFG, f21)
    p_db = 10 * np.log10(p / p.max())
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    ax.plot(r_ax, p_db, color=C1, lw=1.0)
    ax.axvline(21.0, color=MUTED, lw=0.7, ls="--")
    ax.annotate("commanded 21 m", xy=(21.0, -2), xytext=(27, -8), fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlim(0, CFG.max_range)
    ax.set_ylim(-70, 3)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Normalised power (dB)")
    fig.savefig(FIG / "range_profile_21m.pdf")
    fig.savefig(FIG / "range_profile_21m.png", dpi=200)
    plt.close(fig)

    # --- Figure 2: range RMSE vs per-sample SNR -----------------------------
    xs = [c["snr_db"] for c in curve]
    ys = [c["rmse_cm"] for c in curve]
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    ax.plot(xs, ys, color=C1, lw=1.0, marker="o", ms=3)
    bin_cm = CFG.range_resolution * 100
    ax.axhline(bin_cm, color=MUTED, lw=0.7, ls="--")
    ax.text(9.5, bin_cm * 1.25, "one range bin (30 cm)", ha="right", va="bottom", fontsize=7, color=MUTED)
    ax.set_yscale("log")
    ax.set_xlim(-35, 11)
    ax.set_xlabel("Per-sample SNR (dB)")
    ax.set_ylabel("Range RMSE (cm)")
    fig.savefig(FIG / "range_rmse_vs_snr.pdf")
    fig.savefig(FIG / "range_rmse_vs_snr.png", dpi=200)
    plt.close(fig)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

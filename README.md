# vdgs-radar-sim

Physics-based synthetic FMCW radar data for evaluating visual docking guidance systems (VDGS) in low visibility, with an end-to-end test chain from radar samples to A-CDM milestones.

Companion code for the paper *Generating Synthetic FMCW Radar Data for Evaluating Visual Docking Guidance Systems in Low Visibility* (K. Haider).

## Setup

```bash
pip install -e ".[dev]"     # numpy, scipy, matplotlib, pyyaml, pytest
```

## Run

```bash
pytest -q                          # 76 tests (75 pass, 1 expected failure: ghost echo, S11)
python -m experiments.run_m1       # Sec. VI-B: static range accuracy  -> results_m1.json, 2 figures
python -m experiments.run_all      # Sec. VI-C..G: scenarios, faults, SNR sweep (~15-20 min) -> results_all.json
python -m experiments.make_figures # redraw the scenario figures from results_all.json
```

All runs use fixed seeds, so every number in the paper is reproduced exactly.

## Layout

| Folder | Layer | Contents |
|---|---|---|
| `scenarios/` | 1 | One YAML file per scenario (S3-S13), incl. faults |
| `vdgs_sim/scenario/` | 1 | Docking kinematics, crossing actors, ghost reflectors, YAML loader |
| `vdgs_sim/radar/` | 2 | Radar parameters, FMCW sample generator, RCS and path loss, aircraft model |
| `vdgs_sim/processing/` | 3 | Range FFT; range-Doppler map, 2D CA-CFAR, clustering |
| `vdgs_sim/interface/` | 4 | JSON message format, loopback UDP link |
| `vdgs_sim/integration/` | 4-5 | Reference docking logic (v1/v2 rules), mock AODB with TOBT rule |
| `vdgs_sim/faults/` | - | Frozen output, stream latency, stream dropout |
| `vdgs_sim/pipeline.py` | 1-5 | Runs one scenario end to end |
| `experiments/` | - | Scripts that produce every number and figure in the paper |
| `figures/` | - | Generated figures |
| `paper/` | - | LaTeX source (IEEE conference format) |

## Docking logic versions

- **v1**: capture, association, overshoot, obstacle, on-block (AIBT), off-block (AOBT).
- **v2** = v1 + re-acquisition after track loss + nose re-selection + AOBT backdating.
  Enable with `DockingConfig(recapture=True, nose_reselect=True, aobt_backdate=True)`.

## Known open issue

S11: a strong stationary echo near the stop line merges with the nose in CFAR clustering, so the nose detection disappears. Tracked by the `xfail` test `test_ghost_echo`.
# VDGS-Radar-Simulator

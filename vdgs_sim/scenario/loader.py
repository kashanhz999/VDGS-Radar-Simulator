"""Load scenarios from YAML files (one file per scenario)."""
from __future__ import annotations

from pathlib import Path
import yaml

from .model import Scenario, DockingTrajectory, LateralActor, StaticReflector
from ..faults import Faults, FrozenOutput, StreamLatency, StreamDropout


def load(path: str | Path) -> Scenario:
    d = yaml.safe_load(Path(path).read_text())
    return Scenario(
        sid=d["id"],
        title=d["title"],
        trajectory=DockingTrajectory(**d.get("trajectory", {})),
        duration=float(d["duration"]),
        actors=[LateralActor(**a) for a in d.get("actors", [])],
        reflectors=[StaticReflector(**g) for g in d.get("reflectors", [])],
        snr_ref_db=float(d.get("snr_ref_db", -15.0)),
        faults=_faults(d.get("faults", {})),
    )


def _faults(f: dict) -> Faults:
    return Faults(
        frozen=FrozenOutput(**f["frozen"]) if "frozen" in f else None,
        latency=StreamLatency(**f["latency"]) if "latency" in f else None,
        dropout=StreamDropout(**f["dropout"]) if "dropout" in f else None,
    )


def load_all(folder: str | Path) -> dict[str, Scenario]:
    return {s.sid: s for s in (load(p) for p in sorted(Path(folder).glob("*.yaml")))}

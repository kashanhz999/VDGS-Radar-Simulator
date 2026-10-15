import numpy as np
import pytest

from vdgs_sim.radar import RadarConfig, PointTarget, generate_frame
from vdgs_sim.processing import estimate_range

CFG = RadarConfig()


def test_derived_parameters():
    assert CFG.range_resolution == pytest.approx(0.2998, abs=1e-3)
    assert CFG.max_range > 60.0                  # covers the 5-60 m docking zone
    assert CFG.max_velocity > 10.0               # covers taxi-in speeds


def test_21m_first_try():
    """S2: the base thesis needed ~42,000 tries for 21 m; we need one."""
    frame = generate_frame(CFG, [PointTarget(21.0)])
    assert abs(estimate_range(CFG, frame) - 21.0) < CFG.range_resolution


@pytest.mark.parametrize("r", np.arange(5.0, 61.0, 1.0))
def test_range_sweep_clean(r):
    """S1: every commanded range from 5 to 60 m within one range bin."""
    frame = generate_frame(CFG, [PointTarget(float(r))])
    assert abs(estimate_range(CFG, frame) - r) < CFG.range_resolution


@pytest.mark.parametrize("r", [8.0, 21.0, 45.0, 60.0])
def test_range_with_noise(r):
    rng = np.random.default_rng(42)
    frame = generate_frame(CFG, [PointTarget(r)], snr_db=0.0, rng=rng)
    assert abs(estimate_range(CFG, frame) - r) < CFG.range_resolution

"""Radar parameters for the synthetic FMCW generator.

All values are design choices for simulation, not the parameters of any
commercial product. Derived quantities are computed, never typed by hand,
so the paper table and the code can never disagree.
"""
from dataclasses import dataclass

C = 299_792_458.0  # speed of light, m/s


@dataclass(frozen=True)
class RadarConfig:
    f_carrier: float = 77e9        # Hz
    bandwidth: float = 500e6       # Hz, swept per chirp
    t_chirp: float = 50e-6         # s, ramp (sampling) time
    n_samples: int = 256           # complex ADC samples per chirp
    n_chirps: int = 128            # chirps per frame
    frame_rate: float = 5.0        # Hz

    @property
    def slope(self) -> float:
        """Chirp slope S in Hz/s."""
        return self.bandwidth / self.t_chirp

    @property
    def fs(self) -> float:
        """ADC sample rate in Hz (samples spread over the ramp)."""
        return self.n_samples / self.t_chirp

    @property
    def wavelength(self) -> float:
        return C / self.f_carrier

    @property
    def range_resolution(self) -> float:
        """c / 2B, metres."""
        return C / (2 * self.bandwidth)

    @property
    def max_range(self) -> float:
        """Unambiguous range for complex sampling: fs * c / (2 S), metres."""
        return self.fs * C / (2 * self.slope)

    @property
    def max_velocity(self) -> float:
        """lambda / (4 Tc), m/s."""
        return self.wavelength / (4 * self.t_chirp)

    @property
    def velocity_resolution(self) -> float:
        """lambda / (2 N Tc), m/s."""
        return self.wavelength / (2 * self.n_chirps * self.t_chirp)

    def beat_frequency(self, r: float) -> float:
        """f_b = 2 S R / c, Hz."""
        return 2 * self.slope * r / C

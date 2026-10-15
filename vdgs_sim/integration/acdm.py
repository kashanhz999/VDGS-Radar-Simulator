"""Mock AODB / A-CDM consumer of VDGS events.

Rule used for the target off-block time (as published for Singapore Changi):
TOBT = max(SOBT, AIBT + MTT), where MTT is the minimum turnaround time.
A-CDM times are exchanged at one-minute resolution; here TOBT is rounded up
to the next whole minute. Times are seconds from the scenario epoch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class AcdmConfig:
    mtt_s: float = 45 * 60
    sobt_s: float = 40 * 60     # scheduled off-block; the flight is late, so AIBT + MTT binds


@dataclass
class MockAodb:
    cfg: AcdmConfig = field(default_factory=AcdmConfig)
    aibt: list[float] = field(default_factory=list)
    aobt: list[float] = field(default_factory=list)

    def receive(self, kind: str, t: float):
        if kind == "AIBT":
            self.aibt.append(t)
        elif kind == "AOBT":
            self.aobt.append(t)

    def tobt(self, aibt: float | None = None) -> float | None:
        a = aibt if aibt is not None else (self.aibt[0] if self.aibt else None)
        if a is None:
            return None
        t = max(self.cfg.sobt_s, a + self.cfg.mtt_s)
        return math.ceil(t / 60.0) * 60.0

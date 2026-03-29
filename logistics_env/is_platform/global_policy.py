"""
GlobalPolicy — Global policy of the MAS-DUO system
===================================================
Implements Section 3.6 of the thesis: "Global Policy: Static and Dynamic".

The global parameters (A, B, C, D) weight the 4 reward dimensions:
    R(s, s') = A·Delay(s,s') + B·Cost(s,s') + C·QoS(s,s') + D·Energy(s,s')

The policy can be STATIC (does not change during simulation) or DYNAMIC
(parameters are adjusted in real time by the IS Platform).

Equation 9 of the thesis:
    GlobalPolicyParameters in time T  = (A,  B,  C,  D)
    GlobalPolicyParameters in time T' = (A', B', C', D')

Static mode  → useful for training, simulation, and initial deployments.
Dynamic mode → useful in highly dynamic environments with RFID visibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# PolicyMode — Policy mode enumeration
# ---------------------------------------------------------------------------

class PolicyMode(str, Enum):
    """
    Policy modes (Section 3.6 of the thesis).

    STATIC  → fixed parameters throughout the entire simulation.
              Useful for training and deployments without RFID.
    DYNAMIC → parameters change in response to environment events
              and negotiation in the IS Platform.
    """
    STATIC  = "static"
    DYNAMIC = "dynamic"


# ---------------------------------------------------------------------------
# PolicyParameters — policy snapshot at a given moment
# ---------------------------------------------------------------------------

@dataclass
class PolicyParameters:
    """
    Policy parameters at a given simulation instant.

    A → Delay   weight (time)
    B → Cost    weight (economic)
    C → QoS     weight (quality of service)
    D → Energy  weight (energy sustainability)

    Convention: weights are positive; higher values = more importance.
    """
    A: float = 1.0   # Delay
    B: float = 1.0   # Cost
    C: float = 1.0   # QoS
    D: float = 0.5   # Energy

    step: int = 0    # simulation step at which this parameters apply

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.A, self.B, self.C, self.D)

    def normalized(self) -> "PolicyParameters":
        """Normalises the weights so they sum to 1."""
        total = self.A + self.B + self.C + self.D
        if total == 0:
            return PolicyParameters(A=0.25, B=0.25, C=0.25, D=0.25, step=self.step)
        return PolicyParameters(
            A=self.A / total,
            B=self.B / total,
            C=self.C / total,
            D=self.D / total,
            step=self.step,
        )

    def to_dict(self) -> dict:
        return {"A": self.A, "B": self.B, "C": self.C, "D": self.D, "step": self.step}

    def __repr__(self) -> str:
        return f"Policy(A={self.A:.2f}, B={self.B:.2f}, C={self.C:.2f}, D={self.D:.2f})"


# ---------------------------------------------------------------------------
# GlobalPolicy — global policy manager
# ---------------------------------------------------------------------------

class GlobalPolicy:
    """
    Manages the global policy of the MAS-DUO system.

    In STATIC mode, parameters are fixed from the configuration.
    In DYNAMIC mode, the IS Platform may modify them at any step.

    The policy is unique and shared by all Physical BDI Agents
    (pre-assumption of Section 3.4: "equal decision parameters
    for all agents").
    """

    def __init__(
        self,
        mode:    PolicyMode = PolicyMode.STATIC,
        initial: Optional[PolicyParameters] = None,
    ):
        self.mode: PolicyMode = mode
        self._current: PolicyParameters = initial or PolicyParameters()
        self._history: List[PolicyParameters] = [self._current]

        # Scheduled policy change events (dynamic mode)
        # List of (step, PolicyParameters)
        self._scheduled_changes: List[Tuple[int, PolicyParameters]] = []

    # -----------------------------------------------------------------------
    # Access to current parameters
    # -----------------------------------------------------------------------

    @property
    def A(self) -> float: return self._current.A
    @property
    def B(self) -> float: return self._current.B
    @property
    def C(self) -> float: return self._current.C
    @property
    def D(self) -> float: return self._current.D
    @property
    def params(self) -> PolicyParameters: return self._current
    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self._current.as_tuple()

    # -----------------------------------------------------------------------
    # Policy modification (IS Platform — Sec 3.5)
    # -----------------------------------------------------------------------

    def update(
        self,
        step:   int,
        A:      Optional[float] = None,
        B:      Optional[float] = None,
        C:      Optional[float] = None,
        D:      Optional[float] = None,
        reason: str             = "",
    ) -> PolicyParameters:
        """
        Updates policy parameters (only in DYNAMIC mode).

        In STATIC mode raises ValueError to protect the fixed policy.
        """
        if self.mode == PolicyMode.STATIC:
            raise ValueError(
                "Cannot modify policy in STATIC mode. "
                "Switch to PolicyMode.DYNAMIC to enable dynamic policy."
            )

        new_params = PolicyParameters(
            A    = A if A is not None else self._current.A,
            B    = B if B is not None else self._current.B,
            C    = C if C is not None else self._current.C,
            D    = D if D is not None else self._current.D,
            step = step,
        )
        self._current = new_params
        self._history.append(new_params)
        return new_params

    def schedule_change(
        self, at_step: int, A: float, B: float, C: float, D: float
    ) -> None:
        """
        Schedules a policy change at a future step.
        Only valid in DYNAMIC mode.
        """
        if self.mode == PolicyMode.STATIC:
            raise ValueError("schedule_change requiere PolicyMode.DYNAMIC")
        self._scheduled_changes.append(
            (at_step, PolicyParameters(A=A, B=B, C=C, D=D, step=at_step))
        )
        # Mantener ordenados por step
        self._scheduled_changes.sort(key=lambda x: x[0])

    def tick(self, step: int) -> Optional[PolicyParameters]:
        """
        Call at each step to apply scheduled changes.
        Returns the new PolicyParameters if a change occurred, otherwise None.
        """
        if self.mode == PolicyMode.STATIC:
            return None

        changed = None
        remaining = []
        for at_step, params in self._scheduled_changes:
            if at_step <= step:
                self._current = PolicyParameters(
                    A=params.A, B=params.B, C=params.C, D=params.D, step=step
                )
                self._history.append(self._current)
                changed = self._current
            else:
                remaining.append((at_step, params))
        self._scheduled_changes = remaining
        return changed

    # -----------------------------------------------------------------------
    # Creation from configuration
    # -----------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "GlobalPolicy":
        """
        Creates a GlobalPolicy from a JSON configuration dict:

        {
            "mode":    "static" | "dynamic",
            "initial": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5},
            "scheduled_changes": [
                {"at_step": 100, "A": 2.0, "B": 0.5, "C": 1.0, "D": 0.5},
                ...
            ]
        }
        """
        mode = PolicyMode(cfg.get("mode", "static"))
        init_cfg = cfg.get("initial", {})
        initial  = PolicyParameters(
            A=init_cfg.get("A", 1.0),
            B=init_cfg.get("B", 1.0),
            C=init_cfg.get("C", 1.0),
            D=init_cfg.get("D", 0.5),
        )
        policy = cls(mode=mode, initial=initial)
        for change in cfg.get("scheduled_changes", []):
            if mode == PolicyMode.DYNAMIC:
                policy.schedule_change(
                    at_step=change["at_step"],
                    A=change.get("A", initial.A),
                    B=change.get("B", initial.B),
                    C=change.get("C", initial.C),
                    D=change.get("D", initial.D),
                )
        return policy

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def history_summary(self) -> List[dict]:
        return [p.to_dict() for p in self._history]

    def __repr__(self) -> str:
        return f"GlobalPolicy(mode={self.mode.value}, current={self._current})"

"""
RewardMatrices — Reward matrices per state transition
=====================================================
Implements Equation 6 of the MAS-DUO thesis:

    R(s, s') = Σ w_i · x_i(s, s')

    where:
        w_i       → global policy parameters  (A, B, C, D)
        x_i(s,s') → reward matrices per dimension

The matrices are square N×N where N = number of MDP states.
Each element matrix[s][s'] represents the expected reward when
transitioning from state s to state s'.

The thesis (Sec 3.4.2) defines 4 reward dimensions:
  · Delay   (A) → normalised transition time (negative = bad)
  · Cost    (B) → economic cost of the transition
  · QoS     (C) → expected quality of service
  · Energy  (D) → energy consumed in the transition

Sign convention: HIGHER values = more DESIRABLE.
  - Delay:  0 = no delay, -1 = maximum delay
  - Cost:   0 = no cost, negative values indicate cost
  - QoS:    1 = optimal, 0 = unacceptable
  - Energy: 0 = no consumption, negative values indicate consumption
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# TransitionReward — scalar reward for an individual transition
# ---------------------------------------------------------------------------

@dataclass
class TransitionReward:
    """
    Reward decomposed into the 4 MAS-DUO dimensions for the
    transition (state_from, state_to).

    Attributes
    ----------
    delay   : float   Delay reward. 0 = no delay; negative = delayed.
    cost    : float   Cost reward. 0 = free; negative = has cost.
    qos     : float   Quality-of-Service. 1 = maximum quality; 0 = minimum.
    energy  : float   Energy reward. 0 = no consumption; negative = consumes.
    """
    delay:  float = 0.0
    cost:   float = 0.0
    qos:    float = 1.0
    energy: float = 0.0

    def weighted_sum(self, A: float, B: float, C: float, D: float) -> float:
        """
        R(s,s') = A·delay + B·cost + C·qos + D·energy
        (Equation 6 and Equation 8 of the MAS-DUO thesis)
        """
        return A * self.delay + B * self.cost + C * self.qos + D * self.energy

    def to_dict(self) -> dict:
        return {
            "delay":  self.delay,
            "cost":   self.cost,
            "qos":    self.qos,
            "energy": self.energy,
        }


# ---------------------------------------------------------------------------
# RewardMatrices — N×N matrices for all transitions of an agent
# ---------------------------------------------------------------------------

class RewardMatrices:
    """
    Square N×N reward matrices for a given agent class.

    Follows Equation 6 of the thesis:

        Cost(s,s')   → matrix_cost[s][s']
        Delay(s,s')  → matrix_delay[s][s']
        QoS(s,s')    → matrix_qos[s][s']
        Energy(s,s') → matrix_energy[s][s']

    The diagonal is 0 (no useful self-transition).
    """

    def __init__(self, n_states: int, agent_class: str = "generic"):
        """
        Parameters
        ----------
        n_states    : total number of MDP states for the agent
        agent_class : agent class name (for logging)
        """
        self.n_states:    int = n_states
        self.agent_class: str = agent_class

        # N×N matrices initialised to neutral values
        self._delay  = np.zeros((n_states, n_states), dtype=np.float32)
        self._cost   = np.zeros((n_states, n_states), dtype=np.float32)
        self._qos    = np.ones((n_states, n_states),  dtype=np.float32)   # default QoS = 1
        self._energy = np.zeros((n_states, n_states), dtype=np.float32)

        # Mask of valid transitions (True = transition possible)
        self._valid  = np.zeros((n_states, n_states), dtype=bool)
        np.fill_diagonal(self._valid, False)

    # -----------------------------------------------------------------------
    # Individual transition setters
    # -----------------------------------------------------------------------

    def set_transition(
        self,
        s_from: int,
        s_to:   int,
        delay:  float = 0.0,
        cost:   float = 0.0,
        qos:    float = 1.0,
        energy: float = 0.0,
        valid:  bool  = True,
    ) -> None:
        """Sets the reward parameters for the transition s_from → s_to."""
        self._validate_idx(s_from, s_to)
        self._delay [s_from, s_to] = delay
        self._cost  [s_from, s_to] = cost
        self._qos   [s_from, s_to] = qos
        self._energy[s_from, s_to] = energy
        self._valid [s_from, s_to] = valid

    def get_transition_reward(self, s_from: int, s_to: int) -> TransitionReward:
        """Returns the TransitionReward for the transition s_from → s_to."""
        self._validate_idx(s_from, s_to)
        return TransitionReward(
            delay  = float(self._delay [s_from, s_to]),
            cost   = float(self._cost  [s_from, s_to]),
            qos    = float(self._qos   [s_from, s_to]),
            energy = float(self._energy[s_from, s_to]),
        )

    def is_valid(self, s_from: int, s_to: int) -> bool:
        """Checks whether the transition s_from → s_to is permitted."""
        self._validate_idx(s_from, s_to)
        return bool(self._valid[s_from, s_to])

    def valid_transitions(self, s_from: int) -> List[int]:
        """List of state indices reachable from s_from."""
        return list(np.where(self._valid[s_from])[0])

    # -----------------------------------------------------------------------
    # Update via learning (learning model, Sec 3.7.4)
    # -----------------------------------------------------------------------

    def update_from_experience(
        self,
        s_from: int,
        s_to:   int,
        observed: TransitionReward,
        learning_rate: float = 0.1,
    ) -> None:
        """
        Updates the matrices with observed experience (online learning).
        Uses EMA: matrix = (1 - α)·matrix + α·observed
        """
        self._validate_idx(s_from, s_to)
        lr = learning_rate
        self._delay [s_from, s_to] = (1 - lr) * self._delay[s_from, s_to]  + lr * observed.delay
        self._cost  [s_from, s_to] = (1 - lr) * self._cost[s_from, s_to]   + lr * observed.cost
        self._qos   [s_from, s_to] = (1 - lr) * self._qos[s_from, s_to]    + lr * observed.qos
        self._energy[s_from, s_to] = (1 - lr) * self._energy[s_from, s_to] + lr * observed.energy

    # -----------------------------------------------------------------------
    # Build reward vector with global policy
    # -----------------------------------------------------------------------

    def reward_matrix(self, A: float, B: float, C: float, D: float) -> np.ndarray:
        """
        Computes the composite matrix R = A·Delay + B·Cost + C·QoS + D·Energy
        Returns an ndarray (n_states, n_states).
        """
        R = (A * self._delay
           + B * self._cost
           + C * self._qos
           + D * self._energy)
        # Mask invalid transitions with -inf
        R[~self._valid] = -np.inf
        return R

    # -----------------------------------------------------------------------
    # Creation from configuration (dict)
    # -----------------------------------------------------------------------

    @classmethod
    def from_config(cls, n_states: int, agent_class: str, transitions: list) -> "RewardMatrices":
        """
        Creates the matrices from a list of transition definitions:
            transitions = [
                {"from": 0, "to": 1, "delay": -0.1, "cost": -0.5, "qos": 0.9, "energy": -0.2},
                ...
            ]
        """
        rm = cls(n_states=n_states, agent_class=agent_class)
        for t in transitions:
            rm.set_transition(
                s_from = t["from"],
                s_to   = t["to"],
                delay  = t.get("delay",  0.0),
                cost   = t.get("cost",   0.0),
                qos    = t.get("qos",    1.0),
                energy = t.get("energy", 0.0),
                valid  = t.get("valid",  True),
            )
        return rm

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _validate_idx(self, s_from: int, s_to: int) -> None:
        if not (0 <= s_from < self.n_states and 0 <= s_to < self.n_states):
            raise IndexError(
                f"State out of range: s_from={s_from}, s_to={s_to}, "
                f"n_states={self.n_states}"
            )

    def __repr__(self) -> str:
        return (
            f"RewardMatrices(agent_class={self.agent_class!r}, "
            f"n_states={self.n_states})"
        )

"""
MDPEngine — MDP Engine with Q-learning for Physical BDI Agents
=============================================================
Implements Equations 7 and 8 of the MAS-DUO thesis:

    Q^π(s, a) = R(s) + γ · Σ P(s'|s,a) · V^π(s')
    V^π(s)    = max_{a ∈ A(s)} Q^π(s, a)

    R(s, s')  = A·Delay(s,s') + B·Cost(s,s') + C·QoS(s,s') + D·Energy(s,s')

MDP states are pairs (ReadPoint, BusinessStep) — Equation 4:
    States(p) = ReadPoint × BusinessStep

The agent uses Q-learning (Sec 3.7.4) as the reinforcement learning
technique to learn transition probabilities and refine reward matrices
as it operates in the environment.

The MDPEngine is internal to each Physical BDI Agent; the IS Platform
provides the global policy parameters (A, B, C, D).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from logistics_env.mdp.reward_matrices import RewardMatrices, TransitionReward


# ---------------------------------------------------------------------------
# MDPState — MDP state as a (ReadPoint × BusinessStep) pair
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MDPState:
    """
    Agent state in the MDP state space.

    According to Equation 4 of the thesis:
        States(p) = { RP × BS }

    Attributes
    ----------
    read_point    : str   Zone/reader identifier (Where of the EPC)
    business_step : str   Business step (Why of the EPC)
    """
    read_point:    str
    business_step: str

    def __str__(self) -> str:
        return f"[{self.read_point}|{self.business_step}]"


# ---------------------------------------------------------------------------
# MDPEngine — Q-learning over the RP × BS state space
# ---------------------------------------------------------------------------

class MDPEngine:
    """
    MDP decision engine for a Physical BDI Agent.

    Maintains:
    · Q(s, a) table — action values for each state
    · Transition probability table P(s'|s, a)
    · Reward matrices (RewardMatrices) for computing R(s, s')

    Bellman equation parameters (Eq. 7):
    · gamma (γ) : discount factor (Sec 3.4.2, "discount rate")
    · alpha (α) : Q-learning rate
    · epsilon   : ε-greedy for exploration
    """

    def __init__(
        self,
        states:          List[MDPState],
        reward_matrices: RewardMatrices,
        gamma:           float = 0.95,
        alpha:           float = 0.1,
        epsilon:         float = 0.1,
    ):
        self.states:          List[MDPState]  = states
        self.reward_matrices: RewardMatrices  = reward_matrices
        self.gamma:           float           = gamma
        self.alpha:           float           = alpha
        self.epsilon:         float           = epsilon

        self.n_states: int = len(states)

        # State → index mapping
        self._state_index: Dict[MDPState, int] = {
            s: i for i, s in enumerate(states)
        }

        # Q(s, a=s') table — dimension n_states × n_states
        self._Q: np.ndarray = np.zeros((self.n_states, self.n_states), dtype=np.float64)

        # Transition counter table to estimate P(s'|s,a)
        self._transition_counts: np.ndarray = np.zeros(
            (self.n_states, self.n_states, self.n_states), dtype=np.float64
        )

        # Historial de decisiones: [(s_from, s_to, reward)]
        self._decision_history: List[Tuple[int, int, float]] = []

    # -----------------------------------------------------------------------
    # Action selection (BDI deliberation — Point 3.b of the thesis)
    # -----------------------------------------------------------------------

    def select_action(
        self,
        current_state: MDPState,
        goal_state:    MDPState,
        A: float, B: float, C: float, D: float,
    ) -> Optional[MDPState]:
        """
        Selects the best transition from current_state toward goal_state.

        Implements BDI deliberation:
            · Generate options (valid transitions)
            · Compute reward with global parameters (A, B, C, D)
            · Select the transition with highest V^π
            · ε-greedy for exploration (learning model)

        Returns the next MDPState or None if no valid transitions exist.
        """
        s_idx = self._state_index.get(current_state)
        if s_idx is None:
            return None

        valid_nexts = self.reward_matrices.valid_transitions(s_idx)
        if not valid_nexts:
            return None

        # Compute composite R with global policy parameters
        R_matrix = self.reward_matrices.reward_matrix(A, B, C, D)

        # Q(s, a) = R(s, a) + γ · V(s') according to Eq. 7
        q_values = {}
        for s_next in valid_nexts:
            r = R_matrix[s_idx, s_next]
            if np.isinf(r):
                continue
            v_next = self._V(s_next)
            q_values[s_next] = r + self.gamma * v_next

        if not q_values:
            return None

        # ε-greedy exploration (learning model, Sec 3.7.4)
        if np.random.random() < self.epsilon:
            chosen_idx = np.random.choice(list(q_values.keys()))
        else:
            chosen_idx = max(q_values, key=lambda k: q_values[k])

        return self.states[chosen_idx]

    # -----------------------------------------------------------------------
    # Q-learning update (post-execution)
    # -----------------------------------------------------------------------

    def update_q(
        self,
        s_from:  MDPState,
        s_to:    MDPState,
        observed: TransitionReward,
        A: float, B: float, C: float, D: float,
        success: bool = True,
    ) -> float:
        """
        Updates the Q table with the observed result after executing the action.

        Q(s,a) ← Q(s,a) + α · [R(s,a) + γ · V(s') - Q(s,a)]

        Also updates the RewardMatrix with the observed reward (online learning).
        """
        i = self._state_index.get(s_from)
        j = self._state_index.get(s_to)
        if i is None or j is None:
            return 0.0

        r_scalar = observed.weighted_sum(A, B, C, D)

        # Update reward matrices via learning (Sec 3.7.4)
        self.reward_matrices.update_from_experience(
            s_from=i, s_to=j, observed=observed, learning_rate=self.alpha
        )

        # Transition counters (to estimate P)
        if success:
            self._transition_counts[i, j, j] += 1
        else:
            self._transition_counts[i, j, i] += 1   # stayed in same state

        # Q-update
        v_next = self._V(j)
        td_error = r_scalar + self.gamma * v_next - self._Q[i, j]
        self._Q[i, j] += self.alpha * td_error

        self._decision_history.append((i, j, r_scalar))
        return self._Q[i, j]

    # -----------------------------------------------------------------------
    # Visibility gap detection (Equation 10 of the thesis)
    # -----------------------------------------------------------------------

    def detect_visibility_gaps(
        self, task: List[MDPState], threshold_factor: float = 2.0
    ) -> List[Tuple[MDPState, MDPState]]:
        """
        Detects transitions with a reward abnormally different from the average.
        Equation 10: if T(Smn, Sxy) >> Avg(T(S, S')) → visibility gap.

        Returns a list of (s_from, s_to) pairs where a gap is detected.
        """
        if len(task) < 2:
            return []

        # Compute rewards for the task transitions
        rewards = []
        pairs   = []
        for s, s_next in zip(task[:-1], task[1:]):
            i = self._state_index.get(s)
            j = self._state_index.get(s_next)
            if i is not None and j is not None:
                rewards.append(abs(self._Q[i, j]))
                pairs.append((s, s_next))

        if not rewards:
            return []

        avg = np.mean(rewards)
        gaps = [
            pair for pair, r in zip(pairs, rewards)
            if r > threshold_factor * avg
        ]
        return gaps

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _V(self, s_idx: int) -> float:
        """V^π(s) = max_{a} Q(s,a) — Equation 7."""
        valid = self.reward_matrices.valid_transitions(s_idx)
        if not valid:
            return 0.0
        return float(np.max(self._Q[s_idx, valid]))

    def state_by_rp_bs(self, read_point: str, business_step: str) -> Optional[MDPState]:
        """Looks up an MDPState by its RP and BS components."""
        target = MDPState(read_point=read_point, business_step=business_step)
        return target if target in self._state_index else None

    def state_index(self, state: MDPState) -> Optional[int]:
        return self._state_index.get(state)

    def add_state(self, state: MDPState) -> int:
        """Dynamically adds a new state (Sec 3.6 — flexibility)."""
        if state in self._state_index:
            return self._state_index[state]

        idx = self.n_states
        self.states.append(state)
        self._state_index[state] = idx
        self.n_states += 1

        # Expandir matrices
        self._Q = np.pad(self._Q, ((0, 1), (0, 1)), constant_values=0.0)
        old_rm_n = self.reward_matrices.n_states
        self.reward_matrices.n_states = self.n_states

        # Expand reward matrices
        for attr in ("_delay", "_cost", "_qos", "_energy"):
            mat = getattr(self.reward_matrices, attr)
            setattr(
                self.reward_matrices, attr,
                np.pad(mat, ((0, 1), (0, 1)), constant_values=0.0)
            )
        self.reward_matrices._valid = np.pad(
            self.reward_matrices._valid, ((0, 1), (0, 1)), constant_values=False
        )
        # Default QoS = 1 for new rows/columns
        self.reward_matrices._qos[old_rm_n:, :] = 1.0
        self.reward_matrices._qos[:, old_rm_n:] = 1.0

        return idx

    @property
    def avg_reward(self) -> float:
        """Limit Reward = lim (Σ R(t)) / T — Equation 8."""
        if not self._decision_history:
            return 0.0
        return float(np.mean([r for _, _, r in self._decision_history]))

    def __repr__(self) -> str:
        return (
            f"MDPEngine(n_states={self.n_states}, "
            f"gamma={self.gamma}, alpha={self.alpha})"
        )

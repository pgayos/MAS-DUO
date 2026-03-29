"""
MAS-DUO · MDP Module
====================
Implements the Markov Decision Process (MDP) decision engine
as described in the MAS-DUO doctoral thesis (Chapter 3, Section 3.4.2).

The reward function follows Equation 6:
    R(s, s') = A·Delay(s,s') + B·Cost(s,s') + C·QoS(s,s') + D·Energy(s,s')

The global parameters (A, B, C, D) are provided by the IS Platform.
"""

from logistics_env.mdp.reward_matrices import RewardMatrices, TransitionReward
from logistics_env.mdp.mdp_engine      import MDPEngine, MDPState

__all__ = [
    "RewardMatrices",
    "TransitionReward",
    "MDPEngine",
    "MDPState",
]

from __future__ import annotations

import unittest

from logistics_env.mdp.mdp_engine import MDPEngine, MDPState
from logistics_env.mdp.reward_matrices import RewardMatrices, TransitionReward


class RewardTests(unittest.TestCase):
    def test_weighted_reward_matches_mas_duo_equation(self) -> None:
        reward = TransitionReward(delay=-2.0, cost=-3.0, qos=0.8, energy=-1.0)
        self.assertAlmostEqual(
            reward.weighted_sum(A=0.5, B=0.4, C=0.0, D=0.1),
            -2.3,
        )

    def test_reward_matrix_masks_invalid_transitions(self) -> None:
        matrices = RewardMatrices(2)
        matrices.set_transition(0, 1, delay=-1.0, qos=1.0)
        composite = matrices.reward_matrix(A=1.0, B=0.0, C=1.0, D=0.0)
        self.assertEqual(composite[0, 1], 0.0)
        self.assertEqual(composite[1, 0], float("-inf"))


class MDPEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = [MDPState("A", "idle"), MDPState("B", "finished")]
        self.matrices = RewardMatrices(2)
        self.matrices.set_transition(0, 1, delay=-1.0, cost=0.0, qos=1.0)

    def test_q_update_uses_observed_transition_reward(self) -> None:
        engine = MDPEngine(
            self.states, self.matrices, gamma=0.0, alpha=0.5, epsilon=0.0, seed=7
        )
        value = engine.update_q(
            self.states[0], self.states[1],
            TransitionReward(delay=-2.0, qos=1.0),
            A=1.0, B=0.0, C=0.0, D=0.0,
        )
        self.assertAlmostEqual(value, -1.0)

    def test_engine_rng_is_reproducible(self) -> None:
        matrices = RewardMatrices(3)
        matrices.set_transition(0, 1)
        matrices.set_transition(0, 2)
        states = [MDPState(str(i), "state") for i in range(3)]
        first = MDPEngine(states, matrices, epsilon=1.0, seed=8)
        second = MDPEngine(states, matrices, epsilon=1.0, seed=8)
        choices_a = [first.select_action(states[0], states[2], 1, 1, 1, 1) for _ in range(8)]
        choices_b = [second.select_action(states[0], states[2], 1, 1, 1, 1) for _ in range(8)]
        self.assertEqual(choices_a, choices_b)


if __name__ == "__main__":
    unittest.main()

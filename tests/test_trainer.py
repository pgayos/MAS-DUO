from __future__ import annotations

import unittest

from train.train_ql import ALPHA, QTableAgent


class QTableTrainerTests(unittest.TestCase):
    def test_terminal_update_does_not_bootstrap(self) -> None:
        learner = QTableAgent(n_actions=2, seed=1)
        learner.Q[("next",)][0] = 100.0

        learner.update(
            s=("current",), a=0, r=2.0,
            s_next=("next",), terminal=True,
        )

        self.assertAlmostEqual(learner.Q[("current",)][0], ALPHA * 2.0)

    def test_action_selection_is_reproducible(self) -> None:
        first = QTableAgent(n_actions=4, seed=12)
        second = QTableAgent(n_actions=4, seed=12)
        actions_a = [first.select(("state",), epsilon=1.0) for _ in range(20)]
        actions_b = [second.select(("state",), epsilon=1.0) for _ in range(20)]
        self.assertEqual(actions_a, actions_b)


if __name__ == "__main__":
    unittest.main()

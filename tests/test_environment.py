from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from pettingzoo.test import api_test, seed_test

from logistics_env.logistics_maenv import LogisticsMaEnv, env


ROOT = Path(__file__).resolve().parents[1]
FACTORY_CONFIG = ROOT / "config" / "factory_example.json"


class EnvironmentClockTests(unittest.TestCase):
    def test_one_cycle_advances_operational_clock_once(self) -> None:
        environment = LogisticsMaEnv(FACTORY_CONFIG)
        environment.reset(seed=11)
        initial_agents = len(environment.agents)

        for _ in range(initial_agents):
            agent = environment.agent_selection
            environment.step(0)

        self.assertEqual(environment.simulation_step, 1)
        self.assertEqual(environment.agent_turns, initial_agents)
        self.assertEqual(environment.state_snapshot["simulation_step"], 1)
        environment.close()

    def test_local_seed_does_not_modify_global_numpy_rng(self) -> None:
        np.random.seed(1234)
        expected = np.random.random()
        np.random.seed(1234)

        environment = LogisticsMaEnv(FACTORY_CONFIG)
        environment.reset(seed=99)
        actual = np.random.random()

        self.assertEqual(actual, expected)
        environment.close()

    def test_action_space_sampling_is_reproducible(self) -> None:
        first = LogisticsMaEnv(FACTORY_CONFIG)
        second = LogisticsMaEnv(FACTORY_CONFIG)
        first.reset(seed=42)
        second.reset(seed=42)

        first_actions = [first.action_space(a).sample() for a in first.agents]
        second_actions = [second.action_space(a).sample() for a in second.agents]
        self.assertEqual(first_actions, second_actions)
        first.close()
        second.close()


class PettingZooComplianceTests(unittest.TestCase):
    def test_aec_api(self) -> None:
        api_test(env(FACTORY_CONFIG), num_cycles=100, verbose_progress=False)

    def test_seed_contract(self) -> None:
        seed_test(lambda: env(FACTORY_CONFIG), num_cycles=10)


if __name__ == "__main__":
    unittest.main()

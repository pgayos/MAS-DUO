from __future__ import annotations

import copy
import unittest
from pathlib import Path

from logistics_env.config_loader import load_factory_config
from logistics_env.objects.order_manager import OrderManager, OrderStatus


ROOT = Path(__file__).resolve().parents[1]


class OrderDeadlineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = copy.deepcopy(
            load_factory_config(ROOT / "config" / "factory_example.json")
        )
        self.config.orders = self.config.orders[:1]
        self.config.orders[0].deadline_steps = 1
        self.manager = OrderManager(self.config)
        self.epcs = self.manager.initialize_orders()

    def test_deadline_is_not_failed_at_exact_boundary(self) -> None:
        penalty = self.manager.check_deadlines(1)
        order = self.manager.orders[self.config.orders[0].order_id]
        self.assertEqual(penalty, 0.0)
        self.assertEqual(order.status, OrderStatus.PENDING)

    def test_overdue_order_fails_once_with_incremental_lateness(self) -> None:
        penalty = self.manager.check_deadlines(2)
        order = self.manager.orders[self.config.orders[0].order_id]
        self.assertEqual(penalty, self.config.reward_weights.late_penalty_per_step)
        self.assertEqual(order.status, OrderStatus.FAILED)
        self.assertEqual(self.manager.check_deadlines(3), 0.0)

    def test_on_time_completion_gets_full_reward(self) -> None:
        reward = None
        for epc in self.epcs:
            reward = self.manager.register_product_dispatched(epc.pure_identity_uri, 1)
        expected = (
            self.config.reward_weights.on_time_delivery
            + self.config.reward_weights.full_order_bonus
        )
        self.assertEqual(reward, expected)


if __name__ == "__main__":
    unittest.main()

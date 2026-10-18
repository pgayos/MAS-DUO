"""End-to-end tests: products travel their route and orders close correctly."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from logistics_env.agents import ConveyorAction, ProductAction, RobotAction, WorkerAction
from logistics_env.logistics_maenv import LogisticsMaEnv
from logistics_env.objects import OrderStatus


ROOT = Path(__file__).resolve().parents[1]
MINI_LINE = ROOT / "tests" / "fixtures" / "mini_line.json"
BELT_LINE = ROOT / "tests" / "fixtures" / "belt_line.json"


def greedy_action(environment: LogisticsMaEnv, agent_id: str) -> int:
    """Robots fetch the nearest available product and drop it in its next zone."""
    atype = environment._agent_type(agent_id)
    if atype == "product":
        return int(ProductAction.REQUEST_MOVE)
    if atype == "conveyor":
        return int(ConveyorAction.RUN)
    if atype == "worker":
        return int(WorkerAction.IDLE)

    robot = environment._robots[agent_id]
    if robot.carrying:
        product = environment._products[robot.carrying]
        zone = environment._grid.zone_of(robot.position)
        if zone and zone.id == product.current_zone_target:
            return int(RobotAction.DROP)
        return int(RobotAction.NAVIGATE)
    products_here = environment._build_grid_state()["products_at"]
    if products_here.get((robot.position.x, robot.position.y)):
        return int(RobotAction.LIFT)
    return int(RobotAction.NAVIGATE)


def run_episode(environment: LogisticsMaEnv, seed: int = 0, on_turn=None) -> dict:
    """Runs a greedy episode and returns the cumulative reward per agent."""
    environment.reset(seed=seed)
    totals: dict = {}
    for agent_id in environment.agent_iter():
        _, reward, termination, truncation, _ = environment.last()
        totals[agent_id] = totals.get(agent_id, 0.0) + reward
        if on_turn is not None:
            on_turn(environment)
        action = None if termination or truncation else greedy_action(environment, agent_id)
        environment.step(action)
    return totals


class ProductRouteTests(unittest.TestCase):
    def test_product_travels_full_route_and_order_completes_on_time(self) -> None:
        environment = LogisticsMaEnv(MINI_LINE)
        run_episode(environment)

        product = next(iter(environment._products.values()))
        order = environment._order_mgr.orders["ORD-1"]
        self.assertEqual(product.route_index, len(product.route))
        self.assertTrue(product.is_dispatched)
        self.assertEqual(order.status, OrderStatus.COMPLETE)
        self.assertLessEqual(order.finish_step, order.deadline)

        visited = [s.zone_id for s in product.state_trail]
        for zone_id in ("IN", "PROC", "OUT"):
            self.assertIn(zone_id, visited)
        self.assertEqual(product.position, product.state_trail[-1].where)
        environment.close()

    def test_order_is_not_complete_at_intermediate_waypoint(self) -> None:
        environment = LogisticsMaEnv(MINI_LINE)
        seen = []

        def check(env: LogisticsMaEnv) -> None:
            product = next(iter(env._products.values()))
            if 0 < product.route_index < len(product.route):
                seen.append(env._order_mgr.orders["ORD-1"].status)

        run_episode(environment, on_turn=check)
        self.assertTrue(seen)
        self.assertNotIn(OrderStatus.COMPLETE, seen)
        environment.close()

    def test_processing_holds_product_for_process_time(self) -> None:
        environment = LogisticsMaEnv(MINI_LINE)
        steps_in_processing = set()

        def check(env: LogisticsMaEnv) -> None:
            product = next(iter(env._products.values()))
            if product.is_processing:
                self.assertIsNone(product.carried_by)
                steps_in_processing.add(env.simulation_step)

        run_episode(environment, on_turn=check)
        self.assertEqual(len(steps_in_processing), 2)  # process_time_steps
        environment.close()

    def test_carried_product_follows_robot(self) -> None:
        environment = LogisticsMaEnv(MINI_LINE)

        def check(env: LogisticsMaEnv) -> None:
            robot = env._robots["ROB-001"]
            if robot.carrying:
                self.assertEqual(env._products[robot.carrying].position, robot.position)

        run_episode(environment, on_turn=check)
        environment.close()

    def test_carried_product_cannot_be_picked_again(self) -> None:
        environment = LogisticsMaEnv(MINI_LINE)

        def check(env: LogisticsMaEnv) -> None:
            products_at = env._build_grid_state()["products_at"]
            for epc_uri, product in env._products.items():
                if product.carried_by is not None:
                    self.assertFalse(any(epc_uri in ids for ids in products_at.values()))

        run_episode(environment, on_turn=check)
        environment.close()


class ConveyorTests(unittest.TestCase):
    def test_belt_advances_once_per_round_and_delivers(self) -> None:
        environment = LogisticsMaEnv(BELT_LINE)
        environment.reset(seed=0)
        epc_uri, product = next(iter(environment._products.items()))
        belt = environment._conveyors["CB-001"]
        self.assertTrue(belt.load_product(epc_uri))
        product.carried_by = belt.agent_id

        slots = []
        while not product.is_dispatched and environment.simulation_step < 20:
            start = environment.simulation_step
            while environment.simulation_step == start:
                agent_id = environment.agent_selection
                _, _, termination, truncation, _ = environment.last()
                atype = environment._agent_type(agent_id)
                if termination or truncation:
                    action = None
                elif atype == "conveyor":
                    action = int(ConveyorAction.RUN)
                elif atype == "product":
                    action = int(ProductAction.WAIT)
                else:
                    action = 0  # IDLE
                environment.step(action)
            if epc_uri in belt.slots:
                slots.append(belt.slots.index(epc_uri))

        self.assertEqual(slots, list(range(1, len(belt.path))))
        self.assertTrue(product.is_dispatched)
        self.assertEqual(product.position, belt.end_pos)
        self.assertEqual(environment._order_mgr.orders["ORD-1"].status, OrderStatus.COMPLETE)
        environment.close()


class DeadlineTests(unittest.TestCase):
    def test_missed_deadline_fails_order_and_terminates_product(self) -> None:
        environment = LogisticsMaEnv(MINI_LINE)
        environment.factory_cfg.orders[0].deadline_steps = 3
        run_episode(environment)

        product = next(iter(environment._products.values()))
        order = environment._order_mgr.orders["ORD-1"]
        self.assertTrue(product.is_lost)
        self.assertEqual(order.status, OrderStatus.FAILED)
        self.assertLess(len(product.state_trail), 10)
        self.assertLess(environment.simulation_step, 10)
        environment.close()

    def test_failed_order_never_becomes_complete(self) -> None:
        environment = LogisticsMaEnv(MINI_LINE)
        environment.factory_cfg.orders[0].deadline_steps = 3
        run_episode(environment)

        manager = environment._order_mgr
        epc_uri = next(iter(environment._products))
        self.assertIsNone(manager.register_product_dispatched(epc_uri, 99))
        self.assertEqual(manager.orders["ORD-1"].status, OrderStatus.FAILED)
        environment.close()


class PolicyAndLearningTests(unittest.TestCase):
    def _product_return(self, A: float, B: float, C: float, D: float) -> float:
        environment = LogisticsMaEnv(MINI_LINE)
        gp = environment.factory_cfg.global_policy
        gp.A, gp.B, gp.C, gp.D = A, B, C, D
        totals = run_episode(environment, seed=3)
        epc_uri = next(iter(environment._products))
        environment.close()
        return totals[epc_uri]

    def test_global_policy_weights_change_rewards(self) -> None:
        delay_only = self._product_return(1.0, 0.0, 0.0, 0.0)
        qos_only = self._product_return(0.0, 0.0, 1.0, 0.0)
        self.assertNotAlmostEqual(delay_only, qos_only)

    def test_is_negotiation_happens_and_q_table_learns(self) -> None:
        environment = LogisticsMaEnv(MINI_LINE)
        run_episode(environment)

        summary = environment._is_platform.get_summary()
        self.assertGreater(summary["total_negotiations"], 0)
        product = next(iter(environment._products.values()))
        self.assertTrue(np.any(product._mdp_engine._Q != 0))
        environment.close()


class SetupTests(unittest.TestCase):
    def test_agents_do_not_share_space_instances(self) -> None:
        environment = LogisticsMaEnv(ROOT / "config" / "factory_example.json")
        environment.reset(seed=0)
        workers = list(environment._workers)
        self.assertIsNot(environment.action_space(workers[0]), environment.action_space(workers[1]))
        environment.close()

    def test_overlapping_start_positions_are_rejected(self) -> None:
        environment = LogisticsMaEnv(MINI_LINE)
        environment.factory_cfg.workers[0].start_position = (3, 2)  # robot cell
        with self.assertRaises(ValueError):
            environment.reset(seed=0)

    def test_energy_counts_only_consumption(self) -> None:
        environment = LogisticsMaEnv(MINI_LINE)
        environment.reset(seed=0)
        for _ in range(3 * len(environment.agents)):
            environment.step(0)  # WAIT / IDLE for everyone
        self.assertEqual(environment.state_snapshot["total_energy_consumed"], 0.0)
        environment.close()


if __name__ == "__main__":
    unittest.main()

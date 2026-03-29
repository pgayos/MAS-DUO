"""
airport_gh_demo.py — Advanced Ground Handling Demo with Greedy Policy
=====================================================================
Scenario: Ciudad Real Central Airport (CRC) — 8 simultaneous flights

Runs a ground handling simulation with:
  · 8 flights (B737/A320/LCC/CARGO) with different deadlines and priorities
  · Greedy policy: prioritises products/flights with least remaining time (deadline)
  · pygame rendering at 3 FPS (human-readable speed)
  · Side panel with real-time flight status
  · Final performance summary vs. random baseline

Based on the doctoral thesis of Pablo García Ansola (2024):
  Chapter 4.1 — Ciudad Real Central Airport
  Equation 12 — Reward = 0.5·Delay + 0.4·Cost + 0.0·QoS + 0.1·Energy

Usage:
  python examples/airport_gh_demo.py [--headless] [--seed 42] [--fps 3]

"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Configure project path ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from logistics_env.logistics_maenv import LogisticsMaEnv
from logistics_env.agents import (
    ProductAction, WorkerAction, RobotAction, ConveyorAction,
)


# ══════════════════════════════════════════════════════════════════════════════
# Greedy Policy for Ground Handling
# ══════════════════════════════════════════════════════════════════════════════

class GreedyGHPolicy:
    """
    Greedy Ground Handling policy for the MAS-DUO environment.

    Principle: Earliest Deadline First (EDF) with priority adjustment.
    ─────────────────────────────────────────────────────────────────
    · Products   → REQUEST_MOVE if not in target zone, REQUEST_PROCESS if so
    · Robots     → NAVIGATE toward the product with the nearest deadline
    · Workers    → PROCESS in current zone, or MOVE toward the most urgent product
    · Conveyors  → RUN always (maximum throughput)

    Parameters
    ----------
    priority_weights : dict
        Weights for urgency scoring: A=delay, B=cost, D=energy
    """

    def __init__(self, priority_weights: Optional[Dict[str, float]] = None):
        self.weights = priority_weights or {"A": 0.5, "B": 0.4, "D": 0.1}
        self._step   = 0

    def reset(self):
        self._step = 0

    def select_action(
        self,
        agent_id: str,
        observation: np.ndarray,
        env: "LogisticsMaEnv",
    ) -> int:
        """
        Selects the action for the given agent according to the greedy policy.

        Returns an integer (action index).
        """
        self._step = env._step_count

        # ── Determine agent type ────────────────────────────────────────────
        if agent_id in env._products:
            return self._product_action(agent_id, env)
        if agent_id in env._robots:
            return self._robot_action(agent_id, env)
        if agent_id in env._workers:
            return self._worker_action(agent_id, env)
        if agent_id in env._conveyors:
            return int(ConveyorAction.RUN)   # belts always running

        return 0  # fallback WAIT/IDLE

    # ── Producto ─────────────────────────────────────────────────────────────

    def _product_action(self, agent_id: str, env: "LogisticsMaEnv") -> int:
        prod = env._products[agent_id]
        if prod.is_dispatched:
            return int(ProductAction.WAIT)

        # If already in target zone → request processing (IATA step)
        current_zone = env._grid.zone_of(prod.position)
        if current_zone and current_zone.id == prod.current_zone_target:
            return int(ProductAction.REQUEST_PROCESS)

        # If route has remaining steps → request move
        if prod.route_index < len(prod.route) - 1:
            return int(ProductAction.REQUEST_MOVE)

        # At the end of the route → signal ready for dispatch
        return int(ProductAction.SIGNAL_READY)

    # ── Robot ─────────────────────────────────────────────────────────────

    def _robot_action(self, agent_id: str, env: "LogisticsMaEnv") -> int:
        robot = env._robots[agent_id]

        # Low charge → recharge
        if robot.energy_fraction < 0.15:
            return int(RobotAction.CHARGE)

        # Carrying a product → navigate toward its destination
        if robot.is_carrying and robot.carrying in env._products:
            prod = env._products[robot.carrying]
            target_zone = prod.current_zone_target
            if target_zone:
                center = env._grid.zone_center(target_zone)
                if center:
                    if robot.position.x == center.x and robot.position.y == center.y:
                        return int(RobotAction.DROP)
                    return int(RobotAction.NAVIGATE)

        # Carrying nothing → find the most urgent product
        target_prod = self._most_urgent_product(env, exclude_carried=True)
        if target_prod:
            if robot.position.distance_to(target_prod.position) <= 1:
                return int(RobotAction.LIFT)
            return int(RobotAction.NAVIGATE)

        return int(RobotAction.IDLE)

    # ── Worker ──────────────────────────────────────────────────────────────

    def _worker_action(self, agent_id: str, env: "LogisticsMaEnv") -> int:
        worker = env._workers[agent_id]

        # Rest if very tired
        if worker.energy < 0.2:
            return int(WorkerAction.REST)

        # Product in the same cell → process or pick up
        pos = worker.position
        prods_here = [
            p for p in env._products.values()
            if p.position.x == pos.x and p.position.y == pos.y
            and not p.is_dispatched and p.carried_by is None
        ]
        if prods_here:
            # Prefer PROCESS if in the product's target zone
            prod = prods_here[0]
            zone = env._grid.zone_of(pos)
            if zone and zone.id == prod.current_zone_target:
                return int(WorkerAction.PROCESS)
            else:
                return int(WorkerAction.PICK)

        # Carrying a product → move it toward destination
        if worker.is_carrying and worker.carrying in env._products:
            prod = env._products[worker.carrying]
            target_zone = prod.current_zone_target
            if target_zone:
                center = env._grid.zone_center(target_zone)
                if center:
                    if worker.position.x == center.x and worker.position.y == center.y:
                        return int(WorkerAction.PLACE)
                    # Calculate direction toward target
                    dx = center.x - worker.position.x
                    dy = center.y - worker.position.y
                    if abs(dx) >= abs(dy):
                        return int(WorkerAction.MOVE_EAST if dx > 0 else WorkerAction.MOVE_WEST)
                    else:
                        return int(WorkerAction.MOVE_SOUTH if dy > 0 else WorkerAction.MOVE_NORTH)

        # Scan the zone for information
        return int(WorkerAction.SCAN)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _most_urgent_product(self, env: "LogisticsMaEnv", exclude_carried: bool = False):
        """Returns the product with the least remaining time until deadline (EDF)."""
        best     = None
        best_urgency = -1.0

        summary = env._order_mgr.get_summary() if env._order_mgr else {}

        for epc_uri, prod in env._products.items():
            if prod.is_dispatched:
                continue
            if exclude_carried and prod.carried_by is not None:
                continue

            # Urgency = 1 / (steps_to_deadline + 1)
            order_id = getattr(prod, "order_id", "")
            deadline = getattr(prod, "deadline", 0)
            remaining = max(1, deadline - self._step)
            urgency = 1.0 / remaining

            # Boost by order priority
            info = summary.get(order_id, {})
            priority = info.get("priority", "normal")
            if priority == "high":
                urgency *= 2.0
            elif priority == "low":
                urgency *= 0.5

            if urgency > best_urgency:
                best_urgency = urgency
                best         = prod

        return best

    def score_snapshot(self, env: "LogisticsMaEnv") -> dict:
        """Computes performance metrics for the solver at the current step."""
        summary = env._order_mgr.get_summary() if env._order_mgr else {}
        total   = len(summary)
        done    = sum(1 for i in summary.values() if i.get("status") == "complete")
        failed  = sum(1 for i in summary.values() if i.get("status") == "failed")
        ontime  = sum(
            1 for oid, i in summary.items()
            if i.get("status") == "complete"
            and i.get("dispatched", 0) >= i.get("needed", 1)
            and (i.get("deadline", 999) or 999) >= env._step_count
        )
        energy  = env._total_energy
        cumr    = sum(env._cumulative_rewards.values())
        return {
            "total": total,
            "done":  done,
            "failed":failed,
            "ontime":ontime,
            "energy":energy,
            "cum_reward": cumr,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Console display utilities
# ══════════════════════════════════════════════════════════════════════════════

IATA_BS = {
    "OMS": "Order Management",
    "STM": "Station Management",
    "LOD": "Load Control",
    "PAX": "Pax & Crew Handling",
    "BAG": "Baggage Handling",
    "HDL": "ULD/Container Handling",
    "AGM": "Aircraft Ground Movement",
    "CGM": "Cargo & Mail",
}

ZONE_LABEL = {
    "PARK1": "Stand P1 (B737)",
    "PARK2": "Stand P2 (A320)",
    "PARK3": "Stand P3 (LCC)",
    "PARK4": "Stand P4 (CARGO)",
    "TRANSIT": "Taxiway / Airside",
    "TERMINAL": "Terminal / Departures",
    "HANGAR": "Hangar (GH Base)",
}


def print_banner():
    print("\n" + "═" * 68)
    print("  MAS-DUO ✈  Ground Handling Demo — Ciudad Real Central Airport")
    print("  Policy: Greedy EDF (Earliest Deadline First)")
    print("  Thesis: Pablo García Ansola (2024), Ch. 4.1 — Equation 12")
    print("  Reward = 0.5·Delay + 0.4·Cost + 0.0·QoS + 0.1·Energy")
    print("═" * 68)
    print()
    print("  IATA Ground Handling Steps:")
    for code, name in IATA_BS.items():
        print(f"    [{code}] {name}")
    print()
    print("  ReadPoints (EPCIS Zone):")
    for zid, zlabel in ZONE_LABEL.items():
        print(f"    • {zid:<12s} → {zlabel}")
    print()


def print_step_summary(step: int, env: "LogisticsMaEnv", solver: "GreedyGHPolicy"):
    summary = env._order_mgr.get_summary() if env._order_mgr else {}
    step_secs  = getattr(getattr(env.factory_cfg, 'sim_params', None), 'step_duration_seconds', 60)
    sim_min    = (step * step_secs) // 60
    sim_sec    = (step * step_secs) % 60

    print(f"\n  ── Step {step:>3d} | T+{sim_min:02d}:{sim_sec:02d} | "
          f"Score: {solver.score_snapshot(env)} ──")
    for oid, info in summary.items():
        status   = info.get("status", "?")
        disp     = info.get("dispatched", 0)
        needed   = info.get("needed", 1)
        deadline = info.get("deadline", 0)
        remaining = (deadline or 0) - step
        prio     = info.get("priority", "normal")
        tag = {
            "complete": "✓ OK",
            "failed":   "✗ DELAY",
            "pending":  f"T-{max(0, remaining):02d}",
        }.get(status, "?")
        bar = "█" * min(10, disp * 10 // max(needed, 1))
        print(f"    {oid:<12s} [{prio[0].upper()}] {disp}/{needed} {bar:<10s} {tag}")


def print_final_report(env: "LogisticsMaEnv", solver: "GreedyGHPolicy",
                        elapsed_wall: float, steps_run: int):
    s = solver.score_snapshot(env)
    summary = env._order_mgr.get_summary() if env._order_mgr else {}

    print("\n" + "═" * 68)
    print("  FINAL REPORT — MAS-DUO Ground Handling Demo")
    print("═" * 68)
    print(f"  Steps run        : {steps_run}")
    print(f"  Wall time        : {elapsed_wall:.2f}s")
    print(f"  Cumulative reward: {s['cum_reward']:+.2f}")
    print(f"  Total energy     : {s['energy']:.2f} units")
    print()
    print(f"  Flights completed  : {s['done']}/{s['total']}")
    print(f"  On time (deadline) : {s['ontime']}/{s['total']}")
    print(f"  Delayed            : {s['failed']}/{s['total']}")
    print()
    print("  Flight detail:")
    print("  " + "─" * 54)
    print(f"  {'ID':<12} {'Type':<8} {'Status':<10} {'Deadline':<10} {'Priority':<10}")
    print("  " + "─" * 54)
    for oid, info in summary.items():
        ptype    = info.get("product_type", "?")
        status   = info.get("status", "?")
        deadline = info.get("deadline", 0)
        priority = info.get("priority", "normal")
        mark     = "✓" if status == "complete" else ("✗" if status == "failed" else "~")
        print(f"  {mark} {oid:<10} {ptype:<8} {status:<10} {str(deadline):<10} {priority:<10}")

    print()
    print("  IS Platform:")
    is_summary = env._is_platform.get_summary() if env._is_platform else {}
    for k, v in is_summary.items():
        print(f"    {k}: {v}")
    print()
    print("  Global Policy (Equation 12 — MAS-DUO):")
    gp = env._global_policy
    if gp:
        print(f"    A (Delay)  = {gp.A:.2f}")
        print(f"    B (Cost)   = {gp.B:.2f}")
        print(f"    C (QoS)    = {gp.C:.2f}  ← 0.0 = Common Use (no airline preference)")
        print(f"    D (Energy) = {gp.D:.2f}")
    print("═" * 68)


# ══════════════════════════════════════════════════════════════════════════════
# Main demo
# ══════════════════════════════════════════════════════════════════════════════

def run_demo(
    config_path: str,
    headless: bool = False,
    seed: int = 42,
    fps: int = 3,
    verbose_every: int = 10,
):
    """
    Runs the Ground Handling demo with the greedy policy.

    Parameters
    ----------
    config_path    : path to the airport configuration JSON
    headless       : if True, does not open a pygame window
    seed           : seed for reproducibility
    fps            : frames per second for rendering (3 = human-readable speed)
    verbose_every  : print a console summary every N steps
    """

    print_banner()

    render_mode = None if headless else "human"
    cfg_path    = Path(config_path)

    print(f"  Loading scenario : {cfg_path.name}")
    print(f"  Rendering        : {'HEADLESS' if headless else f'pygame @ {fps} FPS'}")
    print(f"  Seed             : {seed}")
    print()

    # ── Construir entorno ──────────────────────────────────────────────────
    raw_env = LogisticsMaEnv(config_path=cfg_path, render_mode=render_mode)
    raw_env.reset(seed=seed)

    # ── Si hay pygame, reconfigurar el renderer con FPS personalizado ──────
    if not headless and raw_env._renderer is not None:
        raw_env._renderer._fps = fps
        raw_env._renderer._title_prefix = "MAS-DUO ✈ Airport GH"

    # ── Greedy Policy ───────────────────────────────────────────────────────────
    policy = GreedyGHPolicy(
        priority_weights={
            "A": raw_env.factory_cfg.global_policy.A,
            "B": raw_env.factory_cfg.global_policy.B,
            "D": raw_env.factory_cfg.global_policy.D,
        }
    )
    policy.reset()

    # ── Info extra para el panel de renderer ──────────────────────────────
    extra_info = {
        "scenario": f"{raw_env.factory_cfg.name[:32]}",
        "solver":   "Greedy EDF",
        "policy":   f"A={raw_env.factory_cfg.global_policy.A} "
                    f"B={raw_env.factory_cfg.global_policy.B} "
                    f"C={raw_env.factory_cfg.global_policy.C} "
                    f"D={raw_env.factory_cfg.global_policy.D}",
        "reward":   0.0,
    }

    print(f"  Active agents  : {len(raw_env.agents)}")
    print(f"  Flights (orders): {len(raw_env.factory_cfg.orders)}")
    print(f"  Max steps      : {raw_env.factory_cfg.sim_params.max_steps}")
    print()
    print("  Starting simulation...")
    time.sleep(0.5)

    # ── Main simulation loop ────────────────────────────────────────────────
    t_start      = time.time()
    steps_run    = 0
    cum_reward   = 0.0
    last_printed = -1  # track which steps have already been printed

    try:
        while raw_env.agents:
            agent_id = raw_env.agent_selection

            # Verificar si el entorno ha terminado
            obs, reward, term, trunc, info = raw_env.last()

            if term or trunc:
                # Terminated/truncated agent: pass None as action
                # Handle the case where PettingZoo already removed it from the list
                try:
                    raw_env.step(None)
                except ValueError:
                    # Agent was already removed internally; continue
                    break
                continue

            # Select greedy action
            action = policy.select_action(agent_id, obs, raw_env)
            raw_env.step(action)
            cum_reward += reward

            # Update renderer info
            extra_info["reward"] = cum_reward

            # Render con info extra aeroportuaria
            if not headless and raw_env._renderer is not None:
                raw_env._renderer.render(
                    grid      = raw_env._grid,
                    products  = raw_env._products,
                    workers   = raw_env._workers,
                    robots    = raw_env._robots,
                    conveyors = raw_env._conveyors,
                    step      = raw_env._step_count,
                    order_mgr = raw_env._order_mgr,
                    mode      = "human",
                    fps       = fps,
                    extra_info= extra_info,
                )

            # Print periodic console summary (once per step)
            current_step = raw_env._step_count
            if current_step % verbose_every == 0 and current_step != last_printed:
                last_printed = current_step
                print_step_summary(current_step, raw_env, policy)

            steps_run = raw_env._step_count

    except KeyboardInterrupt:
        print("\n  [Interrupted by user]")

    finally:
        elapsed = time.time() - t_start
        print_final_report(raw_env, policy, elapsed, steps_run)
        if not headless:
            raw_env.close()

    return {
        "steps":      steps_run,
        "cum_reward": cum_reward,
        "elapsed":    elapsed,
        "score":      policy.score_snapshot(raw_env),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MAS-DUO Airport Ground Handling Demo (Greedy EDF policy)"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "airport_gh_demo.json"),
        help="Path to the airport config JSON (default: config/airport_gh_demo.json)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a pygame window (console only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=3,
        help="pygame render FPS (default: 3 = human-readable speed)",
    )
    parser.add_argument(
        "--verbose-every",
        type=int,
        default=10,
        dest="verbose_every",
        help="Print console summary every N steps (default: 10)",
    )

    args = parser.parse_args()

    result = run_demo(
        config_path   = args.config,
        headless      = args.headless,
        seed          = args.seed,
        fps           = args.fps,
        verbose_every = args.verbose_every,
    )

    sys.exit(0 if result["score"]["failed"] == 0 else 1)

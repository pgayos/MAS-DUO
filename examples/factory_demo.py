"""
factory_demo.py — Visual warehouse demo (human render mode)
===========================================================
Shows the factory simulation in real-time using a smart policy:

  · Workers   — greedy rule-based policy (or Q-table if a trained
                policy is found at train/ql_policy.pkl)
  · Robots    — always NAVIGATE  (built-in A* navigation)
  · Products  — always REQUEST_MOVE
  · Conveyors — always RUN

Visual features:
  · Product movement trails (fading orange dots)
  · Robot A* navigation paths (blue dotted lines)
  · Rich info panel: orders, product routes, worker/robot/conveyor status,
    reward sparkline history

Controls while running:
  SPACE    — pause / resume
  +  / =   — increase simulation speed (FPS)
  -        — decrease simulation speed (FPS)
  Q / ESC  — quit

Usage:
    python examples/factory_demo.py
    python examples/factory_demo.py --fps 6
    python examples/factory_demo.py --policy train/ql_policy.pkl
    python examples/factory_demo.py --config config/factory_example.json --steps 300
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from logistics_env import LogisticsMaEnv
from logistics_env.agents.base_agent     import Position
from logistics_env.agents.conveyor_agent import ConveyorAction
from logistics_env.agents.product_agent  import ProductAction
from logistics_env.agents.robot_agent    import RobotAction
from logistics_env.agents.worker_agent   import WorkerAction
from logistics_env.rendering.renderer    import LogisticsRenderer

# ===========================================================================
# Greedy policy helpers
# ===========================================================================

def _compass(src: Position, dst: Position) -> int:
    """Return the WorkerAction int to take one step from src toward dst."""
    dx = dst.x - src.x
    dy = dst.y - src.y
    if dx == 0 and dy == 0:
        return WorkerAction.IDLE
    if abs(dx) >= abs(dy):
        return WorkerAction.MOVE_EAST if dx > 0 else WorkerAction.MOVE_WEST
    return WorkerAction.MOVE_SOUTH if dy > 0 else WorkerAction.MOVE_NORTH


def _greedy_worker(worker_id: str, env: LogisticsMaEnv) -> int:
    """
    Smart rule-based action policy for one worker:

    1. Rest if energy < 15 % or fatigue > 85 %.
    2. If carrying a product:
       a. Find the conveyor whose END zone matches the product's next route zone.
       b. Navigate to that conveyor's start cell and PLACE (auto-loads belt).
       c. If no conveyor matches, navigate to the zone centre directly.
       d. If already at the target zone: PROCESS (process zone) or PLACE.
    3. If not carrying: walk to the nearest uncarried product and PICK it.
    """
    w     = env._workers[worker_id]
    grid  = env._grid
    prods = env._products

    # Recover when depleted
    if w.energy < 0.15 or w.fatigue > 0.85:
        return WorkerAction.REST

    if w.is_carrying:
        epc  = w.carrying
        prod = prods.get(epc)
        if prod is None:
            return WorkerAction.PLACE          # orphan product — drop it

        zone_target = prod.current_zone_target
        if zone_target is None:
            return WorkerAction.PLACE          # route complete

        cur_zone = grid.zone_of(w.position)
        if cur_zone:
            if cur_zone.id == zone_target:
                return (WorkerAction.PROCESS
                        if cur_zone.type == "process"
                        else WorkerAction.PLACE)

            # Standing on a conveyor start that leads to the target?
            cb_id = grid.conveyor_at(w.position)
            if cb_id and cb_id in env._conveyors:
                cb_end_zone = grid.zone_of(env._conveyors[cb_id].end_pos)
                if cb_end_zone and cb_end_zone.id == zone_target:
                    return WorkerAction.PLACE  # loads the belt

        # Find the belt whose END zone is the product's next target
        target_pos: Optional[Position] = None
        for cb in env._conveyors.values():
            end_zone = grid.zone_of(cb.end_pos)
            if end_zone and end_zone.id == zone_target and cb.can_accept():
                target_pos = cb.start_pos
                break

        if target_pos is None:
            target_pos = grid.zone_center(zone_target)  # fall back to zone centre

        if target_pos is None:
            return WorkerAction.IDLE

        return _compass(w.position, target_pos)

    else:
        # Find nearest uncarried, non-dispatched product
        best: Optional[object] = None
        best_dist = float("inf")
        for _epc, p in prods.items():
            if p.carried_by is None and not p.is_dispatched:
                d = abs(p.position.x - w.position.x) + abs(p.position.y - w.position.y)
                if d < best_dist:
                    best_dist = d
                    best = p

        if best is None:
            return WorkerAction.IDLE

        if w.position.x == best.position.x and w.position.y == best.position.y:
            return WorkerAction.PICK

        return _compass(w.position, best.position)


# Q-Learning state discretisation (must match train_ql.py)
def _disc_worker(obs: np.ndarray) -> tuple:
    x, y, en, fat, carry, zone = obs
    return (
        min(9, int(x * 10)),
        min(9, int(y * 10)),
        min(4, int(en * 5)),
        min(2, int(fat * 3)),
        int(round(carry)),
        min(6, int(zone * 7)),
    )


# ===========================================================================
# Unified policy
# ===========================================================================

class FactoryPolicy:
    """
    Dispatch table for all agent types:
      · workers   — Q-table (if loaded) or greedy rule-based fallback
      · robots    — always NAVIGATE
      · products  — always REQUEST_MOVE
      · conveyors — always RUN
    """

    def __init__(self, ql_policy: Optional[dict] = None):
        self._Q_workers = None
        self._name      = "Greedy Policy"
        if ql_policy is not None:
            self._Q_workers = ql_policy.get("Q_workers")
            if self._Q_workers:
                n_ep = ql_policy.get("n_episodes", "?")
                print(f"  Q-Learning policy loaded  "
                      f"({len(self._Q_workers):,} states, {n_ep} training episodes)")
                self._name = f"Q-Learning ({n_ep} eps)"

    @property
    def name(self) -> str:
        return self._name

    def select_action(self, agent_id: str, obs: np.ndarray,
                      env: LogisticsMaEnv) -> int:
        atype = env._agent_type(agent_id)

        if atype == "product":
            return ProductAction.REQUEST_MOVE

        if atype == "conveyor":
            return ConveyorAction.RUN

        if atype == "robot":
            return RobotAction.NAVIGATE

        if atype == "worker":
            if self._Q_workers:
                s = _disc_worker(obs)
                q = self._Q_workers.get(s)
                if q:
                    return max(q, key=q.get)
            return _greedy_worker(agent_id, env)

        return 0   # safety fallback


# ===========================================================================
# Demo runner
# ===========================================================================

def run_demo(
    config_path: Path,
    policy:      FactoryPolicy,
    max_steps:   int,
    fps:         int,
    seed:        int,
) -> dict:
    """
    Runs one episode with the supplied policy and renders every step.
    Returns a stats dict when the episode finishes or the user quits.
    """
    # Headless env — we drive the renderer manually for full extra_info control
    env = LogisticsMaEnv(config_path=config_path, render_mode=None)
    env.reset(seed=seed)

    renderer = LogisticsRenderer(
        factory_cfg  = env.factory_cfg,
        fps          = fps,
        title_prefix = "MAS-DUO Factory",
    )

    # Product movement trail buffer: last N positions per product
    TRAIL_LEN = 30
    trails: dict = defaultdict(lambda: deque(maxlen=TRAIL_LEN))

    cumulative_reward = 0.0
    step_count        = 0

    print(f"\n  Warehouse : {env.factory_cfg.name}")
    print(f"  Policy    : {policy.name}")
    print(f"  FPS       : {fps}   Max steps: {max_steps}")
    print(f"  Controls  : SPACE=pause  +/-=speed  Q=quit\n")

    while env.agents and step_count < max_steps:
        agent = env.agent_selection
        obs, cum_rew, term, trunc, _info = env.last()

        action = policy.select_action(agent, obs, env)
        cumulative_reward += cum_rew
        env.step(action)
        step_count += 1

        # Update product trails
        for epc_uri, prod in env._products.items():
            if not prod.is_dispatched:
                trails[epc_uri].append((prod.position.x, prod.position.y))

        # Robot A* paths for visualisation
        robot_paths = {
            rid: [(p.x, p.y) for p in r._path]
            for rid, r in env._robots.items()
        }

        # Build extra_info for the renderer panel
        extra = {
            "panel_mode":  "factory",
            "title":       env.factory_cfg.name,
            "solver":      policy.name,
            "reward":      cumulative_reward,
            "trails":      {k: list(v) for k, v in trails.items()},
            "robot_paths": robot_paths,
            "paused":      renderer.paused,
        }

        renderer.render(
            grid      = env._grid,
            products  = env._products,
            workers   = env._workers,
            robots    = env._robots,
            conveyors = env._conveyors,
            step      = env._step_count,
            order_mgr = env._order_mgr,
            mode      = "human",
            extra_info= extra,
        )

        if renderer.quit_requested:
            print("  Quit by user.")
            break

    # Final stats
    snap = env.state_snapshot
    env.close()
    renderer.close()

    return {
        "steps":  step_count,
        "reward": cumulative_reward,
        "orders": snap["orders"],
        "energy": snap["total_energy_consumed"],
    }


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MAS-DUO Factory — visual demo with smart / Q-Learning policy"
    )
    parser.add_argument(
        "--config", type=str, default="config/factory_example.json",
        help="Path to factory JSON config."
    )
    parser.add_argument(
        "--policy", type=str, default="train/ql_policy.pkl",
        help="Path to trained Q-policy pickle (falls back to greedy if not found)."
    )
    parser.add_argument(
        "--fps", type=int, default=4,
        help="Render speed in frames per second (default 4, good for watching)."
    )
    parser.add_argument(
        "--steps", type=int, default=0,
        help="Maximum simulation steps. 0 = use config default."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root        = Path(__file__).parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    policy_path = Path(args.policy)
    if not policy_path.is_absolute():
        policy_path = root / policy_path

    # Load Q-policy if available
    ql_policy = None
    if policy_path.exists():
        with policy_path.open("rb") as f:
            ql_policy = pickle.load(f)
    else:
        print(f"  No trained policy found at {policy_path}.")
        print(f"  Tip: run  python train/train_ql.py  first to train workers.\n")

    policy = FactoryPolicy(ql_policy=ql_policy)

    # Determine max steps from config if not overridden
    if args.steps > 0:
        max_steps = args.steps
    else:
        from logistics_env.config_loader import load_factory_config
        cfg       = load_factory_config(config_path)
        max_steps = cfg.sim_params.max_steps

    stats = run_demo(
        config_path = config_path,
        policy      = policy,
        max_steps   = max_steps,
        fps         = args.fps,
        seed        = args.seed,
    )

    # ── Final report ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  DEMO COMPLETE")
    print(f"{'='*55}")
    print(f"  Steps run    : {stats['steps']}")
    print(f"  Total reward : {stats['reward']:+.2f}")
    print(f"  Energy used  : {stats['energy']:.1f} units")
    print()
    print("  Order results:")
    for oid, info in stats["orders"].items():
        done_n = info.get("dispatched", 0)
        needed = info.get("needed",    1)
        status = info.get("status",   "?")
        finish = info.get("finish_step")
        finish_str = f"step {finish}" if finish else "—"
        print(f"    {oid:<14s}  {done_n}/{needed}  [{status}]  finished: {finish_str}")
    print()


if __name__ == "__main__":
    main()

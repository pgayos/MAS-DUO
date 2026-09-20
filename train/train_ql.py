"""
train_ql.py — Tabular Q-Learning training for the factory warehouse
====================================================================
Trains a Q-table-based policy on the MAS-DUO factory environment.

Learning agents:
  · Workers   — epsilon-greedy Q-Learning (tabular, discretised observation)
  · Robots    — fixed: always NAVIGATE  (built-in A* from grid_state)
  · Products  — fixed: always REQUEST_MOVE
  · Conveyors — fixed: always RUN

The trained Q-tables are saved to train/ql_policy.pkl, and the per-episode
stats are also saved as train/ql_policy.json for external plotting.

Usage:
    python train/train_ql.py
    python train/train_ql.py --episodes 80
    python train/train_ql.py --episodes 60 --render-every 20
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from logistics_env import LogisticsMaEnv

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

ALPHA       = 0.15    # Q-Learning rate
GAMMA       = 0.95    # discount factor
EPSILON_0   = 1.0     # initial exploration rate
EPSILON_MIN = 0.05    # minimum exploration rate

# ---------------------------------------------------------------------------
# Observation discretisation
# ---------------------------------------------------------------------------

def _disc_worker(obs: np.ndarray) -> Tuple:
    """
    Discretise a 6-D worker observation into a hashable Q-table key.
    obs = [x_norm, y_norm, energy, fatigue, is_carrying, zone_type_enc]
    Produces ≈ 10×10×5×3×2×7 = 21 000 possible states.
    """
    x, y, en, fat, carry, zone = obs
    return (
        min(9, int(x * 10)),
        min(9, int(y * 10)),
        min(4, int(en * 5)),
        min(2, int(fat * 3)),
        int(round(carry)),
        min(6, int(zone * 7)),
    )


def _disc_product(obs: np.ndarray) -> Tuple:
    """
    Discretise a 6-D product observation.
    obs = [x_norm, y_norm, route_progress, process_ticks_norm,
           deadline_remaining_norm, is_carried]
    """
    x, y, rp, pt, dl, carried = obs
    return (
        min(9, int(x * 10)),
        min(9, int(y * 10)),
        min(4, int(rp * 5)),
        min(2, int(pt * 3)),
        min(4, int(dl * 5)),
        int(round(carried)),
    )


def _disc_robot(obs: np.ndarray) -> Tuple:
    """
    Discretise a 5-D robot observation.
    obs = [x_norm, y_norm, energy_frac, is_carrying, needs_charge]
    """
    x, y, en, carry, charge = obs
    return (
        min(9, int(x * 10)),
        min(9, int(y * 10)),
        min(4, int(en * 5)),
        int(round(carry)),
        int(round(charge)),
    )


def _disc_conveyor(obs: np.ndarray) -> Tuple:
    """
    Discretise a 5-D conveyor observation.
    obs = [occupancy_fraction, is_running, is_jammed, is_reversed, energy_rate]
    """
    occ, run, jam, rev, _rate = obs
    return (
        min(2, int(occ * 3)),
        int(round(run)),
        int(round(jam)),
        int(round(rev)),
    )


_DISC_FN = {
    "worker":   _disc_worker,
    "product":  _disc_product,
    "robot":    _disc_robot,
    "conveyor": _disc_conveyor,
}

# ---------------------------------------------------------------------------
# Fixed smart actions for non-learning agent types
# ---------------------------------------------------------------------------

_FIXED_ACTIONS = {
    "product":  1,   # REQUEST_MOVE
    "conveyor": 1,   # RUN
    "robot":    8,   # NAVIGATE  (env's grid_state supplies A* targets)
}

# ---------------------------------------------------------------------------
# Q-Table agent
# ---------------------------------------------------------------------------

class QTableAgent:
    """Stores the Q-table and implements epsilon-greedy action selection."""

    def __init__(self, n_actions: int, seed: int | None = None):
        self.n_actions   = n_actions
        self.Q: Dict     = defaultdict(lambda: {a: 0.0 for a in range(n_actions)})
        self.n_updates   = 0
        self.total_reward = 0.0
        self.rng = np.random.default_rng(seed)

    def select(self, state: tuple, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return max(self.Q[state], key=self.Q[state].get)

    def update(
        self, s: tuple, a: int, r: float, s_next: tuple | None, terminal: bool = False
    ) -> None:
        q_next  = 0.0 if terminal or s_next is None else max(self.Q[s_next].values())
        old_q   = self.Q[s][a]
        self.Q[s][a] = old_q + ALPHA * (r + GAMMA * q_next - old_q)
        self.n_updates    += 1
        self.total_reward += r

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    env: LogisticsMaEnv,
    n_episodes: int,
    seed: int = 7,
    progress_callback: Optional[Callable[[dict], bool | None]] = None,
) -> dict:
    """
    Runs the Q-Learning training loop and returns results dict.
    """
    cfg       = env.factory_cfg
    max_steps = cfg.sim_params.max_steps

    # Q-tables — workers are the primary learning agents
    q_worker  = QTableAgent(n_actions=10, seed=seed)   # 10 WorkerActions

    # Per-agent memory: {agent_id: (prev_discrete_state, prev_action)}
    memory: Dict = {}

    episode_rewards:    list = []
    episode_lengths:    list = []
    episode_agent_turns: list = []
    order_completions:  list = []

    # Epsilon decay schedule: linear over ~80 % of training
    decay_episodes = max(1, int(n_episodes * 0.80))

    print(f"\n{'='*60}")
    print(f"  MAS-DUO Q-Learning Trainer")
    print(f"  Factory  : {cfg.name}")
    print(f"  Episodes : {n_episodes}   Max steps/ep : {max_steps}")
    print(f"  α={ALPHA}  γ={GAMMA}  ε: {EPSILON_0:.2f} → {EPSILON_MIN:.2f}")
    print(f"  Learning : workers  |  Fixed : robots / conveyors / products")
    print(f"{'='*60}\n")

    for ep in range(n_episodes):
        # Epsilon: linear decay
        frac    = ep / max(decay_episodes, 1)
        epsilon = max(EPSILON_MIN, EPSILON_0 - frac * (EPSILON_0 - EPSILON_MIN))

        env.reset(seed=seed + ep * 13)
        memory.clear()

        ep_reward  = 0.0
        t0         = time.time()

        while env.agents:
            agent = env.agent_selection
            obs, cum_rew, term, trunc, _info = env.last()

            atype   = env._agent_type(agent)
            disc_fn = _DISC_FN[atype]
            s_prime = disc_fn(obs)

            # Q-update for this agent using its accumulated reward since last visit
            if agent in memory and atype == "worker":
                s_prev, a_prev = memory[agent]
                q_worker.update(
                    s_prev, a_prev, cum_rew, s_prime,
                    terminal=term or trunc,
                )

            ep_reward += cum_rew

            if term or trunc:
                memory.pop(agent, None)
                env.step(None)
                continue

            # Action selection
            if atype == "worker":
                action = q_worker.select(s_prime, epsilon)
            else:
                action = _FIXED_ACTIONS.get(atype, 0)

            # Store this step's state for next update
            memory[agent] = (s_prime, action)
            env.step(action)

        ep_done = sum(
            1 for v in env.state_snapshot["orders"].values()
            if v["status"] == "complete"
        )
        episode_rewards.append(ep_reward)
        episode_lengths.append(env.simulation_step)
        episode_agent_turns.append(env.agent_turns)
        order_completions.append(ep_done)

        # Presentation/UI integration point.  Keeping this callback generic
        # means the trainer remains usable without pygame and with other UIs.
        w5 = episode_rewards[max(0, ep - 4): ep + 1]
        if progress_callback is not None:
            keep_training = progress_callback({
                "episode": ep + 1,
                "total_episodes": n_episodes,
                "reward": ep_reward,
                "moving_average": float(np.mean(w5)),
                "epsilon": epsilon,
                "completed_orders": ep_done,
                "cycles": env.simulation_step,
                "agent_turns": env.agent_turns,
                "q_states": len(q_worker.Q),
                "q_updates": q_worker.n_updates,
                "factory_name": cfg.name,
                "algorithm": "Q-Learning tabular · workers",
                "episode_seconds": time.time() - t0,
            })
            if keep_training is False:
                break

        # Print progress every 5 episodes
        if (ep + 1) % 5 == 0 or ep == 0:
            avg5  = np.mean(w5)
            fill  = int((ep + 1) / n_episodes * 28)
            bar   = "█" * fill + "░" * (28 - fill)
            t_ep  = time.time() - t0
            print(
                f"  Ep {ep+1:>3d}/{n_episodes}  [{bar}]  "
                f"ε={epsilon:.3f}  R(avg5)={avg5:+8.1f}  "
                f"orders={ep_done}  {t_ep:.1f}s"
            )

    print(f"\n  Q-table states visited (workers): {len(q_worker.Q):,}")
    print(f"  Total Q-updates                 : {q_worker.n_updates:,}")

    return {
        "Q_workers":         dict(q_worker.Q),
        "episode_rewards":   episode_rewards,
        "episode_lengths":   episode_lengths,
        "episode_agent_turns": episode_agent_turns,
        "order_completions": order_completions,
        "n_episodes":        len(episode_rewards),
        "factory_name":      cfg.name,
        "alpha":             ALPHA,
        "gamma":             GAMMA,
        "seed":              seed,
    }

# ---------------------------------------------------------------------------
# Summary & stats
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    rewards = results["episode_rewards"]
    n       = len(rewards)
    window  = max(1, n // 5)

    first_avg = np.mean(rewards[:window])
    last_avg  = np.mean(rewards[-window:])
    best_ep   = int(np.argmax(rewards)) + 1
    best_r    = max(rewards)
    orders    = results["order_completions"]
    comp_pct  = np.mean(orders) * 100

    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  Episodes          : {n}")
    print(f"  First {window:2d} ep avg R : {first_avg:+.2f}")
    print(f"  Last  {window:2d} ep avg R : {last_avg:+.2f}")
    print(f"  Improvement       : {last_avg - first_avg:+.2f}")
    print(f"  Best episode      : #{best_ep}  (R = {best_r:+.1f})")
    print(f"  Order completion  : {comp_pct:.1f} %  "
          f"({sum(1 for x in orders if x > 0)}/{n} eps with ≥1 completed order)")

    # ASCII sparkline of reward history
    hist     = rewards[-50:]
    lo, hi   = min(hist), max(hist)
    rng      = max(hi - lo, 1.0)
    bar_chars = "▁▂▃▄▅▆▇█"
    spark    = "".join(
        bar_chars[min(7, int((v - lo) / rng * 7.99))] for v in hist
    )
    print(f"\n  Reward curve (last {len(hist)} eps):")
    print(f"  {spark}")
    print(f"  min={lo:+.1f}  max={hi:+.1f}\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MAS-DUO Q-Learning Trainer")
    parser.add_argument(
        "--config", type=str, default="config/factory_example.json",
        help="Path to factory JSON config."
    )
    parser.add_argument(
        "--episodes", type=int, default=60,
        help="Number of training episodes (default 60)."
    )
    parser.add_argument(
        "--out", type=str, default="train/ql_policy.pkl",
        help="Output path for the trained policy pickle."
    )
    parser.add_argument(
        "--seed", type=int, default=7,
        help="Base seed for reproducible environment and exploration RNGs."
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show the live client-facing training dashboard."
    )
    parser.add_argument(
        "--visual-fps", type=int, default=12,
        help="Dashboard refresh speed (default 12 episodes/second)."
    )
    args = parser.parse_args()

    root        = Path(__file__).parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    dashboard = None
    if args.visualize:
        from logistics_env.rendering import TrainingRenderer
        dashboard = TrainingRenderer(
            fps=args.visual_fps,
            screenshot_dir=root / "artifacts" / "screenshots",
        )

    env = LogisticsMaEnv(config_path=config_path, render_mode=None)
    results = train(
        env, n_episodes=args.episodes, seed=args.seed,
        progress_callback=dashboard.update if dashboard else None,
    )
    env.close()

    print_summary(results)

    # Save trained policy
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as f:
        pickle.dump(results, f)
    print(f"  Policy saved  → {out_path}")

    # Also save episode stats as JSON
    json_path = out_path.with_suffix(".json")
    with json_path.open("w") as f:
        json.dump(
            {
                "episode_rewards":   results["episode_rewards"],
                "episode_lengths":   results["episode_lengths"],
                "episode_agent_turns": results["episode_agent_turns"],
                "order_completions": results["order_completions"],
                "factory_name":      results["factory_name"],
            },
            f, indent=2,
        )
    print(f"  Stats saved   → {json_path}")
    print()

    if dashboard is not None:
        dashboard.finish(wait=True)
        dashboard.close()


if __name__ == "__main__":
    main()

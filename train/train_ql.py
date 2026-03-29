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
from typing import Dict, Tuple

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

    def __init__(self, n_actions: int):
        self.n_actions   = n_actions
        self.Q: Dict     = defaultdict(lambda: {a: 0.0 for a in range(n_actions)})
        self.n_updates   = 0
        self.total_reward = 0.0

    def select(self, state: tuple, epsilon: float) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        return max(self.Q[state], key=self.Q[state].get)

    def update(self, s: tuple, a: int, r: float, s_next: tuple) -> None:
        q_next  = max(self.Q[s_next].values())
        old_q   = self.Q[s][a]
        self.Q[s][a] = old_q + ALPHA * (r + GAMMA * q_next - old_q)
        self.n_updates    += 1
        self.total_reward += r

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(env: LogisticsMaEnv, n_episodes: int) -> dict:
    """
    Runs the Q-Learning training loop and returns results dict.
    """
    cfg       = env.factory_cfg
    max_steps = cfg.sim_params.max_steps

    # Q-tables — workers are the primary learning agents
    q_worker  = QTableAgent(n_actions=10)   # 10 WorkerActions

    # Per-agent memory: {agent_id: (prev_discrete_state, prev_action)}
    memory: Dict = {}

    episode_rewards:    list = []
    episode_lengths:    list = []
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

        env.reset(seed=ep * 13 + 7)
        memory.clear()

        ep_reward  = 0.0
        step_count = 0
        t0         = time.time()

        while env.agents and step_count < max_steps:
            agent = env.agent_selection
            obs, cum_rew, term, trunc, _info = env.last()

            atype   = env._agent_type(agent)
            disc_fn = _DISC_FN[atype]
            s_prime = disc_fn(obs)

            # Q-update for this agent using its accumulated reward since last visit
            if agent in memory and not term and not trunc and atype == "worker":
                s_prev, a_prev = memory[agent]
                q_worker.update(s_prev, a_prev, cum_rew, s_prime)

            # Action selection
            if atype == "worker":
                action = q_worker.select(s_prime, epsilon)
            else:
                action = _FIXED_ACTIONS.get(atype, 0)

            # Store this step's state for next update
            memory[agent] = (s_prime, action)
            ep_reward    += cum_rew
            step_count   += 1

            env.step(action)

        ep_done = sum(
            1 for v in env.state_snapshot["orders"].values()
            if v["status"] == "complete"
        )
        episode_rewards.append(ep_reward)
        episode_lengths.append(step_count)
        order_completions.append(ep_done)

        # Print progress every 5 episodes
        if (ep + 1) % 5 == 0 or ep == 0:
            w5    = episode_rewards[max(0, ep - 4): ep + 1]
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
        "order_completions": order_completions,
        "n_episodes":        n_episodes,
        "factory_name":      cfg.name,
        "alpha":             ALPHA,
        "gamma":             GAMMA,
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
    args = parser.parse_args()

    root        = Path(__file__).parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    env     = LogisticsMaEnv(config_path=config_path, render_mode=None)
    results = train(env, n_episodes=args.episodes)
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
                "order_completions": results["order_completions"],
                "factory_name":      results["factory_name"],
            },
            f, indent=2,
        )
    print(f"  Stats saved   → {json_path}")
    print()


if __name__ == "__main__":
    main()

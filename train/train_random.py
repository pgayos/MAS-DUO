"""
train_random.py — Random policy training script (baseline)
============================================================
Runs several episodes with random actions to verify that the environment
works correctly and gather baseline statistics.

Usage:
    python train/train_random.py --config config/factory_example.json --episodes 5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Add root directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from logistics_env import LogisticsMaEnv


def run_episode(env: LogisticsMaEnv, seed: int) -> dict:
    """Runs one episode with random policy."""
    env.reset(seed=seed)

    total_rewards = {}
    step_count    = 0
    start_time    = time.time()

    while env.agents:
        agent = env.agent_selection

        # Random policy
        act_space = env.action_space(agent)
        obs, rew, term, trunc, info = env.last()
        action = None if term or trunc else act_space.sample()
        env.step(action)

        total_rewards[agent]  = total_rewards.get(agent, 0.0) + rew
        step_count           += 1

    elapsed = time.time() - start_time
    snapshot = env.state_snapshot

    return {
        "steps":           env.simulation_step,
        "agent_turns":     step_count,
        "elapsed_sec":     elapsed,
        "total_rewards":   total_rewards,
        "order_summary":   snapshot["orders"],
        "energy_consumed": snapshot["total_energy_consumed"],
        "agent_trails":    {
            "products": {
                aid: len(a.state_trail)
                for aid, a in env._products.items()
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="MAS-DUO Random Baseline")
    parser.add_argument("--config",   type=str, default="config/factory_example.json")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render",   action="store_true", help="Renderizar con Pygame")
    parser.add_argument("--seed",     type=int, default=7)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / args.config

    render_mode = "human" if args.render else None
    env = LogisticsMaEnv(config_path=config_path, render_mode=render_mode)
    print(f"\n{'='*60}")
    print(f"  MAS-DUO — Multi-Agent Logistics Environment")
    print(f"  Factory  : {env.factory_cfg.name}")
    print(f"  Grid     : {env.factory_cfg.grid.width}x{env.factory_cfg.grid.height}")
    print(f"  Agents   : {len(env.factory_cfg.workers)} workers, "
          f"{len(env.factory_cfg.robots)} robots, "
          f"{len(env.factory_cfg.conveyor_belts)} conveyors")
    print(f"  Orders  : {len(env.factory_cfg.orders)}")
    print(f"{'='*60}\n")

    all_results = []

    for ep in range(args.episodes):
        print(f"── Episode {ep+1}/{args.episodes} ──")
        result = run_episode(env, seed=args.seed + ep)
        all_results.append(result)

        print(f"  Steps       : {result['steps']}")
        print(f"  Time        : {result['elapsed_sec']:.2f}s")
        print(f"  Total energy: {result['energy_consumed']:.2f}")
        print(f"  Orders:")
        for oid, info in result["order_summary"].items():
            print(f"    {oid}: {info['dispatched']}/{info['needed']} "
                  f"[{info['status']}] "
                  f"(deadline={info['deadline']}, finish={info['finish_step']})")

        avg_r = np.mean(list(result["total_rewards"].values())) if result["total_rewards"] else 0.0
        print(f"  Avg reward  : {avg_r:.2f}")
        print()

    env.close()

    # ── Resumen final ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    completed = sum(
        1 for r in all_results
        if any(v["status"] == "complete" for v in r["order_summary"].values())
    )
    avg_energy = np.mean([r["energy_consumed"] for r in all_results])
    print(f"  Episodes with ≥1 completed order: {completed}/{args.episodes}")
    print(f"  Average energy per episode       : {avg_energy:.2f}")
    print()

    # Guardar resultados
    out_path = Path(__file__).parent / "results_random.json"
    with out_path.open("w") as f:
        # Hacer serializable
        for r in all_results:
            r["total_rewards"] = {k: float(v) for k, v in r["total_rewards"].items()}
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()

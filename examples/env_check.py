"""
example_env_check.py — Quick environment check
==============================================
Verifies that the environment is compatible with the PettingZoo API
and displays the 4W state of all agents after a reset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from logistics_env import LogisticsMaEnv
from logistics_env.agents.base_agent import AgentWhy


def main():
    config_path = Path(__file__).parent.parent / "config" / "factory_example.json"
    env = LogisticsMaEnv(config_path=config_path, render_mode=None)
    env.reset(seed=42)

    print(f"\n{'='*70}")
    print(f" MAS-DUO — Initial environment state")
    print(f" Factory: {env.factory_cfg.name}")
    print(f"{'='*70}\n")

    print(f"Agentes activos ({len(env.agents)}):")
    for aid in env.agents:
        atype = env._agent_type(aid)
        obs_shape = env.observation_space(aid).shape
        act_size  = env.action_space(aid).n
        print(f"  [{atype:10s}] {aid[:50]:<50} obs={obs_shape}  act={act_size}")

    print(f"\n{'─'*70}")
    print(f" 4W STATE (What / Where / When / Why) — all agents")
    print(f"{'─'*70}\n")

    snapshot = env.state_snapshot
    sections = {
        "Products (EPC)": snapshot["products"],
        "Workers":        snapshot["workers"],
        "Robots":         snapshot["robots"],
        "Conveyors":      snapshot["conveyors"],
    }

    for section_name, agents_dict in sections.items():
        print(f"  ▶ {section_name}")
        for aid, state in agents_dict.items():
            print(f"    What : {state['what'][:60]}")
            print(f"    Where: ({state['where']['x']}, {state['where']['y']})  "
                  f"Zone={state['zone_id']}")
            print(f"    When : step {state['when']}")
            print(f"    Why  : {state['why']}")
            print()

    print(f"{'─'*70}")
    print(f" ORDERS")
    print(f"{'─'*70}\n")
    for oid, info in snapshot["orders"].items():
        print(f"  Order {oid}")
        print(f"    Status   : {info['status']}")
        print(f"    Products : {info['dispatched']}/{info['needed']}")
        print(f"    Deadline : step {info['deadline']}")
        print()

    # Ejecutar 5 pasos aleatorios
    print(f"\n{'─'*70}")
    print(f" FIRST 5 STEPS (random actions)")
    print(f"{'─'*70}\n")

    import numpy as np
    for i in range(5):
        if not env.agents:
            break
        agent = env.agent_selection
        obs, rew, term, trunc, info = env.last()
        action = env.action_space(agent).sample()
        env.step(action)
        atype = env._agent_type(agent) if agent in (
            list(env._products) + list(env._workers) +
            list(env._robots) + list(env._conveyors)
        ) else "?"
        print(f"  step={i+1:02d} | agent={agent[:40]:<40} | action={action} | "
              f"rew={rew:+.2f} | term={term}")

    env.close()
    print(f"\n✓ Environment verified successfully.\n")


if __name__ == "__main__":
    main()

"""
MAS-DUO Logistics Environment
==============================
Multi-agent logistics environment based on PettingZoo + Gymnasium.

Agents:
  - ProductAgent   : each product is an agent with EPC state (What/Where/When/Why)
  - WorkerAgent    : human operators
  - RobotAgent     : AGV robots
  - ConveyorAgent  : conveyor belts

The state of all agents follows the standard:
  What  → identifier (EPC for products / agent_id for others)
  Where → position (x, y) in the factory grid
  When  → simulation step / timestamp
  Why   → current reason/action of the agent
"""

from logistics_env.logistics_maenv import LogisticsMaEnv

__all__ = ["LogisticsMaEnv"]
__version__ = "0.1.0"

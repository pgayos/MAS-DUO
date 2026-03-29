"""
ConveyorAgent — Conveyor belt agent
============================================
The belt is a passive/active agent: it can be running or stopped,
and in each step advances the products along its path.

Actions:
  0 STOP       - stop the belt (energy saving)
  1 RUN        - start running (normal speed)
  2 RUN_FAST   - increase speed (more energy)
  3 REVERSE    - reverse direction (emergency)

4W State:
  What  → conveyor_id
  Where → starting cell of the path
  When  → sim_step
  Why   → STOPPED / CONVEYING / JAMMED
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logistics_env.agents.base_agent import BaseAgent, AgentWhy, Position
from logistics_env.config_loader import ConveyorConfig


class ConveyorAction(IntEnum):
    STOP     = 0
    RUN      = 1
    RUN_FAST = 2
    REVERSE  = 3


_N_ACTIONS = len(ConveyorAction)


class ConveyorAgent(BaseAgent):
    """
    Conveyor belt agent.

    Maintains a queue of products (EPC URIs) that move
    cell by cell on each step while the belt is running.

    4W State
    --------
    What  → conveyor_id  (e.g. "CB-001")
    Where → position of the first path cell
    When  → simulation step
    Why   → STOPPED / CONVEYING / JAMMED
    """

    def __init__(self, cfg: ConveyorConfig):
        sx, sy = cfg.path[0] if cfg.path else (0, 0)
        super().__init__(agent_id=cfg.id, start_pos=Position(sx, sy))

        self.cfg:            ConveyorConfig = cfg
        self.path:           List[Position] = [Position(x, y) for (x, y) in cfg.path]
        self.is_running:     bool           = True
        self.is_reversed:    bool           = False
        self.speed:          int            = cfg.speed_cells_per_step

        # slot[i] = EPC URI del producto en path[i], o None
        self.slots:          List[Optional[str]] = [None] * len(self.path)
        self.energy_consumed: float = 0.0
        self.total_jams:      int   = 0

        self.update_state(step=0, why=AgentWhy.CONVEYING if self.is_running else AgentWhy.STOPPED)

    # -----------------------------------------------------------------------
    # Propiedades
    # -----------------------------------------------------------------------

    @property
    def start_pos(self) -> Position:
        return self.path[0] if self.path else Position(0, 0)

    @property
    def end_pos(self) -> Position:
        return self.path[-1] if self.path else Position(0, 0)

    @property
    def is_jammed(self) -> bool:
        # Considered jammed if the last slot is occupied
        return self.slots[-1] is not None if self.slots else False

    @property
    def occupancy_fraction(self) -> float:
        occupied = sum(1 for s in self.slots if s is not None)
        return occupied / len(self.slots) if self.slots else 0.0

    # -----------------------------------------------------------------------
    # Belt product management
    # -----------------------------------------------------------------------

    def can_accept(self) -> bool:
        """True if the first slot is free."""
        return len(self.slots) > 0 and self.slots[0] is None

    def load_product(self, epc_uri: str) -> bool:
        """Loads a product into the first slot. Returns True if successful."""
        if self.can_accept():
            self.slots[0] = epc_uri
            return True
        return False

    def advance(self, sim_step: int) -> List[str]:
        """
        Advances all slots by `speed` positions.
        Returns the list of EPCs that have reached the end (exit the belt).
        """
        if not self.is_running:
            return []

        discharged: List[str] = []
        steps = self.speed if not self.is_reversed else -self.speed

        if steps > 0:
            # Advance towards the end
            for _ in range(abs(steps)):
                # Last slot is unloaded if occupied
                if self.slots and self.slots[-1] is not None:
                    discharged.append(self.slots[-1])
                    self.slots[-1] = None

                # Shift → each slot moves to the next
                if len(self.slots) > 1:
                    for i in range(len(self.slots) - 1, 0, -1):
                        if self.slots[i] is None and self.slots[i-1] is not None:
                            self.slots[i]   = self.slots[i-1]
                            self.slots[i-1] = None
        else:
            # Reverse → backward
            for _ in range(abs(steps)):
                if self.slots and self.slots[0] is not None:
                    discharged.append(self.slots[0])
                    self.slots[0] = None
                if len(self.slots) > 1:
                    for i in range(len(self.slots) - 1):
                        if self.slots[i] is None and self.slots[i+1] is not None:
                            self.slots[i]   = self.slots[i+1]
                            self.slots[i+1] = None

        # Check jam
        if self.is_jammed:
            self.total_jams += 1

        # Energy consumption
        energy = self.cfg.energy_per_step * (1.5 if self.speed > self.cfg.speed_cells_per_step else 1.0)
        self.energy_consumed += energy

        why = AgentWhy.JAMMED if self.is_jammed else AgentWhy.CONVEYING
        self.update_state(step=sim_step, why=why)

        return discharged

    # -----------------------------------------------------------------------
    # BaseAgent interface
    # -----------------------------------------------------------------------

    def step(self, action: int, grid_state: dict, sim_step: int) -> dict:
        ca     = ConveyorAction(action) if action < _N_ACTIONS else ConveyorAction.RUN
        reward = 0.0
        done   = False
        info: dict = {"conveyor_id": self.agent_id, "action": ca.name}

        if ca == ConveyorAction.STOP:
            self.is_running = False
            self.speed      = 0
            self.update_state(step=sim_step, why=AgentWhy.STOPPED)
            reward -= 0.3   # penalty for stopping the flow

        elif ca == ConveyorAction.RUN:
            self.is_running  = True
            self.is_reversed = False
            self.speed       = self.cfg.speed_cells_per_step
            reward += 0.1

        elif ca == ConveyorAction.RUN_FAST:
            self.is_running  = True
            self.is_reversed = False
            self.speed       = self.cfg.speed_cells_per_step + 1
            reward -= self.cfg.energy_per_step * 0.5   # energy penalty for excess

        elif ca == ConveyorAction.REVERSE:
            self.is_running  = True
            self.is_reversed = True
            self.speed       = self.cfg.speed_cells_per_step
            reward -= 0.5   # reversing is always costly

        # Advance products on the belt
        discharged = self.advance(sim_step)
        if discharged:
            reward += len(discharged) * 1.0   # each product that exits is positive
            info["discharged"] = discharged

        if self.is_jammed:
            reward -= 2.0
            info["jammed"] = True

        # Continuous energy penalty
        reward += self.cfg.energy_per_step * -0.05

        return {"reward": reward, "done": done, "info": info}

    def get_observation(self, grid_state: dict) -> np.ndarray:
        """
        Observation vector of the belt:
          [occupancy_fraction, is_running, is_jammed, is_reversed, energy_rate]
        """
        return np.array([
            self.occupancy_fraction,
            1.0 if self.is_running else 0.0,
            1.0 if self.is_jammed  else 0.0,
            1.0 if self.is_reversed else 0.0,
            self.cfg.energy_per_step / 2.0,   # normalizado asumiendo max ~2
        ], dtype=np.float32)

    def action_space_size(self) -> int:
        return _N_ACTIONS

    # -----------------------------------------------------------------------
    # Real-time position of each product on the belt
    # -----------------------------------------------------------------------

    def product_positions(self) -> Dict[str, Position]:
        """Returns {epc_uri: grid_position} for each product on the belt."""
        result = {}
        for i, epc in enumerate(self.slots):
            if epc is not None and i < len(self.path):
                result[epc] = self.path[i]
        return result

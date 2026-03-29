"""
WorkerAgent — Human operator agent
===================================
Available actions:
  0 IDLE        - do nothing
  1 MOVE_NORTH  - move north
  2 MOVE_SOUTH  - move south
  3 MOVE_EAST   - move east
  4 MOVE_WEST   - move west
  5 PICK        - pick up product at current position
  6 PLACE       - place product at current position
  7 PROCESS     - process carried product (if in a valid zone)
  8 SCAN        - scan product EPC (generates traceability event)
  9 REST        - rest (recovers energy)

4W State:
  What  → worker_id
  Where → position (x, y)
  When  → sim_step
  Why   → IDLE / MOVING / PICKING / PLACING / PROCESSING / SCANNING / RESTING
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Optional

import numpy as np

from logistics_env.agents.base_agent import BaseAgent, AgentWhy, Position
from logistics_env.config_loader import WorkerConfig


class WorkerAction(IntEnum):
    IDLE         = 0
    MOVE_NORTH   = 1
    MOVE_SOUTH   = 2
    MOVE_EAST    = 3
    MOVE_WEST    = 4
    PICK         = 5
    PLACE        = 6
    PROCESS      = 7
    SCAN         = 8
    REST         = 9


_MOVE_DELTAS = {
    WorkerAction.MOVE_NORTH: Position(0, -1),
    WorkerAction.MOVE_SOUTH: Position(0,  1),
    WorkerAction.MOVE_EAST:  Position(1,  0),
    WorkerAction.MOVE_WEST:  Position(-1, 0),
}

_N_ACTIONS = len(WorkerAction)


class WorkerAgent(BaseAgent):
    """
    Human operator agent in the logistics environment.

    4W State
    --------
    What  → worker_id  (e.g. "WRK-001")
    Where → position on the grid
    When  → simulation step
    Why   → current motive/action (AgentWhy)
    """

    def __init__(self, cfg: WorkerConfig):
        sx, sy = cfg.start_position
        super().__init__(
            agent_id  = cfg.id,
            start_pos = Position(sx, sy),
        )
        self.cfg:           WorkerConfig    = cfg
        self.fatigue:       float           = 0.0      # 0.0 = fresh, 1.0 = exhausted
        self.energy:        float           = 1.0      # available energy fraction
        self.carrying:      Optional[str]   = None     # EPC URI of the carried product
        self.total_energy_used: float       = 0.0
        self.scanned_epcs:  List[str]       = []       # EPC scan registry

    # -----------------------------------------------------------------------
    # Propiedades
    # -----------------------------------------------------------------------

    @property
    def is_tired(self) -> bool:
        return self.fatigue >= self.cfg.rest_threshold

    @property
    def is_carrying(self) -> bool:
        return self.carrying is not None

    # -----------------------------------------------------------------------
    # Action logic
    # -----------------------------------------------------------------------

    def _consume_energy(self) -> None:
        self.energy = max(0.0, self.energy - self.cfg.energy_per_action)
        self.fatigue = min(1.0, self.fatigue + self.cfg.fatigue_rate)
        self.total_energy_used += self.cfg.energy_per_action

    def step(self, action: int, grid_state: dict, sim_step: int) -> dict:
        wa      = WorkerAction(action) if action < _N_ACTIONS else WorkerAction.IDLE
        reward  = 0.0
        done    = False
        info: dict = {"worker_id": self.agent_id, "action": wa.name}

        # ── Rest ────────────────────────────────────────────────────────────
        if wa == WorkerAction.REST or self.energy <= 0.0:
            self.fatigue  = max(0.0, self.fatigue - 0.05)
            self.energy   = min(1.0, self.energy  + 0.1)
            reward -= 0.3   # penalty for not producing
            self.update_state(step=sim_step, why=AgentWhy.RESTING)
            return {"reward": reward, "done": done, "info": info}

        # ── Movement ───────────────────────────────────────────────────────
        if wa in _MOVE_DELTAS:
            delta    = _MOVE_DELTAS[wa]
            new_pos  = Position(self.position.x + delta.x, self.position.y + delta.y)
            grid     = grid_state["grid_obj"]
            moved    = grid.move_agent(self.agent_id, self.position, new_pos)

            if moved:
                zone   = grid.zone_of(new_pos)
                zid    = zone.id if zone else None
                # Check if the zone is allowed
                if zid and self.cfg.allowed_zones and zid not in self.cfg.allowed_zones:
                    # revert
                    grid.move_agent(self.agent_id, new_pos, self.position)
                    reward -= 1.0
                    info["warning"] = f"Zone {zid} not allowed for {self.agent_id}"
                else:
                    self._consume_energy()
                    self.update_state(
                        step=sim_step, where=new_pos,
                        why=AgentWhy.MOVING, zone_id=zid,
                    )
                    reward += 0.05
            else:
                reward -= 0.2   # collision / out of bounds
            return {"reward": reward, "done": done, "info": info}

        # ── Pick ────────────────────────────────────────────────────────────
        if wa == WorkerAction.PICK:
            if self.is_carrying:
                reward -= 0.5   # ya lleva algo
            else:
                products_here = grid_state.get("products_at", {}).get(
                    (self.position.x, self.position.y), []
                )
                if products_here:
                    epc_uri = products_here[0]
                    self.carrying = epc_uri
                    self._consume_energy()
                    self.update_state(step=sim_step, why=AgentWhy.PICKING)
                    reward += 1.0
                    info["picked"] = epc_uri
                else:
                    reward -= 0.3
            return {"reward": reward, "done": done, "info": info}

        # ── Place ───────────────────────────────────────────────────────────
        if wa == WorkerAction.PLACE:
            if not self.is_carrying:
                reward -= 0.5
            else:
                info["placed"] = self.carrying
                self.carrying = None
                self._consume_energy()
                self.update_state(step=sim_step, why=AgentWhy.PLACING)
                reward += 0.5
            return {"reward": reward, "done": done, "info": info}

        # ── Process ─────────────────────────────────────────────────────────
        if wa == WorkerAction.PROCESS:
            zone = grid_state["grid_obj"].zone_of(self.position)
            if zone and zone.type == "process" and self.is_carrying:
                self._consume_energy()
                self.update_state(step=sim_step, why=AgentWhy.PROCESSING)
                reward += 2.0
                info["processing"] = self.carrying
            else:
                reward -= 0.5
            return {"reward": reward, "done": done, "info": info}

        # ── Scan (generates EPC event) ─────────────────────────────────────
        if wa == WorkerAction.SCAN:
            if self.is_carrying:
                self.scanned_epcs.append(self.carrying)
                self._consume_energy()
                self.update_state(step=sim_step, why=AgentWhy.SCANNING)
                reward += 0.3
                info["scanned_epc"] = self.carrying
            else:
                reward -= 0.1
            return {"reward": reward, "done": done, "info": info}

        # ── Idle ────────────────────────────────────────────────────────────
        reward -= 0.5
        self.update_state(step=sim_step, why=AgentWhy.IDLE)
        return {"reward": reward, "done": done, "info": info}

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------

    def get_observation(self, grid_state: dict) -> np.ndarray:
        """
        Local observation vector for the worker:
          [x, y, energy, fatigue, is_carrying, allowed_zone_flags...(up to 6)]
        """
        zone = grid_state["grid_obj"].zone_of(self.position)
        zone_type_enc = 0
        if zone:
            zone_types = ["input","storage","process","packing","output","charging"]
            zone_type_enc = zone_types.index(zone.type) + 1 if zone.type in zone_types else 0

        return np.array([
            self.position.x / grid_state.get("grid_w", 20),
            self.position.y / grid_state.get("grid_h", 15),
            self.energy,
            self.fatigue,
            1.0 if self.is_carrying else 0.0,
            zone_type_enc / 6.0,
        ], dtype=np.float32)

    def action_space_size(self) -> int:
        return _N_ACTIONS

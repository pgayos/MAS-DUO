"""
RobotAgent — AGV Robot Agent (Automated Guided Vehicle)
=========================================================
Actions:
  0 IDLE
  1 MOVE_NORTH
  2 MOVE_SOUTH
  3 MOVE_EAST
  4 MOVE_WEST
  5 LIFT      - pick up product under the robot
  6 DROP      - place product at current position
  7 CHARGE    - start recharging (only in charging zone)
  8 NAVIGATE  - navigate automatically towards target (requires target_pos in grid_state)

4W State:
  What  → robot_id
  Where → position (x, y)
  When  → sim_step
  Why   → IDLE / NAVIGATING / LIFTING / DROPPING / CHARGING
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np

from logistics_env.agents.base_agent import BaseAgent, AgentWhy, Position
from logistics_env.config_loader import RobotConfig


class RobotAction(IntEnum):
    IDLE       = 0
    MOVE_NORTH = 1
    MOVE_SOUTH = 2
    MOVE_EAST  = 3
    MOVE_WEST  = 4
    LIFT       = 5
    DROP       = 6
    CHARGE     = 7
    NAVIGATE   = 8


_MOVE_DELTAS = {
    RobotAction.MOVE_NORTH: Position(0, -1),
    RobotAction.MOVE_SOUTH: Position(0,  1),
    RobotAction.MOVE_EAST:  Position(1,  0),
    RobotAction.MOVE_WEST:  Position(-1, 0),
}

_N_ACTIONS = len(RobotAction)


class RobotAgent(BaseAgent):
    """
    AGV robot agent in the logistics environment.

    4W State
    --------
    What  → robot_id   (e.g. "ROB-001")
    Where → position on the grid
    When  → simulation step
    Why   → current motive/action
    """

    def __init__(self, cfg: RobotConfig):
        sx, sy = cfg.start_position
        super().__init__(agent_id=cfg.id, start_pos=Position(sx, sy))

        self.cfg:              RobotConfig     = cfg
        self.energy:           float           = cfg.energy_capacity
        self.carrying:         Optional[str]   = None    # EPC URI
        self.total_energy_used: float          = 0.0

        # Current A* path cache
        self._path:            List[Position]  = []
        self._path_target:     Optional[Position] = None

    # -----------------------------------------------------------------------
    # Propiedades
    # -----------------------------------------------------------------------

    @property
    def is_carrying(self) -> bool:
        return self.carrying is not None

    @property
    def energy_fraction(self) -> float:
        return self.energy / self.cfg.energy_capacity

    @property
    def needs_charge(self) -> bool:
        return self.energy <= self.cfg.recharge_threshold

    # -----------------------------------------------------------------------
    # Action logic
    # -----------------------------------------------------------------------

    def _consume(self, amount: float) -> None:
        self.energy = max(0.0, self.energy - amount)
        self.total_energy_used += amount

    def step(self, action: int, grid_state: dict, sim_step: int) -> dict:
        ra     = RobotAction(action) if action < _N_ACTIONS else RobotAction.IDLE
        reward = 0.0
        done   = False
        info: dict = {"robot_id": self.agent_id, "action": ra.name}

        # ── Low battery → force charge ─────────────────────────────────────
        if self.energy <= 0.0:
            reward -= 1.0
            self.update_state(step=sim_step, why=AgentWhy.IDLE)
            info["warning"] = "Out of energy"
            return {"reward": reward, "done": done, "info": info}

        # ── Carga ───────────────────────────────────────────────────────────
        if ra == RobotAction.CHARGE:
            zone = grid_state["grid_obj"].zone_of(self.position)
            if zone and zone.type == "charging":
                self.energy = min(
                    self.cfg.energy_capacity,
                    self.energy + self.cfg.recharge_rate
                )
                self.update_state(step=sim_step, why=AgentWhy.CHARGING)
                reward += 0.1 if self.needs_charge else -0.3
            else:
                reward -= 0.5   # not in a charging zone
            return {"reward": reward, "done": done, "info": info}

        # ── Movimiento manual ───────────────────────────────────────────────
        if ra in _MOVE_DELTAS:
            return self._do_move(_MOVE_DELTAS[ra], grid_state, sim_step, reward, done, info)

        # ── Automatic navigation (A*) ──────────────────────────────────────
        if ra == RobotAction.NAVIGATE:
            target = grid_state.get("robot_targets", {}).get(self.agent_id)
            if target:
                target_pos = Position(target[0], target[1])
                if not self._path or self._path_target != target_pos:
                    self._path = grid_state["grid_obj"].astar(
                        self.position, target_pos, self.agent_id
                    ) or []
                    self._path_target = target_pos

                if self._path:
                    next_pos = self._path[0]
                    delta    = Position(next_pos.x - self.position.x,
                                        next_pos.y - self.position.y)
                    result   = self._do_move(delta, grid_state, sim_step, reward, done, info)
                    if result["info"].get("moved"):
                        self._path.pop(0)
                    return result
            reward -= 0.2
            return {"reward": reward, "done": done, "info": info}

        # ── Lift ────────────────────────────────────────────────────────────
        if ra == RobotAction.LIFT:
            if self.is_carrying:
                reward -= 0.5
            else:
                products_here = grid_state.get("products_at", {}).get(
                    (self.position.x, self.position.y), []
                )
                if products_here:
                    self.carrying = products_here[0]
                    self._consume(self.cfg.energy_per_lift)
                    self.update_state(step=sim_step, why=AgentWhy.LIFTING)
                    reward += 1.5
                    info["lifted"] = self.carrying
                else:
                    reward -= 0.3
            return {"reward": reward, "done": done, "info": info}

        # ── Drop ────────────────────────────────────────────────────────────
        if ra == RobotAction.DROP:
            if not self.is_carrying:
                reward -= 0.5
            else:
                info["dropped"] = self.carrying
                self.carrying = None
                self._consume(self.cfg.energy_per_lift * 0.5)
                self.update_state(step=sim_step, why=AgentWhy.DROPPING)
                reward += 1.0
            return {"reward": reward, "done": done, "info": info}

        # ── Idle ────────────────────────────────────────────────────────────
        reward -= 0.5
        self.update_state(step=sim_step, why=AgentWhy.IDLE)
        return {"reward": reward, "done": done, "info": info}

    def _do_move(
        self, delta: Position, grid_state: dict,
        sim_step: int, reward: float, done: bool, info: dict
    ) -> dict:
        new_pos = Position(self.position.x + delta.x, self.position.y + delta.y)
        grid    = grid_state["grid_obj"]
        moved   = grid.move_agent(self.agent_id, self.position, new_pos)

        if moved:
            self._consume(self.cfg.energy_per_move)
            zone = grid.zone_of(new_pos)
            self.update_state(
                step=sim_step, where=new_pos,
                why=AgentWhy.NAVIGATING,
                zone_id=zone.id if zone else None,
            )
            reward += 0.1
            info["moved"] = True
        else:
            reward -= 0.2
            info["moved"] = False

        return {"reward": reward, "done": done, "info": info}

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------

    def get_observation(self, grid_state: dict) -> np.ndarray:
        """
        Robot observation vector:
          [x_norm, y_norm, energy_frac, is_carrying, needs_charge]
        """
        return np.array([
            self.position.x / grid_state.get("grid_w", 20),
            self.position.y / grid_state.get("grid_h", 15),
            self.energy_fraction,
            1.0 if self.is_carrying else 0.0,
            1.0 if self.needs_charge else 0.0,
        ], dtype=np.float32)

    def action_space_size(self) -> int:
        return _N_ACTIONS

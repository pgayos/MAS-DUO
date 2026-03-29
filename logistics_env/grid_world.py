"""
GridWorld — Grid representation for the logistics factory
==========================================================
Maintains the physical layer of the environment: cells, zones, belts,
collisions and A* navigation.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple

from logistics_env.agents.base_agent import Position
from logistics_env.config_loader import FactoryConfig, ZoneConfig, ConveyorConfig


# ---------------------------------------------------------------------------
# Tipo de celda
# ---------------------------------------------------------------------------

class CellType:
    FLOOR      = 0
    WALL       = 1
    CONVEYOR   = 2
    CHARGING   = 3


class GridWorld:
    """
    2D grid of the factory.

    The cell layer distinguishes: free floor, wall, conveyor belt
    and charging zone. Provides:
      - Fast zone → cells lookup
      - Agent occupancy check
      - A* pathfinding for robot/worker movement
    """

    def __init__(self, cfg: FactoryConfig):
        self.width  = cfg.grid.width
        self.height = cfg.grid.height
        self.cfg    = cfg

        # cell_type[y][x]
        self.cell_type: List[List[int]] = [
            [CellType.FLOOR] * self.width for _ in range(self.height)
        ]

        # Zone_id → list of (x, y)
        self.zone_cells: Dict[str, List[Tuple[int, int]]] = {}

        # Belt cells → conveyor_id
        self.conveyor_cells: Dict[Tuple[int, int], str] = {}

        # Occupancy: (x,y) → agent_id o None
        self.occupancy: Dict[Tuple[int, int], Optional[str]] = {}

        self._build_from_config()

    # -----------------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------------

    def _build_from_config(self) -> None:
        # Mark zones
        for zone in self.cfg.zones:
            cells = []
            for dy in range(zone.height):
                for dx in range(zone.width):
                    cx, cy = zone.x + dx, zone.y + dy
                    if 0 <= cx < self.width and 0 <= cy < self.height:
                        cells.append((cx, cy))
                        if zone.type == "charging":
                            self.cell_type[cy][cx] = CellType.CHARGING
            self.zone_cells[zone.id] = cells

        # Mark conveyor belts
        for cb in self.cfg.conveyor_belts:
            for (cx, cy) in cb.path:
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    self.cell_type[cy][cx] = CellType.CONVEYOR
                    self.conveyor_cells[(cx, cy)] = cb.id

    # -----------------------------------------------------------------------
    # Occupancy
    # -----------------------------------------------------------------------

    def is_occupied(self, x: int, y: int) -> bool:
        return self.occupancy.get((x, y)) is not None

    def move_agent(self, agent_id: str, old: Position, new: Position) -> bool:
        """
        Moves an agent from old to new if new is not occupied.
        Returns True if the move was successful.
        """
        ox, oy = old.x, old.y
        nx, ny = new.x, new.y

        if not self.in_bounds(nx, ny):
            return False
        if self.is_occupied(nx, ny):
            return False

        # Release previous position
        self.occupancy.pop((ox, oy), None)
        self.occupancy[(nx, ny)] = agent_id
        return True

    def place_agent(self, agent_id: str, pos: Position) -> None:
        self.occupancy[(pos.x, pos.y)] = agent_id

    def remove_agent(self, agent_id: str, pos: Position) -> None:
        key = (pos.x, pos.y)
        if self.occupancy.get(key) == agent_id:
            del self.occupancy[key]

    # -----------------------------------------------------------------------
    # Navigation
    # -----------------------------------------------------------------------

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, pos: Position, passable_conveyors: bool = True) -> List[Position]:
        """Navigable orthogonal neighbours (excluding cells occupied by others)."""
        result = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = pos.x + dx, pos.y + dy
            if not self.in_bounds(nx, ny):
                continue
            if self.cell_type[ny][nx] == CellType.WALL:
                continue
            if not passable_conveyors and self.cell_type[ny][nx] == CellType.CONVEYOR:
                continue
            result.append(Position(nx, ny))
        return result

    def astar(
        self,
        start: Position,
        goal:  Position,
        agent_id: str = "",
        ignore_occupancy: bool = False,
    ) -> Optional[List[Position]]:
        """
        A* pathfinding from start to goal.
        Returns the list of positions (excludes start, includes goal),
        or None if no path exists.
        """
        def h(p: Position) -> int:
            return abs(p.x - goal.x) + abs(p.y - goal.y)

        counter   = 0   # tie-breaker para evitar comparar Position objects
        open_heap: list = []
        heapq.heappush(open_heap, (h(start), counter, 0, start))
        came_from: Dict[Position, Optional[Position]] = {start: None}
        g_score:   Dict[Position, int]                = {start: 0}

        while open_heap:
            _, _tb, cost, current = heapq.heappop(open_heap)

            if current == goal:
                # Reconstruct path
                path = []
                node: Optional[Position] = current
                while node is not None and node != start:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                return path

            for nb in self.neighbors(current):
                occupied = (
                    self.is_occupied(nb.x, nb.y)
                    and self.occupancy.get((nb.x, nb.y)) != agent_id
                )
                if occupied and not ignore_occupancy and nb != goal:
                    continue
                new_g = g_score[current] + 1
                if nb not in g_score or new_g < g_score[nb]:
                    g_score[nb] = new_g
                    came_from[nb] = current
                    f = new_g + h(nb)
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, new_g, nb))

        return None  # no path

    # -----------------------------------------------------------------------
    # Zone queries
    # -----------------------------------------------------------------------

    def zone_of(self, pos: Position) -> Optional[ZoneConfig]:
        return self.cfg.zone_at(pos.x, pos.y)

    def zone_center(self, zone_id: str) -> Optional[Position]:
        z = self.cfg.zone_by_id(zone_id)
        if z is None:
            return None
        return Position(z.x + z.width // 2, z.y + z.height // 2)

    def conveyor_at(self, pos: Position) -> Optional[str]:
        return self.conveyor_cells.get((pos.x, pos.y))

    # -----------------------------------------------------------------------
    # Snapshot para observaciones
    # -----------------------------------------------------------------------

    def to_observation_array(self) -> List[List[int]]:
        """
        Devuelve una matrix (height x width) con:
          0=libre, 1=pared, 2=cinta, 3=carga, 9=ocupado por agente.
        """
        grid = [
            [self.cell_type[y][x] for x in range(self.width)]
            for y in range(self.height)
        ]
        for (ox, oy) in self.occupancy:
            if self.occupancy[(ox, oy)] is not None:
                grid[oy][ox] = 9
        return grid

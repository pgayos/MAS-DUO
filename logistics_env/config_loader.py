"""
ConfigLoader — Load and validate factory configuration from JSON
===============================================================
Loads the JSON configuration file and exposes it as typed and validated
Python objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GridConfig:
    width:            int
    height:           int
    cell_size_meters: float = 1.0


@dataclass
class ZoneConfig:
    id:     str
    name:   str
    x:      int
    y:      int
    width:  int
    height: int
    type:   str   # "input" | "storage" | "process" | "packing" | "output" | "charging"


@dataclass
class ConveyorConfig:
    id:                  str
    name:                str
    energy_per_step:     float
    speed_cells_per_step: int
    path:                List[Tuple[int, int]]   # lista de (x, y)
    direction:           str
    capacity:            int


@dataclass
class RobotConfig:
    id:                  str
    name:                str
    start_position:      Tuple[int, int]
    energy_capacity:     float
    energy_per_move:     float
    energy_per_lift:     float
    speed_cells_per_step: int
    carrying_capacity:   int
    recharge_rate:       float
    recharge_threshold:  float


@dataclass
class WorkerConfig:
    id:                str
    name:              str
    start_position:    Tuple[int, int]
    role:              str
    energy_per_action: float
    fatigue_rate:      float
    rest_threshold:    float
    speed_cells_per_step: int
    allowed_zones:     List[str]


@dataclass
class ProductTypeConfig:
    item_reference:    str
    name:              str
    size:              int
    weight_kg:         float
    requires_processing: bool
    processing_steps:  List[str]     # secuencia de zone_ids
    process_time_steps: int


@dataclass
class OrderProductSpec:
    item_reference: str
    quantity:       int


@dataclass
class OrderConfig:
    order_id:       str
    products:       List[OrderProductSpec]
    deadline_steps: int
    priority:       str
    destination:    str


@dataclass
class GlobalPolicyConfig:
    """Global policy configuration (Equation 9 of the MAS-DUO thesis)."""
    mode: str = "static"             # "static" | "dynamic"
    A: float = 1.0                   # delay weight (Delay)
    B: float = 1.0                   # cost weight (Cost)
    C: float = 1.0                   # quality-of-service weight (QoS)
    D: float = 0.5                   # energy weight (Energy)
    scheduled_changes: List[dict] = field(default_factory=list)
    # scheduled_changes: list of {"step": int, "A": float, "B": float, "C": float, "D": float}


@dataclass
class ISPlatformConfig:
    """IS Platform configuration (Section 3.5 of the MAS-DUO thesis)."""
    max_allowed_cost: float = float("inf")     # ERPAgent: maximum allowed cost
    max_allowed_delay: float = float("inf")    # ERPAgent: maximum allowed delay
    production_capacity: float = 1.0           # ERPAgent: production capacity
    min_qos_threshold: float = 0.0             # CRMAgent: minimum QoS threshold
    client_priority: float = 1.0               # CRMAgent: client priority [0, 1]
    min_reward_threshold: float = float("-inf") # ExpertSystemAgent: minimum reward


@dataclass
class RewardWeights:
    on_time_delivery:    float =  100.0
    late_penalty_per_step: float = -5.0
    energy_penalty:      float =  -0.1
    wrong_route_penalty: float = -10.0
    idle_penalty:        float =  -0.5
    partial_order_bonus: float =   20.0
    full_order_bonus:    float =   50.0


@dataclass
class SimParams:
    max_steps:             int   = 500
    step_duration_seconds: int   = 5
    random_order_arrival:  bool  = False
    order_arrival_rate:    float = 0.05


@dataclass
class FactoryConfig:
    id:               str
    name:             str
    company_prefix:   str
    grid:             GridConfig
    zones:            List[ZoneConfig]
    conveyor_belts:   List[ConveyorConfig]
    robots:           List[RobotConfig]
    workers:          List[WorkerConfig]
    product_types:    List[ProductTypeConfig]
    orders:           List[OrderConfig]
    reward_weights:   RewardWeights
    sim_params:       SimParams
    global_policy:    GlobalPolicyConfig = field(default_factory=GlobalPolicyConfig)
    is_platform:      ISPlatformConfig = field(default_factory=ISPlatformConfig)

    # -----------------------------------------------------------------------
    # Quick-access helpers
    # -----------------------------------------------------------------------

    def zone_by_id(self, zone_id: str) -> Optional[ZoneConfig]:
        return next((z for z in self.zones if z.id == zone_id), None)

    def product_type_by_ref(self, ref: str) -> Optional[ProductTypeConfig]:
        return next((p for p in self.product_types if p.item_reference == ref), None)

    def zone_at(self, x: int, y: int) -> Optional[ZoneConfig]:
        """Returns the zone containing cell (x, y), if any."""
        for z in self.zones:
            if z.x <= x < z.x + z.width and z.y <= y < z.y + z.height:
                return z
        return None


# ---------------------------------------------------------------------------
# Load function
# ---------------------------------------------------------------------------

def load_factory_config(path: str | Path) -> FactoryConfig:
    """
    Loads and validates factory configuration from a JSON file.

    Parameters
    ----------
    path : str | Path
        Path to the JSON configuration file.

    Returns
    -------
    FactoryConfig
        Validated configuration ready to use.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data: dict = json.load(f)

    # The JSON has a root key "factory"
    raw: dict = data.get("factory", data)

    # --- Grid ---
    g = raw["grid"]
    grid = GridConfig(
        width            = g["width"],
        height           = g["height"],
        cell_size_meters = g.get("cell_size_meters", 1.0),
    )

    # --- Zones ---
    zones = [
        ZoneConfig(
            id=z["id"], name=z["name"],
            x=z["x"], y=z["y"],
            width=z["width"], height=z["height"],
            type=z["type"],
        )
        for z in raw.get("zones", [])
    ]

    # --- Conveyor belts ---
    conveyors = [
        ConveyorConfig(
            id=c["id"], name=c["name"],
            energy_per_step=c["energy_per_step"],
            speed_cells_per_step=c["speed_cells_per_step"],
            path=[(p["x"], p["y"]) for p in c["path"]],
            direction=c["direction"],
            capacity=c["capacity"],
        )
        for c in raw.get("conveyor_belts", [])
    ]

    # --- Robots ---
    robots = [
        RobotConfig(
            id=r["id"], name=r["name"],
            start_position=(r["start_position"]["x"], r["start_position"]["y"]),
            energy_capacity=r["energy_capacity"],
            energy_per_move=r["energy_per_move"],
            energy_per_lift=r["energy_per_lift"],
            speed_cells_per_step=r["speed_cells_per_step"],
            carrying_capacity=r["carrying_capacity"],
            recharge_rate=r["recharge_rate"],
            recharge_threshold=r["recharge_threshold"],
        )
        for r in raw.get("robots", [])
    ]

    # --- Workers ---
    workers = [
        WorkerConfig(
            id=w["id"], name=w["name"],
            start_position=(w["start_position"]["x"], w["start_position"]["y"]),
            role=w["role"],
            energy_per_action=w["energy_per_action"],
            fatigue_rate=w["fatigue_rate"],
            rest_threshold=w["rest_threshold"],
            speed_cells_per_step=w["speed_cells_per_step"],
            allowed_zones=w.get("allowed_zones", []),
        )
        for w in raw.get("workers", [])
    ]

    # --- Product types ---
    product_types = [
        ProductTypeConfig(
            item_reference=p["item_reference"],
            name=p["name"],
            size=p["size"],
            weight_kg=p["weight_kg"],
            requires_processing=p["requires_processing"],
            processing_steps=p["processing_steps"],
            process_time_steps=p.get("process_time_steps", 0),
        )
        for p in raw.get("product_types", [])
    ]

    # --- Orders ---
    orders = [
        OrderConfig(
            order_id=o["order_id"],
            products=[
                OrderProductSpec(
                    item_reference=ps["item_reference"],
                    quantity=ps["quantity"],
                )
                for ps in o["products"]
            ],
            deadline_steps=o["deadline_steps"],
            priority=o["priority"],
            destination=o["destination"],
        )
        for o in raw.get("orders", [])
    ]

    # --- Reward weights ---
    rw = raw.get("reward_weights", {})
    reward_weights = RewardWeights(
        on_time_delivery     = rw.get("on_time_delivery",     100.0),
        late_penalty_per_step= rw.get("late_penalty_per_step", -5.0),
        energy_penalty       = rw.get("energy_penalty",        -0.1),
        wrong_route_penalty  = rw.get("wrong_route_penalty",  -10.0),
        idle_penalty         = rw.get("idle_penalty",          -0.5),
        partial_order_bonus  = rw.get("partial_order_bonus",   20.0),
        full_order_bonus     = rw.get("full_order_bonus",      50.0),
    )

    # --- Sim params ---
    sp = raw.get("sim_params", {})
    sim_params = SimParams(
        max_steps            = sp.get("max_steps",             500),
        step_duration_seconds= sp.get("step_duration_seconds",   5),
        random_order_arrival = sp.get("random_order_arrival",  False),
        order_arrival_rate   = sp.get("order_arrival_rate",    0.05),
    )

    # --- Global policy (MAS-DUO Equation 9) ---
    gp = raw.get("global_policy", {})
    gp_initial = gp.get("initial", gp)  # supports {initial: {A,B,C,D}} or flat {A,B,C,D}
    global_policy = GlobalPolicyConfig(
        mode               = gp.get("mode", "static"),
        A                  = float(gp_initial.get("A", 1.0)),
        B                  = float(gp_initial.get("B", 1.0)),
        C                  = float(gp_initial.get("C", 1.0)),
        D                  = float(gp_initial.get("D", 0.5)),
        scheduled_changes  = gp.get("scheduled_changes", []),
    )

    # --- IS Platform config (MAS-DUO Section 3.5) ---
    isp = raw.get("is_platform", {})
    is_platform = ISPlatformConfig(
        max_allowed_cost      = float(isp.get("max_allowed_cost",   float("inf"))),
        max_allowed_delay     = float(isp.get("max_allowed_delay",  float("inf"))),
        production_capacity   = float(isp.get("production_capacity", 1.0)),
        min_qos_threshold     = float(isp.get("min_qos_threshold",   0.0)),
        client_priority       = float(isp.get("client_priority",     1.0)),
        min_reward_threshold  = float(isp.get("min_reward_threshold", float("-inf"))),
    )

    cfg = FactoryConfig(
        id=raw["id"], name=raw["name"],
        company_prefix=raw["company_prefix"],
        grid=grid, zones=zones,
        conveyor_belts=conveyors,
        robots=robots, workers=workers,
        product_types=product_types,
        orders=orders,
        reward_weights=reward_weights,
        sim_params=sim_params,
        global_policy=global_policy,
        is_platform=is_platform,
    )

    _validate(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------------

def _validate(cfg: FactoryConfig) -> None:
    """Raises ValueError if the configuration has inconsistencies."""
    zone_ids = {z.id for z in cfg.zones}

    # Verify that product processing steps are valid zones
    for pt in cfg.product_types:
        for step in pt.processing_steps:
            if step not in zone_ids:
                raise ValueError(
                    f"ProductType '{pt.item_reference}': zone '{step}' does not exist in config."
                )

    # Verify that conveyors stay within the grid
    for cb in cfg.conveyor_belts:
        for (cx, cy) in cb.path:
            if not (0 <= cx < cfg.grid.width and 0 <= cy < cfg.grid.height):
                raise ValueError(
                    f"Conveyor '{cb.id}': cell ({cx},{cy}) outside grid "
                    f"({cfg.grid.width}x{cfg.grid.height})."
                )

    # Verify robots/workers are within the grid
    for r in cfg.robots:
        rx, ry = r.start_position
        if not (0 <= rx < cfg.grid.width and 0 <= ry < cfg.grid.height):
            raise ValueError(f"Robot '{r.id}' start_position outside grid.")

    for w in cfg.workers:
        wx, wy = w.start_position
        if not (0 <= wx < cfg.grid.width and 0 <= wy < cfg.grid.height):
            raise ValueError(f"Worker '{w.id}' start_position outside grid.")

    # Verify orders reference existing product_types
    pt_refs = {pt.item_reference for pt in cfg.product_types}
    for order in cfg.orders:
        for ps in order.products:
            if ps.item_reference not in pt_refs:
                raise ValueError(
                    f"Order '{order.order_id}': item_reference '{ps.item_reference}' "
                    f"does not exist in product_types."
                )

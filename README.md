# MAS-DUO — Multi-Agent System for Dynamic Use and Optimization

Python implementation of the **MAS-DUO** multi-agent system described in the doctoral thesis:

> **"Improving the Decision Support in Shop Floor Operations by Using Agent-based Systems and Visibility Frameworks"**
> Pablo García Ansola — University of Castilla-La Mancha (UCLM), 2012

The system models production and logistics environments (factory, airport) as **PettingZoo AEC** environments where multiple agent types with **4W state** (What/Where/When/Why) collaborate to optimise orders/services under a configurable global policy.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Project Structure](#2-project-structure)
3. [Installation](#3-installation)
4. [Core Concepts](#4-core-concepts)
   - 4.1 [4W State](#41-4w-state)
   - 4.2 [Agent Types](#42-agent-types)
   - 4.3 [BDI and MDP](#43-bdi-and-mdp)
   - 4.4 [IS Platform](#44-is-platform-erp--crm--expert-system)
   - 4.5 [Global Policy — Equation 12](#45-global-policy--equation-12)
   - 4.6 [Reward System](#46-reward-system)
5. [JSON Configuration](#5-json-configuration)
6. [Environment API (PettingZoo AEC)](#6-environment-api-pettingzoo-aec)
7. [Pygame Renderer](#7-pygame-renderer)
8. [Examples — Airport Scenario](#8-examples--airport-scenario)
   - 8.1 [airport\_gh\_check.py — Scenario Validation](#81-airport_gh_checkpy--scenario-validation)
   - 8.2 [airport\_gh\_demo.py — Greedy Policy Demo](#82-airport_gh_demopy--greedy-policy-demo)
9. [Configuration Files](#9-configuration-files)
10. [Module Reference](#10-module-reference)

---

## 1. Architecture

MAS-DUO implements three architectural layers corresponding to the thesis chapters:

```
┌─────────────────────────────────────────────────────────────────┐
│                    IS PLATFORM (Layer 3)                         │
│   ERP (cost/capacity)  ·  CRM (QoS/clients)  ·  Expert System   │
│              MDP Proposal Negotiation                            │
│              Global Policy R(s,s')                               │
├────────────────────┬────────────────────────────────────────────┤
│  BDI Agent Loop    │          MDP Engine (Layer 2)               │
│  (Layer 1)         │                                             │
│  Beliefs (4W)  ─── │──► States   ──► Policy (A,B,C,D)           │
│  Desires           │   Actions   ──► Reward R(s,s')             │
│  Intentions   ─────│──► IS Proposal ──► Negotiation             │
├────────────────────┴────────────────────────────────────────────┤
│                   ENVIRONMENT (PettingZoo AEC)                   │
│   ProductAgent · WorkerAgent · RobotAgent · ConveyorAgent        │
│   GridWorld · OrderManager · LogisticsRenderer                   │
└─────────────────────────────────────────────────────────────────┘
```

Main flow per step:

1. The agent **observes** the environment (normalised 4W observation vector).
2. The **BDI loop** generates beliefs (`GeneratedBelief`) and selects an intention.
3. The **MDP Engine** evaluates the state transition and computes R(s,s').
4. If a proposal is pending, it is **negotiated** with the IS Platform.
5. The IS Platform may accept, reject, or propose a **new global policy**.
6. The updated policy is **propagated** to all product agents.

---

## 2. Project Structure

```
MAS-DUO/
├── config/
│   ├── factory_example.json        # Generic factory scenario
│   ├── airport_gh_example.json     # CRC Airport — 4 flights
│   └── airport_gh_demo.json        # CRC Airport — 8 flights (demo)
│
├── examples/
│   ├── env_check.py                # Factory environment validation
│   ├── airport_gh_check.py         # Airport scenario validation
│   └── airport_gh_demo.py          # Greedy demo with pygame renderer
│
├── logistics_env/
│   ├── __init__.py                 # Exports LogisticsMaEnv
│   ├── logistics_maenv.py          # Main PettingZoo AEC environment
│   ├── config_loader.py            # Loads and validates JSON configuration
│   ├── grid_world.py               # 2D grid with A* and occupancy tracking
│   │
│   ├── agents/
│   │   ├── base_agent.py           # BaseAgent: 4W state, BDI loop, MDP
│   │   ├── product_agent.py        # ProductAgent + ProductAction
│   │   ├── robot_agent.py          # RobotAgent + RobotAction
│   │   ├── worker_agent.py         # WorkerAgent + WorkerAction
│   │   └── conveyor_agent.py       # ConveyorAgent + ConveyorAction
│   │
│   ├── objects/
│   │   ├── epc.py                  # RFID EPC code (Pure Identity URI)
│   │   └── order_manager.py        # Work orders and tracking
│   │
│   ├── is_platform/
│   │   ├── __init__.py             # Exports ISPlatform, GlobalPolicy, PolicyParameters
│   │   ├── global_policy.py        # GlobalPolicy and PolicyParameters
│   │   └── is_platform.py          # IS agents and negotiation data structures
│   │
│   └── rendering/
│       └── renderer.py             # LogisticsRenderer (Pygame)
│
├── train/
│   ├── train_random.py             # Random baseline
│   └── train_ql.py                 # Tabular Q-learning baseline
├── tests/                          # Unit and PettingZoo compliance tests
│
├── setup.py
└── requirements.txt
```

---

## 3. Installation

### Requirements

- Python **3.10+** (recommended 3.12+, tested with 3.14.2 arm64)
- macOS / Linux

### Virtual environment (recommended)

```bash
# From the project root
python -m venv .venv
source .venv/bin/activate

pip install -e .
```

### Main dependencies

| Package | Minimum version | Purpose |
|---------|----------------|---------|
| `pettingzoo` | ≥ 1.24 | AEC multi-agent environment framework |
| `gymnasium` | ≥ 0.29 | Observation/action spaces |
| `numpy` | ≥ 1.24 | Observation arrays |
| `pygame` | ≥ 2.5 | 2D rendering (optional) |

> **macOS Apple Silicon note:** Ensure you use the native arm64 interpreter. The x86_64-based conda `RL` environment will raise `libffi.8.dylib` errors incompatible with arm64.

---

## 4. Core Concepts

### 4.1 4W State

All agents in the system maintain a **4W state** (inspired by the EPCIS standard):

| Dimension | Description | Example |
|-----------|-------------|---------|
| **What** (`what`) | Object identity (EPC URI or ID) | `urn:epc:id:sgtin:0614150.B737.001` |
| **Where** (`where`) | Grid position `(x, y)` | `Position(x=5, y=3)` |
| **When** (`when`) | Current simulation step | `step=15` |
| **Why** (`why`) | Business state (BusinessStep) | `PhysicalBDIState.IN_TRANSIT` |

In code:

```python
# 4W state of any agent
agent.state.what   # → str (identity)
agent.state.where  # → Position(x, y)
agent.state.when   # → int (step)
agent.state.why    # → str (BDI state)
agent.state.zone_id  # → str ("PARK1", "TERMINAL", …)
```

### 4.2 Agent Types

#### ProductAgent — Flights / products

Represents the physical object that needs to be processed (in the airport scenario: the flight; in the factory scenario: the product). Implements the full **Physical BDI loop**.

**Actions (`ProductAction`):**

| Action | Value | Description |
|--------|-------|-------------|
| `WAIT` | 0 | Wait without moving |
| `REQUEST_MOVE` | 1 | Request advance to the next route waypoint |
| `REQUEST_PROCESS` | 2 | Request processing at the current zone |
| `SIGNAL_READY` | 3 | Signal ready for dispatch |

#### RobotAgent — Equipment / AGVs

Represents robots, guided vehicles, or ground handling equipment. Has a battery that recharges at a designated charging zone.

**Actions (`RobotAction`):**

| Action | Value | Description |
|--------|-------|-------------|
| `IDLE` | 0 | No action |
| `MOVE_NORTH/SOUTH/EAST/WEST` | 1-4 | Cardinal movement |
| `LIFT` | 5 | Pick up product in the same cell |
| `DROP` | 6 | Deposit product |
| `CHARGE` | 7 | Recharge battery |
| `NAVIGATE` | 8 | Navigate via A* towards assigned target |

#### WorkerAgent — Operational personnel

Represents human workers (ramp agents, supervisors, crew chiefs).

**Actions (`WorkerAction`):**

| Action | Value | Description |
|--------|-------|-------------|
| `IDLE` | 0 | No action |
| `MOVE_NORTH/SOUTH/EAST/WEST` | 1-4 | Cardinal movement |
| `PICK` | 5 | Pick up product |
| `PLACE` | 6 | Deposit product |
| `PROCESS` | 7 | Process product at current zone |
| `SCAN` | 8 | Scan zone (RFID/EPCIS) |
| `REST` | 9 | Rest (recover energy) |

#### ConveyorAgent — Belt conveyors

Represents baggage belts, production lines, or automated conveyors.

**Actions (`ConveyorAction`):**

| Action | Value | Description |
|--------|-------|-------------|
| `STOP` | 0 | Stop the belt |
| `RUN` | 1 | Normal speed |
| `RUN_FAST` | 2 | High speed (higher energy consumption) |
| `REVERSE` | 3 | Reverse direction |

### 4.3 BDI and MDP

Each `ProductAgent` (and optionally other agent types) implements a **BDI** (Belief-Desire-Intention) loop over a **MDP** (Markov Decision Process):

```
Observation → Beliefs (GeneratedBelief) → Desires → Intentions
                         ↓
              MDPEngine.select_action(...) / update_q(...)
                         ↓
              Reward R(s,s') = f(A, B, C, D)
                         ↓
              IS Proposal (if applicable) → NegotiationProposal
```

**Exported BDI structures:**

```python
from logistics_env.agents import (
    BusinessStep,       # Enum of business steps (IATA)
    PhysicalBDIState,  # Enum of physical BDI states
    GeneratedBelief,   # Belief generated by the agent
    BDIContext,        # Full context for a BDI round
)
```

### 4.4 IS Platform (ERP / CRM / Expert System)

The `ISPlatform` acts as the central arbiter of the system. It evaluates agent proposals and can modify the global policy at runtime.

Three sub-components:

| Module | Function | Key parameters |
|--------|----------|----------------|
| **ERP** | Controls costs and capacity | `max_cost`, `max_delay`, `capacity` |
| **CRM** | Manages QoS and client priority | `min_qos`, `client_priority` |
| **Expert System** | Evaluates minimum acceptable reward | `min_reward` |

**API:**

```python
from logistics_env.is_platform import ISPlatform, GlobalPolicy, PolicyParameters

# Instantiate with initial policy
policy = GlobalPolicy(
    mode    = PolicyMode.STATIC,
    initial = PolicyParameters(A=0.5, B=0.4, C=0.0, D=0.1),
)
is_platform = ISPlatform(global_policy=policy)

# Configure sub-components
is_platform.configure_erp(max_cost=200.0, max_delay=5.0, capacity=1.0)
is_platform.configure_crm(min_qos=0.0, client_priority=0.5)
is_platform.configure_expert(min_reward=-5.0)

# Evaluate a proposal (called automatically by the environment)
result = is_platform.evaluate(proposal, step=15)
# result.outcome → NegotiationOutcome.APPROVED / REJECTED
# result.new_policy → PolicyParameters or None

# Platform status
summary = is_platform.get_summary()
# → {"total_negotiations": N, "approved": N, "approval_rate": 0.87, ...}
```

### 4.5 Global Policy — Equation 12

The global reward function (Equation 12 of the thesis) weights four dimensions:

```
R(s, s') = A·Delay + B·Cost + C·QoS + D·Energy
```

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Delay weight | `A` | Penalty for delays. Typical value: **0.5** |
| Cost weight | `B` | Economic cost of operations. Typical value: **0.4** |
| QoS weight | `C` | Quality of service / client preference. **0.0** in Common Use Model (no airline preference) |
| Energy weight | `D` | Energy efficiency. Typical value: **0.1** |

> The values A=0.5, B=0.4, C=0.0, D=0.1 are those obtained in the thesis for the Ciudad Real Central Airport scenario (Common Use Model, no airline preference).

In the JSON configuration:

```json
"global_policy": {
  "mode": "static",
  "initial": { "A": 0.5, "B": 0.4, "C": 0.0, "D": 0.1 },
  "scheduled_changes": []
}
```

The policy can change dynamically during the simulation via `scheduled_changes`:

```json
"scheduled_changes": [
  {"step": 50, "A": 0.3, "B": 0.5, "C": 0.1, "D": 0.1}
]
```

### 4.6 Reward System

The environment emits per-agent rewards at each step:

| Event | Reward | JSON parameter |
|-------|--------|----------------|
| On-time delivery | `+200.0` | `on_time_delivery` |
| Full order (bonus) | `+100.0` | `full_order_bonus` |
| Partial delivery | `+50.0` | `partial_order_bonus` |
| Late penalty | `-10.0` / step | `late_penalty_per_step` |
| Energy consumption | `-0.1` × units | `energy_penalty` |
| Wrong route | `-20.0` | `wrong_route_penalty` |
| Idle action | `-1.0` | `idle_penalty` |

---

## 5. JSON Configuration

The entire scenario is defined in a single JSON file. Full structure:

```json
{
  "factory": {
    "id":             "Unique scenario ID",
    "name":           "Descriptive name",
    "company_prefix": "GS1 Company Prefix (for EPCs)",

    "grid": {
      "width": 20,   "height": 15,
      "cell_size_meters": 5.0
    },

    "zones": [
      {
        "id": "PARK1", "name": "Stand P1",
        "x": 4, "y": 2, "width": 4, "height": 5,
        "type": "process"
      }
    ],

    "conveyor_belts": [
      {
        "id": "BAGBELT-P1",
        "path": [{"x": 5, "y": 6}, {"x": 5, "y": 7}],
        "speed_cells_per_step": 1,
        "energy_per_step": 0.3,
        "capacity": 5
      }
    ],

    "robots": [
      {
        "id": "PUSHBACK-1",
        "start_position": {"x": 1, "y": 2},
        "energy_capacity": 500.0,
        "energy_per_move": 2.0,
        "energy_per_lift": 5.0,
        "speed_cells_per_step": 2,
        "carrying_capacity": 1,
        "recharge_rate": 10.0,
        "recharge_threshold": 50.0
      }
    ],

    "workers": [
      {
        "id": "RAMP-1",
        "start_position": {"x": 1, "y": 2},
        "role": "ramp_agent",
        "energy_per_action": 0.5,
        "fatigue_rate": 0.01,
        "rest_threshold": 0.3,
        "speed_cells_per_step": 2,
        "allowed_zones": ["HANGAR", "PARK1", "PARK2"]
      }
    ],

    "product_types": [
      {
        "item_reference": "B737",
        "name": "Boeing 737-800",
        "processing_steps": ["TRANSIT", "PARK1", "TERMINAL"],
        "process_time_steps": 30,
        "requires_processing": true,
        "size": 5,
        "weight_kg": 70000.0
      }
    ],

    "orders": [
      {
        "order_id": "VY-1234",
        "products": [{"item_reference": "B737", "quantity": 1}],
        "deadline_steps": 30,
        "priority": "high",
        "destination": "TERMINAL"
      }
    ],

    "reward_weights": {
      "on_time_delivery": 200.0,
      "full_order_bonus": 100.0,
      "partial_order_bonus": 50.0,
      "late_penalty_per_step": -10.0,
      "energy_penalty": -0.1,
      "wrong_route_penalty": -20.0,
      "idle_penalty": -1.0
    },

    "sim_params": {
      "max_steps": 200,
      "step_duration_seconds": 60
    },

    "global_policy": {
      "mode": "static",
      "initial": {"A": 0.5, "B": 0.4, "C": 0.0, "D": 0.1},
      "scheduled_changes": []
    },

    "is_platform": {
      "max_allowed_cost": 200.0,
      "max_allowed_delay": 5.0,
      "production_capacity": 1.0,
      "min_qos_threshold": 0.0,
      "client_priority": 0.5,
      "min_reward_threshold": -5.0
    }
  }
}
```

**Available zone types:**

| Type | Airport term | Renderer colour |
|------|-------------|----------------|
| `input` | Entry / Reception | Light green |
| `storage` | Storage / Hangar | Light blue |
| `process` | Stand / Processing zone | Cream |
| `output` | Terminal / Departure | Sky blue |
| `charging` | Charging zone | Yellow |
| `taxiway` | Taxiways | Asphalt grey |
| `stand` | Parking stand | Cream |
| `gate` | Boarding gate | Light blue |
| `apron` | Apron / Ramp | Light grey |
| `hangar` | GH Hangar | Mauve |

---

## 6. Environment API (PettingZoo AEC)

The environment follows the PettingZoo **AEC** (Agent-Environment-Cycle) API:

### Time model

MAS-DUO distinguishes two counters:

- `agent_turns`: individual actions submitted through the AEC API.
- `simulation_step`: completed cycles in which every active agent had a turn.

Deadlines, scheduled policy changes and `step_duration_seconds` use
`simulation_step`. Training loops must therefore terminate on environment
`truncations`, not after `max_steps` individual actions.

```python
from logistics_env import LogisticsMaEnv

# Create environment
env = LogisticsMaEnv(
    config_path  = "config/airport_gh_demo.json",
    render_mode  = "human",   # "human" | "rgb_array" | None
)

# Reset
env.reset(seed=42)

# Simulation loop
while env.agents:
    agent_id = env.agent_selection
    obs, reward, terminated, truncated, info = env.last()

    if terminated or truncated:
        env.step(None)
        continue

    action = env.action_space(agent_id).sample()  # or your policy
    env.step(action)

# State snapshot
snapshot = env.state_snapshot
# → {"simulation_step": N, "agent_turns": M, "products": {...}, ...}

env.close()
```

### Observation spaces

| Agent type | Shape | Description |
|------------|-------|-------------|
| `product` | `(6,)` float32 [0,1] | Normalised position, route progress, deadline, energy |
| `worker` | `(6,)` float32 [0,1] | Position, energy, fatigue, zone, previous action |
| `robot` | `(5,)` float32 [0,1] | Position, battery, load, speed |
| `conveyor` | `(5,)` float32 [0,1] | State, occupancy, speed, energy |

### `state_snapshot`

Property returning the complete serialisable state:

```python
snapshot = env.state_snapshot
# {
#   "step": 15,
#   "simulation_step": 15,
#   "agent_turns": 225,
#   "products":  { "urn:epc:...": {"what": ..., "where": ..., "when": ..., "why": ..., "zone_id": ...} },
#   "workers":   { "RAMP-1":      {...} },
#   "robots":    { "PUSHBACK-1":  {...} },
#   "conveyors": { "BAGBELT-P1":  {...} },
#   "orders":    { "VY-1234":     {"status": "complete", "dispatched": 1, "needed": 1, "deadline": 30, "priority": "high", "product_type": "B737"} },
#   "total_energy_consumed": 831.0,
#   "is_platform": {"total_negotiations": 5, "approved": 4, ...},
#   "global_policy": {"A": 0.5, "B": 0.4, "C": 0.0, "D": 0.1, "mode": "static"},
# }
```

---

## 7. Pygame Renderer

`LogisticsRenderer` provides real-time visualisation of the environment.

### Instantiation

```python
from logistics_env.rendering.renderer import LogisticsRenderer

renderer = LogisticsRenderer(
    factory_cfg   = env.factory_cfg,
    fps           = 3,              # 3 FPS = human-readable speed
    title_prefix  = "MAS-DUO",
)
```

### Render call

```python
renderer.render(
    grid      = env._grid,
    products  = env._products,
    workers   = env._workers,
    robots    = env._robots,
    conveyors = env._conveyors,
    step      = env._step_count,
    order_mgr = env._order_mgr,
    mode      = "human",      # "human" | "rgb_array"
    fps       = 3,            # per-frame FPS override
    extra_info = {            # additional info for the panel
        "scenario": "CRC Airport — Demo",
        "solver":   "Greedy EDF",
        "policy":   "A=0.5 B=0.4 C=0.0 D=0.1",
        "reward":   287.07,
    },
)
```

### Visual legend

| Element | Colour | Representation |
|---------|--------|---------------|
| `storage` / hangar zones | Blue/mauve | Rectangle with border |
| `process` / stand zones | Cream/yellow | Rectangle |
| `output` / terminal zones | Sky blue | Rectangle |
| `input` / taxiway zones | Light green / grey | Rectangle |
| Robots / GH equipment | Blue `(60,100,200)` | Rounded rectangle |
| Workers / personnel | Green `(60,160,60)` | Rounded rectangle |
| Supervisors / chiefs | Dark green `(30,130,30)` | Rounded rectangle |
| Products / flights | Orange `(230,120,20)` | Circle with label |
| Completed products | Green `(100,200,120)` | Circle |
| Baggage belts | Dark grey `(100,100,100)` | Row of cells with arrow |
| Low battery (<20%) | Red border | Border on robot |

### Side panel

The right-hand panel (360 px) displays in real time:

- **Header:** scenario name, simulated clock (`T+hh:mm`), step, cumulative reward, active solver
- **Executive view:** completed orders, active resources, battery and the agent/action currently being executed
- **FLIGHTS:** status of each order — time remaining to deadline (`T-XX`), completed (`OK`), delayed (`DELAY`), priority `[H/N/L]`, progress bar
- **EQUIPMENT:** list of robots with battery bar and carried load
- **PERSONNEL:** list of workers with energy level
- **BELTS:** state (RUN/STOP) and products in transit

### Live training dashboard

The tabular Q-Learning trainer includes an optional presentation mode with
episode progress, reward and moving-average curves, order completion, exploration,
Q-table growth and agent/cycle counters:

```bash
python train/train_ql.py --episodes 60 --visualize
```

The dashboard does not change the learning algorithm and is disabled by default,
so automated runs remain headless. Controls: `SPACE` pauses, `+/-` changes the
display rate, `S` saves a screenshot under `artifacts/screenshots`, and `Q` closes it.

Once a policy has been trained, use the same visual language for the operational
simulation:

```bash
python examples/factory_demo.py --policy train/ql_policy.pkl --fps 4
```

---

## 8. Examples — Airport Scenario

Both examples are based on **Chapter 4.1** of the thesis, which describes Ciudad Real Central Airport (CRC) as a real-world MAS-DUO use case, with ground handling operations for B737-800 turnarounds within 30-45 minute windows.

### Airport grid (20 x 15 cells, 5 m/cell = 100 x 75 m)

```
col ->  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
row
  0  [ H  H  H  H | T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  ]  <- TRANSIT
  1  [ H  H  H  H | T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  ]    (Taxiway)
  2  [ H  H  H  H | P1 P1 P1 P1| P2 P2 P2 P2| P3 P3 P3 P3| P4 P4 P4 P4]
  3  [ H  H  H  H | P1 P1 P1 P1| P2 P2 P2 P2| P3 P3 P3 P3| P4 P4 P4 P4]
  4  [ H  H  H  H | P1 P1 P1 P1| P2 P2 P2 P2| P3 P3 P3 P3| P4 P4 P4 P4]
  5  [ H  H  H  H | P1 P1 P1 P1| P2 P2 P2 P2| P3 P3 P3 P3| P4 P4 P4 P4]
  6  [ H  H  H  H | P1 P1 P1 P1| P2 P2 P2 P2| P3 P3 P3 P3| P4 P4 P4 P4]
  7  [ H  H  H  H | TRM TRM TRM TRM TRM TRM TRM TRM TRM TRM TRM TRM TRM]
  8  [ H  H  H  H | TRM TRM TRM TRM TRM TRM TRM TRM TRM TRM TRM TRM TRM]
  ...                                (Passenger terminal)
 14  [ H  H  H  H | TRM TRM TRM TRM TRM TRM TRM TRM TRM TRM TRM TRM TRM]

  H   = HANGAR  (GH equipment base)
  T   = TRANSIT (taxiways / airside roads)
  P1-P4 = PARK1-PARK4 (parking stands)
  TRM = TERMINAL (boarding gates / departures)
```

### ReadPoints and states (Equation 11 of the thesis)

```
ReadPoints (RP) = { HANGAR, PARK1, PARK2, PARK3, PARK4 }  -> 5 RPs

Resource BusinessSteps (BS_Resource):
  { Free, Busy, InTransit, NotAvailable }  -> 4 BS

Resource system state space:
  |S_resource| = |RP| x |BS_resource| = 5 x 4 = 20 states

IATA Flight BusinessSteps (BS_Flight):
  { OMS, STM, LOD, PAX, BAG, HDL, AGM, CGM, ... }  -> 8+ BS

Flight system state space:
  |S_flight| approx 4 stands x 10 BS = 40 states
```

---

### 8.1 `airport_gh_check.py` — Scenario Validation

**File:** `examples/airport_gh_check.py`
**Config:** `config/airport_gh_example.json` (4 flights)

A **validation and exploration** script for the airport scenario. It does not optimise anything — it executes random actions to verify that all system components work correctly: config loading, agent creation, 4W states, IS negotiations, and order summaries.

#### Usage

```bash
python examples/airport_gh_check.py
```

#### Step-by-step walkthrough

1. **Loads the configuration** `airport_gh_example.json` and creates the environment.
2. **Displays the airport grid** (zones, sizes) using airport terminology (TRANSIT -> taxiways, PARK1-4 -> stands, TERMINAL -> boarding gates).
3. **Lists the ReadPoints** and BusinessSteps (Equation 11 of the thesis) with the 20 resource states and ~40 flight states.
4. **Prints the initial state of all agents** — type, observation shape, action space size:
   - `ProductAgent` (flights: B737, A320, LCC, CARGO)
   - `RobotAgent` (pushbacks, stairs, fuel trucks, cargo loaders, pax buses)
   - `WorkerAgent` (ramp agents, supervisors, crew chiefs)
   - `ConveyorAgent` (baggage belts gate to terminal)
5. **Displays the initial airport state** using real-world terminology:
   - Active flights with type, position, and BDI state
   - GH equipment with position and assigned zone
   - Ramp personnel with position and role
   - Active baggage belts
   - Scheduled flights with deadline and status
6. **IS Platform status** — global policy, negotiations, approval rate.
7. **Simulates 10 environment cycles** with random actions, showing rewards and IS negotiations when they occur.
8. **Final summary** — energy consumed, flight-by-flight status, IS platform status.

#### Expected output (excerpt)

```
================================================================
  MAS-DUO -- Airport Ground Handling
  Airport: Ciudad Real Central Airport -- Ground Handling MAS-DUO
  GH Policy: R = 0.5*Delay + 0.4*Cost + 0.0*QoS + 0.1*Energy
  (Equation 12 -- Ciudad Real Central Airport)
================================================================

  Max steps    : 120 steps (120 minutes of operations)
  Step duration: 60 s/step (= 1 min/step)
  Scheduled flt: 4
  Total agents : 21

--------------------------------
  SYSTEM STATES (ReadPoint x BusinessStep -- Equation 11)

  GH resource ReadPoints:
    RP[HANGAR    ] -> 'GH Equipment Hangar'     (0,0) 4x15
    RP[PARK1     ] -> 'Gate 1 -- Stand'         (4,2) 4x5
    ...

  -> States per resource = 5 RP x 4 BS = 20 states (Equation 11)

  Flight BusinessSteps (IATA, Section 4.1.2):
    1. OMS  -- Organisation & Management System
    2. STM  -- Station Management System
    ...

--------------------------------
  FLIGHTS IN OPERATIONS  (Physical BDI Product Agents)

  FLT  urn:epc:id:sgtin:0614150.B737.0000001
       Type     : Boeing 737-800
       Position : (2, 1)  ->  Taxiways / Airside
       Step     : 0
       State    : CREATED

  ...

--------------------------------
  IS PLATFORM  (ERP-SITA / CRM / Expert System)

  Global policy (Equation 12 of the thesis):
    R(s,s') = 0.5*Delay + 0.4*Cost + 0.0*QoS + 0.1*Energy
      -> C=0.0 (no airline preference -- 'Common Use Model')
```

---

### 8.2 `airport_gh_demo.py` — Greedy Policy Demo

**File:** `examples/airport_gh_demo.py`
**Config:** `config/airport_gh_demo.json` (8 flights)

Advanced demo with a **Greedy EDF** (Earliest Deadline First) policy and pygame rendering at human-readable speed. Displays the airport state in real time with a fully annotated side panel.

#### Usage

```bash
# Pygame demo at 3 FPS (human-readable speed)
python examples/airport_gh_demo.py

# Headless (console only, faster)
python examples/airport_gh_demo.py --headless

# Custom FPS, seed, and log frequency
python examples/airport_gh_demo.py --fps 2 --seed 100 --verbose-every 5

# Alternative config
python examples/airport_gh_demo.py --config config/airport_gh_example.json --headless
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config/airport_gh_demo.json` | Path to configuration JSON |
| `--headless` | False | Run without pygame window |
| `--seed` | `42` | Seed for reproducibility |
| `--fps` | `3` | Frames per second for pygame renderer |
| `--verbose-every` | `10` | Print console summary every N steps |

#### Scenario — 8 concurrent flights

`config/airport_gh_demo.json` sets up a scenario with:

**Flights (8 orders):**

| ID | Type | Deadline | Priority |
|----|------|----------|----------|
| `VY-1234` | B737-800 | 30 min | High |
| `IB-4567` | A320 | 35 min | High |
| `FR-8901` | Generic LCC | 25 min | Normal |
| `UX-2345` | Cargo flight | 60 min | Low |
| `VY-5678` | B737-800 | 40 min | High |
| `IB-9012` | A320 | 45 min | High |
| `W6-3456` | Generic LCC | 30 min | Normal |
| `DHL-7890` | Cargo flight | 55 min | Low |

**GH equipment (12 robots):**

| ID | Type | Energy |
|----|------|--------|
| `PUSHBACK-1/2/3` | Pushback tractor | 500 u |
| `STAIRS-1/2/3` | Hydraulic stairs | 300 u |
| `FUEL-TRK-1/2` | Fuel truck | 800 u |
| `CARGO-LOADER-1/2` | ULD Loader | 400 u |
| `BUS-PAX-1/2` | Passenger bus | 600 u |

**GH personnel (7 workers):**

| ID | Role |
|----|------|
| `RAMP-1` to `RAMP-5` | Ramp agents |
| `SUPERVISOR-1` | Operations supervisor |
| `CREW-CHIEF-1` | Ground crew chief |

**Baggage belts (4 conveyors):**
`BAGBELT-P1` to `BAGBELT-P4` — Gate N to Terminal (3-cell length each)

#### Greedy EDF policy — how it works

The `GreedyGHPolicy` class implements **Earliest Deadline First** with a priority-based urgency boost:

```python
# Base urgency: inverse of remaining time until deadline
urgency = 1.0 / max(1, deadline - current_step)

# Order priority boost
if priority == "high":   urgency *= 2.0
if priority == "low":    urgency *= 0.5
```

Rules per agent type:

- **ProductAgent (flight):**
  - In target zone -> `REQUEST_PROCESS` (advance IATA step)
  - Otherwise -> `REQUEST_MOVE` (request transfer)
  - At end of route -> `SIGNAL_READY`

- **RobotAgent (GH equipment):**
  - Battery < 15% -> `CHARGE`
  - Carrying a flight -> `NAVIGATE` towards destination zone, then `DROP`
  - Idle -> find most urgent flight (EDF), `NAVIGATE` + `LIFT`

- **WorkerAgent (personnel):**
  - Energy < 20% -> `REST`
  - Flight in same cell -> `PROCESS` (in target zone) or `PICK`
  - Carrying a flight -> move towards destination zone (`MOVE_*`) and `PLACE`
  - Otherwise -> `SCAN` (RFID scan for visibility)

- **ConveyorAgent (belt):**
  - Always -> `RUN` (maximum throughput)

#### Console output

```
================================================================
  MAS-DUO  Ground Handling Demo -- Ciudad Real Central Airport
  Policy: Greedy EDF (Earliest Deadline First)
  Thesis: Pablo Garcia Ansola (2012), Ch. 4.1 -- Equation 12
  Reward = 0.5*Delay + 0.4*Cost + 0.0*QoS + 0.1*Energy
================================================================

  Active agents: 31
  Flights (orders): 8
  Max steps: 200

  -- Step   0 | T+00:00 | Score: {'total': 8, 'done': 0, ...} --
    VY-1234      [H] 0/1            T-30
    IB-4567      [H] 0/1            T-35
    FR-8901      [N] 0/1            T-25
    UX-2345      [L] 0/1            T-60
    ...

  -- Step  20 | T+20:00 | Score: {'total': 8, 'done': 3, ...} --
    VY-1234      [H] 1/1 ##########  OK
    IB-4567      [H] 1/1 ##########  OK
    FR-8901      [N] 0/1            T-05
    ...
```

#### Final report

```
================================================================
  FINAL REPORT -- MAS-DUO Ground Handling Demo
================================================================
  Steps executed   : 30
  Cumulative reward: +287.07
  Total energy     : 831.00 units

  Completed flights   : 4/8
  On-time (deadline)  : 4/8
  Delayed             : 1/8

  Per-flight detail:
  OK  VY-1234    B737     complete   30    high
  OK  IB-4567    A320     complete   35    high
  !!  FR-8901    LCC      failed     25    normal
  ~   UX-2345    CARGO    pending    60    low
  ...

  Global Policy (Equation 12 -- MAS-DUO):
    A (Delay)  = 0.50
    B (Cost)   = 0.40
    C (QoS)    = 0.00  <- 0.0 = Common Use (no airline preference)
    D (Energy) = 0.10
================================================================
```

**Exit code:** `0` if all flights complete on time, `1` if any delay is detected.

---

## 9. Configuration Files

| File | Flights | Grid | Resources | Use |
|------|---------|------|-----------|-----|
| `config/factory_example.json` | — | 20x15 | Generic products | Factory validation |
| `config/airport_gh_example.json` | 4 | 20x15 | 8 robots, 5 workers, 3 belts | Airport validation (`check`) |
| `config/airport_gh_demo.json` | 8 | 20x15 | 12 robots, 7 workers, 4 belts | Full demo with renderer |

---

## 10. Module Reference

### `logistics_env.logistics_maenv`

Main class `LogisticsMaEnv(AECEnv)`:

| Method / Property | Description |
|-------------------|-------------|
| `reset(seed)` | Initialises the environment, creates all agents |
| `step(action)` | Executes the action for the current agent |
| `observe(agent)` | Returns the agent's observation |
| `last()` | `(obs, reward, term, trunc, info)` for the current agent |
| `state_snapshot` | Complete dict of all agents' 4W state |
| `render()` | Draws a pygame frame if `render_mode="human"` |
| `close()` | Shuts down the renderer |
| `_step_count` | Current simulation step |
| `simulation_step` | Public operational clock: completed AEC cycles |
| `agent_turns` | Number of individual agent actions since reset |
| `_total_energy` | Cumulative total energy consumed |
| `factory_cfg` | `FactoryConfig` object (loaded from JSON) |
| `_products / _workers / _robots / _conveyors` | Agent dictionaries |
| `_order_mgr` | `OrderManager` with all order states |
| `_is_platform` | Active `ISPlatform` |
| `_global_policy` | Active `GlobalPolicy` |

### `logistics_env.config_loader`

```python
from logistics_env.config_loader import load_factory_config, FactoryConfig

cfg = load_factory_config("config/airport_gh_demo.json")
cfg.name           # str
cfg.grid           # GridConfig(width, height, cell_size_meters)
cfg.zones          # List[ZoneConfig]
cfg.robots         # List[RobotConfig]
cfg.workers        # List[WorkerConfig]
cfg.conveyor_belts # List[ConveyorConfig]
cfg.product_types  # List[ProductTypeConfig]
cfg.orders         # List[OrderConfig]
cfg.reward_weights # RewardWeights
cfg.sim_params     # SimParams(max_steps, step_duration_seconds)
cfg.global_policy  # GlobalPolicyConfig(mode, A, B, C, D, scheduled_changes)
cfg.is_platform    # ISPlatformConfig

# Helpers
cfg.zone_by_id("PARK1")         # -> ZoneConfig or None
cfg.product_type_by_ref("B737") # -> ProductTypeConfig or None
```

### `logistics_env.is_platform`

```python
from logistics_env.is_platform import (
    ISPlatform,
    GlobalPolicy, PolicyMode, PolicyParameters,
    NegotiationProposal, NegotiationResult,
)
```

### `logistics_env.grid_world`

```python
from logistics_env.grid_world import GridWorld

grid = GridWorld(factory_cfg)
grid.zone_of(pos)          # -> ZoneConfig or None
grid.zone_center("PARK1")  # -> Position(x, y) or None
grid.astar(start, goal, agent_id)  # -> List[Position] or None
grid.is_occupied(x, y)     # -> bool
grid.conveyor_at(pos)      # -> str (conveyor_id) or None
```

---

## Quick Reference — Commands

Run the complete regression and PettingZoo API suite:

```bash
python -m unittest discover -s tests -v
# or, with the test extra installed
pytest
```

```bash
# Activate virtual environment
source .venv/bin/activate

# Validate airport scenario (4 flights, console only)
python examples/airport_gh_check.py

# Full demo with pygame renderer (8 flights, 3 FPS)
python examples/airport_gh_demo.py --fps 3

# Headless demo (8 flights, console only)
python examples/airport_gh_demo.py --headless --seed 42

# Custom demo
python examples/airport_gh_demo.py \
    --config config/airport_gh_demo.json \
    --fps 2 \
    --seed 100 \
    --verbose-every 5

# Validate generic factory environment
python examples/env_check.py

# Reproducible Q-learning baseline
python train/train_ql.py --episodes 60 --seed 7

# Reproducible random baseline
python train/train_random.py --episodes 3 --seed 7
```

---

## Licence and Authorship

Implementation based on the MAS-DUO architecture described in the doctoral thesis of **Pablo Garcia Ansola** (UCLM, 2012). The code is a reference implementation of the system described in Chapters 3 (BDI/MDP/IS architecture) and 4.1 (airport use case).

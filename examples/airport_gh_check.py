"""
airport_gh_check.py — Airport Ground Handling Scenario
=======================================================
Validation of the MAS-DUO environment applied to the Ground Handling scenario
of Ciudad Real Central Airport, as described in Chapter 4.1 of the thesis:

  «Improving the Decision Support in Shop Floor Operations by Using
   Agent-based Systems and Visibility Frameworks»
   — Pablo García Ansola, UCLM

MAS-DUO architecture applied to the airport:
─────────────────────────────────────────────
  · Product agents    → Flights (Boeing 737, A320, LCC, Cargo)
  · Resource agents   → GH Equipment (pushback, stairs, buses, fuel trucks)
  · Worker agents     → Ramp personnel, supervisors, crew chiefs
  · Conveyor belts    → Baggage belts gate→terminal
  · IS Platform       → ERP (SITA), CRM, Expert System
  · Global policy     → R(s,s') = 0.5·Delay + 0.4·Cost + 0.1·Energy
                         (Equation 12 of thesis, no QoS — no airline preference
                          in the 'Common Use Model')

ReadPoints (Equation 11):
  RP = {Hangar, Park1, Park2, Park3, Park4}

BusinessSteps IATA (Section 4.1.2):
  BSFlight = {OMS, STM, LOD, PAX, BAG, HDL, AGM, CGM, ...}
  BSResource = {Free, Busy, InTransit, NotAvailable}

Airport grid (20 × 15 cells, 5 m/cell):
  ┌────┬─────────────────────────────────┐
  │    │  TRANSIT (taxiways)                    │
  │    ├────────┬────────┬──────┬────────┤
  │ H  │ PARK1  │ PARK2  │PARK3 │ PARK4  │
  │ A  │(Stand1)│(Stand2)│(S.3) │(Stand4)│
  │ N  ├────────┴────────┴──────┴────────┤
  │ G  │                                 │
  │ A  │      PASSENGER TERMINAL          │
  │ R  │       (Departure gates)          │
  └────┴─────────────────────────────────┘
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from logistics_env import LogisticsMaEnv
from logistics_env.agents import BusinessStep

# ──────────────────────────────────────────────────────────────────────────────
# Airport terminology mapping
# ──────────────────────────────────────────────────────────────────────────────

ZONE_LABELS = {
    "HANGAR":   "GH Hangar (equipment base)",
    "TRANSIT":  "Taxiways / Airside",
    "PARK1":    "Gate 1 — Stand P1",
    "PARK2":    "Gate 2 — Stand P2",
    "PARK3":    "Gate 3 — Stand P3",
    "PARK4":    "Gate 4 — Stand P4",
    "TERMINAL": "Passenger terminal / Departures",
}

RESOURCE_TYPE = {
    "PUSHBACK":      "Pushback tug",
    "STAIRS":        "Hydraulic stairs",
    "FUEL-TRK":      "Fuel tanker",
    "CARGO-LOADER":  "ULD/container loader",
    "BUS-PAX":       "Passenger bus",
    "RAMP-AGT":      "Ramp agent",
    "SUPERVISOR":    "GH Supervisor",
    "CREW-CHIEF":    "Crew chief",
    "BAGBELT":       "Baggage belt",
}

FLIGHT_TYPE = {
    "B737":  "Boeing 737-800",
    "A320":  "Airbus A320",
    "LCC":   "Generic LCC",
    "CARGO": "Cargo flight",
}

def resource_label(aid: str) -> str:
    """Returns the descriptive label of a GH resource by its ID."""
    for prefix, label in RESOURCE_TYPE.items():
        if aid.startswith(prefix):
            return label
    return aid


# ──────────────────────────────────────────────────────────────────────────────
# Display functions
# ──────────────────────────────────────────────────────────────────────────────

def print_header(text: str, width: int = 72) -> None:
    print(f"\n{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}\n")


def print_section(text: str, width: int = 72) -> None:
    print(f"\n{'─' * width}")
    print(f"  {text}")
    print(f"{'─' * width}\n")


def print_airport_state(env: LogisticsMaEnv) -> None:
    """Displays the full scenario state using airport terminology."""
    snapshot = env.state_snapshot
    cfg      = env.factory_cfg

    # ── Flights (product agents = Physical BDI Agents) ────────────────────────
    print_section("✈  FLIGHTS IN OPERATIONS  (Physical BDI Product Agents)")
    if snapshot["products"]:
        for aid, st in snapshot["products"].items():
            # Determine flight type by EPC reference
            flight_ref = "unknown"
            for ref, label in FLIGHT_TYPE.items():
                if ref in st["what"]:
                    flight_ref = label
                    break

            zone_label = ZONE_LABELS.get(st["zone_id"] or "—", st["zone_id"] or "—")
            print(f"  FLT  {aid[:55]}")
            print(f"       Type     : {flight_ref}")
            print(f"       Position : ({st['where']['x']}, {st['where']['y']})  →  {zone_label}")
            print(f"       Step     : {st['when']}")
            print(f"       State    : {st['why']}")
            print()
    else:
        print("  (No active flights)")

    # ── GH Equipment → robots ────────────────────────────────────────────
    print_section("🔧  GH EQUIPMENT  (Physical BDI Resource Agents — Robots)")
    for aid, st in snapshot["robots"].items():
        zone_label = ZONE_LABELS.get(st["zone_id"] or "—", st["zone_id"] or "—")
        rlabel = resource_label(aid)
        print(f"  [{aid:<15}]  {rlabel:<28}  "
              f"pos=({st['where']['x']:2d},{st['where']['y']:2d})  zone={zone_label}")

    # ── Ramp personnel → workers ─────────────────────────────────────────
    print_section("👷  GH PERSONNEL  (Physical BDI Resource Agents — Workers)")
    for aid, st in snapshot["workers"].items():
        zone_label = ZONE_LABELS.get(st["zone_id"] or "—", st["zone_id"] or "—")
        rlabel = resource_label(aid)
        print(f"  [{aid:<15}]  {rlabel:<28}  "
              f"pos=({st['where']['x']:2d},{st['where']['y']:2d})  zone={zone_label}")

    # ── Baggage belts → conveyors ───────────────────────────────────────
    print_section("🧳  BAGGAGE BELTS  (Conveyor Agents)")
    for aid, st in snapshot["conveyors"].items():
        print(f"  [{aid:<15}]  Baggage belt  pos=({st['where']['x']},{st['where']['y']})")

    # ── Flights / Active orders ─────────────────────────────────────────
    print_section("📋  SCHEDULED FLIGHTS  (Service orders)")
    for oid, info in snapshot["orders"].items():
        status_icon = {"pending": "🕐", "in_progress": "🟡", "complete": "✅",
                       "failed": "❌"}.get(info["status"], "❓")
        print(f"  {status_icon} Vuelo {oid:<12} "
              f"Estado={info['status']:<12} "
              f"Servidos={info['dispatched']}/{info['needed']}  "
              f"Deadline=step {info['deadline']}")
    print()


def print_is_platform_state(env: LogisticsMaEnv) -> None:
    """Displays the IS Platform state and the global policy."""
    snapshot = env.state_snapshot
    print_section("🏢  IS PLATFORM  (ERP-SITA / CRM / Expert System)")

    gp = snapshot.get("global_policy", {})
    print(f"  Global policy (Equation 12 of thesis):")
    print(f"    R(s,s') = {gp.get('A',0.5)}·Delay + {gp.get('B',0.4)}·Cost"
          f" + {gp.get('C',0.0)}·QoS + {gp.get('D',0.1)}·Energy")
    print(f"    Mode: {gp.get('mode','static').upper()}")
    print(f"    → A=0.5 (delay priority), B=0.4 (costs),")
    print(f"      C=0.0 (no airline preference — 'Common Use Model'),")
    print(f"      D=0.1 (energy efficiency)")
    print()

    is_info = snapshot.get("is_platform", {})
    if is_info:
        print(f"  IS Negotiations:")
        print(f"    Total        : {is_info.get('total_negotiations', 0)}")
        print(f"    Approved     : {is_info.get('approved', 0)}")
        print(f"    Rejected     : {is_info.get('rejected', 0)}")
        rate = is_info.get('approval_rate', 1.0)
        print(f"    Approval rate: {rate:.1%}")
        print(f"    ES rejections: {is_info.get('rejection_count', 0)}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    config_path = Path(__file__).parent.parent / "config" / "airport_gh_example.json"
    env = LogisticsMaEnv(config_path=config_path, render_mode=None)
    env.reset(seed=2026)

    # ── Cabecera ────────────────────────────────────────────────────────────
    print_header(
        f"MAS-DUO — Airport Ground Handling\n"
        f"  Airport: {env.factory_cfg.name}\n"
        f"  GH Policy: R = 0.5·Delay + 0.4·Cost + 0.0·QoS + 0.1·Energy\n"
        f"  (Equation 12 — Ciudad Real Central Airport)"
    )

    cfg = env.factory_cfg
    print(f"  Max steps    : {cfg.sim_params.max_steps} steps "
          f"({cfg.sim_params.max_steps} minutes of operations)")
    print(f"  Step duration: {cfg.sim_params.step_duration_seconds} s/step (= 1 min/step)")
    print(f"  Sched. flights: {len(cfg.orders)}")
    print(f"  Total agents : {len(env.agents)}")
    print()

    # ── ReadPoints and BusinessSteps (Equation 11 of thesis) ───────────────────
    print_section("📡  SYSTEM STATES (ReadPoint × BusinessStep — Equation 11)")
    print("  GH resource ReadPoints:")
    for z in cfg.zones:
        print(f"    RP[{z.id:<10}] → '{z.name}'   ({z.x},{z.y}) {z.width}×{z.height}")
    print()
    print("  Resource BusinessSteps:")
    print("    BS: Free | Busy | InTransit | NotAvailable")
    print(f"\n  → States per resource = 5 RP × 4 BS = 20 states (Equation 11)")
    print()
    print("  Flight BusinessSteps (IATA, Section 4.1.2 thesis):")
    iata_steps = [
        "1. OMS  — Organisation & Management System",
        "2. STM  — Station Management System",
        "3. LOD  — Load Control",
        "4. PAX  — Passenger Handling",
        "5. BAG  — Baggage Handling",
        "6. HDL  — Aircraft Handling & Loading",
        "7. AGM  — Aircraft Ground Movement",
        "8. CGM  — Cargo & Mail Handling",
    ]
    for s in iata_steps:
        print(f"    {s}")
    print(f"\n  → States per flight = 4 stands × 10 BS ≈ 40 states (thesis, Table 3)")
    print()

    # ── Initial agent state ────────────────────────────────────────────────────
    print_section("📊  INITIAL STATE OF ALL AGENTS")
    for aid in env.agents[:6]:               # show first 6
        atype     = env._agent_type(aid)
        obs_shape = env.observation_space(aid).shape
        act_size  = env.action_space(aid).n
        label     = resource_label(aid) if atype in ("robot", "worker") else atype
        print(f"  [{atype:10s}] {aid[:45]:<45}  obs={obs_shape}  act={act_size}")
    if len(env.agents) > 6:
        print(f"  ... ({len(env.agents) - 6} more agents)")

    # ── Initial airport state ─────────────────────────────────────────────────
    print_airport_state(env)
    print_is_platform_state(env)

    # ── 10-step simulation ───────────────────────────────────────────────────
    print_section("🔄  SIMULATION — 10 STEPS (random actions)")
    print("  Simulates a 10-minute GH operations scenario.\n")

    rewards_acc: dict[str, float] = {}
    for step_i in range(10):
        if not env.agents:
            print("  ⚠  All agents have finished.")
            break

        agent = env.agent_selection
        obs, rew, term, trunc, info = env.last()
        action = env.action_space(agent).sample()
        env.step(action)

        rewards_acc[agent] = rewards_acc.get(agent, 0.0) + rew
        atype = env._agent_type(agent)

        # Display notable events
        is_neg = info.get("is_negotiation") if info else None
        if is_neg:
            outcome = is_neg.get("outcome", "")
            icon = "✅" if outcome == "APPROVED" else "⚠"
            print(f"  [{step_i+1:02d}] {icon} IS Negotiation — {agent[:35]}: {outcome}")
        elif step_i < 5:
            print(f"  [{step_i+1:02d}] {atype:10s} {agent[:35]:<35} "
                  f"rew={rew:+.2f}  act={action}")

    print(f"\n  Accumulated rewards (first {len(rewards_acc)} agents with action):")
    for aid, r in list(rewards_acc.items())[:5]:
        atype = env._agent_type(aid)
        print(f"    [{atype:10s}] {aid[:40]:<40}  reward={r:+.2f}")

    # ── Final summary ─────────────────────────────────────────────────────
    print_section(f"📈  FINAL SUMMARY — Step {env._step_count}")
    print(f"  Total energy consumed : {env._total_energy:.2f} units")
    print(f"  Agents still active   : {len(env.agents)}")

    snap_fin = env.state_snapshot
    print(f"\n  Flight status:")
    for oid, info in snap_fin["orders"].items():
        icon = "✅" if info["status"] == "complete" else (
               "❌" if info["status"] == "failed" else "🕐")
        print(f"    {icon} {oid:<12}  {info['status']:<12}  "
              f"{info['dispatched']}/{info['needed']} served  "
              f"deadline=step {info['deadline']}")

    print_is_platform_state(env)

    env.close()
    print("  Environment closed. End of GH simulation.\n")


if __name__ == "__main__":
    main()

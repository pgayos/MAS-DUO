"""
AgentState — Canonical state of all system agents
===================================================
Implements the 4W state model:

  What  → IDENTIFIER of the agent (EPC for products, ID for others)
  Where → POSITION in the logistics grid  (x, y)
  When  → SIMULATION TIME at which the state is recorded (step / timestamp)
  Why   → CURRENT REASON or intention of the agent (Business Step of EPC vocab)

For products, "What" follows the GS1 EPC standard (SGTIN-96).
For other agents (workers, robots, belts) it is an agent identifier.

MAS-DUO: States as a ReadPoint × BusinessStep matrix
----------------------------------------------------
According to Equation 4 of the MAS-DUO doctoral thesis:

    ReadPoint(p)    = {RP1, RP2, ... RPn}       ← zone_id (Where EPC)
    BusinessStep(p) = {BS1, BS2, ... BSm}       ← why     (Why EPC)
    States(p)       = {RP × BS}                 ← MDP state of the agent

This module defines:
  · PhysicalBDIState — formal MDP state as a (ReadPoint, BusinessStep) pair
  · GeneratedBelief  — beliefs generated during BDI reasoning (Point 3.b)
  · BDIContext       — beliefs, desires, intentions of the agent
  · BaseAgent        — abstract class with BDI support
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
from enum import Enum


# ---------------------------------------------------------------------------
# Tipos auxiliares
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Position:
    """2D position in the factory grid."""
    x: int
    y: int

    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y)

    def distance_to(self, other: "Position") -> float:
        return abs(self.x - other.x) + abs(self.y - other.y)   # Manhattan

    def __repr__(self) -> str:
        return f"({self.x},{self.y})"


class BusinessStep(str, Enum):
    """
    Business Steps vocabulary from the EPC standard (Why of EPC vocab).

    According to Equation 4 of MAS-DUO:
        BusinessStep(p) = {BS1, BS2, ... BSm}

    The thesis (Sec 3.4): "The states of the product are defined and detected
    by the 'Where?' (Read point) and 'Why?' (Business step) EPC information."
    """
    # Generic
    IDLE          = "idle"
    MOVING        = "moving"
    WAITING       = "waiting"
    BLOCKED       = "blocked"
    FINISHED      = "finished"
    ERROR         = "error"

    # Product — logistics lifecycle
    UNPROCESSED   = "unprocessed"
    IN_TRANSIT    = "in_transit"
    PROCESSING    = "processing"
    PACKED        = "packed"
    DISPATCHED    = "dispatched"
    LOST          = "lost"

    # Worker — operator steps
    PICKING       = "picking"
    PLACING       = "placing"
    SCANNING      = "scanning"
    RESTING       = "resting"

    # Robot — AGV steps
    LIFTING       = "lifting"
    DROPPING      = "dropping"
    CHARGING      = "charging"
    NAVIGATING    = "navigating"

    # Conveyor belt
    CONVEYING     = "conveying"
    STOPPED       = "stopped"
    JAMMED        = "jammed"


# Backward-compatibility alias
AgentWhy = BusinessStep


# ---------------------------------------------------------------------------
# PhysicalBDIState — estado MDP formal (ReadPoint × BusinessStep)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysicalBDIState:
    """
    Formal state of a Physical BDI Agent in the MDP state space.

    Equation 4 of the MAS-DUO thesis:
        States(p) = {RP × BS}

    Attributes
    ----------
    read_point    : str   Zone/ReadPoint of the EPC (Where of EPC vocab)
    business_step : str   BusinessStep of the EPC   (Why of EPC vocab)
    """
    read_point:    str
    business_step: str

    def __str__(self) -> str:
        return f"[{self.read_point}|{self.business_step}]"

    @classmethod
    def from_agent_state(cls, state: "AgentState") -> "PhysicalBDIState":
        """Builds a PhysicalBDIState from a standard AgentState."""
        why = state.why
        return cls(
            read_point    = state.zone_id or "UNKNOWN",
            business_step = why if isinstance(why, str) else why.value,
        )


# ---------------------------------------------------------------------------
# GeneratedBelief — belief generada durante razonamiento BDI (Punto 3.b)
# ---------------------------------------------------------------------------

@dataclass
class GeneratedBelief:
    """
    Belief generated internally by the Physical BDI Agent during
    reasoning (Section 3.4.1, point 3.b of the BDI algorithm).

    "New Generated Beliefs → Cost? Energy? QoS? Safety?"

    These values feed the MDP reward matrices (Equation 6).
    """
    delay:    float = 0.0   # estimated transition delay
    cost:     float = 0.0   # estimated economic cost
    qos:      float = 1.0   # estimated quality of service [0, 1]
    energy:   float = 0.0   # estimated energy consumed
    safety:   float = 1.0   # estimated safety [0, 1]

    def to_dict(self) -> dict:
        return {
            "delay":  self.delay,
            "cost":   self.cost,
            "qos":    self.qos,
            "energy": self.energy,
            "safety": self.safety,
        }

    def __repr__(self) -> str:
        return (
            f"Belief(delay={self.delay:.2f}, cost={self.cost:.2f}, "
            f"qos={self.qos:.2f}, energy={self.energy:.2f})"
        )


# ---------------------------------------------------------------------------
# BDIContext — Beliefs, Desires, Intentions of the agent (Section 3.4)
# ---------------------------------------------------------------------------

@dataclass
class BDIContext:
    """
    Full BDI context of a Physical BDI Agent.

    According to Table 2 of the thesis (Physical BDI Agents):
    · Beliefs:    Fixed — view of environment as 'ready' or 'busy'
    · Desires:    Simple — goal: finish process/operation
    · Intentions: Low level — atomic physical actions
    """
    # BELIEFS — estado actual del entorno percibido por el agente
    current_state:   Optional["PhysicalBDIState"] = None
    epc_data:        Dict[str, Any]               = field(default_factory=dict)
    generated:       GeneratedBelief              = field(default_factory=GeneratedBelief)
    environment:     Dict[str, Any]               = field(default_factory=dict)

    # DESIRES — estado objetivo que el agente quiere alcanzar
    goal_state:      Optional["PhysicalBDIState"] = None

    # INTENTIONS — action selected for execution
    selected_action: Optional[str]                = None
    selected_next:   Optional["PhysicalBDIState"] = None

    # History of executed intentions
    executed_intentions: List[str]                = field(default_factory=list)

    # Actitudes descartadas (puntos 3.f y 3.g del algoritmo BDI)
    dropped_unsuccessful: List[str]               = field(default_factory=list)
    dropped_impossible:   List[str]               = field(default_factory=list)

    # Confirmation from IS Platform
    proposal_approved: bool = True

    def update_beliefs_from_epc(self, epc_event: dict) -> None:
        """
        Actualiza beliefs a partir de un evento EPC entrante.
        Punto 3.a del algoritmo BDI (Fig. 37):
        'Where? When? What? Why?' → identificar estado actual.
        """
        self.epc_data.update(epc_event)
        rp = epc_event.get("read_point") or epc_event.get("zone_id", "UNKNOWN")
        bs = epc_event.get("business_step") or epc_event.get("why", "idle")
        self.current_state = PhysicalBDIState(
            read_point    = rp,
            business_step = bs if isinstance(bs, str) else bs.value,
        )

    def drop_unsuccessful(self, attitude: str) -> None:
        """Punto 3.f del algoritmo BDI."""
        self.dropped_unsuccessful.append(attitude)

    def drop_impossible(self, attitude: str) -> None:
        """Punto 3.g del algoritmo BDI."""
        self.dropped_impossible.append(attitude)

    def commit_intention(
        self, action: str, next_state: "PhysicalBDIState"
    ) -> None:
        """Punto 3.c — Update-intentions & Confirmation."""
        self.selected_action = action
        self.selected_next   = next_state
        self.executed_intentions.append(action)


# ---------------------------------------------------------------------------
# AgentState  — the core of the 4W model
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """
    Complete state of an agent at a simulation instant (4W model).

    Fields
    ------
    what : str
        Unique identifier of the agent.
        · Products  → EPC URI  (urn:epc:id:sgtin:company.item.serial)
        · Others    → agent_id (ROB-001, WRK-002, CB-003, …)
    where : Position
        Position (x, y) in the factory grid.
        In EPC terms → coordinates of the physical Read Point.
    when : int
        Simulation step. In EPC terms → Event Time.
    why : str | BusinessStep
        Business Step: reason/business context of the agent.
        Together with zone_id forms the MDP state (RP × BS).
    zone_id : str, optional
        Logical Read Point (zone name).
    payload : dict, optional
        Additional data (full EPC dict, energy, load, …).
    """
    what:     str
    where:    Position
    when:     int
    why:      str
    zone_id:  Optional[str] = None
    payload:  dict          = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Conversion to the MDP state space (RP × BS)
    # -----------------------------------------------------------------------

    def to_mdp_state(self) -> PhysicalBDIState:
        """Converts this AgentState to the MDP state (ReadPoint × BusinessStep)."""
        return PhysicalBDIState.from_agent_state(self)

    def to_epc_event(self) -> dict:
        """
        Generates an 'EPC event' with the 4W data.
        Simulates the information that an EPCIS subscription would publish.
        """
        why = self.why
        return {
            "what":          self.what,
            "read_point":    self.zone_id or "UNKNOWN",
            "event_time":    self.when,
            "business_step": why if isinstance(why, str) else why.value,
            "where":         {"x": self.where.x, "y": self.where.y},
            "payload":       self.payload,
        }

    # -----------------------------------------------------------------------
    # Utilidades
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "what":    self.what,
            "where":   {"x": self.where.x, "y": self.where.y},
            "when":    self.when,
            "why":     self.why,
            "zone_id": self.zone_id,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentState":
        return cls(
            what    = d["what"],
            where   = Position(d["where"]["x"], d["where"]["y"]),
            when    = d["when"],
            why     = d["why"],
            zone_id = d.get("zone_id"),
            payload = d.get("payload", {}),
        )

    def __repr__(self) -> str:
        return (
            f"AgentState("
            f"what={self.what!r}, "
            f"where={self.where}, "
            f"when={self.when}, "
            f"why={self.why!r})"
        )


# ---------------------------------------------------------------------------
# BaseAgent — clase abstracta para todos los agentes
# ---------------------------------------------------------------------------

class BaseAgent(abc.ABC):
    """
    Abstract base class for all Physical BDI Agents in the logistics environment.

    All agents maintain:
    · A 4W state history (trail) for EPCIS traceability
    · A BDIContext with the current beliefs/desires/intentions
    · The formal MDP state (PhysicalBDIState) as a (RP × BS) pair

    The BDI cycle (Figure 37 of the MAS-DUO thesis) is implemented in subclasses.
    """

    def __init__(self, agent_id: str, start_pos: Position):
        self.agent_id:    str           = agent_id
        self._state:      AgentState    = AgentState(
            what    = agent_id,
            where   = start_pos,
            when    = 0,
            why     = BusinessStep.IDLE,
        )
        self.state_trail: List[AgentState] = [self._state]

        # BDI context of the agent (Sec 3.4)
        self.bdi: BDIContext = BDIContext(
            current_state = self._state.to_mdp_state(),
            epc_data      = self._state.to_epc_event(),
        )

    # -----------------------------------------------------------------------
    # State property with automatic history
    # -----------------------------------------------------------------------

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def mdp_state(self) -> PhysicalBDIState:
        """Current MDP state of the agent (ReadPoint × BusinessStep)."""
        return self._state.to_mdp_state()

    def update_state(
        self,
        step:    int,
        where:   Optional[Position]    = None,
        why:     Optional[str]         = None,
        zone_id: Optional[str]         = None,
        payload: Optional[dict]        = None,
    ) -> None:
        """
        Updates the agent state and records it in the 4W history.

        After the update, generates an EPC event and updates the BDIContext.
        Simulates point 3.e of the BDI algorithm: 'Get-new-external-events'.
        """
        self._state = AgentState(
            what    = self._state.what,          # "what" never changes
            where   = where   if where   is not None else self._state.where,
            when    = step,
            why     = why     if why     is not None else self._state.why,
            zone_id = zone_id if zone_id is not None else self._state.zone_id,
            payload = payload if payload is not None else self._state.payload,
        )
        self.state_trail.append(self._state)
        # Update BDI beliefs with new EPC event (Point 3.e)
        self.bdi.update_beliefs_from_epc(self._state.to_epc_event())

    # -----------------------------------------------------------------------
    # BDI interface (Figure 37 thesis)
    # -----------------------------------------------------------------------

    def generate_beliefs(self, grid_state: dict) -> GeneratedBelief:
        """
        Generates beliefs from the environment state.
        Point 3.b of the BDI algorithm: 'New Generated Beliefs → Cost? Energy? QoS?'

        Default implementation → override in each agent type.
        """
        return GeneratedBelief()

    # -----------------------------------------------------------------------
    # Abstract interface
    # -----------------------------------------------------------------------

    @abc.abstractmethod
    def get_observation(self, grid_state: dict) -> Any:
        """Returns the local observation of the agent."""
        ...

    @abc.abstractmethod
    def step(self, action: int, grid_state: dict, sim_step: int) -> dict:
        """
        Executes an action and returns step info:
          {'reward': float, 'done': bool, 'info': dict}
        """
        ...

    @abc.abstractmethod
    def action_space_size(self) -> int:
        """Number of available discrete actions."""
        ...

    # -----------------------------------------------------------------------
    # Common utilities
    # -----------------------------------------------------------------------

    @property
    def position(self) -> Position:
        return self._state.where

    @property
    def current_why(self) -> str:
        return self._state.why

    @property
    def current_zone(self) -> Optional[str]:
        """Current ReadPoint of the agent (logical zone)."""
        return self._state.zone_id

    def get_trail_summary(self) -> list[dict]:
        """Returns the full 4W history of the agent."""
        return [s.to_dict() for s in self.state_trail]

    def get_epc_trail(self) -> List[dict]:
        """Returns the history as a sequence of EPC events (4W → EPC vocab)."""
        return [s.to_epc_event() for s in self.state_trail]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id!r}, state={self._state})"

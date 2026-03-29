"""
ISPlatform — Information System Platform (Section 3.5)
=======================================================
Second platform of the MAS-DUO architecture.

Contains the IS BDI Agents that are the INTERFACE between the Physical Platform
and business systems (ERP, CRM, Expert System).

According to the thesis (Table 2), IS BDI Agents have:
  - Beliefs:    Complex (multiple sources: DB, other IT, physical, constraints)
  - Desires:    Dynamic and flexible (change frequently)
  - Intentions: Complex, compound → tactical/strategic decision-making

Negotiation protocol (Section 3.7.5 — Coordination Model):
  1. The Physical Agent sends a proposal (NegotiationProposal) to the IS Platform
     with the computed reward and newly generated beliefs
  2. IS BDI Agents negotiate among themselves using Contract Net Protocol (FIPA)
  3. If minimum consensus → approval (NegotiationResult.APPROVED)
  4. If rejected → new policy parameters are sent
     and the Physical Agent recomputes its plan

The IS Platform is NOT coupled to the Physical Platform; it only shares
results and policy parameters (uncoupling, Sec 3.3).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from logistics_env.is_platform.global_policy import GlobalPolicy, PolicyParameters


# ---------------------------------------------------------------------------
# NegotiationProposal — propuesta del Physical Agent a la IS Platform
# ---------------------------------------------------------------------------

@dataclass
class NegotiationProposal:
    """
    Proposal sent by a Physical BDI Agent to the IS Platform.

    Equivalent to the FIPA PROPOSE message in the Contract Net Protocol.
    Contains the computed reward and the beliefs generated during the
    BDI reasoning cycle (Section 3.4.1, point 3.c).
    """
    agent_id:       str            # ID del Physical BDI Agent
    state_from:     str            # Estado actual (RP|BS)
    state_to:       str            # Estado objetivo (RP|BS)
    reward:         float          # R(s,s') calculado con A,B,C,D
    beliefs:        Dict[str, Any] # Beliefs generadas: cost, energy, qos, delay
    order_id:       Optional[str] = None
    step:           int            = 0

    def to_dict(self) -> dict:
        return {
            "agent_id":   self.agent_id,
            "from":       self.state_from,
            "to":         self.state_to,
            "reward":     self.reward,
            "beliefs":    self.beliefs,
            "order_id":   self.order_id,
            "step":       self.step,
        }


# ---------------------------------------------------------------------------
# NegotiationResult — result of the IS Platform negotiation
# ---------------------------------------------------------------------------

class NegotiationOutcome(str, Enum):
    APPROVED = "approved"    # Plan aprobado → Physical Agent ejecuta
    REJECTED = "rejected"    # Plan rejected → recompute with new policy
    PENDING  = "pending"     # Under negotiation


@dataclass
class NegotiationResult:
    """
    Result returned by the IS Platform to the Physical BDI Agent.

    If APPROVED: the agent may execute the action.
    If REJECTED: new policy parameters are attached for recomputation.
    """
    outcome:        NegotiationOutcome
    new_policy:     Optional[PolicyParameters] = None   # solo si REJECTED
    message:        str                        = ""
    approved_by:    List[str]                  = field(default_factory=list)


# ---------------------------------------------------------------------------
# IS BDI Agent base abstracta
# ---------------------------------------------------------------------------

class ISBDIAgent(abc.ABC):
    """
    Base class for all IS BDI Agents (Section 3.5).

    BDI characteristics (Table 2):
    · Beliefs:    Complex — multi-source (DB, ERPs, physical, clients)
    · Desires:    Dynamic and flexible
    · Intentions: Complex, compound (tactical/strategic level)
    """

    def __init__(self, agent_id: str, name: str):
        self.agent_id: str = agent_id
        self.name:     str = name
        self._beliefs: Dict[str, Any] = {}
        self._history: List[Dict] = []

    @abc.abstractmethod
    def evaluate_proposal(
        self,
        proposal:       NegotiationProposal,
        global_policy:  GlobalPolicy,
        step:           int,
    ) -> Tuple[bool, str]:
        """
        Evaluates a proposal from the Physical Platform.

        Returns (approved: bool, reason: str).
        """
        ...

    def update_beliefs(self, key: str, value: Any) -> None:
        self._beliefs[key] = value

    def get_belief(self, key: str, default: Any = None) -> Any:
        return self._beliefs.get(key, default)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id!r})"


# ---------------------------------------------------------------------------
# ERPAgent — ERP interface agent (Section 3.5)
# ---------------------------------------------------------------------------

class ERPAgent(ISBDIAgent):
    """
    IS Agent that interfaces with the ERP.

    Responsibilities:
    · Order control (incomings and order recipients)
    · Trigger initialisation of Physical BDI Agents (point 1 of algorithm)
    · Plan validation against production and stock constraints
    · Sourcing contracts, production decisions, inventory (Sec 3.5 list)
    """

    def __init__(self):
        super().__init__(agent_id="IS-ERP-001", name="ERP Agent")
        # Beliefs del ERP
        self._beliefs = {
            "open_orders":        [],    # open orders
            "production_capacity": 1.0,  # productive capacity [0,1]
            "inventory_levels":   {},    # stock by zone_id
            "max_allowed_cost":   float("inf"),   # cost limit per transaction
            "max_allowed_delay":  float("inf"),   # acceptable delay limit
        }

    def evaluate_proposal(
        self,
        proposal:      NegotiationProposal,
        global_policy: GlobalPolicy,
        step:          int,
    ) -> Tuple[bool, str]:
        """
        The ERP approves if the reward and beliefs satisfy constraints.

        Checks:
        1. Transition cost does not exceed the ERP limit
        2. Delay does not exceed the order deadline
        3. Production capacity is compatible
        """
        beliefs = proposal.beliefs

        # Verify cost
        cost = abs(beliefs.get("cost", 0.0))
        max_cost = self._beliefs["max_allowed_cost"]
        if cost > max_cost:
            return False, f"ERPAgent: cost {cost:.2f} exceeds limit {max_cost:.2f}"

        # Verify delay
        delay = abs(beliefs.get("delay", 0.0))
        max_delay = self._beliefs["max_allowed_delay"]
        if delay > max_delay:
            return False, f"ERPAgent: delay {delay:.2f} exceeds limit {max_delay:.2f}"

        # Verify production capacity
        if self._beliefs["production_capacity"] <= 0.0:
            return False, "ERPAgent: sin capacidad productiva disponible"

        return True, "ERPAgent: aprobado"

    def register_order_completion(self, order_id: str, step: int, on_time: bool) -> None:
        """Registers order completion to update KPIs."""
        self._history.append({
            "event":   "order_complete",
            "order":   order_id,
            "step":    step,
            "on_time": on_time,
        })

    def configure(
        self,
        max_cost:  float = float("inf"),
        max_delay: float = float("inf"),
        capacity:  float = 1.0,
    ) -> None:
        self._beliefs["max_allowed_cost"]   = max_cost
        self._beliefs["max_allowed_delay"]  = max_delay
        self._beliefs["production_capacity"] = capacity


# ---------------------------------------------------------------------------
# CRMAgent — CRM interface agent (Section 3.5)
# ---------------------------------------------------------------------------

class CRMAgent(ISBDIAgent):
    """
    IS Agent that interfaces with the CRM.

    Responsibilities:
    · Client priority → adjustment of QoS weights in the policy
    · Demand management and client behaviour tracking
    · Milestone payments and client scoring
    """

    def __init__(self):
        super().__init__(agent_id="IS-CRM-001", name="CRM Agent")
        self._beliefs = {
            "client_priorities":  {},   # {order_id: priority (0-1)}
            "min_qos_threshold":  0.0,  # minimum acceptable QoS
            "client_satisfaction": 1.0, # current global satisfaction
        }

    def evaluate_proposal(
        self,
        proposal:      NegotiationProposal,
        global_policy: GlobalPolicy,
        step:          int,
    ) -> Tuple[bool, str]:
        """
        The CRM approves if the QoS of the proposal exceeds the minimum threshold
        weighted by client priority.
        """
        qos = beliefs = proposal.beliefs.get("qos", 1.0)
        min_qos = self._beliefs["min_qos_threshold"]

        # Adjust threshold by order priority
        if proposal.order_id:
            priority = self._beliefs["client_priorities"].get(
                proposal.order_id, 0.5
            )
            # Clientes de alta prioridad requieren mejor QoS
            adjusted_min = min_qos + priority * 0.2
        else:
            adjusted_min = min_qos

        if qos < adjusted_min:
            return False, (
                f"CRMAgent: QoS {qos:.2f} < required minimum {adjusted_min:.2f}"
            )

        return True, "CRMAgent: QoS aceptable"

    def set_client_priority(self, order_id: str, priority: float) -> None:
        """Establece la prioridad de un cliente/pedido [0, 1]."""
        self._beliefs["client_priorities"][order_id] = max(0.0, min(1.0, priority))

    def configure(self, min_qos: float = 0.0) -> None:
        self._beliefs["min_qos_threshold"] = min_qos


# ---------------------------------------------------------------------------
# ExpertSystemAgent — Expert system / benchmarking (Section 3.5)
# ---------------------------------------------------------------------------

class ExpertSystemAgent(ISBDIAgent):
    """
    IS Agent implementing the strategic expert system.

    Responsibilities:
    · Operational benchmarking vs. competitors
    · Dynamic adjustment of policy parameters (when IS Platform rejects)
    · Detection of strategic misalignments
    · Plant KPIs → continuous improvement of global parameters
    """

    def __init__(self):
        super().__init__(agent_id="IS-EXPERT-001", name="Expert System Agent")
        self._beliefs = {
            "kpi_on_time_rate":   1.0,   # % on-time orders
            "kpi_avg_cost":       0.0,   # average cost per transaction
            "kpi_avg_energy":     0.0,   # average energy
            "rejection_count":    0,     # accumulated rejections
            "min_reward_threshold": -float("inf"),  # minimum acceptable reward
        }
        # Historial de KPIs para aprendizaje
        self._kpi_window: List[dict] = []

    def evaluate_proposal(
        self,
        proposal:      NegotiationProposal,
        global_policy: GlobalPolicy,
        step:          int,
    ) -> Tuple[bool, str]:
        """
        The expert system checks whether the reward exceeds the minimum threshold
        computed by historical benchmarking.
        """
        min_r = self._beliefs["min_reward_threshold"]
        if proposal.reward < min_r:
            return False, (
                f"ExpertSystem: reward {proposal.reward:.3f} < threshold {min_r:.3f}"
            )
        return True, "ExpertSystem: reward aceptable"

    def suggest_policy_adjustment(
        self,
        rejected_proposal: NegotiationProposal,
        current_policy:    GlobalPolicy,
    ) -> PolicyParameters:
        """
        Suggests policy parameter adjustments when a plan is rejected.

        Heuristic logic based on expert system beliefs:
        - If cost was the rejection reason → reduce B
        - If delay was the reason → increase A
        - If QoS was the reason → increase C
        """
        self._beliefs["rejection_count"] += 1

        beliefs = rejected_proposal.beliefs
        A, B, C, D = current_policy.as_tuple

        # Heuristic adjustment
        cost   = abs(beliefs.get("cost",   0.0))
        delay  = abs(beliefs.get("delay",  0.0))
        qos    =     beliefs.get("qos",    1.0)
        energy = abs(beliefs.get("energy", 0.0))

        # Adjustment factor inversely proportional to number of rejections
        factor = max(0.05, 0.2 / (1 + self._beliefs["rejection_count"] * 0.1))

        if cost > 0.3:
            B = max(0.1, B - factor)
        if delay > 0.3:
            A = min(3.0, A + factor)
        if qos < 0.5:
            C = min(3.0, C + factor)
        if energy > 0.3:
            D = max(0.1, D - factor)

        return PolicyParameters(A=A, B=B, C=C, D=D, step=rejected_proposal.step)

    def record_kpis(
        self, on_time: float, avg_cost: float, avg_energy: float, step: int
    ) -> None:
        """Registra KPIs para benchmarking y ajuste de umbral de reward."""
        record = {
            "step":       step,
            "on_time":    on_time,
            "avg_cost":   avg_cost,
            "avg_energy": avg_energy,
        }
        self._kpi_window.append(record)
        # Mantener ventana deslizante de 50 registros
        if len(self._kpi_window) > 50:
            self._kpi_window.pop(0)

        # Actualizar beliefs
        self._beliefs["kpi_on_time_rate"] = on_time
        self._beliefs["kpi_avg_cost"]     = avg_cost
        self._beliefs["kpi_avg_energy"]   = avg_energy


# ---------------------------------------------------------------------------
# ISPlatform — IS platform coordinator (Section 3.5)
# ---------------------------------------------------------------------------

class ISPlatform:
    """
    IS Platform coordinator in MAS-DUO.

    Orchestrates negotiation among IS BDI Agents using a variant
    of the Contract Net Protocol (FIPA), as described in Section 3.7.5.

    Negotiation protocol:
        1. Physical Agent → ISPlatform.evaluate(proposal)
        2. ISPlatform consults ERP, CRM, ExpertSystem
        3. If minimum consensus → NegotiationOutcome.APPROVED
        4. If rejected → ExpertSystem suggests new policy
           → NegotiationOutcome.REJECTED + new_policy

    The IS Platform modifies the global policy parameters (A,B,C,D)
    and sends them to the physical agent to recompute its plan.
    """

    def __init__(self, global_policy: GlobalPolicy):
        self.global_policy:   GlobalPolicy         = global_policy
        self.erp_agent:       ERPAgent             = ERPAgent()
        self.crm_agent:       CRMAgent             = CRMAgent()
        self.expert_agent:    ExpertSystemAgent     = ExpertSystemAgent()

        self._is_agents: List[ISBDIAgent] = [
            self.erp_agent,
            self.crm_agent,
            self.expert_agent,
        ]

        # Log de todas las negociaciones
        self._negotiation_log: List[dict] = []

    # -----------------------------------------------------------------------
    # Negotiation protocol (Section 3.7.5)
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        proposal: NegotiationProposal,
        step:     int,
    ) -> NegotiationResult:
        """
        Evaluates a proposal from the Physical Platform via IS Agent consensus.

        Minimum consensus: all IS BDI agents must approve
        (the thesis defines consensus as agreement of the full IS agent set).
        """
        votes  = []
        reasons = []

        for agent in self._is_agents:
            approved, reason = agent.evaluate_proposal(
                proposal=proposal,
                global_policy=self.global_policy,
                step=step,
            )
            votes.append(approved)
            reasons.append(f"[{agent.name}] {reason}")

        all_approved = all(votes)
        approved_by  = [
            agent.name for agent, v in zip(self._is_agents, votes) if v
        ]

        if all_approved:
            result = NegotiationResult(
                outcome     = NegotiationOutcome.APPROVED,
                approved_by = approved_by,
                message     = "; ".join(reasons),
            )
        else:
            # Expert system suggests new policy
            new_policy = self.expert_agent.suggest_policy_adjustment(
                rejected_proposal=proposal,
                current_policy=self.global_policy,
            )
            # In DYNAMIC mode, update the global policy
            if self.global_policy.mode.value == "dynamic":
                try:
                    self.global_policy.update(
                        step=step,
                        A=new_policy.A, B=new_policy.B,
                        C=new_policy.C, D=new_policy.D,
                        reason="ExpertSystem adjustment after rejection",
                    )
                except ValueError:
                    pass  # modo STATIC protegido

            result = NegotiationResult(
                outcome     = NegotiationOutcome.REJECTED,
                new_policy  = new_policy,
                approved_by = approved_by,
                message     = "; ".join(reasons),
            )

        # Log
        self._negotiation_log.append({
            "step":       step,
            "agent_id":   proposal.agent_id,
            "outcome":    result.outcome.value,
            "reward":     proposal.reward,
            "approved_by": approved_by,
        })

        return result

    # -----------------------------------------------------------------------
    # Initialisation trigger for Physical BDI Agents (steps 1 and 2 of alg.)
    # -----------------------------------------------------------------------

    def initialize_agent(
        self, agent_id: str, order_id: str, epc: str, step: int
    ) -> Dict[str, Any]:
        """
        The ERP triggers initialisation of a Physical BDI Agent when
        a new product or order is received (step 1 of BDI algorithm, Fig.37).

        Returns the EPC subscription and initial policy parameters.
        """
        self.erp_agent._beliefs.setdefault("open_orders", []).append(order_id)
        return {
            "epc_subscription": epc,
            "order_id":         order_id,
            "policy":           self.global_policy.params.as_tuple(),
            "step":             step,
        }

    # -----------------------------------------------------------------------
    # KPI feedback desde Physical Platform (alineamiento, Sec 3.6)
    # -----------------------------------------------------------------------

    def record_physical_result(
        self,
        agent_id:  str,
        order_id:  Optional[str],
        on_time:   bool,
        cost:      float,
        energy:    float,
        step:      int,
    ) -> None:
        """
        Registra el resultado de un Physical Agent para actualizar KPIs
        en el sistema experto (feedback de la IS Platform, Sec 3.5).
        """
        self.expert_agent.record_kpis(
            on_time    = 1.0 if on_time else 0.0,
            avg_cost   = cost,
            avg_energy = energy,
            step       = step,
        )
        if order_id:
            self.erp_agent.register_order_completion(order_id, step, on_time)

    # -----------------------------------------------------------------------
    # Tick (apply scheduled policy changes)
    # -----------------------------------------------------------------------

    def tick(self, step: int) -> Optional[PolicyParameters]:
        """Call at each step to apply scheduled policy changes."""
        return self.global_policy.tick(step)

    # -----------------------------------------------------------------------
    # Access to current global policy
    # -----------------------------------------------------------------------

    @property
    def policy_params(self) -> Tuple[float, float, float, float]:
        return self.global_policy.as_tuple

    @property
    def A(self) -> float: return self.global_policy.A
    @property
    def B(self) -> float: return self.global_policy.B
    @property
    def C(self) -> float: return self.global_policy.C
    @property
    def D(self) -> float: return self.global_policy.D

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def get_log_summary(self) -> List[dict]:
        return list(self._negotiation_log)

    def configure_erp(self, **kwargs) -> None:
        self.erp_agent.configure(**kwargs)

    def configure_crm(
        self,
        min_qos: float = 0.0,
        client_priority: float = 1.0,
        **client_priorities,
    ) -> None:
        self.crm_agent.configure(min_qos=min_qos)
        for order_id, priority in client_priorities.items():
            self.crm_agent.set_client_priority(order_id, priority)

    def configure_expert(self, min_reward: float = float("-inf")) -> None:
        """Sets the minimum reward threshold for the expert system."""
        self.expert_agent._beliefs["min_reward_threshold"] = min_reward

    def get_summary(self) -> dict:
        """Summary of IS Platform state for logging/traceability."""
        total = len(self._negotiation_log)
        approved = sum(
            1 for r in self._negotiation_log if r["outcome"] == "approved"
        )
        return {
            "total_negotiations": total,
            "approved": approved,
            "rejected": total - approved,
            "approval_rate": approved / total if total > 0 else 1.0,
            "policy": {
                "A": self.global_policy.A,
                "B": self.global_policy.B,
                "C": self.global_policy.C,
                "D": self.global_policy.D,
                "mode": self.global_policy.mode.value,
            },
            "rejection_count": self.expert_agent._beliefs.get("rejection_count", 0),
        }

    def __repr__(self) -> str:
        return (
            f"ISPlatform(mode={self.global_policy.mode.value}, "
            f"policy={self.global_policy.params})"
        )

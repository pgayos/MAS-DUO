"""
ProductAgent — Physical BDI Agent for the product (Section 3.4 MAS-DUO thesis)
============================================================================
Each physical product unit is a Physical BDI Agent whose identifier
"What" is its EPC (urn:epc:id:sgtin:…).

Implements the customised BDI algorithm (Figure 37 of the thesis):

  1. Initialize-state()
  2. Subscription(EPC)  → receives EPCIS events
  3. Repeat while new EPC event or new task or proactive trigger:
     a. Options: option-generator & state identification (Where? When? What? Why?)
     b. Selected-options: deliberate(options)
        · New Generated Beliefs → Cost? Energy? QoS?
        · New Reward Selection  → MDP (A, B, C, D)
     c. Update-intentions & Confirmation (send proposal to IS Platform)
     d. Execution & Results
     e. Get-new-external-events (Where? When? What? Why?)
     f. Drop-unsuccessful-attitudes()
     g. Drop-impossible-attitudes()
  4. End repeat

The MDP state of the product follows Equation 4:
    States(p) = {ReadPoint × BusinessStep}
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logistics_env.agents.base_agent import (
    BaseAgent, AgentWhy, BusinessStep, Position,
    GeneratedBelief, PhysicalBDIState,
)
from logistics_env.objects.epc import EPC
from logistics_env.config_loader import ProductTypeConfig
from logistics_env.mdp.reward_matrices import RewardMatrices, TransitionReward
from logistics_env.mdp.mdp_engine import MDPEngine, MDPState


# ---------------------------------------------------------------------------
# Product action space
# ---------------------------------------------------------------------------

class ProductAction(IntEnum):
    """
    Actions of the Physical BDI Product Agent.
    They represent the agent's 'intentions' in the MDP action space.
    """
    WAIT            = 0   # Idle — wait without moving
    REQUEST_MOVE    = 1   # Request advance to the next route state
    REQUEST_PROCESS = 2   # Request processing in the current zone
    SIGNAL_READY    = 3   # Signal that it is ready for dispatch


_N_ACTIONS = len(ProductAction)


# ---------------------------------------------------------------------------
# ProductAgent
# ---------------------------------------------------------------------------

class ProductAgent(BaseAgent):
    """
    Agent representing a physical product unit.

    The agent's 4W state:
    ─────────────────────────────────────────────────────
    What  → EPC URI  (urn:epc:id:sgtin:<company>.<item>.<serial>)
    Where → position (x, y) on the grid
    When  → current simulation step
    Why   → logical state: UNPROCESSED / IN_TRANSIT / PROCESSING /
            PACKED / DISPATCHED / LOST
    ─────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        epc:          EPC,
        product_type: ProductTypeConfig,
        start_pos:    Position,
        order_id:     str,
        deadline:     int,
    ):
        # The agent's "what" is the EPC URI
        super().__init__(agent_id=epc.pure_identity_uri, start_pos=start_pos)

        self.epc:           EPC                 = epc
        self.product_type:  ProductTypeConfig   = product_type
        self.order_id:      str                 = order_id
        self.deadline:      int                 = deadline

        # Route of zones the product must traverse
        self.route:         List[str]           = list(product_type.processing_steps)
        self.route_index:   int                 = 0   # current zone in the route

        # Remaining processing time in current zone
        self.process_ticks_remaining: int       = 0

        # Energy consumed in transport for this product
        self.energy_consumed: float             = 0.0

        # Who is carrying the product now (agent_id or None)
        self.carried_by:    Optional[str]       = None

        # Actualizar estado inicial con EPC payload
        self._state = self._state.__class__(
            what    = epc.pure_identity_uri,
            where   = start_pos,
            when    = 0,
            why     = AgentWhy.UNPROCESSED,
            zone_id = self.route[0] if self.route else None,
            payload = epc.to_dict(),
        )
        self.state_trail = [self._state]

        # MDP engine and global policy parameters (Section 3.4.2 / 3.6)
        # Initialised until set_policy_params() is called by the IS Platform
        self._mdp_engine: Optional[MDPEngine] = None
        self._policy_A: float = 1.0   # Delay weight
        self._policy_B: float = 1.0   # Cost weight
        self._policy_C: float = 1.0   # QoS weight
        self._policy_D: float = 0.5   # Energy weight
        # Initialise the MDP engine with the product type configuration
        self._init_mdp_engine(product_type)

    # -----------------------------------------------------------------------
    # MDP engine initialisation (Section 3.4.2)
    # -----------------------------------------------------------------------

    def _init_mdp_engine(self, product_type: ProductTypeConfig) -> None:
        """
        Creates the MDPEngine for this product.

        The MDP states are all (zone × business_step) pairs from the route.
        Equation 4: States(p) = {RP × BS}

        Transitions between consecutive route states are
        initialised with default values that are refined through learning.
        """
        # Business steps that can occur at each zone
        bs_sequence = [
            BusinessStep.UNPROCESSED,
            BusinessStep.IN_TRANSIT,
            BusinessStep.PROCESSING,
            BusinessStep.PACKED,
            BusinessStep.DISPATCHED,
        ]

        # Create MDP states: (zone, business_step) pair for each route zone
        mdp_states: List[MDPState] = []
        for zone in product_type.processing_steps:
            for bs in bs_sequence:
                mdp_states.append(MDPState(read_point=zone, business_step=bs.value))

        # Special start and end states
        start_zone = product_type.processing_steps[0] if product_type.processing_steps else "RECEIVING"
        end_zone   = product_type.processing_steps[-1] if product_type.processing_steps else "DISPATCH"
        mdp_states.append(MDPState(read_point="UNKNOWN",  business_step=BusinessStep.IDLE.value))
        mdp_states.append(MDPState(read_point=end_zone,   business_step=BusinessStep.LOST.value))

        n = len(mdp_states)
        rm = RewardMatrices(n_states=n, agent_class=f"product:{product_type.item_reference}")

        # Define main route transitions with default rewards
        # (Refined through learning - Sec 3.7.4)
        for i in range(n - 3):  # skip extra states (UNKNOWN, LOST)
            if i + 1 < n - 2:
                rm.set_transition(
                    s_from = i,
                    s_to   = i + 1,
                    delay  = -0.1 * (i + 1),    # delay grows with the route
                    cost   = -0.2,              # cost per movement
                    qos    = 0.9,               # default quality high
                    energy = -0.1,              # consumption per movement
                    valid  = True,
                )

        self._mdp_engine = MDPEngine(
            states          = mdp_states,
            reward_matrices = rm,
        )

    def set_policy_params(self, A: float, B: float, C: float, D: float) -> None:
        """
        Updates the global policy parameters received from the IS Platform.
        (Equation 9 of the thesis: GlobalPolicyParameters)
        """
        self._policy_A = A
        self._policy_B = B
        self._policy_C = C
        self._policy_D = D

    # -----------------------------------------------------------------------
    # Propiedades
    # -----------------------------------------------------------------------

    @property
    def current_zone_target(self) -> Optional[str]:
        if self.route_index < len(self.route):
            return self.route[self.route_index]
        return None

    @property
    def is_dispatched(self) -> bool:
        return self._state.why == BusinessStep.DISPATCHED

    @property
    def is_at_final_zone(self) -> bool:
        return self.route_index >= len(self.route) - 1

    # -----------------------------------------------------------------------
    # Avance de ruta
    # -----------------------------------------------------------------------

    def arrive_at_zone(self, zone_id: str, pos: Position, step: int) -> None:
        """Marks that the product has arrived at a route zone."""
        if self.route_index < len(self.route) and self.route[self.route_index] == zone_id:
            self.route_index += 1
            why = BusinessStep.IN_TRANSIT

            if zone_id == "DISPATCH" or self.route_index >= len(self.route):
                why = BusinessStep.DISPATCHED
            elif self.product_type.requires_processing and "PROCESSING" in zone_id:
                why = BusinessStep.PROCESSING
                self.process_ticks_remaining = self.product_type.process_time_steps
            elif "PACKING" in zone_id:
                why = BusinessStep.PACKED

            self.update_state(
                step=step, where=pos, why=why, zone_id=zone_id,
                payload={**self.epc.to_dict(), "route_index": self.route_index},
            )

    def tick_processing(self) -> bool:
        """
        Advances one processing tick. Returns True when complete.
        """
        if self.process_ticks_remaining > 0:
            self.process_ticks_remaining -= 1
        return self.process_ticks_remaining == 0

    # -----------------------------------------------------------------------
    # Belief generation (Point 3.b of BDI algorithm, Figure 37)
    # -----------------------------------------------------------------------

    def generate_beliefs(self, grid_state: dict) -> GeneratedBelief:
        """
        Generates focused beliefs for MDP decision-making.

        Section 3.4.1 point 3.b:
        'New Generated Beliefs → Cost? Energy? QoS? Safety?'

        These beliefs feed the reward matrices (Equation 6).
        """
        step = grid_state.get("step", 0)
        remaining = max(0, self.deadline - step)
        n_agents  = max(1, len(grid_state.get("products_at", {})))

        # Delay: proportional to remaining time vs. deadline
        delay = -max(0.0, 1.0 - remaining / max(1, self.deadline))

        # Cost: per movement and per carry (carried_by != None)
        cost = -0.2 if self.carried_by is None else -0.1

        # QoS: decreases as deadline approaches
        qos = min(1.0, remaining / max(1, self.deadline))

        # Energy: proxy — worse the more moves accumulated
        energy = -self.energy_consumed * 0.01

        belief = GeneratedBelief(
            delay=delay, cost=cost, qos=qos, energy=energy
        )
        # Save to BDI context
        self.bdi.generated = belief
        return belief

    # -----------------------------------------------------------------------
    # MDP deliberation (Point 3.b — select best instruction via MDP)
    # -----------------------------------------------------------------------

    def deliberate_mdp(self) -> Optional[PhysicalBDIState]:
        """
        Selects the best state transition using the MDPEngine.

        Section 3.4.2 of the thesis:
        Q^\u03c0(s,a) = R(s) + γ · Σ P(s'|s,a) · V^\u03c0(s')
        """
        if self._mdp_engine is None:
            return None

        current = self._state.to_mdp_state()
        # Convert to MDPState
        current_mdp = MDPState(
            read_point=current.read_point,
            business_step=current.business_step,
        )
        goal_zone = self.route[-1] if self.route else "DISPATCH"
        goal_mdp  = MDPState(
            read_point    = goal_zone,
            business_step = BusinessStep.DISPATCHED.value,
        )
        # Convert goal to PhysicalBDIState for return
        next_mdp: Optional[MDPState] = self._mdp_engine.select_action(
            current_state = current_mdp,
            goal_state    = goal_mdp,
            A = self._policy_A,
            B = self._policy_B,
            C = self._policy_C,
            D = self._policy_D,
        )
        if next_mdp is None:
            return None
        return PhysicalBDIState(
            read_point    = next_mdp.read_point,
            business_step = next_mdp.business_step,
        )

    def update_q_after_execution(
        self,
        s_from:   PhysicalBDIState,
        s_to:     PhysicalBDIState,
        belief:   GeneratedBelief,
        success:  bool,
    ) -> None:
        """
        Updates the Q-table with observed results (Learning Model, Sec 3.7.4).
        Call after the action has been executed (point 3.d).
        """
        if self._mdp_engine is None:
            return
        from_mdp = MDPState(read_point=s_from.read_point, business_step=s_from.business_step)
        to_mdp   = MDPState(read_point=s_to.read_point,   business_step=s_to.business_step)
        tr = TransitionReward(
            delay  = belief.delay,
            cost   = belief.cost,
            qos    = belief.qos,
            energy = belief.energy,
        )
        self._mdp_engine.update_q(
            s_from=from_mdp, s_to=to_mdp, observed=tr,
            A=self._policy_A, B=self._policy_B,
            C=self._policy_C, D=self._policy_D,
            success=success,
        )

    # -----------------------------------------------------------------------
    # Recompensa local del producto
    # -----------------------------------------------------------------------

    def compute_reward(self, sim_step: int) -> float:
        """
        Instantaneous reward for the product.
        Follows the MAS-DUO reward scheme:
          + advance in route / deliver on time
          - excessive idle
          - energy consumption
          - approaching deadline without dispatching
        """
        reward = 0.0

        if self.is_dispatched:
            steps_saved = max(0, self.deadline - sim_step)
            reward += 100.0 + steps_saved * 2.0

        elif self._state.why == BusinessStep.IDLE:
            reward -= 0.5

        reward -= self.energy_consumed * 0.01
        return reward

    # -----------------------------------------------------------------------
    # BaseAgent interface
    # -----------------------------------------------------------------------

    def get_observation(self, grid_state: dict) -> np.ndarray:
        """
        Physical BDI Agent observation (numeric vector):
          [x_norm, y_norm, route_progress, process_ticks_norm,
           deadline_remaining_norm, is_carried]
        """
        grid_w = grid_state.get("grid_w", 20)
        grid_h = grid_state.get("grid_h", 15)
        step   = grid_state.get("step", 0)
        local = np.array([
            self.position.x / max(1, grid_w),
            self.position.y / max(1, grid_h),
            self.route_index / max(1, len(self.route)),
            self.process_ticks_remaining / max(1, self.product_type.process_time_steps),
            max(0, self.deadline - step) / max(1, self.deadline),
            1.0 if self.carried_by is not None else 0.0,
        ], dtype=np.float32)
        return local

    def step(self, action: int, grid_state: dict, sim_step: int) -> dict:
        """
        Product BDI cycle (Figure 37, points 3.a → 3.g).

        The product:
        a. Identifies its MDP state (Where/When/What/Why via EPC event)
        b. Generates beliefs (Cost, Energy, QoS)
        c. Deliberates with the MDPEngine and selects intention
        d. Returns result for the environment to process the interaction
        e. Discards failed/impossible attitudes
        """
        reward   = 0.0
        done     = self.is_dispatched
        info: dict = {
            "epc":        self.epc.pure_identity_uri,
            "action":     action,
            "mdp_state":  str(self.mdp_state),
            "bdi_belief": self.bdi.generated.to_dict(),
        }

        if done:
            return {"reward": 0.0, "done": True, "info": info}

        # Point 3.a: State identification (EPC event already in bdi.current_state)
        pa = ProductAction(action) if action < _N_ACTIONS else ProductAction.WAIT

        # Point 3.b: Generate beliefs
        belief = self.generate_beliefs(grid_state)

        # Point 3.b: Deliberate with MDP
        next_state = self.deliberate_mdp()
        if next_state:
            self.bdi.commit_intention(
                action=pa.name,
                next_state=next_state,
            )
            info["next_mdp_state"] = str(next_state)

        # Point 3.d: Execute action and compute reward
        if pa == ProductAction.WAIT:
            reward -= 0.2   # penalty for idle
            self.bdi.drop_unsuccessful("wait")
        elif pa == ProductAction.REQUEST_MOVE:
            reward += 0.1   # signal for attempting to advance
        elif pa == ProductAction.REQUEST_PROCESS:
            if self._state.why == BusinessStep.PROCESSING:
                reward += 0.3
        elif pa == ProductAction.SIGNAL_READY:
            if self.is_at_final_zone:
                reward += 1.0

        # Deadline check (Point 3.f: drop impossible attitudes)
        remaining = self.deadline - sim_step
        if remaining <= 0:
            reward -= 5.0
            done = True
            self.update_state(step=sim_step, why=BusinessStep.LOST)
            self.bdi.drop_impossible("any_action")
        elif remaining < 20:
            reward -= 0.5

        # Incorporate MDP avg reward as a plan quality signal
        if self._mdp_engine is not None:
            avg_r = self._mdp_engine.avg_reward
            reward += avg_r * 0.05   # contribution from accumulated learning

        return {"reward": reward, "done": done, "info": info}

    def action_space_size(self) -> int:
        return _N_ACTIONS

    def __repr__(self) -> str:
        return (
            f"ProductAgent("
            f"epc={self.epc.short_id!r}, "
            f"route={self.route_index}/{len(self.route)}, "
            f"why={self._state.why!r})"
        )

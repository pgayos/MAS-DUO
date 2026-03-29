"""
Order / OrderManager — Order management and lifecycle
=====================================================
An order groups several products (with their EPCs) and has:
  - Deadline
  - Priority
  - Status (PENDING → IN_PROGRESS → COMPLETE | FAILED)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from logistics_env.config_loader import OrderConfig, FactoryConfig
from logistics_env.objects.epc import EPC, EPCFactory


class OrderStatus(str, Enum):
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE    = "complete"
    FAILED      = "failed"


@dataclass
class Order:
    """Pedido en curso con referencias a los EPCs de sus productos."""
    config:         OrderConfig
    status:         OrderStatus             = OrderStatus.PENDING
    dispatched_epcs: Set[str]              = field(default_factory=set)
    start_step:     int                     = 0
    finish_step:    Optional[int]           = None

    @property
    def order_id(self) -> str:
        return self.config.order_id

    @property
    def deadline(self) -> int:
        return self.config.deadline_steps

    @property
    def priority(self) -> str:
        return self.config.priority

    @property
    def is_complete(self) -> bool:
        return self.status == OrderStatus.COMPLETE

    @property
    def total_products_needed(self) -> int:
        return sum(ps.quantity for ps in self.config.products)

    @property
    def products_dispatched(self) -> int:
        return len(self.dispatched_epcs)

    def register_dispatch(self, epc_uri: str, step: int) -> None:
        self.dispatched_epcs.add(epc_uri)
        if self.products_dispatched >= self.total_products_needed:
            self.status      = OrderStatus.COMPLETE
            self.finish_step = step

    def mark_failed(self, step: int) -> None:
        if self.status != OrderStatus.COMPLETE:
            self.status      = OrderStatus.FAILED
            self.finish_step = step

    def compute_reward(self, reward_weights, sim_step: int) -> float:
        """Reward for completing or failing an order."""
        rw = reward_weights
        if self.status == OrderStatus.COMPLETE:
            if self.finish_step is not None and self.finish_step <= self.deadline:
                return rw.on_time_delivery + rw.full_order_bonus
            else:
                late = max(0, (self.finish_step or sim_step) - self.deadline)
                return rw.partial_order_bonus + rw.late_penalty_per_step * late
        elif self.status == OrderStatus.FAILED:
            return rw.late_penalty_per_step * self.deadline
        return 0.0


class OrderManager:
    """
    Manages EPC creation for each order and tracks
    the order lifecycle.
    """

    def __init__(self, factory_cfg: FactoryConfig):
        self.factory_cfg = factory_cfg
        self.epc_factory = EPCFactory(factory_cfg.company_prefix)
        self.orders:     Dict[str, Order]         = {}
        self.epc_to_order: Dict[str, str]         = {}   # epc_uri → order_id
        self.epc_to_item:  Dict[str, str]         = {}   # epc_uri → item_reference
        self.all_epcs:    List[EPC]               = []

    def initialize_orders(self) -> List[EPC]:
        """
        Creates EPCs for all products in all orders.
        Returns the full list of generated EPC objects.
        """
        self.epc_factory.reset()
        self.orders.clear()
        self.epc_to_order.clear()
        self.epc_to_item.clear()
        self.all_epcs.clear()

        for order_cfg in self.factory_cfg.orders:
            order = Order(config=order_cfg)
            self.orders[order_cfg.order_id] = order

            for ps in order_cfg.products:
                for _ in range(ps.quantity):
                    epc = self.epc_factory.create(ps.item_reference)
                    self.all_epcs.append(epc)
                    self.epc_to_order[epc.pure_identity_uri] = order_cfg.order_id
                    self.epc_to_item[epc.pure_identity_uri]  = ps.item_reference

        return list(self.all_epcs)

    def register_product_dispatched(self, epc_uri: str, step: int) -> Optional[float]:
        """
        Registers that a product has been dispatched.
        Returns the order reward if the order is completed.
        """
        order_id = self.epc_to_order.get(epc_uri)
        if not order_id:
            return None
        order = self.orders[order_id]
        order.register_dispatch(epc_uri, step)
        if order.is_complete:
            return order.compute_reward(self.factory_cfg.reward_weights, step)
        return None

    def check_deadlines(self, sim_step: int) -> float:
        """
        Checks overdue orders. Returns the accumulated penalty.
        """
        penalty = 0.0
        rw = self.factory_cfg.reward_weights
        for order in self.orders.values():
            if order.status == OrderStatus.PENDING or order.status == OrderStatus.IN_PROGRESS:
                if sim_step > order.deadline:
                    order.mark_failed(sim_step)
                    penalty += rw.late_penalty_per_step * max(1, sim_step - order.deadline)
        return penalty

    def get_summary(self) -> dict:
        return {
            oid: {
                "status":       o.status.value,
                "dispatched":   o.products_dispatched,
                "needed":       o.total_products_needed,
                "deadline":     o.deadline,
                "finish_step":  o.finish_step,
                "priority":     o.priority,
                "product_type": o.config.products[0].item_reference if o.config.products else "?",
            }
            for oid, o in self.orders.items()
        }

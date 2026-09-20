from __future__ import annotations

import unittest

from logistics_env.is_platform import GlobalPolicy, ISPlatform, PolicyMode, PolicyParameters
from logistics_env.is_platform.is_platform import NegotiationOutcome, NegotiationProposal


def rejected_proposal() -> NegotiationProposal:
    return NegotiationProposal(
        agent_id="product-1",
        state_from="A",
        state_to="B",
        reward=-10.0,
        beliefs={"cost": -1.0, "delay": -1.0, "qos": 0.4, "energy": -0.5},
        order_id="urgent-order",
        step=3,
    )


class NegotiationTests(unittest.TestCase):
    def test_static_policy_is_never_changed_or_returned_as_applied(self) -> None:
        initial = PolicyParameters(A=1.0, B=1.0, C=1.0, D=1.0)
        policy = GlobalPolicy(PolicyMode.STATIC, initial)
        platform = ISPlatform(policy)
        platform.configure_expert(min_reward=0.0)

        result = platform.evaluate(rejected_proposal(), step=3)

        self.assertEqual(result.outcome, NegotiationOutcome.REJECTED)
        self.assertIsNone(result.new_policy)
        self.assertEqual(policy.params, initial)
        self.assertEqual(len(policy.history_summary()), 1)

    def test_dynamic_policy_applies_recommendation(self) -> None:
        initial = PolicyParameters(A=1.0, B=1.0, C=1.0, D=1.0)
        policy = GlobalPolicy(PolicyMode.DYNAMIC, initial)
        platform = ISPlatform(policy)
        platform.configure_expert(min_reward=0.0)

        result = platform.evaluate(rejected_proposal(), step=3)

        self.assertEqual(result.outcome, NegotiationOutcome.REJECTED)
        self.assertIsNotNone(result.new_policy)
        self.assertEqual(result.new_policy, policy.params)
        self.assertEqual(len(policy.history_summary()), 2)

    def test_default_client_priority_affects_qos_threshold(self) -> None:
        policy = GlobalPolicy(PolicyMode.STATIC, PolicyParameters())
        platform = ISPlatform(policy)
        platform.configure_crm(min_qos=0.5, client_priority=1.0)
        proposal = NegotiationProposal(
            agent_id="product-1", state_from="A", state_to="B", reward=1.0,
            beliefs={"qos": 0.65}, order_id="unconfigured-order",
        )

        approved, _ = platform.crm_agent.evaluate_proposal(proposal, policy, 0)
        self.assertFalse(approved)  # required QoS is 0.5 + 1.0 * 0.2


if __name__ == "__main__":
    unittest.main()

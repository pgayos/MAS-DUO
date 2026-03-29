"""
MAS-DUO · IS Platform (Information System Platform)
====================================================
Second platform of the MAS-DUO architecture (Section 3.3 and 3.5).

The IS Platform acts as the interface between the Physical Platform and
business information systems (ERP, CRM, WMS, expert system).

Contains:
  · GlobalPolicy      — global policy parameters (A, B, C, D)
  · ERPAgent          — agent interfacing with the ERP
  · CRMAgent          — agent interfacing with the CRM (client priorities)
  · ExpertSystemAgent — expert system / strategic benchmarking
  · ISPlatform        — IS platform coordinator
"""

from logistics_env.is_platform.global_policy import GlobalPolicy, PolicyMode, PolicyParameters
from logistics_env.is_platform.is_platform   import (
    ISPlatform,
    ERPAgent,
    CRMAgent,
    ExpertSystemAgent,
    NegotiationProposal,
    NegotiationResult,
)

__all__ = [
    "GlobalPolicy",
    "PolicyMode",
    "PolicyParameters",
    "ISPlatform",
    "ERPAgent",
    "CRMAgent",
    "ExpertSystemAgent",
    "NegotiationProposal",
    "NegotiationResult",
]

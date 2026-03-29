"""
EPC - Electronic Product Code
==============================
Implementation of the GS1 EPC / SGTIN-96 standard.

SGTIN structure:
  urn:epc:id:sgtin:<company_prefix>.<item_reference>.<serial>

Reference: GS1 EPC Tag Data Standard v2.0 (https://www.gs1.org/epc-tds)

The EPC forms the "What" part of the product-agent state:
  What  → EPC URI (unique identifier of the physical item)
  Where → position in the logistics grid
  When  → timestamp / simulation step
  Why   → current reason or action (e.g. "in_transit", "processing", "dispatched")
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# EPC Header constants (SGTIN-96 filter values)
# ---------------------------------------------------------------------------
class EPCFilter:
    ALL_OTHERS      = 0
    POINT_OF_SALE   = 1
    FULL_CASE       = 2
    RESERVED_3      = 3
    INNER_PACK      = 4
    RESERVED_5      = 5
    UNIT_LOAD       = 6
    COMPONENT       = 7


@dataclass
class EPC:
    """
    Electronic Product Code following the GS1 SGTIN-96 scheme.

    Attributes
    ----------
    company_prefix : str
        GS1 company prefix (6–12 digits).
    item_reference : str
        Item reference / object class.
    serial : str
        Unique serial number of the physical item.
    filter_value : int
        SGTIN filter value (see EPCFilter).
    """
    company_prefix: str
    item_reference: str
    serial: str
    filter_value: int = EPCFilter.ALL_OTHERS

    # -----------------------------------------------------------------------
    # Propiedades derivadas
    # -----------------------------------------------------------------------

    @property
    def pure_identity_uri(self) -> str:
        """urn:epc:id:sgtin:<company>.<item>.<serial>"""
        return f"urn:epc:id:sgtin:{self.company_prefix}.{self.item_reference}.{self.serial}"

    @property
    def tag_uri(self) -> str:
        """urn:epc:tag:sgtin-96:<filter>.<company>.<item>.<serial>"""
        return (
            f"urn:epc:tag:sgtin-96:{self.filter_value}."
            f"{self.company_prefix}.{self.item_reference}.{self.serial}"
        )

    @property
    def object_class(self) -> str:
        """Object class = company_prefix + item_reference (GTIN without serial)."""
        return f"{self.company_prefix}.{self.item_reference}"

    @property
    def short_id(self) -> str:
        """Short human-readable identifier for logs and rendering."""
        return f"EPC-{self.item_reference}-{self.serial}"

    # -----------------------------------------------------------------------
    # Serialisation / parsing
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "pure_identity_uri": self.pure_identity_uri,
            "tag_uri": self.tag_uri,
            "company_prefix": self.company_prefix,
            "item_reference": self.item_reference,
            "serial": self.serial,
            "filter_value": self.filter_value,
        }

    @classmethod
    def from_uri(cls, uri: str) -> "EPC":
        """
        Parsea un URI EPC puro o de tag:
          urn:epc:id:sgtin:<company>.<item>.<serial>
          urn:epc:tag:sgtin-96:<filter>.<company>.<item>.<serial>
        """
        pattern_id  = r"urn:epc:id:sgtin:(\S+)\.(\S+)\.(\S+)"
        pattern_tag = r"urn:epc:tag:sgtin-96:(\d+)\.(\S+)\.(\S+)\.(\S+)"

        m = re.match(pattern_tag, uri)
        if m:
            return cls(
                filter_value=int(m.group(1)),
                company_prefix=m.group(2),
                item_reference=m.group(3),
                serial=m.group(4),
            )
        m = re.match(pattern_id, uri)
        if m:
            return cls(
                company_prefix=m.group(1),
                item_reference=m.group(2),
                serial=m.group(3),
            )
        raise ValueError(f"Unrecognised EPC URI: {uri!r}")

    def __str__(self) -> str:
        return self.pure_identity_uri

    def __repr__(self) -> str:
        return f"EPC(short_id={self.short_id!r})"


# ---------------------------------------------------------------------------
# EPC factory with auto-incrementing serial
# ---------------------------------------------------------------------------
class EPCFactory:
    """Generates unique EPCs for a given company and item class."""

    def __init__(self, company_prefix: str):
        self.company_prefix = company_prefix
        self._counters: dict[str, int] = {}

    def create(self, item_reference: str, filter_value: int = EPCFilter.ALL_OTHERS) -> EPC:
        count = self._counters.get(item_reference, 0) + 1
        self._counters[item_reference] = count
        serial = str(count).zfill(10)
        return EPC(
            company_prefix=self.company_prefix,
            item_reference=item_reference,
            serial=serial,
            filter_value=filter_value,
        )

    def reset(self) -> None:
        self._counters.clear()

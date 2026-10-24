"""Status vocabulary shared by every analysis section."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

PASS, WARN, FAIL, UNKNOWN = "PASS", "WARN", "FAIL", "UNKNOWN"
_RANK = {PASS: 0, UNKNOWN: 1, WARN: 2, FAIL: 3}


@dataclass
class Check:
    name: str
    status: str
    detail: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def worst(statuses, *, empty: str = UNKNOWN) -> str:
    """Worst status. UNKNOWN outranks PASS: an unchecked item cannot make a section pass."""
    statuses = list(statuses)
    return max(statuses, key=_RANK.__getitem__) if statuses else empty


def section_status(checks: list[Check]) -> str:
    """FAIL/WARN dominate; a section with only PASS and UNKNOWN items is WARN-free but not a clean PASS
    unless every item was actually checked."""
    return worst(c.status for c in checks)

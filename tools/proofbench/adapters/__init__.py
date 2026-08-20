"""Reusable domain semantics for the proof-session core."""
from typing import Any

from .additive_basis import AdditiveBasisAdapter


def adapter_for_specification(specification: dict[str, Any]):
    """Select a registered runtime adapter from its structural input shape."""
    if {"n", "forbidden", "max_selected"}.issubset(specification):
        return AdditiveBasisAdapter(specification)
    return None

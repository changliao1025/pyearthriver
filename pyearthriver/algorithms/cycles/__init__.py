"""
Cycle detection and loop removal algorithms for river networks.
"""

from .cycle_resolver import remove_cycles_from_graph

__all__ = [
    "remove_cycles_from_graph",
]

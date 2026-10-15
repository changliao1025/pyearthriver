"""
Braided channel detection and resolution algorithms for river networks.
"""

from .braided_detector import (
    find_braided_channels,
    find_parallel_paths,
    find_partial_parallel_sections,
    find_reconvergence_points,
    find_first_common_vertex,
    get_reachable_vertices,
)
from .braided_resolver import (
    remove_braided_river,
    remove_parallel_river,
)

__all__ = [
    "find_braided_channels",
    "find_parallel_paths",
    "find_partial_parallel_sections",
    "find_reconvergence_points",
    "find_first_common_vertex",
    "get_reachable_vertices",
    "remove_braided_river",
    "remove_parallel_river",
]

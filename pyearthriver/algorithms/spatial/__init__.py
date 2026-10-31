"""
Spatial algorithms and spatial indexing dispatchers for pyearthriver.
"""

from .indexing import (
    CYTHON_ACCELERATED,
    add_unique_vertex,
    find_vertex_in_list,
    find_vertex_on_edge,
)

__all__ = [
    "find_vertex_in_list",
    "find_vertex_on_edge",
    "add_unique_vertex",
    "CYTHON_ACCELERATED",
]

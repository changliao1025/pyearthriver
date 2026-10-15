"""
Unified spatial indexing and query dispatcher.

Safely imports Cython-accelerated spatial routines if compiled,
or falls back to pure-Python implementations in auxiliary/.
"""

try:
    from pyearthriver.algorithms.cython.kernel import (
        find_vertex_in_list,
        find_vertex_on_edge,
        add_unique_vertex,
    )
    CYTHON_ACCELERATED = True
except (ImportError, ModuleNotFoundError):
    from pyearthriver.algorithms.auxiliary.find_vertex_in_list import find_vertex_in_list
    from pyearthriver.algorithms.auxiliary.find_index_in_list import (
        find_vertex_on_edge,
        add_unique_vertex,
    )
    CYTHON_ACCELERATED = False

__all__ = [
    "find_vertex_in_list",
    "find_vertex_on_edge",
    "add_unique_vertex",
    "CYTHON_ACCELERATED",
]

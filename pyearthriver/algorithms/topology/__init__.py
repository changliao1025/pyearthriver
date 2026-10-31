"""
Topology analysis, pruning, segmentation, and connectivity algorithms.
"""

from pyearthriver.algorithms.topology.connectivity import (
    define_stream_segment,
    define_stream_topology,
    find_all_paths,
    find_nearest_outlet,
    find_outlet_vertices,
    get_downstream_indices,
    get_sinks,
    get_sources,
    get_upstream_indices,
    path_to_flowlines,
    sort_flowlines_from_outlet,
    split_graph,
)
from pyearthriver.algorithms.topology.pruning import (
    find_outlet_reachable_vertices,
    remove_disconnected_flowlines,
    remove_duplicate_flowlines,
    remove_duplicate_flowlines_direction_insensitive,
    remove_duplicate_flowlines_direction_sensitive,
    remove_small_river,
    remove_small_river_step,
)
from pyearthriver.algorithms.topology.segmentation import (
    build_bidirectional_segment,
    find_linear_segments,
    merge_flowline,
    merge_flowline_segment,
    split_flowline,
)

__all__ = [
    # Pruning
    "find_outlet_reachable_vertices",
    "remove_disconnected_flowlines",
    "remove_duplicate_flowlines",
    "remove_duplicate_flowlines_direction_sensitive",
    "remove_duplicate_flowlines_direction_insensitive",
    "remove_small_river",
    "remove_small_river_step",
    # Segmentation
    "build_bidirectional_segment",
    "find_linear_segments",
    "merge_flowline_segment",
    "merge_flowline",
    "split_flowline",
    # Connectivity
    "get_sources",
    "get_sinks",
    "find_outlet_vertices",
    "find_nearest_outlet",
    "split_graph",
    "sort_flowlines_from_outlet",
    "define_stream_segment",
    "define_stream_topology",
    "find_all_paths",
    "path_to_flowlines",
    "get_upstream_indices",
    "get_downstream_indices",
]

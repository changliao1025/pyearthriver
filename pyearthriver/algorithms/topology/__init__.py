"""
Topology analysis, pruning, segmentation, and connectivity algorithms.
"""

from pyearthriver.algorithms.topology.pruning import (
    find_outlet_reachable_vertices,
    remove_disconnected_flowlines,
    remove_duplicate_flowlines,
    remove_duplicate_flowlines_direction_sensitive,
    remove_duplicate_flowlines_direction_insensitive,
    remove_small_river,
    remove_small_river_step,
)

from pyearthriver.algorithms.topology.segmentation import (
    build_bidirectional_segment,
    find_linear_segments,
    merge_flowline_segment,
    merge_flowline,
    split_flowline,
)

from pyearthriver.algorithms.topology.connectivity import (
    get_sources,
    get_sinks,
    find_outlet_vertices,
    split_graph,
    sort_flowlines_from_outlet,
    define_stream_segment,
    define_stream_topology,
    find_all_paths,
    path_to_flowlines,
    get_upstream_indices,
    get_downstream_indices,
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
    "split_graph",
    "sort_flowlines_from_outlet",
    "define_stream_segment",
    "define_stream_topology",
    "find_all_paths",
    "path_to_flowlines",
    "get_upstream_indices",
    "get_downstream_indices",
]

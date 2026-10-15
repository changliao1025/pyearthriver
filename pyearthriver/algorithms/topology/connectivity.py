"""
Graph connectivity, topological sorting, and stream segment algorithms.
"""

import logging
from collections import deque
from typing import List, Set, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pyearthriver.classes.rivergraph import pyrivergraph
    from pyearthriver.classes.flowline import pyflowline
    from pyearthriver.classes.vertex import pyvertex

logger = logging.getLogger(__name__)


def get_sources(graph: "pyrivergraph") -> List[int]:
    """
    Get source nodes (headwaters) with no incoming edges.

    Args:
        graph (pyrivergraph): The river graph instance.

    Returns:
        List[int]: List of source vertex IDs.
    """
    return [
        node_id
        for node_id in graph.id_to_vertex.keys()
        if graph.in_degree[node_id] == 0
    ]


def get_sinks(graph: "pyrivergraph") -> List[int]:
    """
    Get sink nodes (outlets) with no outgoing edges.

    Args:
        graph (pyrivergraph): The river graph instance.

    Returns:
        List[int]: List of sink vertex IDs.
    """
    return [
        node_id
        for node_id in graph.id_to_vertex.keys()
        if graph.out_degree[node_id] == 0
    ]


def find_outlet_vertices(graph: "pyrivergraph") -> List["pyvertex"]:
    """
    Detect outlet vertices: one per independent river network.

    Args:
        graph (pyrivergraph): The river graph instance.

    Returns:
        List[pyvertex]: One outlet vertex per detected river network.
    """
    graph.aVertex_outlet = [
        graph.id_to_vertex[iVertex] for iVertex in get_sinks(graph)
    ]
    logger.info(
        f"Detected {len(graph.aVertex_outlet)} outlet vertices (river networks)"
    )
    return graph.aVertex_outlet


def split_graph(graph: "pyrivergraph") -> List["pyrivergraph"]:
    """
    Split the graph into independent river networks, one per outlet vertex.

    Args:
        graph (pyrivergraph): The river graph instance.

    Returns:
        List[pyrivergraph]: One river graph per independent network.
    """
    from pyearthriver.algorithms.topology.pruning import (
        find_outlet_reachable_vertices,
    )

    if not graph.aVertex_outlet:
        find_outlet_vertices(graph)

    if len(graph.aVertex_outlet) <= 1:
        return [graph]

    graph_cls = type(graph)
    aRivergraph: List["pyrivergraph"] = []
    aFlowline_assigned: Set[int] = set()

    for pVertex_outlet in graph.aVertex_outlet:
        iVertex_outlet = graph.vertex_to_id.get(pVertex_outlet)
        if iVertex_outlet is None:
            logger.warning("Outlet vertex not found in graph, skipping")
            continue

        aVertex_reachable = find_outlet_reachable_vertices(graph, iVertex_outlet)

        aFlowline_sub = []
        for iFlowline, (iVertex_start, iVertex_end) in graph.aFlowline_edges.items():
            if iFlowline in aFlowline_assigned:
                continue
            if (
                iVertex_start in aVertex_reachable
                and iVertex_end in aVertex_reachable
            ):
                aFlowline_sub.append(graph.aFlowline[iFlowline])
                aFlowline_assigned.add(iFlowline)

        if len(aFlowline_sub) == 0:
            logger.warning("Outlet vertex has no connected flowlines, skipping")
            continue

        logger.info(
            f"Split network draining to outlet vertex {iVertex_outlet}: "
            f"{len(aFlowline_sub)} flowlines"
        )
        aRivergraph.append(graph_cls(aFlowline_sub, pVertex_outlet))

    return aRivergraph


def sort_flowlines_from_outlet(graph: "pyrivergraph") -> List["pyflowline"]:
    """
    Sort flowlines using BFS traversal from outlet vertex upstream.

    Args:
        graph (pyrivergraph): The river graph instance.

    Returns:
        List[pyflowline]: Flowlines sorted from outlet to headwaters.
    """
    visited_flowlines = set()
    sorted_flowlines = []
    queue = deque()

    outlet_id = graph.pVertex_outlet_id

    if outlet_id is None or outlet_id not in graph.id_to_vertex:
        logger.warning("Outlet vertex ID is invalid, attempting to refresh")
        if hasattr(graph, "pVertex_outlet") and graph.pVertex_outlet is not None:
            graph._set_outlet_vertex_id()
            outlet_id = graph.pVertex_outlet_id

        if outlet_id is None or outlet_id not in graph.id_to_vertex:
            logger.error(
                "Cannot find valid outlet vertex after refresh, returning original flowline order"
            )
            return graph.aFlowline

    outlet_flowlines = []
    for flowline_idx, (start_id, end_id) in graph.aFlowline_edges.items():
        if end_id == outlet_id:
            outlet_flowlines.append(flowline_idx)

    if not outlet_flowlines:
        logger.warning("No flowlines found ending at outlet vertex")
        return graph.aFlowline

    for flowline_idx in outlet_flowlines:
        queue.append(flowline_idx)

    while queue:
        flowline_idx = queue.popleft()

        if flowline_idx in visited_flowlines:
            continue

        visited_flowlines.add(flowline_idx)

        if flowline_idx < len(graph.aFlowline):
            sorted_flowlines.append(graph.aFlowline[flowline_idx])

        start_id, _ = graph.aFlowline_edges[flowline_idx]

        upstream_flowlines = []
        for upstream_idx, (up_start, up_end) in graph.aFlowline_edges.items():
            if up_end == start_id and upstream_idx not in visited_flowlines:
                upstream_flowlines.append(upstream_idx)

        for upstream_idx in upstream_flowlines:
            if upstream_idx not in queue:
                queue.append(upstream_idx)

    excluded_count = len(graph.aFlowline) - len(sorted_flowlines)
    if excluded_count > 0:
        logger.info(
            f"Excluded {excluded_count} disconnected flowlines that do not flow to outlet"
        )
        for i, flowline in enumerate(graph.aFlowline):
            if i not in visited_flowlines:
                logger.debug(
                    f"Excluded disconnected flowline {i} (ID: {getattr(flowline, 'lFlowlineID', 'unknown')})"
                )

    logger.info(
        f"Sorted {len(sorted_flowlines)} connected flowlines using BFS from outlet"
    )
    return sorted_flowlines


def define_stream_segment(
    graph: "pyrivergraph",
) -> Tuple[List["pyflowline"], List[int]]:
    """
    Define stream segments using topological sorting from outlet to headwater.

    Args:
        graph (pyrivergraph): The river graph instance.

    Returns:
        Tuple[List[pyflowline], List[int]]: Tuple of (sorted_flowlines, segment_indices).
    """
    nFlowline = len(graph.aFlowline)

    if nFlowline == 0:
        logger.warning("Data incomplete: empty flowlines")
        return [], []

    try:
        sorted_flowlines = sort_flowlines_from_outlet(graph)

        nFlowline = len(sorted_flowlines)
        for i, flowline in enumerate(sorted_flowlines, start=1):
            flowline.iStream_segment = nFlowline - i + 1

        graph.aFlowline = sorted_flowlines
        graph._update_graph_flowlines(sorted_flowlines)

        aStream_segment = [
            flowline.iStream_segment for flowline in sorted_flowlines
        ]

        logger.info(
            f"Topologically sorted {len(sorted_flowlines)} flowlines from outlet to headwater"
        )
        return sorted_flowlines, aStream_segment

    except Exception as e:
        logger.error(
            f"Error in topological sorting: {e}, using simple enumeration"
        )
        for i, pFlowline in enumerate(graph.aFlowline, start=1):
            pFlowline.iStream_segment = nFlowline - i + 1
        aStream_segment = [
            pFlowline.iStream_segment for pFlowline in graph.aFlowline
        ]
        return graph.aFlowline, aStream_segment


def define_stream_topology(graph: "pyrivergraph") -> List["pyflowline"]:
    """
    Define stream connectivity: sets upstream and downstream references on each flowline.

    Args:
        graph (pyrivergraph): The river graph instance.

    Returns:
        List[pyflowline]: The flowlines with connectivity populated.
    """
    aFlowline = graph.aFlowline
    try:
        for flowline in aFlowline:
            upstream_flowlines = [
                graph.aFlowline[i] for i in get_upstream_indices(graph, flowline)
            ]
            downstream_flowlines = [
                graph.aFlowline[i] for i in get_downstream_indices(graph, flowline)
            ]
            flowline.aFlowline_upstream = upstream_flowlines
            flowline.aFlowline_downstream = downstream_flowlines
    except Exception as e:
        logger.error(f"Error defining stream topology: {e}")

    return aFlowline


def find_all_paths(
    graph: "pyrivergraph", start_id: int, target_id: int, max_depth: int = 10
) -> List[List[int]]:
    """
    Find all directed paths from start to target vertex using DFS.

    Args:
        graph (pyrivergraph): The river graph instance.
        start_id (int): Starting vertex ID.
        target_id (int): Target vertex ID.
        max_depth (int, optional): Maximum search depth. Defaults to 10.

    Returns:
        List[List[int]]: List of paths, where each path is a list of vertex IDs.
    """
    paths = []

    def dfs_paths(
        current_id: int,
        target_id: int,
        path: List[int],
        visited: Set[int],
        depth: int,
    ):
        if depth > max_depth:
            return

        if current_id == target_id:
            paths.append(path.copy())
            return

        if current_id in visited:
            return

        visited.add(current_id)

        for neighbor_id, _ in graph.adjacency_list[current_id]:
            if neighbor_id not in visited:
                path.append(neighbor_id)
                dfs_paths(neighbor_id, target_id, path, visited, depth + 1)
                path.pop()

        visited.remove(current_id)

    if start_id != target_id:
        dfs_paths(start_id, target_id, [start_id], set(), 0)

    return paths


def path_to_flowlines(graph: "pyrivergraph", path: List[int]) -> List[int]:
    """
    Convert a path of vertex IDs to flowline indices.

    Args:
        graph (pyrivergraph): The river graph instance.
        path (List[int]): List of vertex IDs.

    Returns:
        List[int]: List of flowline indices.
    """
    flowline_indices = []

    for i in range(len(path) - 1):
        start_id = path[i]
        end_id = path[i + 1]

        for neighbor_id, flowline_idx in graph.adjacency_list[start_id]:
            if neighbor_id == end_id:
                flowline_indices.append(flowline_idx)
                break

    return flowline_indices


def get_upstream_indices(
    graph: "pyrivergraph", flowline: "pyflowline"
) -> List[int]:
    """
    Get indices of upstream flowlines for a given flowline.

    Args:
        graph (pyrivergraph): The river graph instance.
        flowline (pyflowline): Flowline object.

    Returns:
        List[int]: List of upstream flowline indices.
    """
    upstream_indices = []
    start_vertex_id = graph.vertex_to_id.get(flowline.pVertex_start)
    if start_vertex_id is not None:
        for start_id, neighbors in graph.adjacency_list.items():
            for end_id, flowline_idx in neighbors:
                if end_id == start_vertex_id and flowline_idx < len(
                    graph.aFlowline
                ):
                    upstream_indices.append(flowline_idx)
    return upstream_indices


def get_downstream_indices(
    graph: "pyrivergraph", flowline: "pyflowline"
) -> List[int]:
    """
    Get indices of downstream flowlines for a given flowline.

    Args:
        graph (pyrivergraph): The river graph instance.
        flowline (pyflowline): Flowline object.

    Returns:
        List[int]: List of downstream flowline indices.
    """
    downstream_indices = []
    end_vertex_id = graph.vertex_to_id.get(flowline.pVertex_end)
    if end_vertex_id is not None:
        for neighbor_id, flowline_idx in graph.adjacency_list[end_vertex_id]:
            if flowline_idx < len(graph.aFlowline):
                downstream_indices.append(flowline_idx)
    return downstream_indices

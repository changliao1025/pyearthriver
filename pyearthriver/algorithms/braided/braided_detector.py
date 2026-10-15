"""
Detection algorithms for braided channels, parallel paths, and reconvergence points.
"""

import logging
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional, DefaultDict, TYPE_CHECKING

if TYPE_CHECKING:
    from pyearthriver.classes.rivergraph import pyrivergraph
    from pyearthriver.classes.flowline import pyflowline

logger = logging.getLogger(__name__)


def find_braided_channels(graph: "pyrivergraph") -> List[List["pyflowline"]]:
    """
    Find braided channels (multiple flowlines between the same vertex pair).

    Args:
        graph (pyrivergraph): The river network graph.

    Returns:
        List[List[pyflowline]]: List of braided regions, where each region is
        a list of flowlines connecting the same (start, end) vertex pair.
    """
    channel_groups: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)

    for flowline_idx, (start_id, end_id) in graph.aFlowline_edges.items():
        channel_groups[(start_id, end_id)].append(flowline_idx)

    braided_groups = []
    for (start_id, end_id), indices in channel_groups.items():
        if len(indices) > 1:
            region_flowlines = [graph.aFlowline[idx] for idx in indices]
            braided_groups.append(region_flowlines)

    return braided_groups


def find_parallel_paths(graph: "pyrivergraph") -> List[Dict]:
    """
    Find divergent parallel sections between vertices.

    Args:
        graph (pyrivergraph): The river network graph.

    Returns:
        List[Dict]: List of parallel path group dictionaries containing 'paths',
        'start_vertex', and 'end_vertex'.
    """
    return find_partial_parallel_sections(graph)


def find_partial_parallel_sections(graph: "pyrivergraph") -> List[Dict]:
    """
    Find partial parallel sections within longer paths in the graph.

    Args:
        graph (pyrivergraph): The river network graph.

    Returns:
        List[Dict]: Identified parallel path groups.
    """
    parallel_sections = []

    divergence_vertices = []
    for vertex_id in graph.adjacency_list.keys():
        if graph.out_degree[vertex_id] > 1:
            divergence_vertices.append(vertex_id)

    logger.debug(f"Found {len(divergence_vertices)} potential divergence vertices")

    for divergence_id in divergence_vertices:
        reconvergence_id = find_reconvergence_points(graph, divergence_id)

        if reconvergence_id is not None:
            divergent_paths = graph._find_all_paths(
                divergence_id, reconvergence_id, max_depth=8
            )

            if len(divergent_paths) > 1:
                path_flowline_groups = []
                for path in divergent_paths:
                    path_flowlines = graph._path_to_flowlines(path)
                    if path_flowlines and len(path_flowlines) > 0:
                        path_flowline_groups.append(path_flowlines)

                if len(path_flowline_groups) > 1:
                    unique_paths = []
                    for path_flowlines in path_flowline_groups:
                        path_signature = tuple(sorted(path_flowlines))
                        if path_signature not in [
                            tuple(sorted(up)) for up in unique_paths
                        ]:
                            unique_paths.append(path_flowlines)

                    if len(unique_paths) > 1:
                        logger.debug(
                            f"Found {len(unique_paths)} partial parallel paths between vertices {divergence_id} -> {reconvergence_id}"
                        )
                        parallel_sections.append(
                            {
                                "paths": unique_paths,
                                "start_vertex": divergence_id,
                                "end_vertex": reconvergence_id,
                            }
                        )

    return parallel_sections


def find_reconvergence_points(
    graph: "pyrivergraph", divergence_id: int, max_depth: int = 8
) -> Optional[int]:
    """
    Find vertices where paths from a divergence point reconverge.

    Args:
        graph (pyrivergraph): The river network graph.
        divergence_id (int): The vertex ID where paths diverge.
        max_depth (int): Maximum search depth.

    Returns:
        Optional[int]: The first common reconvergence vertex ID, or None.
    """
    downstream_vertices = [
        neighbor_id for neighbor_id, _ in graph.adjacency_list[divergence_id]
    ]

    if len(downstream_vertices) < 2:
        return None

    common_vertex = None
    for i in range(len(downstream_vertices)):
        for j in range(i + 1, len(downstream_vertices)):
            path1_start = downstream_vertices[i]
            path2_start = downstream_vertices[j]

            path1_reachable = get_reachable_vertices(graph, path1_start, max_depth)
            path2_reachable = get_reachable_vertices(graph, path2_start, max_depth)

            common_vertex = find_first_common_vertex(path1_reachable, path2_reachable)
            if common_vertex is not None:
                return common_vertex

    return common_vertex


def find_first_common_vertex(
    reachable_list1: List[int], reachable_list2: List[int]
) -> Optional[int]:
    """
    Find the first vertex that appears in both lists, respecting topological order.
    """
    set2 = set(reachable_list2)
    for vertex in reachable_list1:
        if vertex in set2:
            return vertex
    return None


def get_reachable_vertices(
    graph: "pyrivergraph", start_id: int, max_depth: int
) -> List[int]:
    """
    Get all vertices reachable from a starting vertex within max_depth.
    """
    reachable = []
    queue = deque([(start_id, 0)])
    visited = {start_id}

    while queue:
        current_id, depth = queue.popleft()

        if depth >= max_depth:
            continue

        for neighbor_id, _ in graph.adjacency_list[current_id]:
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                reachable.append(neighbor_id)
                queue.append((neighbor_id, depth + 1))

    return reachable

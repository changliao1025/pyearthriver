"""
Topology pruning and filtering algorithms for river networks.
"""

import logging
import os
from collections import deque, defaultdict
from typing import List, Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pyearthriver.classes.rivergraph import pyrivergraph
    from pyearthriver.classes.flowline import pyflowline
    from pyearthriver.classes.vertex import pyvertex

logger = logging.getLogger(__name__)


def find_outlet_reachable_vertices(
    graph: "pyrivergraph", outlet_vertex_id: int
) -> Set[int]:
    """
    Find all vertices that can reach the outlet using backward traversal.

    Args:
        graph (pyrivergraph): The river graph instance.
        outlet_vertex_id (int): ID of the outlet vertex.

    Returns:
        Set[int]: Set of vertex IDs that can reach the outlet.
    """
    reachable = set()
    queue = deque([outlet_vertex_id])
    reachable.add(outlet_vertex_id)

    # Build reverse adjacency list for backward traversal
    reverse_adjacency = defaultdict(list)
    for start_id, neighbors in graph.adjacency_list.items():
        for end_id, flowline_idx in neighbors:
            reverse_adjacency[end_id].append((start_id, flowline_idx))

    # Backward BFS from outlet
    while queue:
        current_id = queue.popleft()

        # Add all vertices that flow into current vertex
        for upstream_id, _ in reverse_adjacency[current_id]:
            if upstream_id not in reachable:
                reachable.add(upstream_id)
                queue.append(upstream_id)

    return reachable


def remove_disconnected_flowlines(
    graph: "pyrivergraph", pVertex_outlet: Optional["pyvertex"] = None
) -> List["pyflowline"]:
    """
    Remove flowlines that don't flow out to the specified outlet vertex.

    Performs a backward traversal from the outlet vertex to identify
    all flowlines connected to the drainage network. Flowlines that
    cannot reach the outlet (isolated components, disconnected segments) are removed.

    Args:
        graph (pyrivergraph): The river graph instance.
        pVertex_outlet (Optional[pyvertex]): Optional outlet vertex. Defaults to graph.pVertex_outlet.

    Returns:
        List[pyflowline]: List of flowlines connected to the outlet vertex.
    """
    flowlines = graph.aFlowline

    pVertex_target = (
        pVertex_outlet if pVertex_outlet is not None else graph.pVertex_outlet
    )

    if pVertex_target is None:
        logger.warning(
            "No outlet vertex provided and none stored during initialization"
        )
        return flowlines

    # Use stored outlet vertex ID if it matches, otherwise find it
    outlet_vertex_id = None
    if pVertex_target == graph.pVertex_outlet and graph.pVertex_outlet_id is not None:
        outlet_vertex_id = graph.pVertex_outlet_id
    else:
        for vertex_id, vertex in graph.id_to_vertex.items():
            if vertex == pVertex_target:
                outlet_vertex_id = vertex_id
                break

    if outlet_vertex_id is None:
        logger.warning("Outlet vertex not found in network graph")
        return flowlines

    logger.info(
        f"Removing disconnected flowlines using outlet vertex {outlet_vertex_id}"
    )

    outlet_reachable_vertices = find_outlet_reachable_vertices(
        graph, outlet_vertex_id
    )

    reachable_flowlines = set()
    for flowline_idx, (start_id, end_id) in graph.aFlowline_edges.items():
        if (
            start_id in outlet_reachable_vertices
            and end_id in outlet_reachable_vertices
        ):
            reachable_flowlines.add(flowline_idx)

    connected_flowlines = []
    disconnected_count = 0

    for i, flowline in enumerate(flowlines):
        if i in reachable_flowlines:
            connected_flowlines.append(flowline)
        else:
            disconnected_count += 1
            logger.debug(
                f"Removing disconnected flowline {i}: {flowline.pVertex_start} -> {flowline.pVertex_end}"
            )

    logger.info(
        f"Removed {disconnected_count} disconnected flowlines, kept {len(connected_flowlines)} connected flowlines"
    )

    graph._update_graph_flowlines(connected_flowlines)
    return connected_flowlines


def remove_duplicate_flowlines_direction_sensitive(
    graph: "pyrivergraph",
) -> List["pyflowline"]:
    """
    Remove duplicate flowlines where direction matters (A→B != B→A).

    Args:
        graph (pyrivergraph): The river graph instance.

    Returns:
        List[pyflowline]: Unique flowlines.
    """
    seen_edges = set()
    unique_flowlines = []
    duplicates_count = 0

    for idx, flowline in enumerate(graph.aFlowline):
        start_id = graph.vertex_to_id.get(flowline.pVertex_start)
        end_id = graph.vertex_to_id.get(flowline.pVertex_end)

        if start_id is None or end_id is None:
            logger.warning(
                f"Flowline {idx} has unknown vertices, skipping duplicate check"
            )
            unique_flowlines.append(flowline)
            continue

        edge_key = (start_id, end_id)

        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            unique_flowlines.append(flowline)
        else:
            duplicates_count += 1
            logger.debug(
                f"Removing duplicate flowline {idx} (direction-sensitive): "
                f"{flowline.pVertex_start} -> {flowline.pVertex_end}"
            )

    logger.info(
        f"Removed {duplicates_count} duplicate flowlines (direction-sensitive), "
        f"resulting in {len(unique_flowlines)} unique flowlines"
    )

    graph._update_graph_flowlines(unique_flowlines)
    return unique_flowlines


def remove_duplicate_flowlines_direction_insensitive(
    graph: "pyrivergraph",
) -> List["pyflowline"]:
    """
    Remove duplicate flowlines where direction doesn't matter.
    If two flowlines are in opposite directions (A→B and B→A), resolves by path to outlet.

    Args:
        graph (pyrivergraph): The river graph instance.

    Returns:
        List[pyflowline]: Unique flowlines.
    """
    edge_groups = defaultdict(list)

    for idx, flowline in enumerate(graph.aFlowline):
        start_id = graph.vertex_to_id.get(flowline.pVertex_start)
        end_id = graph.vertex_to_id.get(flowline.pVertex_end)

        if start_id is None or end_id is None:
            logger.warning(
                f"Flowline {idx} has unknown vertices, skipping duplicate check"
            )
            continue

        edge_key = tuple(sorted([start_id, end_id]))
        edge_groups[edge_key].append((idx, flowline, (start_id, end_id)))

    unique_flowlines = []
    duplicates_count = 0
    opposite_pairs_count = 0

    for edge_key, flowline_group in edge_groups.items():
        if len(flowline_group) == 1:
            _, flowline, _ = flowline_group[0]
            unique_flowlines.append(flowline)
        else:
            directions = set()
            for idx, flowline, (start_id, end_id) in flowline_group:
                directions.add((start_id, end_id))

            has_opposite_directions = False
            direction_list = list(directions)
            for i in range(len(direction_list)):
                for j in range(i + 1, len(direction_list)):
                    dir1 = direction_list[i]
                    dir2 = direction_list[j]
                    if dir1 == (dir2[1], dir2[0]):
                        has_opposite_directions = True
                        break
                if has_opposite_directions:
                    break

            if has_opposite_directions:
                if graph.pVertex_outlet_id is not None:
                    flowline_path_lengths = []
                    for idx, flowline, (start_id, end_id) in flowline_group:
                        paths_to_outlet = graph._find_all_paths(
                            end_id, graph.pVertex_outlet_id, max_depth=1000
                        )
                        if paths_to_outlet:
                            shortest_path_length = min(
                                len(path) for path in paths_to_outlet
                            )
                            flowline_path_lengths.append(
                                (
                                    shortest_path_length,
                                    idx,
                                    flowline,
                                    (start_id, end_id),
                                )
                            )
                        else:
                            flowline_path_lengths.append(
                                (float("inf"), idx, flowline, (start_id, end_id))
                            )

                    flowline_path_lengths.sort(key=lambda x: x[0])
                    shortest_path_length, closest_idx, closest_flowline, _ = (
                        flowline_path_lengths[0]
                    )
                    unique_flowlines.append(closest_flowline)

                    for (
                        path_length,
                        idx,
                        flowline,
                        (start_id, end_id),
                    ) in flowline_path_lengths[1:]:
                        opposite_pairs_count += 1
                        logger.debug(
                            f"Removing flowline {idx} (longer path to outlet, path_length={path_length}): "
                            f"{flowline.pVertex_start} -> {flowline.pVertex_end}"
                        )
                else:
                    opposite_pairs_count += len(flowline_group)
                    for idx, flowline, (start_id, end_id) in flowline_group:
                        logger.debug(
                            f"Removing flowline {idx} (opposite direction pair, no outlet): "
                            f"{flowline.pVertex_start} -> {flowline.pVertex_end}"
                        )
            else:
                _, first_flowline, _ = flowline_group[0]
                unique_flowlines.append(first_flowline)
                duplicates_count += len(flowline_group) - 1
                for idx, flowline, _ in flowline_group[1:]:
                    logger.debug(
                        f"Removing duplicate flowline {idx} (same direction): "
                        f"{flowline.pVertex_start} -> {flowline.pVertex_end}"
                    )

    for idx, flowline in enumerate(graph.aFlowline):
        start_id = graph.vertex_to_id.get(flowline.pVertex_start)
        end_id = graph.vertex_to_id.get(flowline.pVertex_end)
        if start_id is None or end_id is None:
            unique_flowlines.append(flowline)

    total_removed = duplicates_count + opposite_pairs_count
    logger.info(
        f"Removed {total_removed} flowlines (direction-insensitive): "
        f"{duplicates_count} same-direction duplicates, "
        f"{opposite_pairs_count} opposite-direction pairs, "
        f"resulting in {len(unique_flowlines)} unique flowlines"
    )

    graph._update_graph_flowlines(unique_flowlines)
    return unique_flowlines


def remove_duplicate_flowlines(
    graph: "pyrivergraph", iFlag_direction_insensitive: bool = False
) -> List["pyflowline"]:
    """
    Remove duplicate flowlines from the network.

    Args:
        graph (pyrivergraph): The river graph instance.
        iFlag_direction_insensitive (bool): If True, treats opposite directions as duplicates.

    Returns:
        List[pyflowline]: Unique flowlines.
    """
    if iFlag_direction_insensitive:
        return remove_duplicate_flowlines_direction_insensitive(graph)
    else:
        return remove_duplicate_flowlines_direction_sensitive(graph)


def remove_small_river_step(
    graph: "pyrivergraph", flowlines: List["pyflowline"], threshold: float
) -> List["pyflowline"]:
    """
    Remove small rivers based on length threshold (first-order streams shorter than threshold).

    Args:
        graph (pyrivergraph): The river graph instance.
        flowlines (List[pyflowline]): Flowlines to filter.
        threshold (float): Length threshold.

    Returns:
        List[pyflowline]: Filtered flowlines.
    """
    filtered_flowlines = []

    for flowline in flowlines:
        length = getattr(flowline, "dLength", 0.0)
        stream_order = getattr(flowline, "iStream_order", None)
        if stream_order != 1 or length >= threshold:
            filtered_flowlines.append(flowline)
        else:
            logger.debug(f"Removing small first-order river with length {length}")

    logger.info(f"Removed {len(flowlines) - len(filtered_flowlines)} small rivers")
    return filtered_flowlines


def remove_small_river(
    graph: "pyrivergraph",
    dThreshold_small_river: float,
    nIterations: int = 3,
    iFlag_debug: int = 0,
    sWorkspace_output_basin: Optional[str] = None,
) -> List["pyflowline"]:
    """
    Remove small rivers iteratively using graph-based approach with graph updates at each step.

    Args:
        graph (pyrivergraph): The river graph instance.
        dThreshold_small_river (float): Length threshold for small river removal.
        nIterations (int, optional): Number of iterations. Defaults to 3.
        iFlag_debug (int, optional): Debug flag for output files. Defaults to 0.
        sWorkspace_output_basin (str, optional): Output workspace for debug files. Defaults to None.

    Returns:
        List[pyflowline]: Updated flowlines after iterative small river removal.
    """
    from pyearthriver.formats import export_flowline_to_vector

    aFlowline_current = graph.aFlowline.copy()

    for i in range(nIterations):
        sStep = "{:02d}".format(i + 1)
        logger.info(
            f"Iteration {sStep}: removing small rivers with threshold {dThreshold_small_river}"
        )

        aFlowline_filtered = remove_small_river_step(
            graph, aFlowline_current, dThreshold_small_river
        )

        if iFlag_debug == 1 and sWorkspace_output_basin is not None:
            sFilename_out = f"flowline_large_{sStep}_before_intersect.geojson"
            sFilename_out = os.path.join(sWorkspace_output_basin, sFilename_out)
            export_flowline_to_vector(aFlowline_filtered, sFilename_out)

        if len(aFlowline_filtered) == len(aFlowline_current):
            logger.info(f"No small rivers found in iteration {sStep}, stopping early.")
            break
        else:
            graph._update_graph_flowlines(aFlowline_filtered)
            aFlowline_filtered = graph.update_headwater_stream_order()
            aFlowline_merged = graph.merge_flowline()

            if iFlag_debug == 1 and sWorkspace_output_basin is not None:
                sFilename_out = f"flowline_merge_{sStep}_before_intersect.geojson"
                sFilename_out = os.path.join(sWorkspace_output_basin, sFilename_out)
                export_flowline_to_vector(aFlowline_merged, sFilename_out)

            aFlowline_current = graph.update_headwater_stream_order()

            if len(aFlowline_current) == 1:
                break

    return aFlowline_current

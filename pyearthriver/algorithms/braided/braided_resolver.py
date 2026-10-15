"""
Algorithms for pruning and resolving braided channels and parallel flowlines.
"""

import logging
from typing import List, TYPE_CHECKING
from .braided_detector import find_braided_channels, find_parallel_paths

if TYPE_CHECKING:
    from pyearthriver.classes.rivergraph import pyrivergraph
    from pyearthriver.classes.flowline import pyflowline

logger = logging.getLogger(__name__)


def remove_braided_river(graph: "pyrivergraph") -> List["pyflowline"]:
    """
    Remove braided channels from a river network graph.

    For each set of flowlines connecting the exact same start and end vertices,
    retains the shortest flowline and removes the secondary redundant branches.

    Args:
        graph (pyrivergraph): The river network graph.

    Returns:
        List[pyflowline]: Flowlines with braided channels removed.
    """
    braided_regions = find_braided_channels(graph)
    if not braided_regions:
        logger.info("No braided channels found in the network")
        return graph.aFlowline

    logger.info(f"Found {len(braided_regions)} braided regions to resolve")

    flowlines_to_remove = set()

    for region_flowlines in braided_regions:
        if len(region_flowlines) <= 1:
            continue

        shortest_flowline = min(
            region_flowlines, key=lambda fl: getattr(fl, "dLength", float("inf"))
        )

        for flowline in region_flowlines:
            if flowline != shortest_flowline:
                flowline_idx = graph.aFlowline.index(flowline)
                flowlines_to_remove.add(flowline_idx)
                logger.debug(
                    f"Marking flowline {flowline_idx} for removal (braided channel, length={getattr(flowline, 'dLength', 'unknown')})"
                )

    simplified_flowlines = [
        flowline
        for idx, flowline in enumerate(graph.aFlowline)
        if idx not in flowlines_to_remove
    ]

    logger.info(
        f"Removed {len(flowlines_to_remove)} braided flowlines, "
        f"resulting in {len(simplified_flowlines)} flowlines"
    )

    graph._update_graph_flowlines(simplified_flowlines)
    graph.merge_flowline()

    return simplified_flowlines


def remove_parallel_river(graph: "pyrivergraph") -> List["pyflowline"]:
    """
    Remove parallel rivers using graph-based path analysis.

    Args:
        graph (pyrivergraph): The river network graph.

    Returns:
        List[pyflowline]: Flowlines with parallel paths resolved.
    """
    if not hasattr(graph, "aFlowline") or not graph.aFlowline:
        logger.warning(
            "No flowlines available in class instance for parallel river removal"
        )
        return []

    if len(graph.aFlowline) <= 1:
        logger.debug("Skipping parallel path resolution: insufficient flowlines")
        return graph.aFlowline.copy()

    logger.info(
        f"Removing parallel rivers from {len(graph.aFlowline)} flowlines using graph-based approach"
    )

    try:
        parallel_groups = find_parallel_paths(graph)

        if not parallel_groups:
            logger.debug("No parallel paths found")
            return graph.aFlowline.copy()

        logger.info(f"Found {len(parallel_groups)} parallel path groups")
    except Exception as e:
        logger.error(f"Error finding parallel paths: {e}")
        return graph.aFlowline.copy()

    flowlines_to_remove = set()

    for group in parallel_groups:
        paths = group["paths"]

        if len(paths) <= 1:
            continue

        path_scores = []
        for path_idx, path_flowlines in enumerate(paths):
            path_length = 0.0
            valid_flowlines = 0

            for flowline_idx in path_flowlines:
                if flowline_idx >= len(graph.aFlowline):
                    logger.warning(
                        f"Invalid flowline index {flowline_idx}, skipping"
                    )
                    continue

                flowline = graph.aFlowline[flowline_idx]
                valid_flowlines += 1

                length = getattr(flowline, "dLength", 0.0)
                length = length if length > 0 else 0.0
                path_length += length

            if valid_flowlines == 0:
                continue

            path_score = path_length if path_length > 0 else 1.0
            path_scores.append((path_score, path_idx, path_flowlines))

        if not path_scores:
            continue

        path_scores.sort(reverse=True)
        for _, path_idx, path_flowlines in path_scores:
            path_flowlines = list(path_flowlines)
            for flowline_idx in path_flowlines[:-1]:
                flowline = graph.aFlowline[flowline_idx]
                end_vertex_id = graph.vertex_to_id.get(flowline.pVertex_end)
                if end_vertex_id is not None and graph.in_degree[end_vertex_id] > 1:
                    flowlines_to_remove.add(flowline_idx)
                    logger.debug(
                        f"Removing flowline {flowline_idx} as its end point is a confluence"
                    )
                    break

    try:
        result = [
            graph.aFlowline[i]
            for i in range(len(graph.aFlowline))
            if i not in flowlines_to_remove
        ]
        logger.info(
            f"Removed {len(flowlines_to_remove)} flowlines to resolve parallel paths"
        )
        if len(result) == 0 and len(graph.aFlowline) > 0:
            logger.warning(
                "All flowlines were removed during parallel path resolution, returning original"
            )
            return graph.aFlowline.copy()

        graph._update_graph_flowlines(result)
        graph.merge_flowline()
        return result
    except Exception as e:
        logger.error(
            f"Error filtering flowlines during parallel path resolution: {e}"
        )
        return graph.aFlowline.copy()

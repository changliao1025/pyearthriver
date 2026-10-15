"""
Topology segmentation and merging algorithms for river networks.
"""

import logging
from typing import List, Optional, Set, TYPE_CHECKING
import numpy as np

from pyearthriver.classes.edge import pyedge
from pyearthriver.classes.flowline import pyflowline
from pyearthriver.algorithms.spatial import find_vertex_on_edge

if TYPE_CHECKING:
    from pyearthriver.classes.rivergraph import pyrivergraph
    from pyearthriver.classes.vertex import pyvertex

logger = logging.getLogger(__name__)


def build_bidirectional_segment(
    graph: "pyrivergraph",
    start_flowline_idx: int,
    start_vertex_id: int,
    end_vertex_id: int,
    merge_vertices: Set[int],
    processed_flowlines: Set[int],
) -> List[int]:
    """
    Build a complete linear segment by extending both upstream and downstream
    from the starting flowline.

    Args:
        graph (pyrivergraph): The river graph instance.
        start_flowline_idx (int): Index of the flowline to start building from.
        start_vertex_id (int): Start vertex ID of the starting flowline.
        end_vertex_id (int): End vertex ID of the starting flowline.
        merge_vertices (Set[int]): Set of degree-2 vertices that can be merged.
        processed_flowlines (Set[int]): Set of already processed flowline indices.

    Returns:
        List[int]: List of flowline indices in upstream-to-downstream order.
    """
    upstream_segment = []
    downstream_segment = [start_flowline_idx]

    # Extend upstream from start vertex
    current_start = start_vertex_id
    while current_start in merge_vertices:
        upstream_flowlines = []
        for fl_idx, (fl_start, fl_end) in graph.aFlowline_edges.items():
            if fl_end == current_start and fl_idx not in processed_flowlines:
                upstream_flowlines.append(fl_idx)

        if len(upstream_flowlines) == 1:
            upstream_flowline_idx = upstream_flowlines[0]
            upstream_segment.insert(0, upstream_flowline_idx)
            current_start, _ = graph.aFlowline_edges[upstream_flowline_idx]
        else:
            break

    # Extend downstream from end vertex
    current_end = end_vertex_id
    while current_end in merge_vertices:
        downstream_flowlines = [
            fl_idx for neighbor_id, fl_idx in graph.adjacency_list[current_end]
        ]

        if len(downstream_flowlines) == 1:
            downstream_flowline_idx = downstream_flowlines[0]
            if downstream_flowline_idx not in processed_flowlines:
                downstream_segment.append(downstream_flowline_idx)
                if downstream_flowline_idx in graph.aFlowline_edges:
                    _, current_end = graph.aFlowline_edges[downstream_flowline_idx]
                else:
                    break
            else:
                break
        else:
            break

    complete_segment = upstream_segment + downstream_segment
    return complete_segment


def find_linear_segments(graph: "pyrivergraph") -> List[List[int]]:
    """
    Find linear segments that can be merged into single flowlines.

    A linear segment is a chain of flowlines where each intermediate vertex
    has exactly one incoming and one outgoing edge (degree 2 vertices).

    Args:
        graph (pyrivergraph): The river graph instance.

    Returns:
        List[List[int]]: List of linear segments (flowline indices ordered upstream to downstream).
    """
    linear_segments = []
    processed_flowlines = set()

    merge_vertices = set()
    for vertex_id in graph.id_to_vertex.keys():
        if graph.in_degree[vertex_id] == 1 and graph.out_degree[vertex_id] == 1:
            merge_vertices.add(vertex_id)

    logger.debug(
        f"Found {len(merge_vertices)} degree-2 vertices for potential merging"
    )

    for flowline_idx, (start_id, end_id) in graph.aFlowline_edges.items():
        if flowline_idx in processed_flowlines:
            continue

        segment = build_bidirectional_segment(
            graph,
            flowline_idx,
            start_id,
            end_id,
            merge_vertices,
            processed_flowlines,
        )

        if len(segment) > 1:
            linear_segments.append(segment)
            processed_flowlines.update(segment)
            logger.debug(
                f"Found linear segment with {len(segment)} flowlines: {segment}"
            )

    logger.info(
        f"Found {len(linear_segments)} linear segments for potential merging"
    )
    return linear_segments


def merge_flowline_segment(
    segment_flowlines: List[pyflowline],
) -> Optional[pyflowline]:
    """
    Merge a chain of flowlines into a single flowline.

    Args:
        segment_flowlines (List[pyflowline]): List of flowlines to merge.

    Returns:
        Optional[pyflowline]: Merged flowline or None if merge failed.
    """
    if not segment_flowlines:
        return None

    try:
        merged_flowline = segment_flowlines[-1]
        for flowline in segment_flowlines[-2::-1]:
            flowline_up = flowline
            merged_flowline = merged_flowline.merge_upstream(flowline_up)

        return merged_flowline
    except Exception as e:
        logger.error(f"Error merging flowline segment: {e}")
        return None


def merge_flowline(graph: "pyrivergraph") -> List[pyflowline]:
    """
    Merge linear segments of flowlines into single flowlines.

    Identifies chains of flowlines where intermediate vertices have degree 2
    and creates new merged flowlines to replace them.

    Args:
        graph (pyrivergraph): The river graph instance.

    Returns:
        List[pyflowline]: List of flowlines with linear segments merged.
    """
    flowlines = graph.aFlowline

    try:
        linear_segments = find_linear_segments(graph)

        if not linear_segments:
            logger.info("No linear segments found for merging")
            return graph.aFlowline

        logger.info(f"Found {len(linear_segments)} linear segments to merge")

        merged_flowlines = []
        processed_indices = set()

        for segment in linear_segments:
            if len(segment) < 2:
                continue

            try:
                segment_flowlines = [
                    flowlines[idx] for idx in segment if idx < len(flowlines)
                ]

                if len(segment_flowlines) < 2:
                    continue

                merged_flowline = merge_flowline_segment(segment_flowlines)
                if merged_flowline is not None:
                    merged_flowlines.append(merged_flowline)
                    processed_indices.update(segment)
                    logger.debug(
                        f"Merged {len(segment_flowlines)} flowlines into single flowline"
                    )

            except Exception as e:
                logger.error(f"Error merging segment {segment}: {e}")
                continue

        for i, flowline in enumerate(flowlines):
            if i not in processed_indices:
                merged_flowlines.append(flowline)

        logger.info(
            f"Merged {len(linear_segments)} segments, "
            f"resulting in {len(merged_flowlines)} flowlines "
            f"(reduced from {len(flowlines)})"
        )

        graph._update_graph_flowlines(merged_flowlines)
        return merged_flowlines

    except Exception as e:
        logger.error(f"Error in merge_flowline: {e}")
        return graph.aFlowline


def split_flowline(
    graph: "pyrivergraph",
    aVertex_in: Optional[List["pyvertex"]] = None,
    iFlag_intersect: Optional[int] = None,
    iFlag_use_id: Optional[int] = None,
) -> List[pyflowline]:
    """
    Split flowlines based on intersection with a list of vertices.

    Args:
        graph (pyrivergraph): The river graph instance.
        aVertex_in (Optional[List[pyvertex]]): List of vertices to split on. Defaults to graph.aVertex.
        iFlag_intersect (Optional[int]): 1 if vertices may lie along edges without being edge endpoints.
        iFlag_use_id (Optional[int]): 1 if matching using flowline IDs.

    Returns:
        List[pyflowline]: Split flowlines.
    """
    iFlag_graph_update = 0
    aFlowline_in = graph.aFlowline
    aFlowline_out = list()
    nFlowline = len(aFlowline_in)

    if aVertex_in is None:
        aVertex_in = graph.aVertex

    aVertex_in_set = set(aVertex_in)

    for i in range(nFlowline):
        pFlowline = aFlowline_in[i]
        iStream_order = pFlowline.iStream_order
        iFlag_dam = pFlowline.iFlag_dam
        nEdge = pFlowline.nEdge
        iPart = 0
        aVertex = list()
        aVertex_all = list()

        for j in range(nEdge):
            pEdge = pFlowline.aEdge[j]
            pVertex = pEdge.pVertex_start
            aVertex_all.append(pVertex)

            if aVertex_in and pVertex in aVertex_in_set:
                iPart += 1
                aVertex.append(pVertex)

            if iFlag_intersect is not None:
                if iFlag_use_id is not None:
                    aDistance = list()
                    iFlag_exist = 0
                    aVertex_dummy = list()
                    aIndex = list()
                    npoint = 0
                    for k in range(len(aVertex_in)) if aVertex_in else []:
                        pVertex0 = aVertex_in[k]
                        if pVertex0.lFlowlineID == pFlowline.lFlowlineID:
                            iFlag_exist = 1
                            distance = pEdge.pVertex_start.calculate_distance(
                                pVertex0
                            )
                            if distance == 0:
                                continue
                            iPart = iPart + 1
                            aDistance.append(distance)
                            aVertex_dummy.append(pVertex0)
                            aIndex.append(k)
                            npoint = npoint + 1

                    if aDistance:
                        aIndex_order = np.argsort(aDistance)
                        for k in aIndex_order:
                            pVertex_dummy = aVertex_in[aIndex[k]]
                            aVertex.append(pVertex_dummy)
                            aVertex_all.append(pVertex_dummy)
                else:
                    iFlag_exist, npoint, aIndex = (
                        find_vertex_on_edge(aVertex_in, pEdge)
                        if aVertex_in
                        else (0, 0, [])
                    )
                    if iFlag_exist == 1:
                        for m in range(npoint):
                            pVertex_dummy = aVertex_in[aIndex[m]]
                            iPart = iPart + 1
                            aVertex.append(pVertex_dummy)
                            aVertex_all.append(pVertex_dummy)

        pVertex = pFlowline.pVertex_end
        aVertex_all.append(pVertex)

        if aVertex_in and pVertex in aVertex_in_set:
            iPart = iPart + 1
            aVertex.append(pVertex)

        if iPart == 0:
            logger.warning("No matching vertices found during split")
        else:
            if iPart >= 2:
                nLine = iPart - 1
                aVertex_index = [
                    aVertex_all.index(pVertex)
                    for pVertex in aVertex
                    if pVertex in aVertex_all
                ]

                for k in range(nLine):
                    t = aVertex_index[k]
                    s = aVertex_index[k + 1]
                    if s != t:
                        aEdge = [
                            pyedge.create(aVertex_all[l], aVertex_all[l + 1])
                            for l in range(t, s)
                        ]
                        aEdge = [edge for edge in aEdge if edge is not None]
                        pFlowline1 = pyflowline(aEdge)
                        pFlowline1.iStream_order = iStream_order
                        pFlowline1.iFlag_dam = iFlag_dam
                        aFlowline_out.append(pFlowline1)
                        nEdge_new = pFlowline1.nEdge
                        if nEdge_new != nEdge:
                            iFlag_graph_update = 1

    if iFlag_graph_update == 1:
        graph._update_graph_flowlines(aFlowline_out)
    return aFlowline_out

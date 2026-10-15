"""
Stream ordering algorithms including Strahler and Shreve methodologies.
"""

import logging
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pyearthriver.classes.rivergraph import pyrivergraph
    from pyearthriver.classes.flowline import pyflowline

logger = logging.getLogger(__name__)


def identify_headwater_flowlines(graph: "pyrivergraph") -> List["pyflowline"]:
    """
    Identify headwater flowlines (sources) in the network.

    Args:
        graph (pyrivergraph): River network graph.

    Returns:
        List[pyflowline]: Flowlines with no upstream connections.
    """
    aFlowline_headwater = []
    for flowline in graph.aFlowline:
        start_vertex_id = graph.vertex_to_id.get(flowline.pVertex_start)
        if start_vertex_id is not None and graph.in_degree[start_vertex_id] == 0:
            aFlowline_headwater.append(flowline)

    logger.debug(f"Identified {len(aFlowline_headwater)} headwater flowlines")
    graph.aFlowline_headwater = aFlowline_headwater
    return aFlowline_headwater


def update_headwater_stream_order(graph: "pyrivergraph") -> List["pyflowline"]:
    """
    Set stream order to 1 for all headwater flowlines.

    Args:
        graph (pyrivergraph): River network graph.

    Returns:
        List[pyflowline]: Updated flowlines list.
    """
    if not hasattr(graph, "aFlowline") or not graph.aFlowline:
        logger.warning(
            "No flowlines available in class instance for stream order update"
        )
        return []

    logger.info(
        f"Updating stream order for {len(graph.aFlowline)} flowlines using graph-based approach"
    )

    try:
        aFlowline_headwater = identify_headwater_flowlines(graph)
        if not aFlowline_headwater:
            logger.warning(
                "No headwater flowlines identified for stream order update"
            )
            return graph.aFlowline.copy()
        else:
            for pFlowline in aFlowline_headwater:
                pFlowline.iStream_order = 1

        return graph.aFlowline
    except Exception as e:
        logger.error(f"Error updating stream orders: {e}")
        return graph.aFlowline.copy()


def process_confluences_iteratively(graph: "pyrivergraph", iFlag_so_method_in: int) -> None:
    """
    Process confluences iteratively to assign downstream stream orders.

    Args:
        graph (pyrivergraph): River network graph.
        iFlag_so_method_in (int): 1 for Strahler, 2 for Shreve.
    """
    if not hasattr(graph, "aConfluence") or not graph.aConfluence:
        return

    unresolved_confluences = set(range(len(graph.aConfluence)))
    iteration = 0

    while unresolved_confluences:
        iteration += 1
        logger.debug(
            f"Stream order iteration {iteration}, "
            f"{len(unresolved_confluences)} confluences remaining"
        )

        resolved_this_iteration = set()

        for idx in unresolved_confluences:
            confluence = graph.aConfluence[idx]
            upstream_orders = [
                fl.iStream_order
                for fl in confluence.aFlowline_upstream
                if fl.iStream_order > 0
            ]

            if len(upstream_orders) == len(confluence.aFlowline_upstream):
                if iFlag_so_method_in == 1:  # Strahler
                    if upstream_orders.count(max(upstream_orders)) > 1:
                        downstream_order = max(upstream_orders) + 1
                    else:
                        downstream_order = max(upstream_orders)
                elif iFlag_so_method_in == 2:  # Shreve
                    downstream_order = sum(upstream_orders)
                else:
                    logger.warning(
                        f"Unknown stream order method: {iFlag_so_method_in}"
                    )
                    continue

                for fl in confluence.aFlowline_downstream:
                    fl.iStream_order = downstream_order

                resolved_this_iteration.add(idx)

        if not resolved_this_iteration:
            logger.warning(
                "No confluences resolved in this iteration, stopping to avoid infinite loop"
            )
            break

        unresolved_confluences -= resolved_this_iteration


def define_stream_order(
    graph: "pyrivergraph", iFlag_so_method_in: int = 1
) -> Tuple[Optional[List["pyflowline"]], Optional[List[int]]]:
    """
    Define stream order for all flowlines using graph-based approach with confluence analysis.

    Args:
        graph (pyrivergraph): River network graph.
        iFlag_so_method_in (int): 1=Strahler, 2=Shreve.

    Returns:
        Tuple[List[pyflowline], List[int]]: Flowlines with updated stream orders and order values list.
    """
    if not hasattr(graph, "aFlowline") or not graph.aFlowline:
        logger.warning("No flowlines available for stream order calculation")
        return None, None

    logger.info(
        f"Defining stream order for {len(graph.aFlowline)} flowlines using "
        f"{'Strahler' if iFlag_so_method_in == 1 else 'Shreve'} method"
    )

    try:
        identify_headwater_flowlines(graph)
        update_headwater_stream_order(graph)
        process_confluences_iteratively(graph, iFlag_so_method_in)

        aStream_order = [flowline.iStream_order for flowline in graph.aFlowline]
        return graph.aFlowline, aStream_order
    except Exception as e:
        logger.error(f"Error defining stream order: {e}")
        return None, None

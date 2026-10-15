"""
Pipeline and high-level workflow functions for river network conditioning.

Provides convenient entry points for running end-to-end river network cleaning,
braided channel pruning, cycle detection, and stream ordering.
"""

import logging
from typing import List, Optional, Tuple
from pyearthriver.classes.vertex import pyvertex
from pyearthriver.classes.flowline import pyflowline
from pyearthriver.classes.rivergraph import pyrivergraph

logger = logging.getLogger(__name__)


def condition_river_network(
    flowlines: List[pyflowline],
    outlet_vertex: Optional[pyvertex] = None,
    remove_duplicates: bool = True,
    remove_disconnected: bool = True,
    remove_braids: bool = True,
    remove_parallels: bool = True,
    remove_cycles: bool = True,
    define_topology: bool = True,
    compute_stream_order: bool = True,
    stream_order_method: int = 1,
) -> Tuple[pyrivergraph, List[pyflowline]]:
    """
    Run an end-to-end topological conditioning pipeline on river flowlines.

    This function performs graph-based cleaning and simplification to transform
    raw river flowline networks into a clean Directed Acyclic Graph (DAG) suitable
    for hydrological modeling and routing.

    Pipeline steps performed (configurable):
        1. Initialize pyrivergraph.
        2. Remove duplicate flowline geometries.
        3. Remove disconnected flowlines (if outlet is specified).
        4. Detect and prune secondary braided river channels.
        5. Remove parallel river channels.
        6. Break loops and topological cycles.
        7. Define stream segments, confluences, and topology.
        8. Compute Strahler stream order.

    Args:
        flowlines (List[pyflowline]): Raw flowline objects.
        outlet_vertex (Optional[pyvertex]): Basin outlet vertex. If None,
            the outlet is automatically inferred from sink vertices.
        remove_duplicates (bool): Remove duplicate flowlines (default: True).
        remove_disconnected (bool): Remove flowlines disconnected from the outlet (default: True).
        remove_braids (bool): Prune secondary braided channels (default: True).
        remove_parallels (bool): Remove parallel flowline reaches (default: True).
        remove_cycles (bool): Break cycles/loops in the network (default: True).
        define_topology (bool): Define stream segments, topology, and confluences (default: True).
        compute_stream_order (bool): Calculate Strahler stream order (default: True).
        stream_order_method (int): Stream order algorithm method (default: 1).

    Returns:
        Tuple[pyrivergraph, List[pyflowline]]: A tuple containing:
            - graph (pyrivergraph): The conditioned graph object.
            - flowlines (List[pyflowline]): The conditioned flowlines.
    """
    if not flowlines:
        raise ValueError("Cannot condition empty flowlines list")

    logger.info("Initializing pyrivergraph with %d flowlines", len(flowlines))
    graph = pyrivergraph(flowlines, outlet_vertex)

    if remove_duplicates:
        logger.info("Removing duplicate flowlines")
        graph.remove_duplicate_flowlines()

    if remove_disconnected and (outlet_vertex is not None or graph.pVertex_outlet is not None):
        logger.info("Removing disconnected flowlines")
        graph.remove_disconnected_flowlines()

    if remove_braids:
        logger.info("Removing braided river channels")
        graph.remove_braided_river()

    if remove_parallels:
        logger.info("Removing parallel river channels")
        graph.remove_parallel_river()

    if remove_cycles:
        logger.info("Detecting and removing cycles")
        graph.remove_cycle()

    if define_topology:
        logger.info("Defining stream segments, confluences, and topology")
        graph.define_stream_segment()
        graph.define_river_confluence()
        graph.define_stream_topology()

    if compute_stream_order:
        logger.info("Computing stream order (method=%d)", stream_order_method)
        graph.define_stream_order(iFlag_so_method_in=stream_order_method)

    return graph, graph.aFlowline


def compute_strahler_stream_order(
    flowlines: List[pyflowline],
    outlet_vertex: Optional[pyvertex] = None,
    method: int = 1,
) -> List[pyflowline]:
    """
    Compute Strahler stream orders for a collection of flowlines.

    Args:
        flowlines (List[pyflowline]): List of flowlines.
        outlet_vertex (Optional[pyvertex]): Outlet vertex if known.
        method (int): Stream order algorithm method (default: 1).

    Returns:
        List[pyflowline]: Flowlines with assigned iStream_order attributes.
    """
    graph = pyrivergraph(flowlines, outlet_vertex)
    graph.define_stream_segment()
    graph.define_river_confluence()
    graph.define_stream_topology()
    res = graph.define_stream_order(iFlag_so_method_in=method)
    if isinstance(res, tuple):
        return res[0]
    return graph.aFlowline


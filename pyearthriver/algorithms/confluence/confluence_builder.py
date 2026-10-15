"""
River confluence construction and identification algorithms.
"""

import logging
from typing import List, Optional, TYPE_CHECKING
from pyearthriver.classes.confluence import pyconfluence
from pyearthriver.classes.vertex import pyvertex

if TYPE_CHECKING:
    from pyearthriver.classes.rivergraph import pyrivergraph

logger = logging.getLogger(__name__)


def define_river_confluence(graph: "pyrivergraph") -> List[pyconfluence]:
    """
    Build confluences in the river network based on in-degree > 1.

    Args:
        graph (pyrivergraph): River network graph.

    Returns:
        List[pyconfluence]: List of constructed confluence objects.
    """
    aConfluence = []
    try:
        for vertex_id, vertex in graph.id_to_vertex.items():
            if graph.in_degree[vertex_id] > 1:
                pConfluence = create_confluence_object(graph, vertex_id, vertex)
                if pConfluence:
                    aConfluence.append(pConfluence)
    except Exception as e:
        logger.error(f"Error defining river confluences: {e}")
        return []

    graph.aConfluence = aConfluence
    return aConfluence


def create_confluence_object(
    graph: "pyrivergraph", vertex_id: int, vertex: pyvertex
) -> Optional[pyconfluence]:
    """
    Create a confluence object for a given vertex with in-degree > 1.

    Args:
        graph (pyrivergraph): River network graph.
        vertex_id (int): Vertex identifier.
        vertex (pyvertex): Vertex point object.

    Returns:
        Optional[pyconfluence]: Constructed pyconfluence object, or None if failed.
    """
    try:
        aFlowline_upstream = []
        for start_id, neighbors in graph.adjacency_list.items():
            for end_id, flowline_idx in neighbors:
                if end_id == vertex_id and flowline_idx < len(graph.aFlowline):
                    aFlowline_upstream.append(graph.aFlowline[flowline_idx])

        aFlowline_downstream = []
        for end_id, flowline_idx in graph.adjacency_list[vertex_id]:
            if flowline_idx < len(graph.aFlowline):
                aFlowline_downstream.append(graph.aFlowline[flowline_idx])

        confluence = pyconfluence(vertex, aFlowline_upstream, aFlowline_downstream)
        confluence.lVertexID = vertex_id
        return confluence
    except Exception as e:
        logger.error(f"Error creating confluence object: {e}")
        return None

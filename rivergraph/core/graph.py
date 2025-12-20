"""
Core graph data structure for river network representation.

This module provides the fundamental graph structure without high-level operations.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional, DefaultDict
from collections import defaultdict, deque

from ..classes.vertex import pyvertex
from ..classes.flowline import pyflowline

logger = logging.getLogger(__name__)


class RiverGraph:
    """
    Core graph data structure for river networks.

    This class manages the fundamental graph representation without high-level
    operations like simplification or analysis. It provides:
    - Vertex and edge ID management
    - Adjacency list maintenance
    - Degree tracking (in/out)
    - Basic graph queries (sources, sinks, vertex lookup)
    """

    def __init__(self, flowlines: List[pyflowline], pVertex_outlet: Optional[pyvertex] = None):
        """
        Initialize the river network graph from flowlines.

        Args:
            flowlines: List of flowline objects representing the river network
            pVertex_outlet: Optional outlet vertex for the drainage network
        """
        self.aFlowline = flowlines
        self.pVertex_outlet = pVertex_outlet
        self.pVertex_outlet_id: Optional[int] = None

        # Vertex mappings
        self.vertex_to_id: Dict[pyvertex, int] = {}
        self.id_to_vertex: Dict[int, pyvertex] = {}
        self.aVertex: List[pyvertex] = []

        # Edge mappings
        self.aFlowline_edges: Dict[int, Tuple[int, int]] = {}

        # Graph structure
        self.adjacency_list: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.in_degree: DefaultDict[int, int] = defaultdict(int)
        self.out_degree: DefaultDict[int, int] = defaultdict(int)

        # Confluence data
        self.aVertex_confluence: List[pyvertex] = []
        self.aConfluence = []

        logger.debug("Initializing RiverGraph with custom graph implementation")
        self._build_graph()

        if pVertex_outlet is not None:
            self._set_outlet_vertex_id()

    def get_sources(self) -> List[int]:
        """Get source nodes (headwaters) with no incoming edges."""
        return [node_id for node_id in self.id_to_vertex.keys() if self.in_degree[node_id] == 0]

    def get_sinks(self) -> List[int]:
        """Get sink nodes (outlets) with no outgoing edges."""
        return [node_id for node_id in self.id_to_vertex.keys() if self.out_degree[node_id] == 0]

    def get_vertices(self) -> List[pyvertex]:
        """
        Extract all unique vertices from the flowline network.

        Returns:
            List[pyvertex]: List of unique vertices with assigned lVertexID values
        """
        logger.debug(f"Extracted {len(self.aVertex)} unique vertices from river graph")
        return self.aVertex.copy()

    def get_vertex_by_id(self, vertex_id: int) -> Optional[pyvertex]:
        """
        Get a vertex by its internal graph ID.

        Args:
            vertex_id: Internal vertex ID (0-based)

        Returns:
            The vertex object, or None if not found
        """
        return self.id_to_vertex.get(vertex_id)

    def get_vertex_id(self, vertex: pyvertex) -> Optional[int]:
        """
        Get the internal graph ID for a vertex.

        Args:
            vertex: The vertex to look up

        Returns:
            Internal vertex ID (0-based), or None if not found
        """
        return self.vertex_to_id.get(vertex)

    def get_vertex_count(self) -> int:
        """
        Get the total number of unique vertices in the network.

        Returns:
            Number of unique vertices
        """
        return len(self.id_to_vertex)

    def _build_graph(self):
        """
        Build the graph structure from flowlines.
        This is the core method that constructs the adjacency list representation.
        """
        # Clear existing graph data
        self.vertex_to_id.clear()
        self.id_to_vertex.clear()
        self.aFlowline_edges.clear()
        self.adjacency_list.clear()
        self.in_degree.clear()
        self.out_degree.clear()
        self.aVertex.clear()

        vertex_counter = 0

        # Process each flowline to build vertex mappings and edges
        for flowline_idx, flowline in enumerate(self.aFlowline):
            if flowline is None:
                continue

            start_vertex = flowline.pVertex_start
            end_vertex = flowline.pVertex_end

            # Add start vertex if not seen before
            if start_vertex not in self.vertex_to_id:
                self.vertex_to_id[start_vertex] = vertex_counter
                self.id_to_vertex[vertex_counter] = start_vertex
                self.aVertex.append(start_vertex)
                start_vertex.lVertexID = vertex_counter
                vertex_counter += 1

            # Add end vertex if not seen before
            if end_vertex not in self.vertex_to_id:
                self.vertex_to_id[end_vertex] = vertex_counter
                self.id_to_vertex[vertex_counter] = end_vertex
                self.aVertex.append(end_vertex)
                end_vertex.lVertexID = vertex_counter
                vertex_counter += 1

            # Get vertex IDs
            start_id = self.vertex_to_id[start_vertex]
            end_id = self.vertex_to_id[end_vertex]

            # Store edge mapping
            self.aFlowline_edges[flowline_idx] = (start_id, end_id)

            # Add to adjacency list
            self.adjacency_list[start_id].append((end_id, flowline_idx))

            # Update degrees
            self.out_degree[start_id] += 1
            self.in_degree[end_id] += 1

        logger.debug(f"Built graph with {len(self.id_to_vertex)} vertices and {len(self.aFlowline_edges)} edges")

    def _set_outlet_vertex_id(self):
        """Set the outlet vertex ID if outlet vertex is provided."""
        if self.pVertex_outlet is not None:
            self.pVertex_outlet_id = self.vertex_to_id.get(self.pVertex_outlet)
            if self.pVertex_outlet_id is not None:
                logger.debug(f"Set outlet vertex ID: {self.pVertex_outlet_id}")
            else:
                logger.warning("Outlet vertex not found in graph")

    def update_graph_flowlines(self, new_flowlines: List[pyflowline]):
        """
        Update the graph with a new set of flowlines.

        Args:
            new_flowlines: New list of flowlines to update the graph with
        """
        self.aFlowline = new_flowlines
        self._build_graph()

        # Update outlet vertex ID after graph rebuild
        if hasattr(self, 'pVertex_outlet') and self.pVertex_outlet is not None:
            self._set_outlet_vertex_id()
            logger.debug(f"Updated outlet vertex ID after graph rebuild: {self.pVertex_outlet_id}")
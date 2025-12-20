"""
Main facade class for river network analysis.

This module provides the pyrivergraph class that maintains backward compatibility
while delegating to specialized modules.
"""

import logging
from typing import List, Optional

from ..classes.flowline import pyflowline
from ..classes.vertex import pyvertex
from ..classes.confluence import pyconfluence
from .graph import RiverGraph
from ..operations.simplification import NetworkSimplifier
from ..operations.modification import NetworkModifier
from ..operations.topology import TopologyManager
from ..analysis.detection import NetworkAnalyzer
from ..analysis.pathfinding import PathFinder

logger = logging.getLogger(__name__)


class pyrivergraph:
    """
    Main facade class for river network analysis.

    Maintains backward compatibility while delegating to specialized modules.
    This class provides the same public API as the original monolithic class.
    """

    def __init__(self, flowlines: List[pyflowline], pVertex_outlet: Optional[pyvertex] = None):
        """
        Initialize the river network graph from flowlines.

        Args:
            flowlines: List of flowline objects representing the river network
            pVertex_outlet: Optional outlet vertex for the drainage network
        """
        # Initialize core graph
        self._graph = RiverGraph(flowlines, pVertex_outlet)

        # Initialize analysis components
        self._analyzer = NetworkAnalyzer(self._graph)
        self._pathfinder = PathFinder(self._graph)

        # Initialize operation components
        self._simplifier = NetworkSimplifier(self._graph, self._analyzer, self._pathfinder)
        self._modifier = NetworkModifier(self._graph, self._analyzer)
        self._topology = TopologyManager(self._graph, self._pathfinder)

        # Expose graph properties for backward compatibility
        self._sync_state()

        # Handle outlet vertex if provided
        if pVertex_outlet is not None:
            self.aFlowline = self._simplifier.remove_disconnected_flowlines()
            self._sync_state()

    def _sync_state(self):
        """Sync state after operations that modify the graph."""
        self.aFlowline = self._graph.aFlowline
        self.aVertex = self._graph.aVertex
        self.pVertex_outlet = self._graph.pVertex_outlet
        self.pVertex_outlet_id = self._graph.pVertex_outlet_id
        self.vertex_to_id = self._graph.vertex_to_id
        self.id_to_vertex = self._graph.id_to_vertex
        self.aFlowline_edges = self._graph.aFlowline_edges
        self.adjacency_list = self._graph.adjacency_list
        self.in_degree = self._graph.in_degree
        self.out_degree = self._graph.out_degree
        self.aVertex_confluence = self._graph.aVertex_confluence
        self.aConfluence = self._graph.aConfluence

    # ========================================================================
    # BASIC GRAPH OPERATIONS
    # ========================================================================

    def get_sources(self) -> List[int]:
        """Get source nodes (headwaters) with no incoming edges."""
        return self._graph.get_sources()

    def get_sinks(self) -> List[int]:
        """Get sink nodes (outlets) with no outgoing edges."""
        return self._graph.get_sinks()

    def get_vertices(self) -> List[pyvertex]:
        """Extract all unique vertices from the flowline network."""
        return self._graph.get_vertices()

    def get_vertex_by_id(self, vertex_id: int) -> Optional[pyvertex]:
        """Get a vertex by its internal graph ID."""
        return self._graph.get_vertex_by_id(vertex_id)

    def get_vertex_id(self, vertex: pyvertex) -> Optional[int]:
        """Get the internal graph ID for a vertex."""
        return self._graph.get_vertex_id(vertex)

    def get_vertex_count(self) -> int:
        """Get the total number of unique vertices in the network."""
        return self._graph.get_vertex_count()

    # ========================================================================
    # NETWORK SIMPLIFICATION OPERATIONS
    # ========================================================================

    def remove_disconnected_flowlines(self, pVertex_outlet: Optional[pyvertex] = None) -> List[pyflowline]:
        """Remove flowlines that don't flow out to the specified outlet vertex."""
        result = self._simplifier.remove_disconnected_flowlines(pVertex_outlet)
        self._sync_state()
        return result

    def remove_braided_river(self) -> List[pyflowline]:
        """Remove braided channels from the river network."""
        result = self._simplifier.remove_braided_river()
        self._sync_state()
        # Also merge after removing braided channels
        result = self._modifier.merge_flowline()
        self._sync_state()
        return result

    def remove_parallel_river(self) -> List[pyflowline]:
        """Remove parallel rivers using graph-based approach."""
        result = self._simplifier.remove_parallel_river()
        self._sync_state()
        # Also merge after removing parallel rivers
        result = self._modifier.merge_flowline()
        self._sync_state()
        return result

    def remove_cycle(self) -> List[pyflowline]:
        """Detect and break cycles in the network."""
        result = self._simplifier.remove_cycle()
        self._sync_state()
        return result

    def remove_small_river(self, dThreshold_small_river, nIterations=3, iFlag_debug=0, sWorkspace_output_basin=None):
        """Remove small rivers iteratively."""
        result = self._simplifier.remove_small_river(dThreshold_small_river, nIterations, iFlag_debug, sWorkspace_output_basin)
        self._sync_state()
        return result

    def remove_duplicate_flowlines(self, iFlag_direction_insensitive: bool = False) -> List[pyflowline]:
        """Remove duplicate flowlines from the network."""
        result = self._simplifier.remove_duplicate_flowlines(iFlag_direction_insensitive)
        self._sync_state()
        return result

    # ========================================================================
    # NETWORK ANALYSIS & DETECTION
    # ========================================================================

    def find_braided_channels(self) -> List[List[pyflowline]]:
        """Find braided channels (multiple flowlines between same vertex pair)."""
        return self._analyzer.find_braided_channels()

    def find_parallel_paths(self) -> List[dict]:
        """Find divergent parallel sections between vertices."""
        return self._analyzer.find_parallel_paths()

    def detect_cycles(self) -> List[List[int]]:
        """Detect cycles in the network using DFS."""
        return self._analyzer.detect_cycles()

    # ========================================================================
    # NETWORK MODIFICATION & PROCESSING
    # ========================================================================

    def split_flowline(self, aVertex_in: Optional[List[pyvertex]] = None,
                        iFlag_intersect = None, iFlag_use_id=None) -> List[pyflowline]:
        """Split flowline based on the intersection with a list of vertex."""
        result = self._modifier.split_flowline(aVertex_in, iFlag_intersect, iFlag_use_id)
        self._sync_state()
        return result

    def merge_flowline(self) -> List[pyflowline]:
        """Merge linear segments of flowlines into single flowlines."""
        result = self._modifier.merge_flowline()
        self._sync_state()
        return result

    # ========================================================================
    # STREAM ANALYSIS & TOPOLOGY
    # ========================================================================

    def update_headwater_stream_order(self) -> List[pyflowline]:
        """Update stream order for head water flowlines."""
        result = self._topology.update_headwater_stream_order()
        self._sync_state()
        return result

    def identify_headwater_flowlines(self) -> List[pyflowline]:
        """Identify headwater flowlines (sources) in the network."""
        result = self._topology.identify_headwater_flowlines()
        self._sync_state()
        return result

    def define_stream_segment(self):
        """Define stream segments using topological sorting."""
        result = self._topology.define_stream_segment()
        self._sync_state()
        return result

    def define_river_confluence(self):
        """Build the confluence using the in_degree."""
        result = self._topology.define_river_confluence()
        self._sync_state()
        return result

    def define_stream_topology(self) -> List[pyflowline]:
        """Define comprehensive stream topology."""
        result = self._topology.define_stream_topology()
        self._sync_state()
        return result

    def define_stream_order(self, iFlag_so_method_in: int = 1):
        """Define stream order for all flowlines."""
        result = self._topology.define_stream_order(iFlag_so_method_in)
        self._sync_state()
        return result

    # ========================================================================
    # PATH FINDING & ANALYSIS
    # ========================================================================

    def get_upstream_indices(self, flowline: pyflowline) -> List[int]:
        """Get indices of upstream flowlines for a given flowline."""
        return self._pathfinder.get_upstream_indices(flowline)

    def get_downstream_indices(self, flowline: pyflowline) -> List[int]:
        """Get indices of downstream flowlines for a given flowline."""
        return self._pathfinder.get_downstream_indices(flowline)

    # ========================================================================
    # PRIVATE METHODS (for backward compatibility)
    # ========================================================================

    def _build_graph(self):
        """Build the graph structure from flowlines."""
        self._graph._build_graph()
        self._sync_state()

    def _set_outlet_vertex_id(self):
        """Set the outlet vertex ID if outlet vertex is provided."""
        self._graph._set_outlet_vertex_id()
        self._sync_state()

    def _update_graph_flowlines(self, new_flowlines: List[pyflowline]):
        """Update the graph with a new set of flowlines."""
        self._graph.update_graph_flowlines(new_flowlines)
        self._sync_state()

    def _find_outlet_reachable_vertices(self, outlet_vertex_id: int):
        """Find all vertices that can reach the outlet."""
        return self._pathfinder.find_outlet_reachable_vertices(outlet_vertex_id)

    def _find_all_paths(self, start_id: int, target_id: int, max_depth: int = 10):
        """Find all paths from start to target vertex."""
        return self._pathfinder.find_all_paths(start_id, target_id, max_depth)

    def _path_to_flowlines(self, path: List[int]) -> List[int]:
        """Convert a path of vertex IDs to flowline indices."""
        return self._pathfinder.path_to_flowlines(path)

    def _find_linear_segments(self):
        """Find linear segments that can be merged."""
        return self._analyzer.find_linear_segments()

    def _create_confluence_object(self, vertex_id: int, vertex: pyvertex):
        """Create a confluence object for a vertex."""
        return self._topology._create_confluence_object(vertex_id, vertex)

    def _process_confluences_iteratively(self, iFlag_so_method_in: int):
        """Process confluences iteratively to update stream orders."""
        self._topology._process_confluences_iteratively(iFlag_so_method_in)
        self._sync_state()

    def _sort_flowlines_from_outlet(self):
        """Core sorting function using BFS traversal from outlet vertex."""
        return self._topology._sort_flowlines_from_outlet()
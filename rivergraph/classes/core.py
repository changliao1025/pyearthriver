
"""
Core pyrivergraph class integrating all modular components.

This module provides the main pyrivergraph class that maintains backward compatibility
with the original monolithic implementation while leveraging the modular architecture
for improved maintainability and performance.
"""

import logging
import os
from typing import List, Dict, Set, Tuple, Optional, DefaultDict
from collections import defaultdict, deque
import numpy as np
from enum import Enum
import time

# Import for spatial indexing (R-tree)
try:
    from rtree.index import Index as RTreeindex
    HAS_RTREE = True
except ImportError:
    HAS_RTREE = False
    logger = logging.getLogger(__name__)
    logger.warning("R-tree not available - using fallback spatial indexing")

HAS_NETWORKX = False

from pyflowline.classes.flowline import pyflowline
from pyflowline.classes.vertex import pyvertex
from pyflowline.classes.edge import pyedge
from pyflowline.classes.confluence import pyconfluence
from pyflowline.formats.export_flowline import export_flowline_to_geojson

# Import split_flowline dependencies
import importlib.util
iFlag_cython = importlib.util.find_spec("cython")
if iFlag_cython is not None:
    from pyflowline.algorithms.cython.kernel import find_vertex_on_edge
else:
    from pyflowline.algorithms.auxiliary.find_index_in_list import find_vertex_on_edge

# Import modular components

from .utils import RiverGraphUtils
from .network_analysis import NetworkAnalyzer
from .stream_analysis import StreamAnalyzer
from .network_operations import NetworkOperations
from .graph_builders import GraphBuilders

# Set up logger
logger = logging.getLogger(__name__)


class pyrivergraph:
    """
    Directed graph representation of a river network for braided channel analysis.

    This class models the flowline network as a directed graph where:
    - Nodes represent vertices (confluence/divergence points)
    - Edges represent flowlines connecting vertices
    - Multiple edges between same nodes represent braided channels

    Uses NetworkX when available for enhanced graph algorithms, falls back to
    custom implementation for compatibility.

    Enhanced with intelligent state management and hybrid update strategies for
    optimal performance with dynamic network modifications.
    """

    # ========================================================================
    # INITIALIZATION & BASIC GRAPH OPERATIONS
    # ========================================================================

    def __init__(self, flowlines: List[pyflowline], outlet_vertex: Optional[pyvertex] = None):
        """
        Initialize the river network graph from flowlines.

        Args:
            flowlines: List of flowline objects representing the river network
            outlet_vertex: Optional outlet vertex for the drainage network. If provided,
                          enables outlet-based operations and optimizations.
        """
        self.aFlowline = flowlines
        self.pVertex_outlet = outlet_vertex
        self.pVertex_outlet_id: Optional[int] = None
        self.vertex_to_id: Dict[pyvertex, int] = {}
        self.id_to_vertex: Dict[int, pyvertex] = {}
        self.aFlowline_edges: Dict[int, Tuple[int, int]] = {}
        self.aVertex: List[pyvertex] = []  # Will be populated during graph building

        # Always use custom implementation
        self.adjacency_list: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.in_degree: DefaultDict[int, int] = defaultdict(int)
        self.out_degree: DefaultDict[int, int] = defaultdict(int)

        # Initialize state management system


        # Initialize modular components
        self.utils = RiverGraphUtils(self)

        self.network_analyzer = NetworkAnalyzer(self)
        self.stream_analyzer = StreamAnalyzer(self)
        self.network_ops = NetworkOperations(self)
        self.graph_builders = GraphBuilders(self)

        logger.debug("Using custom graph implementation with state management")
        self._build_graph()
        # Set outlet vertex ID if outlet was provided
        if self.pVertex_outlet is not None:
            self._set_outlet_vertex_id()

        # Mark initial state as clean
        self.state_manager.clear_pending_changes()

    def get_sources(self) -> List[int]:
        """Get source nodes (headwaters) with no incoming edges."""
        return [node_id for node_id in self.id_to_vertex.keys() if self.in_degree[node_id] == 0]

    def get_sinks(self) -> List[int]:
        """Get sink nodes (outlets) with no outgoing edges."""
        return [node_id for node_id in self.id_to_vertex.keys() if self.out_degree[node_id] == 0]

    def get_vertices(self) -> List[pyvertex]:
        """
        Extract all unique vertices from the flowline network.

        This method provides functionality equivalent to find_flowline_vertex(),
        returning a list of unique vertices with assigned vertex IDs. The vertices
        are returned in the order they were encountered during graph construction.

        Returns:
            List[pyvertex]: List of unique vertices with assigned lVertexID values

        Example:
            >>> river_graph = pyrivergraph(flowlines)
            >>> vertices = river_graph.get_vertices()
            >>> print(f"Found {len(vertices)} unique vertices")
        """
        # Return the stored vertex list which maintains the order from graph construction
        logger.debug(f"Extracted {len(self.aVertex)} unique vertices from river graph")
        return self.aVertex.copy()  # Return a copy to prevent external modification

    def get_vertex_by_id(self, vertex_id: int) -> Optional[pyvertex]:
        """
        Get a vertex by its internal graph ID.

        Args:
            vertex_id (int): Internal vertex ID (0-based)

        Returns:
            Optional[pyvertex]: The vertex object, or None if not found
        """
        return self.id_to_vertex.get(vertex_id)

    def get_vertex_id(self, vertex: pyvertex) -> Optional[int]:
        """
        Get the internal graph ID for a vertex.

        Args:
            vertex (pyvertex): The vertex to look up

        Returns:
            Optional[int]: Internal vertex ID (0-based), or None if not found
        """
        return self.vertex_to_id.get(vertex)

    def get_vertex_count(self) -> int:
        """
        Get the total number of unique vertices in the network.

        Returns:
            int: Number of unique vertices
        """
        return len(self.id_to_vertex)

    # ========================================================================
    # NETWORK SIMPLIFICATION OPERATIONS
    # ========================================================================

    def remove_disconnected_flowlines(self, flowlines: List[pyflowline], outlet_vertex: Optional[pyvertex] = None) -> List[pyflowline]:
        """
        Remove flowlines that don't flow out to the specified outlet vertex.

        This method performs a backward traversal from the outlet vertex to identify
        all flowlines that are connected to the drainage network. Flowlines that
        cannot reach the outlet (isolated components, disconnected segments) are removed.

        Args:
            flowlines: List of input flowlines
            outlet_vertex: Optional outlet vertex. If not provided, uses the outlet vertex
                          from initialization. If neither is available, returns original flowlines.

        Returns:
            List of flowlines that are connected to the outlet vertex
        """
        return self.network_ops.remove_disconnected_flowlines(flowlines, outlet_vertex)

    def remove_braided_river(self) -> List[pyflowline]:
        """
        Remove braided channels from the river network.
        Enhanced with state management and change tracking.

        This method identifies braided channels (multiple flowlines between
        the same vertex pair) and removes redundant ones to create a simplified
        river network without braiding.

        Returns:
            List of flowlines with braided channels removed
        """
        return self.network_ops.remove_braided_river()

    def remove_parallel_river(self) -> List[pyflowline]:
        """
        Remove parallel rivers using graph-based approach with class instance flowlines.
        Enhanced with state management and change tracking.

        This method replaces the standalone resolve_parallel_paths function by leveraging
        the graph structure to identify alternative routes between distant vertices and
        select the path with the highest cumulative hydrological significance.

        Parallel paths are alternative routes between the same start and end vertices.
        This method selects the most significant route based on cumulative path metrics.

        Returns:
            List[pyflowline]: Flowlines with parallel paths resolved

        Example:
            >>> river_graph = pyrivergraph(flowlines, outlet_vertex)
            >>> resolved_flowlines = river_graph.remove_parallel_river()
            >>> print(f"Resolved parallel paths: {len(flowlines)} -> {len(resolved_flowlines)} flowlines")
        """
        return self.network_ops.remove_parallel_river()

    def remove_cycle(self) -> List[pyflowline]:
        """
        Detect and break cycles in the network by removing lowest priority flowlines.
        Enhanced with state management and change tracking.

        This method replaces the standalone break_network_cycles function by leveraging
        the graph structure to detect cycles and remove the flowline with the lowest
        hydrological significance from each cycle.

        Uses depth-first search to detect cycles and removes the flowline with
        the lowest hydrological significance from each cycle.

        Returns:
            List[pyflowline]: Flowlines with cycles broken (acyclic network)

        Example:
            >>> river_graph = pyrivergraph(flowlines, outlet_vertex)
            >>> acyclic_flowlines = river_graph.remove_cycle()
            >>> print(f"Removed cycles: {len(flowlines) - len(acyclic_flowlines)} flowlines removed")
        """
        return self.network_ops.remove_cycle()

    def remove_small_rivers_iterative(self, dThreshold_small_river, nIterations=3, iFlag_debug=0, sWorkspace_output_basin=None):
        """
        Remove small rivers iteratively using graph-based approach with graph updates at each step.
        This method replaces the standalone function loop in basin.py for small river removal.

        Args:
            dThreshold_small_river (float): Length threshold for small river removal
            nIterations (int, optional): Number of iterations. Defaults to 3.
            iFlag_debug (int, optional): Debug flag for output files. Defaults to 0.
            sWorkspace_output_basin (str, optional): Output workspace for debug files. Defaults to None.

        Returns:
            list: Updated flowlines after iterative small river removal
        """
        return self.network_ops.remove_small_rivers_iterative(
            dThreshold_small_river, nIterations, iFlag_debug, sWorkspace_output_basin
        )

    # ========================================================================
    # NETWORK ANALYSIS & DETECTION
    # ========================================================================

    def find_braided_channels(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Find braided channels (multiple flowlines between same vertex pair).
        Uses caching for performance optimization.

        Returns:
            Dictionary mapping (start_vertex_id, end_vertex_id) to list of flowline indices
        """
        return self.network_analyzer.find_braided_channels()

    def find_parallel_paths(self) -> List[Dict]:
        """
        Find parallel paths between vertices (alternative routes).
        Uses caching for performance optimization.

        Parallel paths are defined as multiple different paths that have the
        SAME starting vertex AND the SAME ending vertex.

        Returns:
            List of parallel path group dictionaries, each containing:
            - 'paths': List of paths, where each path is a list of flowline indices
            - 'start_vertex': Start vertex ID
            - 'end_vertex': End vertex ID
        """
        return self.network_analyzer.find_parallel_paths()

    def detect_cycles(self) -> List[List[int]]:
        """
        Custom cycle detection using DFS with recursion stack.
        Uses caching for performance optimization.
        """
        return self.network_analyzer.detect_cycles()

    def find_linear_segments(self) -> List[List[int]]:
        """
        Find linear segments that can be merged into single flowlines.
        Uses caching for performance optimization.

        A linear segment is a chain of flowlines where each intermediate vertex
        has exactly one incoming and one outgoing edge (degree 2 vertices).
        These represent artificially segmented flowlines that should be merged.

        Returns:
            List of linear segments, where each segment is a list of flowline indices
            that should be merged together in order from upstream to downstream.
        """
        return self.network_analyzer.find_linear_segments()

    def find_outlet_connected_components(self, flowlines: List[pyflowline], outlet_vertex: Optional[pyvertex] = None) -> Dict[str, List[int]]:
        """
        Find connected components in the network and classify them relative to the outlet.

        This method identifies different connected components in the river network
        and classifies them as either connected to the main drainage network (outlet)
        or as isolated components that don't contribute to the main flow.

        Args:
            flowlines: List of flowlines to analyze
            outlet_vertex: Optional outlet vertex. If not provided, uses stored outlet.

        Returns:
            Dictionary with keys:
            - 'main_network': List of flowline indices connected to outlet
            - 'isolated_components': List of lists, each containing flowline indices
              for an isolated component
            - 'component_stats': Dictionary with statistics about components
        """
        return self.network_analyzer.find_outlet_connected_components(flowlines, outlet_vertex)

    # ========================================================================
    # NETWORK MODIFICATION & PROCESSING
    # ========================================================================

    def merge_linear_segments(self, flowlines: List[pyflowline]) -> List[pyflowline]:
        """
        Merge linear segments of flowlines into single flowlines.
        Enhanced with state management and change tracking.

        This method identifies chains of flowlines that can be merged into single
        flowlines (linear segments where intermediate vertices have degree 2) and
        creates new merged flowlines to replace them.

        Args:
            flowlines: List of flowlines to process

        Returns:
            List of flowlines with linear segments merged
        """
        return self.network_ops.merge_linear_segments(flowlines)

    # ========================================================================
    # STREAM ANALYSIS & TOPOLOGY
    # ========================================================================

    def update_head_water_stream_order(self) -> List[pyflowline]:
        """
        Update stream order for head water flowlines using graph-based approach.
        Enhanced with state management and change tracking.

        This method replaces the standalone function by leveraging the graph structure
        to identify headwater flowlines and update their stream order based on
        network topology and confluence analysis.

        Returns:
            List[pyflowline]: Updated flowlines with corrected stream orders
        """
        return self.stream_analyzer.update_head_water_stream_order()

    def identify_headwater_flowlines(self) -> List[pyflowline]:
        """
        Identify headwater flowlines (sources) in the network.

        Returns:
            List of flowlines that are headwaters (no upstream connections)
        """
        return self.stream_analyzer.identify_headwater_flowlines()

    def define_stream_topology(self) -> Dict[str, any]:
        """
        Define comprehensive stream topology including confluences and stream segments.
        Enhanced with state management and caching.

        Returns:
            Dictionary containing topology information:
            - confluences: List of confluence objects
            - stream_segments: List of stream segment definitions
            - topology_stats: Statistics about the network topology
        """
        return self.stream_analyzer.define_stream_topology()

    def define_stream_order(self, iFlag_so_method_in: int = 1) -> Dict[str, any]:
        """
        Define stream order for all flowlines using graph-based approach with confluence analysis.
        Enhanced with state management and caching.

        This method adapts the reference implementation to work with the graph structure,
        supporting both Strahler (default) and Shreve stream ordering methods.

        Args:
            iFlag_so_method_in: Stream ordering method (1=Strahler, 2=Shreve)

        Returns:
            Dictionary containing:
            - 'flowlines': Updated flowlines with stream orders
            - 'stream_orders': Array of stream order values
            - 'confluences': List of confluence objects used
            - 'method': Stream ordering method used
            - 'statistics': Statistics about stream order distribution
        """
        return self.stream_analyzer.define_stream_order(iFlag_so_method_in)

 



    # ========================================================================
    # PRIVATE METHODS - GRAPH CONSTRUCTION & MANAGEMENT
    # ========================================================================

    def _build_graph(self):
        """
        Build the graph structure from flowlines.
        This is the core method that constructs the adjacency list representation.
        """
        self.graph_builders.build_graph()

    def _set_outlet_vertex_id(self):
        """Set the outlet vertex ID if outlet vertex is provided."""
        self.graph_builders.set_outlet_vertex_id()

    def _update_graph_flowlines(self, new_flowlines: List[pyflowline]):
        """
        Update the graph with a new set of flowlines using smart update strategy.

        Args:
            new_flowlines: New list of flowlines to update the graph with
        """
        self.graph_builders.update_graph_flowlines(new_flowlines)





    # ========================================================================
    # PRIVATE METHODS - PATH FINDING & ANALYSIS
    # ========================================================================

    def _find_all_paths(self, start_id: int, target_id: int, max_depth: int = 10) -> List[List[int]]:
        """
        Find all paths from start to target vertex using DFS.

        Args:
            start_id: Starting vertex ID
            target_id: Target vertex ID
            max_depth: Maximum search depth to prevent infinite loops

        Returns:
            List of paths, where each path is a list of vertex IDs
        """
        return self.utils.find_all_paths(start_id, target_id, max_depth)

    def _path_to_flowlines(self, path: List[int]) -> List[int]:
        """
        Convert a path of vertex IDs to flowline indices.

        Args:
            path: List of vertex IDs representing a path

        Returns:
            List of flowline indices
        """
        return self.utils.path_to_flowlines(path)

    def _find_outlet_reachable_vertices(self, outlet_vertex_id: int) -> Set[int]:
        """
        Find all vertices that can reach the outlet using backward traversal.

        Args:
            outlet_vertex_id: ID of the outlet vertex

        Returns:
            Set of vertex IDs that can reach the outlet
        """
        return self.utils.find_outlet_reachable_vertices(outlet_vertex_id)

    def _flowline_contributes_to_outlet(self, flowline_idx: int, start_id: int, end_id: int,
                                       outlet_vertex_id: int, outlet_reachable: Set[int]) -> bool:
        """
        Check if a flowline contributes to the drainage network leading to outlet.

        Args:
            flowline_idx: Index of the flowline
            start_id: Start vertex ID
            end_id: End vertex ID
            outlet_vertex_id: Outlet vertex ID
            outlet_reachable: Set of vertices reachable from outlet

        Returns:
            bool: True if flowline contributes to outlet drainage
        """
        return self.utils.flowline_contributes_to_outlet(
            flowline_idx, start_id, end_id, outlet_vertex_id, outlet_reachable
        )

    # ========================================================================
    # PRIVATE METHODS - NETWORK PROCESSING HELPERS
    # ========================================================================

    def _remove_small_rivers(self, flowlines: List[pyflowline], threshold: float) -> List[pyflowline]:
        """
        Remove small rivers based on length threshold.

        Args:
            flowlines: List of flowlines to filter
            threshold: Length threshold for removal

        Returns:
            List of flowlines with small rivers removed
        """
        return self.utils.remove_small_rivers(flowlines, threshold)

    def _topological_sort_flowlines(self) -> List[pyflowline]:
        """
        Perform topological sort of flowlines for stream order calculation.

        Returns:
            List of flowlines in topological order
        """
        return self.utils.topological_sort_flowlines()

    def _calculate_stream_order(self, flowline: pyflowline) -> int:
        """
        Calculate stream order for a flowline based on upstream confluences.

        Args:
            flowline: Flowline to calculate order for

        Returns:
            int: Stream order
        """
        return self.utils.calculate_stream_order(flowline)

    def _merge_flowline_segment(self, segment_flowlines: List[pyflowline]) -> Optional[pyflowline]:
        """
        Merge a segment of flowlines into a single flowline.

        Args:
            segment_flowlines: List of flowlines to merge

        Returns:
            Optional[pyflowline]: Merged flowline or None if merge failed
        """
        return self.utils.merge_flowline_segment(segment_flowlines)

    # ========================================================================
    # PRIVATE METHODS - VERTEX & CONFLUENCE MANAGEMENT
    # ========================================================================

    def _create_confluence_object(self, vertex_id: int, vertex: pyvertex) -> Optional[pyconfluence]:
        """
        Create a confluence object for a vertex.

        Args:
            vertex_id: Vertex ID
            vertex: Vertex object

        Returns:
            Optional[pyconfluence]: Confluence object or None if creation failed
        """
        return self.utils.create_confluence_object(vertex_id, vertex)

    def _define_stream_segments(self, confluences: List[pyconfluence]) -> List[Dict]:
        """
        Define stream segments between confluences.

        Args:
            confluences: List of confluence objects

        Returns:
            List of stream segment definitions
        """
        return self.utils.define_stream_segments(confluences)



    def _initialize_headwater_stream_orders(self):
        """
        Initialize stream orders for headwater flowlines (stream order = 1).
        Adapted from the reference update_head_water_stream_order function.
        """
        return self.stream_analyzer._initialize_headwater_stream_orders()

    def _extract_confluences_for_stream_order(self) -> List[Dict]:
        """
        Extract confluence information from the graph topology for stream order calculation.

        Returns:
            List of confluence dictionaries with upstream/downstream flowline information
        """
        return self.stream_analyzer._extract_confluences_for_stream_order()

    def _build_confluence_spatial_index(self, confluences: List[Dict]):
        """
        Build spatial index for efficient confluence lookup.

        Args:
            confluences: List of confluence dictionaries

        Returns:
            Spatial index object (R-tree if available, otherwise fallback)
        """
        return self.stream_analyzer._build_confluence_spatial_index(confluences)

    def _process_confluences_iteratively(self, confluences: List[Dict], confluence_index, iFlag_so_method_in: int):
        """
        Process confluences iteratively until all stream orders are defined.
        Adapted from the reference define_stream_order function.

        Args:
            confluences: List of confluence dictionaries
            confluence_index: Spatial index for confluence lookup
            iFlag_so_method_in: Stream ordering method (1=Strahler, 2=Shreve)
        """
        return self.stream_analyzer._process_confluences_iteratively(
            confluences, confluence_index, iFlag_so_method_in
        )

    def _update_connected_flowlines_stream_order(self, reference_flowline: pyflowline, stream_order: int,
                                                confluences: List[Dict], confluence_index):
        """
        Update stream order for flowlines connected to the same stream segment.
        Adapted from the reference implementation's confluence update logic.

        Args:
            reference_flowline: The flowline whose order was just set
            stream_order: The stream order to propagate
            confluences: List of confluence dictionaries
            confluence_index: Spatial index for confluence lookup
        """
        return self.stream_analyzer._update_connected_flowlines_stream_order(
            reference_flowline, stream_order, confluences, confluence_index
        )

    def _create_stream_order_result(self, iFlag_so_method_in: int, confluences: List[Dict]) -> Dict[str, any]:
        """
        Create the result dictionary for stream order calculation.

        Args:
            iFlag_so_method_in: Stream ordering method used
            confluences: List of confluence dictionaries

        Returns:
            Dictionary containing results and statistics
        """
        return self.stream_analyzer._create_stream_order_result(iFlag_so_method_in, confluences)
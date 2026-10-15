"""
Graph-based algorithm for removing braided rivers and resolving flowline loops.

This module provides advanced algorithms using directed graph theory to handle
complex river networks with braided channels, loops, and parallel streams.
"""

import logging
import os
from typing import List, Dict, Set, Tuple, Optional, DefaultDict
from collections import defaultdict, deque
import numpy as np
from enum import Enum
import time

# Set up logger
logger = logging.getLogger(__name__)
# Import for spatial indexing (R-tree)
try:
    from rtree.index import Index as RTreeindex
    HAS_RTREE = True
except ImportError:
    HAS_RTREE = False
    logger.warning("R-tree not available - using fallback spatial indexing")

from pyearthriver.algorithms.spatial import find_vertex_on_edge


from pyearthriver.classes.flowline import pyflowline
from pyearthriver.classes.vertex import pyvertex
from pyearthriver.classes.edge import pyedge
from pyearthriver.classes.confluence import pyconfluence
from pyearthriver.formats.export_flowline import export_flowline_to_vector
from pyearthriver.formats.export_vertex import export_vertex_to_vector


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

    def __init__(
        self, flowlines: List[pyflowline], pVertex_outlet: Optional[pyvertex] = None
    ):
        """
        Initialize the river network graph from flowlines.

        Args:
            flowlines: List of flowline objects representing the river network
            outlet_vertex: Optional outlet vertex for the drainage network. If provided,
                          enables outlet-based operations and optimizations.
        """
        self.aFlowline = flowlines
        self.pVertex_outlet_id: Optional[int] = None
        self.vertex_to_id: Dict[pyvertex, int] = {}
        self.id_to_vertex: Dict[int, pyvertex] = {}
        self.aFlowline_edges: Dict[int, Tuple[int, int]] = {}
        self.aVertex: List[pyvertex] = []  # Will be populated during graph building
        self.aVertex_outlet: List[pyvertex] = []  # all the outlet vertices

        # Always use custom implementation
        self.adjacency_list: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.in_degree: DefaultDict[int, int] = defaultdict(int)
        self.out_degree: DefaultDict[int, int] = defaultdict(int)
        self.aVertex_confluence: List[pyvertex] = []
        self.aConfluence: List[pyconfluence] = []

        # Initialize state management system

        logger.debug("Using custom graph implementation with state management")
        self._build_graph()
        # Set outlet vertex ID if outlet was provided
        if pVertex_outlet is not None:
            # only preserve flowlines that can flow out to the outlet
            self.pVertex_outlet = pVertex_outlet
            self.aVertex_outlet = [pVertex_outlet]
            self.aFlowline = self.remove_disconnected_flowlines()
            self._set_outlet_vertex_id()

    def get_sources(self) -> List[int]:
        """Get source nodes (headwaters) with no incoming edges."""
        from pyearthriver.algorithms.topology.connectivity import get_sources
        return get_sources(self)

    def get_sinks(self) -> List[int]:
        """Get sink nodes (outlets) with no outgoing edges."""
        from pyearthriver.algorithms.topology.connectivity import get_sinks
        return get_sinks(self)

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

    def find_outlet_vertices(self) -> List[pyvertex]:
        """
        Detect outlet vertices: one per independent river network.

        An outlet is a sink vertex with no outgoing edges.

        Returns:
            List[pyvertex]: One outlet vertex per detected river network
        """
        from pyearthriver.algorithms.topology.connectivity import find_outlet_vertices
        return find_outlet_vertices(self)

    def split_graph(self) -> List["pyrivergraph"]:
        """
        Split the graph into independent river networks, one per outlet vertex.

        Returns:
            List[pyrivergraph]: One river graph per independent network
        """
        from pyearthriver.algorithms.topology.connectivity import split_graph
        return split_graph(self)

    def export_graph(
        self,
        sFilename_flowline_in: str = None,
        sFilename_vertex_in: str = None,
    ):
        """
        Export the graph to GeoJSON files.

        Use the optional filename arguments to control what to export:
        a file is exported only when its filename is provided.

        Args:
            sFilename_flowline_in (str, optional): Output filename for the
                flowlines. Defaults to None (skip flowline export).
            sFilename_vertex_in (str, optional): Output filename for the
                vertices. Defaults to None (skip vertex export).

        Returns:
            List[str]: Paths of the exported files
        """
        aFilename_out = list()

        if sFilename_flowline_in is None and sFilename_vertex_in is None:
            logger.warning("No output filename provided, nothing to export")
            return aFilename_out

        if sFilename_flowline_in is not None:
            if self.aFlowline:
                sFolder_output = os.path.dirname(sFilename_flowline_in)
                if sFolder_output and not os.path.exists(sFolder_output):
                    os.makedirs(sFolder_output)
                export_flowline_to_vector(self.aFlowline, sFilename_flowline_in)
                aFilename_out.append(sFilename_flowline_in)
                logger.info(
                    f"Exported {len(self.aFlowline)} flowlines to {sFilename_flowline_in}"
                )
            else:
                logger.warning("No flowlines available to export")

        if sFilename_vertex_in is not None:
            if self.aVertex:
                sFolder_output = os.path.dirname(sFilename_vertex_in)
                if sFolder_output and not os.path.exists(sFolder_output):
                    os.makedirs(sFolder_output)
                export_vertex_to_vector(self.aVertex, sFilename_vertex_in)
                aFilename_out.append(sFilename_vertex_in)
                logger.info(
                    f"Exported {len(self.aVertex)} vertices to {sFilename_vertex_in}"
                )
            else:
                logger.warning("No vertices available to export")

        return aFilename_out

    # ========================================================================
    # NETWORK SIMPLIFICATION OPERATIONS
    # ========================================================================

    def remove_disconnected_flowlines(
        self, pVertex_outlet: Optional[pyvertex] = None
    ) -> List[pyflowline]:
        """
        Remove flowlines that don't flow out to the specified outlet vertex.

        Args:
            pVertex_outlet: Optional outlet vertex.

        Returns:
            List of flowlines that are connected to the outlet vertex
        """
        from pyearthriver.algorithms.topology.pruning import (
            remove_disconnected_flowlines,
        )
        return remove_disconnected_flowlines(self, pVertex_outlet=pVertex_outlet)

    def remove_braided_river(self) -> List[pyflowline]:
        """
        Remove braided channels from the river network.
        Enhanced with state management and change tracking.

        This method identifies braided channels (multiple flowlines between
        the same vertex pair) and removes redundant ones to create a simplified
        river network without braiding. For each braided region, keeps the
        shortest flowline and removes the rest.

        Returns:
            List of flowlines with braided channels removed
        """

        from pyearthriver.algorithms.braided import remove_braided_river
        return remove_braided_river(self)

    def remove_parallel_river(self) -> List[pyflowline]:
        """
        Remove parallel rivers using graph-based approach with class instance flowlines.
        Enhanced with state management and change tracking.

        Returns:
            List[pyflowline]: Flowlines with parallel paths resolved
        """
        from pyearthriver.algorithms.braided import remove_parallel_river
        return remove_parallel_river(self)


    def remove_cycle(self) -> List[pyflowline]:
        """
        Remove cycles from the directed flow network using a single-pass
        **reverse-BFS from the outlet**.

        Algorithm
        ---------
        A valid flow network is a DAG where every flowline points toward the
        outlet.  We exploit this by growing a set ``known`` of vertices that
        are confirmed to have a path to the outlet, starting from the outlet
        itself and expanding upstream one flowline at a time.

        For each flowline ``start → end``:
        - If ``end ∈ known`` and ``start ∉ known``:
            The flowline correctly flows toward the outlet.
            Add ``start`` to ``known`` and enqueue it for further expansion.
        - If ``end ∈ known`` and ``start ∈ known``:
            Both endpoints are already confirmed outlet-reachable, but this
            flowline creates a redundant/cyclic connection — it flows from a
            vertex that is already upstream-reachable back toward a vertex
            that is also already known.  **Remove this flowline.**
        - If ``end ∉ known``:
            This flowline's end is not yet confirmed; skip for now (it will
            be processed when its end vertex is added to ``known``).

        This is O(V + E) — a single BFS pass — and requires no separate cycle
        detection step.  It naturally preserves all flowlines that contribute
        to the outlet-connected network and removes only those that create
        directed cycles.

        Fallback
        --------
        If no outlet vertex is available (``pVertex_outlet_id`` is None), cycle
        removal is skipped and the original flowlines are returned unchanged,
        because flow direction cannot be determined without an outlet reference.

        Returns
        -------
        List[pyflowline]
            Flowlines with all directed cycles removed.

        Example
        -------
        >>> river_graph = pyrivergraph(flowlines, outlet_vertex)
        >>> acyclic_flowlines = river_graph.remove_cycle()
        >>> print(f"Removed: {len(flowlines) - len(acyclic_flowlines)} flowlines")
        """
        from pyearthriver.algorithms.cycles import remove_cycles_from_graph
        return remove_cycles_from_graph(self)


    def remove_small_river(
        self,
        dThreshold_small_river,
        nIterations=3,
        iFlag_debug=0,
        sWorkspace_output_basin=None,
    ):
        """
        Remove small rivers iteratively using graph-based approach with graph updates at each step.

        Args:
            dThreshold_small_river (float): Length threshold for small river removal
            nIterations (int, optional): Number of iterations. Defaults to 3.
            iFlag_debug (int, optional): Debug flag for output files. Defaults to 0.
            sWorkspace_output_basin (str, optional): Output workspace for debug files. Defaults to None.

        Returns:
            list: Updated flowlines after iterative small river removal
        """
        from pyearthriver.algorithms.topology.pruning import remove_small_river
        return remove_small_river(
            self,
            dThreshold_small_river,
            nIterations=nIterations,
            iFlag_debug=iFlag_debug,
            sWorkspace_output_basin=sWorkspace_output_basin,
        )

    def remove_duplicate_flowlines(
        self, iFlag_direction_insensitive: bool = False
    ) -> List[pyflowline]:
        """
        Remove duplicate flowlines from the network.

        Args:
            iFlag_direction_insensitive (bool): If True, flowlines with opposite directions
                                              (A→B and B→A) are considered duplicates.
                                              If False (default), direction matters.

        Returns:
            List of flowlines with duplicates removed
        """
        from pyearthriver.algorithms.topology.pruning import (
            remove_duplicate_flowlines,
        )
        return remove_duplicate_flowlines(
            self, iFlag_direction_insensitive=iFlag_direction_insensitive
        )

    def _remove_duplicate_flowlines_direction_sensitive(self) -> List[pyflowline]:
        """
        Remove duplicate flowlines where direction matters (A→B ≠ B→A).
        """
        from pyearthriver.algorithms.topology.pruning import (
            remove_duplicate_flowlines_direction_sensitive,
        )
        return remove_duplicate_flowlines_direction_sensitive(self)

    def _remove_duplicate_flowlines_direction_insensitive(self) -> List[pyflowline]:
        """
        Remove duplicate flowlines where direction doesn't matter.
        """
        from pyearthriver.algorithms.topology.pruning import (
            remove_duplicate_flowlines_direction_insensitive,
        )
        return remove_duplicate_flowlines_direction_insensitive(self)

    # ========================================================================
    # NETWORK ANALYSIS & DETECTION
    # ========================================================================

    def find_braided_channels(self) -> List[List[pyflowline]]:
        """
        Find braided channels (multiple flowlines between same vertex pair).
        """
        from pyearthriver.algorithms.braided import find_braided_channels
        return find_braided_channels(self)

    def find_parallel_paths(self) -> List[Dict]:
        """
        Find divergent parallel sections between vertices.
        """
        from pyearthriver.algorithms.braided import find_parallel_paths
        return find_parallel_paths(self)

    def _find_partial_parallel_sections(self) -> List[Dict]:
        """
        Find partial parallel sections within longer paths.
        """
        from pyearthriver.algorithms.braided import find_partial_parallel_sections
        return find_partial_parallel_sections(self)

    def _find_reconvergence_points(
        self, divergence_id: int, max_depth: int = 8
    ) -> Optional[int]:
        """
        Find vertices where paths from a divergence point reconverge.
        """
        from pyearthriver.algorithms.braided import find_reconvergence_points
        return find_reconvergence_points(self, divergence_id, max_depth)

    def find_first_common_vertex(self, reachable_list1, reachable_list2):
        """
        Find the first vertex that appears in both lists, respecting topological order.
        """
        from pyearthriver.algorithms.braided import find_first_common_vertex
        return find_first_common_vertex(reachable_list1, reachable_list2)

    def _get_reachable_vertices(self, start_id: int, max_depth: int) -> List[int]:
        """
        Get all vertices reachable from a starting vertex within max_depth.
        """
        from pyearthriver.algorithms.braided import get_reachable_vertices
        return get_reachable_vertices(self, start_id, max_depth)


    def _find_linear_segments(self) -> List[List[int]]:
        """
        Find linear segments that can be merged into single flowlines.
        """
        from pyearthriver.algorithms.topology.segmentation import find_linear_segments
        return find_linear_segments(self)

    def _build_bidirectional_segment(
        self,
        start_flowline_idx: int,
        start_vertex_id: int,
        end_vertex_id: int,
        merge_vertices: set,
        processed_flowlines: set,
    ) -> List[int]:
        """
        Build a complete linear segment by extending both upstream and downstream
        from the starting flowline.
        """
        from pyearthriver.algorithms.topology.segmentation import (
            build_bidirectional_segment,
        )
        return build_bidirectional_segment(
            self,
            start_flowline_idx,
            start_vertex_id,
            end_vertex_id,
            merge_vertices,
            processed_flowlines,
        )

    # ========================================================================
    # NETWORK MODIFICATION & PROCESSING
    # ========================================================================

    def split_flowline(
        self,
        aVertex_in: Optional[List[pyvertex]] = None,
        iFlag_intersect=None,
        iFlag_use_id=None,
    ) -> List[pyflowline]:
        """
        Split flowline based on the intersection with a list of vertex.
        """
        from pyearthriver.algorithms.topology.segmentation import split_flowline
        return split_flowline(
            self,
            aVertex_in=aVertex_in,
            iFlag_intersect=iFlag_intersect,
            iFlag_use_id=iFlag_use_id,
        )

    def merge_flowline(self) -> List[pyflowline]:
        """
        Merge linear segments of flowlines into single flowlines.
        """
        from pyearthriver.algorithms.topology.segmentation import merge_flowline
        return merge_flowline(self)

    # ========================================================================
    # STREAM ANALYSIS & TOPOLOGY
    # ========================================================================

    def update_headwater_stream_order(self) -> List[pyflowline]:
        """
        Update stream order for head water flowlines using graph-based approach.
        """
        from pyearthriver.algorithms.stream_order import update_headwater_stream_order
        return update_headwater_stream_order(self)

    def identify_headwater_flowlines(self) -> List[pyflowline]:
        """
        Identify headwater flowlines (sources) in the network.
        """
        from pyearthriver.algorithms.stream_order import identify_headwater_flowlines
        return identify_headwater_flowlines(self)


    def define_stream_segment(self):
        """
        Define stream segments using topological sorting from outlet to headwater.

        Returns:
            Tuple of (sorted_flowlines, segment_indices)
        """
        from pyearthriver.algorithms.topology.connectivity import define_stream_segment
        return define_stream_segment(self)

    def _sort_flowlines_from_outlet(self):
        """
        Core sorting function using BFS traversal from outlet vertex.

        Returns:
            List of flowlines sorted from outlet to headwater
        """
        from pyearthriver.algorithms.topology.connectivity import (
            sort_flowlines_from_outlet,
        )
        return sort_flowlines_from_outlet(self)

    def define_river_confluence(self):
        """
        Build the conflence using the in_degree

        Args:
            aFlowline_basin_in (list [pyflowline]): A list of flowlines in this basin
            aVertex_confluence_in (list [pyconfluence]): A list of vertices in this basin

        Returns:
            list [pyconfluence]: A list of confluences in this basin
        """
        from pyearthriver.algorithms.confluence import define_river_confluence
        return define_river_confluence(self)

    def define_stream_topology(self) -> List[pyflowline]:
        """
        Define comprehensive stream topology including confluences and stream segments.
        """
        from pyearthriver.algorithms.topology.connectivity import define_stream_topology
        return define_stream_topology(self)

    def define_stream_order(self, iFlag_so_method_in: int = 1) -> List[pyflowline]:
        """
        Define stream order for all flowlines using graph-based approach with confluence analysis.
        """
        from pyearthriver.algorithms.stream_order import define_stream_order
        return define_stream_order(self, iFlag_so_method_in)


    # ========================================================================
    # PRIVATE METHODS - GRAPH CONSTRUCTION & MANAGEMENT
    # ========================================================================

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
            if flowline is None:  # Skip removed flowlines
                continue

            start_vertex = flowline.pVertex_start
            end_vertex = flowline.pVertex_end

            # Add start vertex if not seen before
            if start_vertex not in self.vertex_to_id:
                self.vertex_to_id[start_vertex] = vertex_counter
                self.id_to_vertex[vertex_counter] = start_vertex
                self.aVertex.append(start_vertex)
                start_vertex.lVertexID = vertex_counter  # Set vertex ID
                vertex_counter += 1

            # Add end vertex if not seen before
            if end_vertex not in self.vertex_to_id:
                self.vertex_to_id[end_vertex] = vertex_counter
                self.id_to_vertex[vertex_counter] = end_vertex
                self.aVertex.append(end_vertex)
                end_vertex.lVertexID = vertex_counter  # Set vertex ID
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

        logger.debug(
            f"Built graph with {len(self.id_to_vertex)} vertices and {len(self.aFlowline_edges)} edges"
        )

    def _set_outlet_vertex_id(self):
        """Set the outlet vertex ID if outlet vertex is provided."""
        if self.pVertex_outlet is not None:
            self.pVertex_outlet_id = self.vertex_to_id.get(self.pVertex_outlet)
            if self.pVertex_outlet_id is not None:
                logger.debug(f"Set outlet vertex ID: {self.pVertex_outlet_id}")
            else:
                logger.warning("Outlet vertex not found in graph")

    def _update_graph_flowlines(self, new_flowlines: List[pyflowline]):
        """
        Update the graph with a new set of flowlines using smart update strategy.

        Args:
            new_flowlines: New list of flowlines to update the graph with
        """

        start_time = time.time()
        self.aFlowline = new_flowlines
        self._build_graph()

        # Critical: Update outlet vertex ID after graph rebuild
        # The outlet vertex ID can become invalid after graph modifications
        if hasattr(self, "pVertex_outlet") and self.pVertex_outlet is not None:
            self._set_outlet_vertex_id()
            logger.debug(
                f"Updated outlet vertex ID after graph rebuild: {self.pVertex_outlet_id}"
            )

        return

    # ========================================================================
    # PRIVATE METHODS - PATH FINDING & ANALYSIS
    # ========================================================================

    def _find_all_paths(
        self, start_id: int, target_id: int, max_depth: int = 10
    ) -> List[List[int]]:
        """
        Find all paths from start to target vertex using DFS.
        """
        from pyearthriver.algorithms.topology.connectivity import find_all_paths
        return find_all_paths(self, start_id, target_id, max_depth=max_depth)

    def _path_to_flowlines(self, path: List[int]) -> List[int]:
        """
        Convert a path of vertex IDs to flowline indices.
        """
        from pyearthriver.algorithms.topology.connectivity import path_to_flowlines
        return path_to_flowlines(self, path)

    def _find_outlet_reachable_vertices(self, outlet_vertex_id: int) -> Set[int]:
        """
        Find all vertices that can reach the outlet using backward traversal.
        """
        from pyearthriver.algorithms.topology.pruning import (
            find_outlet_reachable_vertices,
        )
        return find_outlet_reachable_vertices(self, outlet_vertex_id)

    def _remove_small_river_step(
        self, flowlines: List[pyflowline], threshold: float
    ) -> List[pyflowline]:
        """
        Remove small rivers based on length threshold.
        """
        from pyearthriver.algorithms.topology.pruning import remove_small_river_step
        return remove_small_river_step(self, flowlines, threshold)

    def _merge_flowline_segment(
        self, segment_flowlines: List[pyflowline]
    ) -> Optional[pyflowline]:
        """
        Merge a segment of flowlines into a single flowline.
        """
        from pyearthriver.algorithms.topology.segmentation import (
            merge_flowline_segment,
        )
        return merge_flowline_segment(segment_flowlines)

    # ========================================================================
    # PRIVATE METHODS - VERTEX & CONFLUENCE MANAGEMENT
    # ========================================================================

    def _create_confluence_object(
        self, vertex_id: int, vertex: pyvertex
    ) -> Optional[pyconfluence]:
        """
        Create a confluence object for a vertex.
        """
        from pyearthriver.algorithms.confluence import create_confluence_object
        return create_confluence_object(self, vertex_id, vertex)

    # ========================================================================
    # PRIVATE METHODS - STREAM ORDER CALCULATION HELPERS
    # ========================================================================

    def _process_confluences_iteratively(self, iFlag_so_method_in: int):
        """
        Process confluences iteratively to update stream orders.
        """
        from pyearthriver.algorithms.stream_order import process_confluences_iteratively
        return process_confluences_iteratively(self, iFlag_so_method_in)

    def get_upstream_indices(self, flowline: pyflowline) -> List[int]:
        """
        Get indices of upstream flowlines for a given flowline.
        """
        from pyearthriver.algorithms.topology.connectivity import get_upstream_indices
        return get_upstream_indices(self, flowline)

    def get_downstream_indices(self, flowline: pyflowline) -> List[int]:
        """
        Get indices of downstream flowlines for a given flowline.
        """
        from pyearthriver.algorithms.topology.connectivity import (
            get_downstream_indices,
        )
        return get_downstream_indices(self, flowline)

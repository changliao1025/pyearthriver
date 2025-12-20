"""
Topology definition and management for river networks.

This module provides operations for defining and managing stream network topology.
"""

import logging
from typing import List, Tuple, Optional
from collections import deque

from ..classes.flowline import pyflowline
from ..classes.confluence import pyconfluence
from ..classes.vertex import pyvertex
from ..core.graph import RiverGraph
from ..analysis.pathfinding import PathFinder

logger = logging.getLogger(__name__)


class TopologyManager:
    """
    Manages stream network topology.

    This class provides methods for:
    - Defining stream segments
    - Defining confluences
    - Defining stream topology
    - Defining stream order
    - Identifying headwater flowlines
    """

    def __init__(self, graph: RiverGraph, pathfinder: PathFinder):
        """
        Initialize the topology manager.

        Args:
            graph: RiverGraph instance to manage
            pathfinder: PathFinder for path analysis
        """
        self.graph = graph
        self.pathfinder = pathfinder

    def define_stream_segment(self):
        """
        Define stream segments using topological sorting from outlet to headwater.

        Returns:
            Tuple of (sorted_flowlines, segment_indices)
        """
        nFlowline = len(self.graph.aFlowline)

        if nFlowline == 0:
            print('data incomplete')
            return [], []

        try:
            sorted_flowlines = self._sort_flowlines_from_outlet()

            nFlowline = len(sorted_flowlines)
            for i, flowline in enumerate(sorted_flowlines, start=1):
                flowline.iStream_segment = nFlowline - i + 1

            self.graph.aFlowline = sorted_flowlines
            self.graph.update_graph_flowlines(sorted_flowlines)

            aStream_segment = [flowline.iStream_segment for flowline in sorted_flowlines]

            logger.info(f"Topologically sorted {len(sorted_flowlines)} flowlines from outlet to headwater")

            return sorted_flowlines, aStream_segment

        except Exception as e:
            logger.error(f"Error in topological sorting: {e}, using simple enumeration")
            for i, pFlowline in enumerate(self.graph.aFlowline, start=1):
                pFlowline.iStream_segment = nFlowline - i + 1
            aStream_segment = [pFlowline.iStream_segment for pFlowline in self.graph.aFlowline]
            return self.graph.aFlowline, aStream_segment

    def _sort_flowlines_from_outlet(self):
        """
        Core sorting function using BFS traversal from outlet vertex.

        Returns:
            List of flowlines sorted from outlet to headwater
        """
        visited_flowlines = set()
        sorted_flowlines = []
        queue = deque()

        outlet_id = self.graph.pVertex_outlet_id

        if outlet_id is None or outlet_id not in self.graph.id_to_vertex:
            logger.warning("Outlet vertex ID is invalid, attempting to refresh")
            if hasattr(self.graph, 'pVertex_outlet') and self.graph.pVertex_outlet is not None:
                self.graph._set_outlet_vertex_id()
                outlet_id = self.graph.pVertex_outlet_id

            if outlet_id is None or outlet_id not in self.graph.id_to_vertex:
                logger.error("Cannot find valid outlet vertex after refresh, returning original flowline order")
                return self.graph.aFlowline

        outlet_flowlines = []
        for flowline_idx, (start_id, end_id) in self.graph.aFlowline_edges.items():
            if end_id == outlet_id:
                outlet_flowlines.append(flowline_idx)

        if not outlet_flowlines:
            logger.warning("No flowlines found ending at outlet vertex")
            return self.graph.aFlowline

        for flowline_idx in outlet_flowlines:
            queue.append(flowline_idx)

        while queue:
            flowline_idx = queue.popleft()

            if flowline_idx in visited_flowlines:
                continue

            visited_flowlines.add(flowline_idx)

            if flowline_idx < len(self.graph.aFlowline):
                sorted_flowlines.append(self.graph.aFlowline[flowline_idx])

            start_id, _ = self.graph.aFlowline_edges[flowline_idx]

            upstream_flowlines = []
            for upstream_idx, (up_start, up_end) in self.graph.aFlowline_edges.items():
                if up_end == start_id and upstream_idx not in visited_flowlines:
                    upstream_flowlines.append(upstream_idx)

            for upstream_idx in upstream_flowlines:
                if upstream_idx not in queue:
                    queue.append(upstream_idx)

        excluded_count = len(self.graph.aFlowline) - len(sorted_flowlines)
        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} disconnected flowlines that do not flow to outlet")

            for i, flowline in enumerate(self.graph.aFlowline):
                if i not in visited_flowlines:
                    logger.debug(f"Excluded disconnected flowline {i}")

        logger.info(f"Sorted {len(sorted_flowlines)} connected flowlines using BFS from outlet")
        return sorted_flowlines

    def define_river_confluence(self):
        """
        Build the confluence using the in_degree.

        Returns:
            List of confluences in this basin
        """
        aConfluence = []
        try:
            for vertex_id, vertex in self.graph.id_to_vertex.items():
                if self.graph.in_degree[vertex_id] > 1:
                    pConfluence = self._create_confluence_object(vertex_id, vertex)
                    if pConfluence:
                        aConfluence.append(pConfluence)
        except:
            logger.error("Error defining river confluences")
            return []

        self.graph.aConfluence = aConfluence
        return aConfluence

    def _create_confluence_object(self, vertex_id: int, vertex: pyvertex) -> Optional[pyconfluence]:
        """
        Create a confluence object for a vertex.

        Args:
            vertex_id: Vertex ID
            vertex: Vertex object

        Returns:
            Confluence object or None if creation failed
        """
        try:
            aFlowline_upstream = []
            for start_id, neighbors in self.graph.adjacency_list.items():
                for end_id, flowline_idx in neighbors:
                    if end_id == vertex_id and flowline_idx < len(self.graph.aFlowline):
                        aFlowline_upstream.append(self.graph.aFlowline[flowline_idx])

            aFlowline_downstream = []
            for end_id, flowline_idx in self.graph.adjacency_list[vertex_id]:
                if flowline_idx < len(self.graph.aFlowline):
                    aFlowline_downstream.append(self.graph.aFlowline[flowline_idx])

            confluence = pyconfluence(vertex, aFlowline_upstream, aFlowline_downstream)
            confluence.lVertexID = vertex_id

            return confluence

        except Exception as e:
            logger.error(f"Error creating confluence object: {e}")
            return None

    def define_stream_topology(self) -> List[pyflowline]:
        """
        Define comprehensive stream topology including confluences and stream segments.

        Returns:
            Updated flowlines with topology information
        """
        aFlowline = self.graph.aFlowline
        try:
            for flowline in aFlowline:
                upstream_flowlines = [self.graph.aFlowline[i] for i in self.pathfinder.get_upstream_indices(flowline)]
                downstream_flowlines = [self.graph.aFlowline[i] for i in self.pathfinder.get_downstream_indices(flowline)]
                flowline.aFlowline_upstream = upstream_flowlines
                flowline.aFlowline_downstream = downstream_flowlines
            pass

        except Exception as e:
            logger.error(f"Error defining stream topology: {e}")

        return aFlowline

    def define_stream_order(self, iFlag_so_method_in: int = 1) -> Tuple[List[pyflowline], List[int]]:
        """
        Define stream order for all flowlines using graph-based approach.

        Args:
            iFlag_so_method_in: Stream ordering method (1=Strahler, 2=Shreve)

        Returns:
            Tuple of (flowlines, stream_orders)
        """
        if not self.graph.aFlowline:
            logger.warning('No flowlines available for stream order calculation')
            return None, None

        logger.info(f"Defining stream order for {len(self.graph.aFlowline)} flowlines using "
                   f"{'Strahler' if iFlag_so_method_in == 1 else 'Shreve'} method")

        try:
            self.identify_headwater_flowlines()
            self.update_headwater_stream_order()
            self._process_confluences_iteratively(iFlag_so_method_in)

            aStream_order = [flowline.iStream_order for flowline in self.graph.aFlowline]

            return self.graph.aFlowline, aStream_order

        except Exception as e:
            logger.error(f"Error defining stream order: {e}")
            return None, None

    def update_headwater_stream_order(self) -> List[pyflowline]:
        """
        Update stream order for head water flowlines.

        Returns:
            Updated flowlines with corrected stream orders
        """
        if not self.graph.aFlowline:
            logger.warning('No flowlines available for stream order update')
            return []

        logger.info(f"Updating stream order for {len(self.graph.aFlowline)} flowlines")

        try:
            aFlowline_headwater = self.identify_headwater_flowlines()
            if not aFlowline_headwater:
                logger.warning("No headwater flowlines identified for stream order update")
                return self.graph.aFlowline.copy()
            else:
                for pFlowline in aFlowline_headwater:
                    pFlowline.iStream_order = 1

            return self.graph.aFlowline

        except Exception as e:
            logger.error(f"Error updating stream orders: {e}")
            return self.graph.aFlowline.copy()

    def identify_headwater_flowlines(self) -> List[pyflowline]:
        """
        Identify headwater flowlines (sources) in the network.

        Returns:
            List of flowlines that are headwaters
        """
        aFlowline_headwater = []
        for i, flowline in enumerate(self.graph.aFlowline):
            start_vertex_id = self.graph.vertex_to_id.get(flowline.pVertex_start)
            if start_vertex_id is not None and self.graph.in_degree[start_vertex_id] == 0:
                aFlowline_headwater.append(flowline)

        logger.debug(f"Identified {len(aFlowline_headwater)} headwater flowlines")
        self.graph.aFlowline_headwater = aFlowline_headwater
        return aFlowline_headwater

    def _process_confluences_iteratively(self, iFlag_so_method_in: int):
        """
        Process confluences iteratively to update stream orders.

        Args:
            iFlag_so_method_in: Stream ordering method (1=Strahler, 2=Shreve)
        """
        unresolved_confluences = set(range(len(self.graph.aConfluence)))
        iteration = 0

        while unresolved_confluences:
            iteration += 1
            logger.debug(f"Stream order iteration {iteration}, "
                         f"{len(unresolved_confluences)} confluences remaining")

            resolved_this_iteration = set()

            for idx in unresolved_confluences:
                confluence = self.graph.aConfluence[idx]
                upstream_orders = [fl.iStream_order for fl in confluence.aFlowline_upstream
                                   if fl.iStream_order > 0]

                if len(upstream_orders) == len(confluence.aFlowline_upstream):
                    if iFlag_so_method_in == 1:  # Strahler
                        if upstream_orders.count(max(upstream_orders)) > 1:
                            downstream_order = max(upstream_orders) + 1
                        else:
                            downstream_order = max(upstream_orders)
                    elif iFlag_so_method_in == 2:  # Shreve
                        downstream_order = sum(upstream_orders)
                    else:
                        logger.warning(f"Unknown stream order method: {iFlag_so_method_in}")
                        continue

                    for fl in confluence.aFlowline_downstream:
                        fl.iStream_order = downstream_order

                    resolved_this_iteration.add(idx)

            if not resolved_this_iteration:
                logger.warning("No confluences resolved in this iteration, stopping to avoid infinite loop")
                break

            unresolved_confluences -= resolved_this_iteration

        return
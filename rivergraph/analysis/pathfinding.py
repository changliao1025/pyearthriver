"""
Path finding and reachability analysis for river networks.

This module provides algorithms for finding paths and analyzing connectivity.
"""

import logging
from typing import List, Set
from collections import deque, defaultdict

from ..classes.flowline import pyflowline
from ..core.graph import RiverGraph

logger = logging.getLogger(__name__)


class PathFinder:
    """
    Path finding algorithms for river networks.

    This class provides methods for:
    - Finding all paths between vertices
    - Converting paths to flowlines
    - Analyzing reachability
    - Getting upstream/downstream connections
    """

    def __init__(self, graph: RiverGraph):
        """
        Initialize the path finder.

        Args:
            graph: RiverGraph instance to analyze
        """
        self.graph = graph

    def find_all_paths(self, start_id: int, target_id: int, max_depth: int = 10) -> List[List[int]]:
        """
        Find all paths from start to target vertex using DFS.

        Args:
            start_id: Starting vertex ID
            target_id: Target vertex ID
            max_depth: Maximum search depth to prevent infinite loops

        Returns:
            List of paths, where each path is a list of vertex IDs
        """
        paths = []

        def dfs_paths(current_id: int, target_id: int, path: List[int], visited: Set[int], depth: int):
            if depth > max_depth:
                return

            if current_id == target_id:
                paths.append(path.copy())
                return

            if current_id in visited:
                return

            visited.add(current_id)

            for neighbor_id, _ in self.graph.adjacency_list[current_id]:
                if neighbor_id not in visited:
                    path.append(neighbor_id)
                    dfs_paths(neighbor_id, target_id, path, visited, depth + 1)
                    path.pop()

            visited.remove(current_id)

        if start_id != target_id:
            dfs_paths(start_id, target_id, [start_id], set(), 0)

        return paths

    def path_to_flowlines(self, path: List[int]) -> List[int]:
        """
        Convert a path of vertex IDs to flowline indices.

        Args:
            path: List of vertex IDs representing a path

        Returns:
            List of flowline indices
        """
        flowline_indices = []

        for i in range(len(path) - 1):
            start_id = path[i]
            end_id = path[i + 1]

            for neighbor_id, flowline_idx in self.graph.adjacency_list[start_id]:
                if neighbor_id == end_id:
                    flowline_indices.append(flowline_idx)
                    break

        return flowline_indices

    def find_outlet_reachable_vertices(self, outlet_vertex_id: int) -> Set[int]:
        """
        Find all vertices that can reach the outlet using backward traversal.

        Args:
            outlet_vertex_id: ID of the outlet vertex

        Returns:
            Set of vertex IDs that can reach the outlet
        """
        reachable = set()
        queue = deque([outlet_vertex_id])
        reachable.add(outlet_vertex_id)

        # Build reverse adjacency list for backward traversal
        reverse_adjacency = defaultdict(list)
        for start_id, neighbors in self.graph.adjacency_list.items():
            for end_id, flowline_idx in neighbors:
                reverse_adjacency[end_id].append((start_id, flowline_idx))

        # Backward BFS from outlet
        while queue:
            current_id = queue.popleft()

            # Add all vertices that flow into current vertex
            for upstream_id, _ in reverse_adjacency[current_id]:
                if upstream_id not in reachable:
                    reachable.add(upstream_id)
                    queue.append(upstream_id)

        return reachable

    def get_upstream_indices(self, flowline: pyflowline) -> List[int]:
        """
        Get indices of upstream flowlines for a given flowline.

        Args:
            flowline: Flowline object

        Returns:
            List of indices of upstream flowlines
        """
        upstream_indices = []
        start_vertex_id = self.graph.vertex_to_id.get(flowline.pVertex_start)

        if start_vertex_id is not None:
            # Find flowlines that end at this flowline's start vertex
            for start_id, neighbors in self.graph.adjacency_list.items():
                for end_id, flowline_idx in neighbors:
                    if end_id == start_vertex_id and flowline_idx < len(self.graph.aFlowline):
                        upstream_indices.append(flowline_idx)

        return upstream_indices

    def get_downstream_indices(self, flowline: pyflowline) -> List[int]:
        """
        Get indices of downstream flowlines for a given flowline.

        Args:
            flowline: Flowline object

        Returns:
            List of indices of downstream flowlines
        """
        downstream_indices = []
        end_vertex_id = self.graph.vertex_to_id.get(flowline.pVertex_end)

        if end_vertex_id is not None:
            # Find flowlines that start at this flowline's end vertex
            for neighbor_id, flowline_idx in self.graph.adjacency_list[end_vertex_id]:
                if flowline_idx < len(self.graph.aFlowline):
                    downstream_indices.append(flowline_idx)

        return downstream_indices
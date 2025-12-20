"""
Network analysis and pattern detection for river networks.

This module provides algorithms for detecting patterns and features in river networks.
"""

import logging
from typing import List, Dict, Set, Tuple, DefaultDict
from collections import defaultdict, deque

from ..classes.flowline import pyflowline
from ..core.graph import RiverGraph

logger = logging.getLogger(__name__)


class NetworkAnalyzer:
    """
    Detects patterns and features in river networks.

    This class provides methods for:
    - Finding braided channels
    - Finding parallel paths
    - Detecting cycles
    - Finding linear segments
    """

    def __init__(self, graph: RiverGraph):
        """
        Initialize the network analyzer.

        Args:
            graph: RiverGraph instance to analyze
        """
        self.graph = graph

    def find_braided_channels(self) -> List[List[pyflowline]]:
        """
        Find braided channels (multiple flowlines between same vertex pair).

        Returns:
            List of braided regions, where each region is a list of braided flowlines
        """
        channel_groups: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)

        for flowline_idx, (start_id, end_id) in self.graph.aFlowline_edges.items():
            channel_groups[(start_id, end_id)].append(flowline_idx)

        # Return only groups with multiple channels (braided) as separate regions
        braided_groups = []
        for (start_id, end_id), indices in channel_groups.items():
            if len(indices) > 1:
                region_flowlines = [self.graph.aFlowline[idx] for idx in indices]
                braided_groups.append(region_flowlines)

        return braided_groups

    def find_parallel_paths(self) -> List[Dict]:
        """
        Find divergent parallel sections between vertices.

        Returns:
            List of parallel path group dictionaries, each containing:
            - 'paths': List of paths, where each path is a list of flowline indices
            - 'start_vertex': Start vertex ID (divergence point)
            - 'end_vertex': End vertex ID (reconvergence point)
        """
        return self._find_partial_parallel_sections()

    def _find_partial_parallel_sections(self) -> List[Dict]:
        """
        Find partial parallel sections within longer paths.

        This identifies divergence and reconvergence points in the network.
        """
        parallel_sections = []

        # Find all vertices that have multiple outgoing paths (divergence points)
        divergence_vertices = []
        for vertex_id in self.graph.adjacency_list.keys():
            if self.graph.out_degree[vertex_id] > 1:
                divergence_vertices.append(vertex_id)

        logger.debug(f"Found {len(divergence_vertices)} potential divergence vertices")

        # For each divergence vertex, find where the paths reconverge
        for divergence_id in divergence_vertices:
            reconvergence_id = self._find_reconvergence_points(divergence_id)

            if reconvergence_id is not None:
                # Find all paths between divergence and reconvergence points
                divergent_paths = self._find_all_paths(divergence_id, reconvergence_id, max_depth=8)

                if len(divergent_paths) > 1:
                    # Convert paths to flowline indices
                    path_flowline_groups = []
                    for path in divergent_paths:
                        path_flowlines = self._path_to_flowlines(path)
                        if path_flowlines and len(path_flowlines) > 0:
                            path_flowline_groups.append(path_flowlines)

                    # Ensure we have multiple unique divergent paths
                    if len(path_flowline_groups) > 1:
                        unique_paths = []
                        for path_flowlines in path_flowline_groups:
                            path_signature = tuple(sorted(path_flowlines))
                            if path_signature not in [tuple(sorted(up)) for up in unique_paths]:
                                unique_paths.append(path_flowlines)

                        if len(unique_paths) > 1:
                            logger.debug(f"Found {len(unique_paths)} partial parallel paths between vertices {divergence_id} -> {reconvergence_id}")
                            parallel_sections.append({
                                'paths': unique_paths,
                                'start_vertex': divergence_id,
                                'end_vertex': reconvergence_id
                            })

        return parallel_sections

    def _find_reconvergence_points(self, divergence_id: int, max_depth: int = 8) -> int:
        """
        Find vertices where paths from a divergence point reconverge.

        Args:
            divergence_id: The vertex ID where paths diverge
            max_depth: Maximum search depth

        Returns:
            Vertex ID where paths reconverge, or None
        """
        # Get all immediate downstream vertices from the divergence point
        downstream_vertices = [neighbor_id for neighbor_id, _ in self.graph.adjacency_list[divergence_id]]

        if len(downstream_vertices) < 2:
            return None

        # For each pair of downstream paths, find where they reconverge
        for i in range(len(downstream_vertices)):
            for j in range(i + 1, len(downstream_vertices)):
                path1_start = downstream_vertices[i]
                path2_start = downstream_vertices[j]

                # Find common downstream vertices reachable from both paths
                path1_reachable = self._get_reachable_vertices(path1_start, max_depth)
                path2_reachable = self._get_reachable_vertices(path2_start, max_depth)

                # Find the first identical vertex in both reachable lists
                common_vertex = self.find_first_common_vertex(path1_reachable, path2_reachable)

        return common_vertex

    def find_first_common_vertex(self, reachable_list1, reachable_list2):
        """
        Find the first vertex that appears in both lists.
        """
        set2 = set(reachable_list2)

        for vertex in reachable_list1:
            if vertex in set2:
                return vertex

        return None

    def _get_reachable_vertices(self, start_id: int, max_depth: int) -> List[int]:
        """
        Get all vertices reachable from a starting vertex within max_depth.

        Args:
            start_id: Starting vertex ID
            max_depth: Maximum search depth

        Returns:
            List of reachable vertex IDs in order
        """
        reachable = []
        queue = deque([(start_id, 0)])
        visited = set([start_id])

        while queue:
            current_id, depth = queue.popleft()

            if depth >= max_depth:
                continue

            reachable.append(current_id)

            for neighbor_id, _ in self.graph.adjacency_list[current_id]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))

        return reachable

    def detect_cycles(self) -> List[List[int]]:
        """
        Detect cycles in the network using DFS.

        Returns:
            List of cycles, where each cycle is a list of vertex IDs
        """
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs_cycle_detection(node_id: int, path: List[int]) -> bool:
            try:
                visited.add(node_id)
                rec_stack.add(node_id)

                neighbors = list(self.graph.adjacency_list[node_id])

                for neighbor_id, _ in neighbors:
                    try:
                        if neighbor_id in rec_stack:
                            try:
                                cycle_start_idx = path.index(neighbor_id)
                                cycle = path[cycle_start_idx:] + [neighbor_id]
                                cycles.append(cycle)
                                logger.debug(f"Detected cycle: {cycle}")
                            except ValueError:
                                cycle = [node_id, neighbor_id]
                                cycles.append(cycle)
                                logger.debug(f"Detected simple back-edge cycle: {cycle}")
                            except Exception as e:
                                logger.warning(f"Error processing cycle from {node_id} to {neighbor_id}: {e}")
                                cycles.append([node_id, neighbor_id])
                            return True
                        elif neighbor_id not in visited:
                            try:
                                new_path = path + [neighbor_id]
                                if dfs_cycle_detection(neighbor_id, new_path):
                                    return True
                            except RecursionError:
                                logger.error(f"Recursion limit reached during cycle detection at node {neighbor_id}")
                                if len(path) > 1:
                                    cycles.append(path + [neighbor_id])
                                return False
                            except Exception as e:
                                logger.error(f"Error in recursive cycle detection for neighbor {neighbor_id}: {e}")
                                return False
                    except Exception as e:
                        logger.error(f"Error processing neighbor {neighbor_id} from node {node_id}: {e}")
                        continue

                try:
                    rec_stack.remove(node_id)
                except KeyError:
                    logger.warning(f"Node {node_id} not found in recursion stack during removal")
                return False

            except Exception as e:
                logger.error(f"Critical error in cycle detection for node {node_id}: {e}")
                try:
                    rec_stack.discard(node_id)
                except Exception:
                    pass
                return False

        try:
            node_ids = list(self.graph.adjacency_list.keys())

            for node_id in node_ids:
                if node_id not in visited:
                    try:
                        dfs_cycle_detection(node_id, [node_id])
                    except Exception as e:
                        logger.error(f"Error starting cycle detection from node {node_id}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Critical error during cycle detection initialization: {e}")

        logger.info(f"Cycle detection completed. Found {len(cycles)} cycles.")
        return cycles

    def find_linear_segments(self) -> List[List[int]]:
        """
        Find linear segments that can be merged into single flowlines.

        Returns:
            List of linear segments, where each segment is a list of flowline indices
        """
        linear_segments = []
        processed_flowlines = set()

        # Find all vertices with degree 2 (one in, one out)
        merge_vertices = set()
        for vertex_id in self.graph.id_to_vertex.keys():
            if self.graph.in_degree[vertex_id] == 1 and self.graph.out_degree[vertex_id] == 1:
                merge_vertices.add(vertex_id)

        logger.debug(f"Found {len(merge_vertices)} degree-2 vertices for potential merging")

        # For each flowline, try to build a complete linear segment
        for flowline_idx, (start_id, end_id) in self.graph.aFlowline_edges.items():
            if flowline_idx in processed_flowlines:
                continue

            segment = self._build_bidirectional_segment(flowline_idx, start_id, end_id, merge_vertices, processed_flowlines)

            if len(segment) > 1:
                linear_segments.append(segment)
                processed_flowlines.update(segment)
                logger.debug(f"Found linear segment with {len(segment)} flowlines: {segment}")

        logger.info(f"Found {len(linear_segments)} linear segments for potential merging")
        return linear_segments

    def _build_bidirectional_segment(self, start_flowline_idx: int, start_vertex_id: int,
                                   end_vertex_id: int, merge_vertices: set,
                                   processed_flowlines: set) -> List[int]:
        """
        Build a complete linear segment by extending both upstream and downstream.

        Args:
            start_flowline_idx: Index of the flowline to start building from
            start_vertex_id: Start vertex ID of the starting flowline
            end_vertex_id: End vertex ID of the starting flowline
            merge_vertices: Set of degree-2 vertices that can be merged
            processed_flowlines: Set of already processed flowline indices

        Returns:
            List of flowline indices in upstream-to-downstream order
        """
        upstream_segment = []
        downstream_segment = [start_flowline_idx]

        # Extend upstream from start vertex
        current_start = start_vertex_id
        while current_start in merge_vertices:
            upstream_flowlines = []
            for fl_idx, (fl_start, fl_end) in self.graph.aFlowline_edges.items():
                if fl_end == current_start and fl_idx not in processed_flowlines:
                    upstream_flowlines.append(fl_idx)

            if len(upstream_flowlines) == 1:
                upstream_flowline_idx = upstream_flowlines[0]
                upstream_segment.insert(0, upstream_flowline_idx)
                current_start, _ = self.graph.aFlowline_edges[upstream_flowline_idx]
            else:
                break

        # Extend downstream from end vertex
        current_end = end_vertex_id
        while current_end in merge_vertices:
            downstream_flowlines = [fl_idx for neighbor_id, fl_idx in self.graph.adjacency_list[current_end]]

            if len(downstream_flowlines) == 1:
                downstream_flowline_idx = downstream_flowlines[0]
                if downstream_flowline_idx not in processed_flowlines:
                    downstream_segment.append(downstream_flowline_idx)
                    if downstream_flowline_idx in self.graph.aFlowline_edges:
                        _, current_end = self.graph.aFlowline_edges[downstream_flowline_idx]
                    else:
                        break
                else:
                    break
            else:
                break

        return upstream_segment + downstream_segment

    def _find_all_paths(self, start_id: int, target_id: int, max_depth: int = 10) -> List[List[int]]:
        """
        Find all paths from start to target vertex using DFS.

        Args:
            start_id: Starting vertex ID
            target_id: Target vertex ID
            max_depth: Maximum search depth

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

    def _path_to_flowlines(self, path: List[int]) -> List[int]:
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
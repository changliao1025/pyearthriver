"""
Network simplification operations for river networks.

This module provides operations that simplify the network by removing elements.
"""

import logging
import os
from typing import List, Set, Tuple, Optional

from ..classes.flowline import pyflowline
from ..classes.vertex import pyvertex
from ..core.graph import RiverGraph
from ..analysis.detection import NetworkAnalyzer
from ..analysis.pathfinding import PathFinder
from ..formats.export_flowline import export_flowline_to_geojson

logger = logging.getLogger(__name__)


class NetworkSimplifier:
    """
    Handles all network simplification operations.

    This class provides methods for:
    - Removing disconnected flowlines
    - Removing braided channels
    - Removing parallel rivers
    - Removing cycles
    - Removing small rivers
    - Removing duplicate flowlines
    """

    def __init__(self, graph: RiverGraph, analyzer: NetworkAnalyzer, pathfinder: PathFinder):
        """
        Initialize the network simplifier.

        Args:
            graph: RiverGraph instance to simplify
            analyzer: NetworkAnalyzer for pattern detection
            pathfinder: PathFinder for path analysis
        """
        self.graph = graph
        self.analyzer = analyzer
        self.pathfinder = pathfinder

    def remove_disconnected_flowlines(self, pVertex_outlet: Optional[pyvertex] = None) -> List[pyflowline]:
        """
        Remove flowlines that don't flow out to the specified outlet vertex.

        Args:
            pVertex_outlet: Optional outlet vertex

        Returns:
            List of flowlines that are connected to the outlet vertex
        """
        flowlines = self.graph.aFlowline

        pVertex_target = pVertex_outlet if pVertex_outlet is not None else self.graph.pVertex_outlet

        if pVertex_target is None:
            logger.warning("No outlet vertex provided and none stored during initialization")
            return flowlines

        outlet_vertex_id = None
        if pVertex_target == self.graph.pVertex_outlet and self.graph.pVertex_outlet_id is not None:
            outlet_vertex_id = self.graph.pVertex_outlet_id
        else:
            for vertex_id, vertex in self.graph.id_to_vertex.items():
                if vertex == pVertex_target:
                    outlet_vertex_id = vertex_id
                    break

        if outlet_vertex_id is None:
            logger.warning("Outlet vertex not found in network graph")
            return flowlines

        logger.info(f"Removing disconnected flowlines using outlet vertex {outlet_vertex_id}")

        outlet_reachable_vertices = self.pathfinder.find_outlet_reachable_vertices(outlet_vertex_id)

        reachable_flowlines = set()
        for flowline_idx, (start_id, end_id) in self.graph.aFlowline_edges.items():
            if start_id in outlet_reachable_vertices and end_id in outlet_reachable_vertices:
                reachable_flowlines.add(flowline_idx)

        connected_flowlines = []
        disconnected_count = 0

        for i, flowline in enumerate(flowlines):
            if i in reachable_flowlines:
                connected_flowlines.append(flowline)
            else:
                disconnected_count += 1
                logger.debug(f"Removing disconnected flowline {i}")

        logger.info(f"Removed {disconnected_count} disconnected flowlines, kept {len(connected_flowlines)} connected flowlines")

        self.graph.update_graph_flowlines(connected_flowlines)
        return connected_flowlines

    def remove_braided_river(self) -> List[pyflowline]:
        """
        Remove braided channels from the river network.

        Returns:
            List of flowlines with braided channels removed
        """
        braided_regions = self.analyzer.find_braided_channels()
        if not braided_regions:
            logger.info("No braided channels found in the network")
            return self.graph.aFlowline

        logger.info(f"Found {len(braided_regions)} braided regions to resolve")

        flowlines_to_remove = set()

        for region_flowlines in braided_regions:
            if len(region_flowlines) <= 1:
                continue

            shortest_flowline = min(region_flowlines, key=lambda fl: getattr(fl, 'dLength', float('inf')))

            for flowline in region_flowlines:
                if flowline != shortest_flowline:
                    flowline_idx = self.graph.aFlowline.index(flowline)
                    flowlines_to_remove.add(flowline_idx)
                    logger.debug(f"Marking flowline {flowline_idx} for removal (braided channel)")

        simplified_flowlines = [
            flowline for idx, flowline in enumerate(self.graph.aFlowline)
            if idx not in flowlines_to_remove
        ]

        logger.info(f"Removed {len(flowlines_to_remove)} braided flowlines, "
                    f"resulting in {len(simplified_flowlines)} flowlines")

        self.graph.update_graph_flowlines(simplified_flowlines)
        return simplified_flowlines

    def remove_parallel_river(self) -> List[pyflowline]:
        """
        Remove parallel rivers using graph-based approach.

        Returns:
            List of flowlines with parallel paths resolved
        """
        if not self.graph.aFlowline:
            logger.warning('No flowlines available for parallel river removal')
            return []

        if len(self.graph.aFlowline) <= 1:
            logger.debug("Skipping parallel path resolution: insufficient flowlines")
            return self.graph.aFlowline.copy()

        logger.info(f"Removing parallel rivers from {len(self.graph.aFlowline)} flowlines")

        try:
            parallel_groups = self.analyzer.find_parallel_paths()

            if not parallel_groups:
                logger.debug("No parallel paths found")
                return self.graph.aFlowline.copy()

            logger.info(f"Found {len(parallel_groups)} parallel path groups")
        except Exception as e:
            logger.error(f"Error finding parallel paths: {e}")
            return self.graph.aFlowline.copy()

        flowlines_to_remove = set()

        for group in parallel_groups:
            paths = group['paths']

            if len(paths) <= 1:
                continue

            path_scores = []
            for path_idx, path_flowlines in enumerate(paths):
                path_length = 0.0
                valid_flowlines = 0

                for flowline_idx in path_flowlines:
                    if flowline_idx >= len(self.graph.aFlowline):
                        logger.warning(f"Invalid flowline index {flowline_idx}, skipping")
                        continue

                    flowline = self.graph.aFlowline[flowline_idx]
                    valid_flowlines += 1

                    length = getattr(flowline, 'dLength', 0.0)
                    length = length if length > 0 else 0.0
                    path_length += length

                if valid_flowlines == 0:
                    continue

                path_score = path_length if path_length > 0 else 1.0
                path_scores.append((path_score, path_idx, path_flowlines))

            if not path_scores:
                continue

            path_scores.sort(reverse=True)

            for _, path_idx, path_flowlines in path_scores:
                path_flowlines = list(path_flowlines)
                for flowline_idx in path_flowlines[:-1]:
                    flowline = self.graph.aFlowline[flowline_idx]
                    end_vertex_id = self.graph.vertex_to_id.get(flowline.pVertex_end)
                    if end_vertex_id is not None and self.graph.in_degree[end_vertex_id] > 1:
                        flowlines_to_remove.add(flowline_idx)
                        logger.debug(f"Removing flowline {flowline_idx} as its end point is a confluence")
                        break

        try:
            result = [self.graph.aFlowline[i] for i in range(len(self.graph.aFlowline)) if i not in flowlines_to_remove]
            logger.info(f"Removed {len(flowlines_to_remove)} flowlines to resolve parallel paths")

            if len(result) == 0 and len(self.graph.aFlowline) > 0:
                logger.warning("All flowlines were removed during parallel path resolution, returning original")
                return self.graph.aFlowline.copy()

            self.graph.update_graph_flowlines(result)
            return result
        except Exception as e:
            logger.error(f"Error filtering flowlines during parallel path resolution: {e}")
            return self.graph.aFlowline.copy()

    def remove_cycle(self) -> List[pyflowline]:
        """
        Detect and break cycles in the network by removing lowest priority flowlines.

        Returns:
            List of flowlines with cycles broken
        """
        if not self.graph.aFlowline:
            logger.warning('No flowlines available for cycle removal')
            return []

        if len(self.graph.aFlowline) <= 1:
            logger.debug("Skipping cycle removal: insufficient flowlines")
            return self.graph.aFlowline.copy()

        logger.info(f"Removing cycles from {len(self.graph.aFlowline)} flowlines")

        try:
            cycles = self.analyzer.detect_cycles()

            if not cycles:
                logger.debug("No cycles detected in network")
                return self.graph.aFlowline.copy()

            logger.info(f"Detected {len(cycles)} cycles in network")
        except Exception as e:
            logger.error(f"Error detecting cycles: {e}")
            return self.graph.aFlowline.copy()

        flowlines_to_remove = set()

        for cycle_vertices in cycles:
            if len(cycle_vertices) < 3:
                continue

            cycle_flowlines = []
            for i in range(len(cycle_vertices) - 1):
                start_vertex_id = cycle_vertices[i]
                end_vertex_id = cycle_vertices[i + 1]

                for flowline_idx, (s_id, e_id) in self.graph.aFlowline_edges.items():
                    if s_id == start_vertex_id and e_id == end_vertex_id:
                        if flowline_idx < len(self.graph.aFlowline):
                            cycle_flowlines.append((flowline_idx, self.graph.aFlowline[flowline_idx]))

            if cycle_flowlines:
                def cycle_priority(flowline_data) -> Tuple:
                    flowline_idx, flowline = flowline_data
                    stream_order = getattr(flowline, 'iStream_order', -1)
                    stream_order = stream_order if stream_order > 0 else 1

                    drainage_area = getattr(flowline, 'dDrainage_area', 0.0)
                    drainage_area = drainage_area if drainage_area > 0 else 0.0

                    length = getattr(flowline, 'dLength', 0.0)
                    length = length if length > 0 else 0.0

                    return (stream_order, drainage_area, length)

                worst_flowline_idx, worst_flowline = min(cycle_flowlines, key=cycle_priority)
                flowlines_to_remove.add(worst_flowline_idx)
                logger.debug(f"Removing flowline {worst_flowline_idx} to break cycle")

        try:
            result = [self.graph.aFlowline[i] for i in range(len(self.graph.aFlowline)) if i not in flowlines_to_remove]
            logger.info(f"Removed {len(flowlines_to_remove)} flowlines to break cycles")

            if len(result) == 0 and len(self.graph.aFlowline) > 0:
                logger.warning("All flowlines were removed during cycle removal, returning original")
                return self.graph.aFlowline.copy()

            self.graph.update_graph_flowlines(result)
            return result
        except Exception as e:
            logger.error(f"Error filtering flowlines during cycle removal: {e}")
            return self.graph.aFlowline.copy()

    def remove_small_river(self, dThreshold_small_river, nIterations=3, iFlag_debug=0, sWorkspace_output_basin=None):
        """
        Remove small rivers iteratively.

        Args:
            dThreshold_small_river: Length threshold for small river removal
            nIterations: Number of iterations
            iFlag_debug: Debug flag for output files
            sWorkspace_output_basin: Output workspace for debug files

        Returns:
            Updated flowlines after iterative small river removal
        """
        aFlowline_current = self.graph.aFlowline.copy()

        for i in range(nIterations):
            sStep = "{:02d}".format(i+1)
            print(f'Iteration {sStep}: removing small rivers with threshold {dThreshold_small_river}')

            aFlowline_filtered = self._remove_small_river_step(aFlowline_current, dThreshold_small_river)

            if iFlag_debug == 1 and sWorkspace_output_basin is not None:
                sFilename_out = f'flowline_large_{sStep}_before_intersect.geojson'
                sFilename_out = os.path.join(sWorkspace_output_basin, sFilename_out)
                export_flowline_to_geojson(aFlowline_filtered, sFilename_out)

            if len(aFlowline_filtered) == len(aFlowline_current):
                print(f'No small rivers found in iteration {sStep}, stopping early.')
                break
            else:
                self.graph.update_graph_flowlines(aFlowline_filtered)
                aFlowline_current = aFlowline_filtered

                if len(aFlowline_current) == 1:
                    break

        return aFlowline_current

    def remove_duplicate_flowlines(self, iFlag_direction_insensitive: bool = False) -> List[pyflowline]:
        """
        Remove duplicate flowlines from the network.

        Args:
            iFlag_direction_insensitive: If True, flowlines with opposite directions are considered duplicates

        Returns:
            List of flowlines with duplicates removed
        """
        if iFlag_direction_insensitive:
            return self._remove_duplicate_flowlines_direction_insensitive()
        else:
            return self._remove_duplicate_flowlines_direction_sensitive()

    def _remove_duplicate_flowlines_direction_sensitive(self) -> List[pyflowline]:
        """Remove duplicate flowlines where direction matters."""
        seen_edges = set()
        unique_flowlines = []
        duplicates_count = 0

        for idx, flowline in enumerate(self.graph.aFlowline):
            start_id = self.graph.vertex_to_id.get(flowline.pVertex_start)
            end_id = self.graph.vertex_to_id.get(flowline.pVertex_end)

            if start_id is None or end_id is None:
                logger.warning(f"Flowline {idx} has unknown vertices, skipping duplicate check")
                unique_flowlines.append(flowline)
                continue

            edge_key = (start_id, end_id)

            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                unique_flowlines.append(flowline)
            else:
                duplicates_count += 1
                logger.debug(f"Removing duplicate flowline {idx} (direction-sensitive)")

        logger.info(f"Removed {duplicates_count} duplicate flowlines (direction-sensitive), "
                    f"resulting in {len(unique_flowlines)} unique flowlines")

        self.graph.update_graph_flowlines(unique_flowlines)
        return unique_flowlines

    def _remove_duplicate_flowlines_direction_insensitive(self) -> List[pyflowline]:
        """Remove duplicate flowlines where direction doesn't matter."""
        from collections import defaultdict

        edge_groups = defaultdict(list)

        for idx, flowline in enumerate(self.graph.aFlowline):
            start_id = self.graph.vertex_to_id.get(flowline.pVertex_start)
            end_id = self.graph.vertex_to_id.get(flowline.pVertex_end)

            if start_id is None or end_id is None:
                logger.warning(f"Flowline {idx} has unknown vertices, skipping duplicate check")
                continue

            edge_key = tuple(sorted([start_id, end_id]))
            edge_groups[edge_key].append((idx, flowline, (start_id, end_id)))

        unique_flowlines = []
        duplicates_count = 0
        opposite_pairs_count = 0

        for edge_key, flowline_group in edge_groups.items():
            if len(flowline_group) == 1:
                _, flowline, _ = flowline_group[0]
                unique_flowlines.append(flowline)
            else:
                directions = set()
                for idx, flowline, (start_id, end_id) in flowline_group:
                    directions.add((start_id, end_id))

                has_opposite_directions = False
                direction_list = list(directions)
                for i in range(len(direction_list)):
                    for j in range(i + 1, len(direction_list)):
                        dir1 = direction_list[i]
                        dir2 = direction_list[j]
                        if dir1 == (dir2[1], dir2[0]):
                            has_opposite_directions = True
                            break
                    if has_opposite_directions:
                        break

                if has_opposite_directions:
                    if self.graph.pVertex_outlet_id is not None:
                        flowline_path_lengths = []
                        for idx, flowline, (start_id, end_id) in flowline_group:
                            paths_to_outlet = self.pathfinder.find_all_paths(end_id, self.graph.pVertex_outlet_id, max_depth=1000)
                            if paths_to_outlet:
                                shortest_path_length = min(len(path) for path in paths_to_outlet)
                                flowline_path_lengths.append((shortest_path_length, idx, flowline, (start_id, end_id)))
                            else:
                                flowline_path_lengths.append((float('inf'), idx, flowline, (start_id, end_id)))

                        flowline_path_lengths.sort(key=lambda x: x[0])
                        shortest_path_length, closest_idx, closest_flowline, _ = flowline_path_lengths[0]
                        unique_flowlines.append(closest_flowline)

                        for path_length, idx, flowline, (start_id, end_id) in flowline_path_lengths[1:]:
                            opposite_pairs_count += 1
                            logger.debug(f"Removing flowline {idx} (longer path to outlet)")
                    else:
                        opposite_pairs_count += len(flowline_group)
                        for idx, flowline, (start_id, end_id) in flowline_group:
                            logger.debug(f"Removing flowline {idx} (opposite direction pair, no outlet)")
                else:
                    _, first_flowline, _ = flowline_group[0]
                    unique_flowlines.append(first_flowline)
                    duplicates_count += len(flowline_group) - 1
                    for idx, flowline, _ in flowline_group[1:]:
                        logger.debug(f"Removing duplicate flowline {idx} (same direction)")

        for idx, flowline in enumerate(self.graph.aFlowline):
            start_id = self.graph.vertex_to_id.get(flowline.pVertex_start)
            end_id = self.graph.vertex_to_id.get(flowline.pVertex_end)
            if start_id is None or end_id is None:
                unique_flowlines.append(flowline)

        total_removed = duplicates_count + opposite_pairs_count
        logger.info(f"Removed {total_removed} flowlines (direction-insensitive)")

        self.graph.update_graph_flowlines(unique_flowlines)
        return unique_flowlines

    def _remove_small_river_step(self, flowlines: List[pyflowline], threshold: float) -> List[pyflowline]:
        """
        Remove small rivers based on length threshold.

        Args:
            flowlines: List of flowlines to filter
            threshold: Length threshold for removal

        Returns:
            List of flowlines with small rivers removed
        """
        filtered_flowlines = []

        for flowline in flowlines:
            length = getattr(flowline, 'dLength', 0.0)
            if length >= threshold:
                filtered_flowlines.append(flowline)
            else:
                logger.debug(f"Removing small river with length {length}")

        logger.info(f"Removed {len(flowlines) - len(filtered_flowlines)} small rivers")
        return filtered_flowlines
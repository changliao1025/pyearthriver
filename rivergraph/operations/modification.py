"""
Network modification operations for river networks.

This module provides operations that modify network structure.
"""

import logging
from typing import List, Optional
import numpy as np

from ..classes.flowline import pyflowline
from ..classes.vertex import pyvertex
from ..classes.edge import pyedge
from ..core.graph import RiverGraph
from ..analysis.detection import NetworkAnalyzer
from ..formats.find_index_in_list import find_vertex_on_edge

logger = logging.getLogger(__name__)


class NetworkModifier:
    """
    Handles network modification operations.

    This class provides methods for:
    - Splitting flowlines at vertices
    - Merging linear flowline segments
    """

    def __init__(self, graph: RiverGraph, analyzer: NetworkAnalyzer):
        """
        Initialize the network modifier.

        Args:
            graph: RiverGraph instance to modify
            analyzer: NetworkAnalyzer for pattern detection
        """
        self.graph = graph
        self.analyzer = analyzer

    def split_flowline(self, aVertex_in: Optional[List[pyvertex]] = None,
                        iFlag_intersect = None, iFlag_use_id=None) -> List[pyflowline]:
        """
        Split flowline based on the intersection with a list of vertex.

        Args:
            aVertex_in: list of vertex (optional)
            iFlag_intersect: 1: a vertex maybe on a line, but it is not a vertex of the line
            iFlag_use_id: 1: use flowline ID

        Returns:
            aFlowline_out: list of flowline
        """
        iFlag_graph_update = 0
        aFlowline_in = self.graph.aFlowline
        aFlowline_out = list()
        nFlowline = len(aFlowline_in)

        if aVertex_in is None:
            aVertex_in = self.graph.aVertex

        aVertex_in_set = set(aVertex_in)

        for i in range(nFlowline):
            pFlowline = aFlowline_in[i]
            iStream_order = pFlowline.iStream_order
            iFlag_dam = pFlowline.iFlag_dam
            nEdge= pFlowline.nEdge
            iPart = 0
            aVertex  = list()
            aVertex_all = list()

            for j in range(nEdge):
                pEdge=pFlowline.aEdge[j]
                pVertex = pEdge.pVertex_start
                aVertex_all.append(pVertex)

                if aVertex_in and pVertex in aVertex_in_set:
                    iPart += 1
                    aVertex.append(pVertex)

                if iFlag_intersect is not None:
                    if iFlag_use_id is not None:
                        aDistance = list()
                        iFlag_exist=0
                        aVertex_dummy=list()
                        aIndex=list()
                        npoint = 0
                        for k in range(len(aVertex_in)) if aVertex_in else []:
                            pVertex0 = aVertex_in[k]
                            if pVertex0.lFlowlineID == pFlowline.lFlowlineID:
                                iFlag_exist =1
                                distance  = pEdge.pVertex_start.calculate_distance(pVertex0)
                                if distance == 0:
                                    continue
                                iPart = iPart + 1
                                aDistance.append(distance)
                                aVertex_dummy.append(pVertex0)
                                aIndex.append(k)
                                npoint= npoint+ 1
                            else:
                                pass

                        if aDistance:
                            aIndex_order = np.argsort(aDistance)
                            for k in aIndex_order:
                                pVertex_dummy = aVertex_in[aIndex[k]]
                                aVertex.append(pVertex_dummy)
                                aVertex_all.append(pVertex_dummy)

                    else:
                        iFlag_exist, npoint, aIndex = find_vertex_on_edge(aVertex_in, pEdge) if aVertex_in else (0, 0, [])
                        if iFlag_exist==1:
                            for m in range(npoint):
                                pVertex_dummy = aVertex_in[aIndex[m]]
                                iPart = iPart + 1
                                aVertex.append(pVertex_dummy)
                                aVertex_all.append(pVertex_dummy)

            # The last ending vertex
            pVertex = pFlowline.pVertex_end
            aVertex_all.append(pVertex)

            if aVertex_in and pVertex in aVertex_in_set:
                iPart = iPart + 1
                aVertex.append(pVertex)

            if iPart == 0 :
                print('Something is wrong')
            else:
                if iPart ==1:
                    if iFlag_use_id is not None:
                        pass
                    pass
                else:
                    if iPart >=2:
                        nLine = iPart-1
                        aVertex_index = [aVertex_all.index(pVertex) for pVertex in aVertex if pVertex in aVertex_all]

                        for k in range(nLine):
                            t = aVertex_index[k]
                            s = aVertex_index[k+1]
                            if s!=t:
                                aEdge = [pyedge.create(aVertex_all[l], aVertex_all[l+1]) for l in range(t, s)]
                                aEdge = [edge for edge in aEdge if edge is not None]
                                pFlowline1 = pyflowline(aEdge)
                                pFlowline1.iStream_order = iStream_order
                                pFlowline1.iFlag_dam = iFlag_dam
                                aFlowline_out.append(pFlowline1)
                                nEdge_new = pFlowline1.nEdge
                                if nEdge_new != nEdge:
                                    iFlag_graph_update = 1
                            pass
                        pass
                    pass

        if iFlag_graph_update == 1:
            self.graph.update_graph_flowlines(aFlowline_out)
        return aFlowline_out

    def merge_flowline(self) -> List[pyflowline]:
        """
        Merge linear segments of flowlines into single flowlines.

        Returns:
            List of flowlines with linear segments merged
        """
        flowlines = self.graph.aFlowline

        try:
            linear_segments = self.analyzer.find_linear_segments()

            if not linear_segments:
                logger.info("No linear segments found for merging")
                return self.graph.aFlowline

            logger.info(f"Found {len(linear_segments)} linear segments to merge")

            merged_flowlines = []
            processed_indices = set()

            for segment in linear_segments:
                if len(segment) < 2:
                    continue

                try:
                    segment_flowlines = [flowlines[idx] for idx in segment if idx < len(flowlines)]

                    if len(segment_flowlines) < 2:
                        continue

                    merged_flowline = self._merge_flowline_segment(segment_flowlines)
                    if merged_flowline is not None:
                        merged_flowlines.append(merged_flowline)
                        processed_indices.update(segment)
                        logger.debug(f"Merged {len(segment_flowlines)} flowlines into single flowline")

                except Exception as e:
                    logger.error(f"Error merging segment {segment}: {e}")
                    continue

            for i, flowline in enumerate(flowlines):
                if i not in processed_indices:
                    merged_flowlines.append(flowline)

            logger.info(f"Merged {len(linear_segments)} segments, "
                       f"resulting in {len(merged_flowlines)} flowlines "
                       f"(reduced from {len(flowlines)})")

            self.graph.update_graph_flowlines(merged_flowlines)
            return merged_flowlines

        except Exception as e:
            logger.error(f"Error during linear segment merging: {e}")
            return flowlines

    def _merge_flowline_segment(self, segment_flowlines: List[pyflowline]) -> Optional[pyflowline]:
        """
        Merge a segment of flowlines into a single flowline.

        Args:
            segment_flowlines: List of flowlines to merge

        Returns:
            Merged flowline or None if merge failed
        """
        if not segment_flowlines:
            return None

        try:
            # Merge using the built-in api in the reverse order
            merged_flowline = segment_flowlines[-1]
            for flowline in segment_flowlines[-2::-1]:
                flowline_up = flowline
                merged_flowline = merged_flowline.merge_upstream(flowline_up)

            return merged_flowline

        except Exception as e:
            logger.error(f"Error merging flowline segment: {e}")
            return None
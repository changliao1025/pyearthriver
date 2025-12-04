"""
Stream analysis module for pyrivergraph.

This module provides stream ordering algorithms, hydrological calculations,
and stream network analysis capabilities including Strahler and Shreve
ordering methods, flow accumulation, and watershed delineation.
"""

import time
import logging
import math
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from collections import defaultdict, deque
from enum import Enum

from .state_management import GraphStateManager


logger = logging.getLogger(__name__)


class StreamOrderMethod(Enum):
    """Available stream ordering methods."""
    STRAHLER = "strahler"
    SHREVE = "shreve"
    HORTON = "horton"


class StreamClassification(Enum):
    """Stream classification categories."""
    HEADWATER = "headwater"
    TRIBUTARY = "tributary"
    MAIN_STEM = "main_stem"
    OUTLET = "outlet"



class StreamAnalyzer:
    """
    Comprehensive stream analyzer for river graphs with stream ordering,
    classification, and hydrological analysis capabilities.
    """

    def __init__(self, graph_instance: Any, state_manager: Optional[GraphStateManager] = None):
        """
        Initialize the stream analyzer.

        Args:
            graph_instance: The pyrivergraph instance to analyze
            state_manager: Optional state manager for tracking analysis history
        """
        self.graph = graph_instance
        self.state_manager = state_manager
        self.tolerance = 1e-6  # Distance tolerance for geometric operations

        # Cached analysis results
        self._stream_order_cache = {}
        self._flow_accumulation_cache = {}
        self._topology_cache = None

        logger.debug("Initialized StreamAnalyzer")

    def calculate_stream_order(self,
                             method: StreamOrderMethod = StreamOrderMethod.STRAHLER,
                             force_recalculate: bool = False) -> StreamAnalysisResult:
        """
        Calculate stream order using the specified method.

        Args:
            method: Stream ordering method to use
            force_recalculate: Whether to force recalculation even if cached

        Returns:
            StreamAnalysisResult with stream orders and analysis
        """
        start_time = time.time()
        result = StreamAnalysisResult()
        result.method_used = method

        try:
            # Check cache first
            cache_key = method.value
            if not force_recalculate and cache_key in self._stream_order_cache:
                cached_result = self._stream_order_cache[cache_key]
                result.stream_orders = cached_result.copy()
                result.execution_time = time.time() - start_time
                logger.debug(f"Using cached stream orders for method {method.value}")
                return result

            logger.info(f"Calculating stream order using {method.value} method")

            if method == StreamOrderMethod.STRAHLER:
                orders = self._calculate_strahler_order()
            elif method == StreamOrderMethod.SHREVE:
                orders = self._calculate_shreve_order()
            elif method == StreamOrderMethod.HORTON:
                orders = self._calculate_horton_order()
            else:
                raise ValueError(f"Unknown stream ordering method: {method}")

            # Store results
            for flowline_id, order in orders.items():
                result.add_stream_order(flowline_id, order)

            # Cache results
            self._stream_order_cache[cache_key] = orders.copy()

            # Update flowline objects with stream orders
            self._update_flowline_stream_orders(orders)

            result.execution_time = time.time() - start_time
            logger.info(f"Stream order calculation completed in {result.execution_time:.3f}s")

            return result

        except Exception as e:
            logger.error(f"Stream order calculation failed: {e}")
            result.execution_time = time.time() - start_time
            return result


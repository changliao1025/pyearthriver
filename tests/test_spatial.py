"""
Unit tests for spatial analysis algorithms in pyearthriver:
- find_vertex_in_list
- find_vertex_on_edge
- Pure-Python vs Cython fallback behavior
"""

import unittest
from unittest.mock import patch
from tests import conftest  # Ensures environment setup before imports

from pyearthriver.classes.vertex import pyvertex
from pyearthriver.classes.edge import pyedge
import pyearthriver.algorithms.auxiliary.find_vertex_in_list as py_find_v
import pyearthriver.algorithms.auxiliary.find_index_in_list as py_find_idx


class TestFindVertexInList(unittest.TestCase):
    """Test suite for find_vertex_in_list (pure-Python implementation)."""

    def setUp(self):
        self.v1 = pyvertex({"dLongitude_degree": -77.0, "dLatitude_degree": 38.0, "lVertexID": 1})
        self.v2 = pyvertex({"dLongitude_degree": -77.1, "dLatitude_degree": 38.1, "lVertexID": 2})
        self.v3 = pyvertex({"dLongitude_degree": -77.2, "dLatitude_degree": 38.2, "lVertexID": 3})
        self.vertex_list = [self.v1, self.v2, self.v3]

    def test_find_vertex_exact_match(self):
        """Test finding a vertex with exact coordinates."""
        target = pyvertex({"dLongitude_degree": -77.1, "dLatitude_degree": 38.1, "lVertexID": 99})
        flag, index = py_find_v.find_vertex_in_list(self.vertex_list, target)
        self.assertEqual(flag, 1)
        self.assertEqual(index, 1)

    def test_find_vertex_within_tolerance(self):
        """Test finding a vertex within coordinate tolerance."""
        # 1e-7 difference is within 1e-6 default tolerance
        target = pyvertex({"dLongitude_degree": -77.0 + 1e-7, "dLatitude_degree": 38.0 - 1e-7})
        flag, index = py_find_v.find_vertex_in_list(self.vertex_list, target, dThreshold_in=1e-5)
        self.assertEqual(flag, 1)
        self.assertEqual(index, 0)

    def test_find_vertex_not_found(self):
        """Test query for a vertex not in the list."""
        target = pyvertex({"dLongitude_degree": -80.0, "dLatitude_degree": 35.0})
        flag, index = py_find_v.find_vertex_in_list(self.vertex_list, target)
        self.assertEqual(flag, 0)
        self.assertEqual(index, -1)

    def test_find_vertex_empty_list(self):
        """Test querying against an empty vertex list."""
        target = pyvertex({"dLongitude_degree": -77.0, "dLatitude_degree": 38.0})
        flag, index = py_find_v.find_vertex_in_list([], target)
        self.assertEqual(flag, 0)
        self.assertEqual(index, -1)


class TestFindVertexOnEdge(unittest.TestCase):
    """Test suite for find_vertex_on_edge."""

    def setUp(self):
        self.v_start = pyvertex({"dLongitude_degree": -77.0, "dLatitude_degree": 38.0})
        self.v_end = pyvertex({"dLongitude_degree": -77.2, "dLatitude_degree": 38.2})
        self.edge = pyedge(self.v_start, self.v_end)

    def test_vertex_on_edge_midpoint(self):
        """Test detecting a vertex placed at the midpoint of an edge."""
        midpoint = self.edge.get_midpoint()
        candidates = [midpoint]
        flag, npoint, order = py_find_idx.find_vertex_on_edge(candidates, self.edge)
        self.assertEqual(flag, 1)
        self.assertEqual(npoint, 1)
        self.assertEqual(order, [0])

    def test_multiple_vertices_on_edge_ordered(self):
        """Test multiple vertices on edge are returned ordered by distance from start."""
        # Midpoint (~50% along edge)
        v_mid = self.edge.get_midpoint()
        # Point near start (~25% along edge)
        v_quarter = pyvertex({"dLongitude_degree": -77.05, "dLatitude_degree": 38.05})
        # Point near end (~75% along edge)
        v_three_quarter = pyvertex({"dLongitude_degree": -77.15, "dLatitude_degree": 38.15})

        candidates = [v_three_quarter, v_quarter, v_mid]
        flag, npoint, order = py_find_idx.find_vertex_on_edge(candidates, self.edge)
        self.assertEqual(flag, 1)
        self.assertEqual(npoint, 3)
        # order should be index of quarter (1), mid (2), three_quarter (0)
        self.assertEqual(order, [1, 2, 0])

    def test_vertex_outside_edge(self):
        """Test querying vertices located outside the edge bounding box or line."""
        v_outside = pyvertex({"dLongitude_degree": -79.0, "dLatitude_degree": 35.0})
        candidates = [v_outside]
        flag, npoint, order = py_find_idx.find_vertex_on_edge(candidates, self.edge)
        self.assertEqual(flag, 0)
        self.assertEqual(npoint, 0)
        self.assertEqual(order, [])

    def test_empty_candidates_on_edge(self):
        """Test finding vertices on edge with an empty candidates list."""
        flag, npoint, order = py_find_idx.find_vertex_on_edge([], self.edge)
        self.assertEqual(flag, 0)
        self.assertEqual(npoint, 0)
        self.assertEqual(order, [])


class TestCythonFallbackBehavior(unittest.TestCase):
    """Test pure-Python vs Cython fallback mechanisms."""

    def test_pure_python_modules_exist(self):
        """Verify that pure-Python fallback implementations are always importable."""
        self.assertTrue(callable(py_find_v.find_vertex_in_list))
        self.assertTrue(callable(py_find_idx.find_vertex_on_edge))

    def test_fallback_when_cython_missing(self):
        """Simulate Cython import failure and verify graceful fallback to pure Python."""
        with patch.dict("sys.modules", {"pyearthriver.algorithms.cython.kernel": None}):
            # Attempt fallback resolution as done in rivergraph.py and find_index_in_list.py
            try:
                from pyearthriver.algorithms.cython.kernel import find_vertex_on_edge
            except (ImportError, ModuleNotFoundError):
                from pyearthriver.algorithms.auxiliary.find_index_in_list import (
                    find_vertex_on_edge as fallback_func,
                )
            self.assertTrue(callable(fallback_func))
            self.assertEqual(fallback_func, py_find_idx.find_vertex_on_edge)

    def test_cython_consistency_if_available(self):
        """If Cython compiled extension is available, test result consistency with pure Python."""
        try:
            from pyearthriver.algorithms.cython.kernel import (
                find_vertex_in_list as cy_find_v,
            )
            have_cython = True
        except (ImportError, ModuleNotFoundError):
            have_cython = False

        if not have_cython:
            self.skipTest("Cython kernel extension not available in this environment")

        v1 = pyvertex({"dLongitude_degree": -77.0, "dLatitude_degree": 38.0, "lVertexID": 1})
        v2 = pyvertex({"dLongitude_degree": -77.1, "dLatitude_degree": 38.1, "lVertexID": 2})
        target = pyvertex({"dLongitude_degree": -77.0, "dLatitude_degree": 38.0})

        py_res = py_find_v.find_vertex_in_list([v1, v2], target)
        cy_res = cy_find_v([v1, v2], target)
        self.assertEqual(py_res, cy_res)


if __name__ == "__main__":
    unittest.main()

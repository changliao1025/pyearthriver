"""
Unit tests for high-level pipeline workflows in pyearthriver.pipeline.
"""

import unittest
from pyearthriver.classes.vertex import pyvertex
from pyearthriver.classes.edge import pyedge
from pyearthriver.classes.flowline import pyflowline
from pyearthriver.pipeline import condition_river_network, compute_strahler_stream_order


class TestPipeline(unittest.TestCase):
    """Test the condition_river_network and compute_strahler_stream_order functions."""

    def setUp(self):
        # Create a simple Y-shaped network:
        # Branch 1: v1 -> v2
        # Branch 2: v3 -> v2
        # Mainstem: v2 -> v4 (outlet)
        self.v1 = pyvertex({"lVertexID": 1, "dLongitude_degree": -77.2, "dLatitude_degree": 41.2})
        self.v2 = pyvertex({"lVertexID": 2, "dLongitude_degree": -77.1, "dLatitude_degree": 41.1})
        self.v3 = pyvertex({"lVertexID": 3, "dLongitude_degree": -77.0, "dLatitude_degree": 41.2})
        self.v4 = pyvertex({"lVertexID": 4, "dLongitude_degree": -77.1, "dLatitude_degree": 41.0})

        self.e1 = pyedge(self.v1, self.v2)
        self.e2 = pyedge(self.v3, self.v2)
        self.e3 = pyedge(self.v2, self.v4)

        self.f1 = pyflowline([self.e1])
        self.f1.set_flowline_id(1)
        self.f2 = pyflowline([self.e2])
        self.f2.set_flowline_id(2)
        self.f3 = pyflowline([self.e3])
        self.f3.set_flowline_id(3)

        self.flowlines = [self.f1, self.f2, self.f3]

    def test_condition_river_network(self):
        """Test full conditioning pipeline on a simple branching network."""
        graph, conditioned = condition_river_network(
            self.flowlines,
            outlet_vertex=self.v4,
            remove_duplicates=True,
            remove_disconnected=True,
            remove_braids=True,
            remove_cycles=True,
            define_topology=True,
            compute_stream_order=True,
        )

        self.assertIsNotNone(graph)
        self.assertGreater(len(conditioned), 0)

    def test_condition_empty_raises(self):
        """Test that passing an empty flowlines list raises ValueError."""
        with self.assertRaises(ValueError):
            condition_river_network([])

    def test_compute_strahler_stream_order(self):
        """Test the compute_strahler_stream_order convenience function."""
        ordered_flowlines = compute_strahler_stream_order(self.flowlines, self.v4)
        self.assertIsNotNone(ordered_flowlines)
        self.assertGreater(len(ordered_flowlines), 0)


if __name__ == "__main__":
    unittest.main()

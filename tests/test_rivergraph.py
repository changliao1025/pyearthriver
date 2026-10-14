"""
Unit tests for pyrivergraph class:
- Initialization with flowlines and outlet
- Basic graph operations (vertex count, flowline count)
- Sources, sinks, and vertex lookups
- Network topology queries
"""

import unittest
from tests import conftest  # Ensures environment setup before imports

from pyearthriver.classes.vertex import pyvertex
from pyearthriver.classes.edge import pyedge
from pyearthriver.classes.flowline import pyflowline
from pyearthriver.classes.rivergraph import pyrivergraph


class TestPyRivergraph(unittest.TestCase):
    """Test suite for pyrivergraph initialization and operations."""

    def setUp(self):
        # Create a simple linear network: v1 -> v2 -> v3
        self.v1 = pyvertex({"dLongitude_degree": -77.0, "dLatitude_degree": 38.0, "lVertexID": 1})
        self.v2 = pyvertex({"dLongitude_degree": -77.1, "dLatitude_degree": 38.1, "lVertexID": 2})
        self.v3 = pyvertex({"dLongitude_degree": -77.2, "dLatitude_degree": 38.2, "lVertexID": 3})

        self.e1 = pyedge(self.v1, self.v2)
        self.e2 = pyedge(self.v2, self.v3)

        self.fl1 = pyflowline([self.e1])
        self.fl1.set_flowline_id(1)
        self.fl2 = pyflowline([self.e2])
        self.fl2.set_flowline_id(2)

        self.linear_flowlines = [self.fl1, self.fl2]

    def test_initialize_with_flowlines_and_outlet(self):
        """Test initializing pyrivergraph with flowlines and an outlet vertex."""
        graph = pyrivergraph(self.linear_flowlines, pVertex_outlet=self.v3)
        self.assertIsNotNone(graph)
        self.assertEqual(graph.pVertex_outlet, self.v3)
        self.assertEqual(len(graph.aFlowline), 2)
        self.assertEqual(graph.get_vertex_count(), 3)

    def test_initialize_without_outlet(self):
        """Test initializing pyrivergraph without specifying an outlet vertex."""
        graph = pyrivergraph(self.linear_flowlines)
        self.assertIsNotNone(graph)
        self.assertEqual(len(graph.aFlowline), 2)
        self.assertEqual(graph.get_vertex_count(), 3)

    def test_vertex_lookups(self):
        """Test vertex lookups by ID and reverse lookup."""
        graph = pyrivergraph(self.linear_flowlines)
        vertices = graph.get_vertices()
        self.assertEqual(len(vertices), 3)

        # Lookup internal ID for a vertex
        id_v1 = graph.get_vertex_id(self.v1)
        self.assertIsNotNone(id_v1)
        self.assertEqual(graph.get_vertex_by_id(id_v1), self.v1)

        id_v2 = graph.get_vertex_id(self.v2)
        self.assertIsNotNone(id_v2)
        self.assertEqual(graph.get_vertex_by_id(id_v2), self.v2)

        # Nonexistent vertex lookup
        v_missing = pyvertex({"dLongitude_degree": -80.0, "dLatitude_degree": 35.0})
        self.assertIsNone(graph.get_vertex_id(v_missing))
        self.assertIsNone(graph.get_vertex_by_id(9999))

    def test_sources_and_sinks(self):
        """Test identifying headwaters (sources) and river mouth (sinks)."""
        graph = pyrivergraph(self.linear_flowlines)
        sources = graph.get_sources()
        sinks = graph.get_sinks()

        # In linear network v1 -> v2 -> v3, v1 is source and v3 is sink
        id_v1 = graph.get_vertex_id(self.v1)
        id_v3 = graph.get_vertex_id(self.v3)

        self.assertEqual(sources, [id_v1])
        self.assertEqual(sinks, [id_v3])

    def test_branching_confluence_network(self):
        """Test a Y-shaped network with two tributaries joining at a confluence."""
        # Trib 1: v_a -> v_conf
        # Trib 2: v_b -> v_conf
        # Main:   v_conf -> v_mouth
        v_a = pyvertex({"dLongitude_degree": -77.0, "dLatitude_degree": 38.2, "lVertexID": 10})
        v_b = pyvertex({"dLongitude_degree": -77.0, "dLatitude_degree": 38.0, "lVertexID": 20})
        v_conf = pyvertex({"dLongitude_degree": -77.1, "dLatitude_degree": 38.1, "lVertexID": 30})
        v_mouth = pyvertex({"dLongitude_degree": -77.3, "dLatitude_degree": 38.1, "lVertexID": 40})

        fl_trib1 = pyflowline([pyedge(v_a, v_conf)])
        fl_trib1.set_flowline_id(101)
        fl_trib2 = pyflowline([pyedge(v_b, v_conf)])
        fl_trib2.set_flowline_id(102)
        fl_main = pyflowline([pyedge(v_conf, v_mouth)])
        fl_main.set_flowline_id(103)

        network = [fl_trib1, fl_trib2, fl_main]
        graph = pyrivergraph(network, pVertex_outlet=v_mouth)

        self.assertEqual(len(graph.aFlowline), 3)
        self.assertEqual(graph.get_vertex_count(), 4)

        sources = graph.get_sources()
        self.assertEqual(len(sources), 2)
        id_va = graph.get_vertex_id(v_a)
        id_vb = graph.get_vertex_id(v_b)
        self.assertIn(id_va, sources)
        self.assertIn(id_vb, sources)

        sinks = graph.get_sinks()
        id_mouth = graph.get_vertex_id(v_mouth)
        self.assertEqual(sinks, [id_mouth])

        outlets = graph.find_outlet_vertices()
        self.assertEqual(len(outlets), 1)
        self.assertEqual(outlets[0], v_mouth)


if __name__ == "__main__":
    unittest.main()

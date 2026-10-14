"""
Unit tests for pyearthriver core classes:
- pyvertex
- pyedge
- pyflowline
- pyconfluence
"""

import unittest
import json
from tests import conftest  # Ensures environment setup before imports
from pyearthriver.classes.vertex import pyvertex
from pyearthriver.classes.edge import pyedge
from pyearthriver.classes.flowline import pyflowline
from pyearthriver.classes.confluence import pyconfluence


class TestPyVertex(unittest.TestCase):
    """Test suite for pyvertex class."""

    def setUp(self):
        self.v_params1 = {
            "dLongitude_degree": -77.0369,
            "dLatitude_degree": 38.9072,
            "lVertexID": 10,
            "lVertexIndex": 0,
            "lFlowlineID": 100,
            "x": 1000.0,
            "y": 2000.0,
            "z": 10.0,
            "dElevation": 50.0,
        }
        self.v_params2 = {
            "dLongitude_degree": -77.0369,
            "dLatitude_degree": 38.9072,
            "lVertexID": 20,
            "lVertexIndex": 1,
        }
        self.v_params3 = {
            "dLongitude_degree": -77.1000,
            "dLatitude_degree": 38.9500,
            "lVertexID": 30,
        }

    def test_vertex_initialization(self):
        """Test vertex initialization and basic properties."""
        v = pyvertex(self.v_params1)
        self.assertAlmostEqual(v.dLongitude_degree, -77.0369)
        self.assertAlmostEqual(v.dLatitude_degree, 38.9072)
        self.assertEqual(v.lVertexID, 10)
        self.assertEqual(v.lVertexIndex, 0)
        self.assertEqual(v.lFlowlineID, 100)
        self.assertAlmostEqual(v.dElevation, 50.0)

    def test_vertex_initialization_defaults(self):
        """Test vertex initialization with default optional parameters."""
        v = pyvertex({"dLongitude_degree": -77.0, "dLatitude_degree": 38.0})
        self.assertEqual(v.lVertexID, 1)
        self.assertEqual(v.lVertexIndex, -1)
        self.assertEqual(v.lFlowlineID, -1)

    def test_vertex_invalid_input(self):
        """Test error handling when initializing with invalid input."""
        with self.assertRaises(TypeError):
            pyvertex("invalid")
        with self.assertRaises(TypeError):
            pyvertex([1, 2, 3])

    def test_vertex_equality_and_hashing(self):
        """Test coordinate-based equality and hash compatibility."""
        v1 = pyvertex(self.v_params1)
        v2 = pyvertex(self.v_params2)  # Same coordinates, different ID
        v3 = pyvertex(self.v_params3)  # Different coordinates

        self.assertEqual(v1, v2)
        self.assertNotEqual(v1, v3)
        self.assertEqual(hash(v1), hash(v2))

        # Check set membership
        vertex_set = {v1}
        self.assertIn(v2, vertex_set)
        self.assertNotIn(v3, vertex_set)

    def test_vertex_mutators(self):
        """Test setter methods and type validation."""
        v = pyvertex(self.v_params1)
        v.set_vertex_id(99)
        self.assertEqual(v.lVertexID, 99)

        v.set_vertex_index(5)
        self.assertEqual(v.lVertexIndex, 5)

        v.set_flowline_id(42)
        self.assertEqual(v.lFlowlineID, 42)

        with self.assertRaises(TypeError):
            v.set_vertex_id("invalid")
        with self.assertRaises(TypeError):
            v.set_vertex_index("invalid")
        with self.assertRaises(TypeError):
            v.set_flowline_id("invalid")

    def test_vertex_is_valid(self):
        """Test vertex validity check."""
        v_valid = pyvertex(self.v_params1)
        self.assertTrue(v_valid.is_valid())

        v_invalid_coords = pyvertex(
            {"dLongitude_degree": 250.0, "dLatitude_degree": 38.0, "lVertexID": 1}
        )
        self.assertFalse(v_invalid_coords.is_valid())

    def test_vertex_copy(self):
        """Test deep copy of pyvertex."""
        v = pyvertex(self.v_params1)
        v_copy = v.copy()
        self.assertEqual(v, v_copy)
        self.assertEqual(v.lVertexID, v_copy.lVertexID)
        self.assertEqual(v.lFlowlineID, v_copy.lFlowlineID)
        # Verify independence
        v_copy.set_vertex_id(999)
        self.assertNotEqual(v.lVertexID, v_copy.lVertexID)

    def test_vertex_tojson(self):
        """Test JSON serialization."""
        v = pyvertex(self.v_params1)
        json_str = v.tojson()
        data = json.loads(json_str)
        self.assertEqual(data["lVertexID"], 10)
        self.assertAlmostEqual(data["dLongitude_degree"], -77.0369)
        self.assertAlmostEqual(data["dLatitude_degree"], 38.9072)

    def test_vertex_str_repr(self):
        """Test string representations."""
        v = pyvertex(self.v_params1)
        self.assertIn("pyvertex", str(v))
        self.assertIn("10", str(v))
        self.assertIn("pyvertex", repr(v))


class TestPyEdge(unittest.TestCase):
    """Test suite for pyedge class."""

    def setUp(self):
        self.v1 = pyvertex({"dLongitude_degree": -77.0, "dLatitude_degree": 38.0, "lVertexID": 1})
        self.v2 = pyvertex({"dLongitude_degree": -77.1, "dLatitude_degree": 38.1, "lVertexID": 2})
        self.v3 = pyvertex({"dLongitude_degree": -77.2, "dLatitude_degree": 38.2, "lVertexID": 3})

    def test_edge_initialization_and_length(self):
        """Test edge initialization, start/end vertices, and length calculation."""
        edge = pyedge(self.v1, self.v2)
        self.assertEqual(edge.pVertex_start, self.v1)
        self.assertEqual(edge.pVertex_end, self.v2)
        self.assertGreater(edge.dLength, 0.0)
        self.assertEqual(edge.lEdgeID, -1)
        self.assertEqual(edge.lIndex_upstream, -1)
        self.assertEqual(edge.lIndex_downstream, -1)

    def test_edge_zero_length_raises(self):
        """Test that identical start and end vertices raise ValueError."""
        v_same = pyvertex({"dLongitude_degree": -77.0, "dLatitude_degree": 38.0, "lVertexID": 10})
        with self.assertRaises(ValueError):
            pyedge(self.v1, v_same)

    def test_edge_none_vertices_raises(self):
        """Test that None vertex inputs raise ValueError."""
        with self.assertRaises(ValueError):
            pyedge(None, self.v2)
        with self.assertRaises(ValueError):
            pyedge(self.v1, None)

    def test_edge_equality(self):
        """Test edge equality based on start and end points."""
        edge1 = pyedge(self.v1, self.v2)
        edge2 = pyedge(self.v1, self.v2)
        edge3 = pyedge(self.v1, self.v3)

        self.assertEqual(edge1, edge2)
        self.assertNotEqual(edge1, edge3)

    def test_edge_midpoint(self):
        """Test midpoint calculation of edge."""
        edge = pyedge(self.v1, self.v2)
        midpoint = edge.get_midpoint()
        self.assertAlmostEqual(midpoint.dLongitude_degree, -77.05, places=5)
        self.assertAlmostEqual(midpoint.dLatitude_degree, 38.05, places=5)

    def test_edge_connectivity(self):
        """Test connectivity and relative direction methods."""
        e1 = pyedge(self.v1, self.v2)
        e2 = pyedge(self.v2, self.v3)
        e3 = pyedge(
            pyvertex({"dLongitude_degree": 0.0, "dLatitude_degree": 0.0}),
            pyvertex({"dLongitude_degree": 1.0, "dLatitude_degree": 1.0}),
        )

        self.assertTrue(e1.is_connected_to(e2))
        self.assertFalse(e1.is_connected_to(e3))

        self.assertTrue(e1.is_upstream_of(e2))
        self.assertTrue(e2.is_downstream_of(e1))
        self.assertFalse(e2.is_upstream_of(e1))

    def test_edge_reverse_and_copy(self):
        """Test reverse and copy methods."""
        edge = pyedge(self.v1, self.v2)
        edge.set_edge_id(10)
        edge.set_upstream_index(1)
        edge.set_downstream_index(2)

        rev = edge.reverse()
        self.assertEqual(rev.pVertex_start, self.v2)
        self.assertEqual(rev.pVertex_end, self.v1)
        self.assertEqual(rev.lIndex_upstream, 2)
        self.assertEqual(rev.lIndex_downstream, 1)

        copy_edge = edge.copy()
        self.assertEqual(copy_edge.lEdgeID, 10)
        self.assertEqual(copy_edge.pVertex_start, edge.pVertex_start)
        self.assertEqual(copy_edge.pVertex_end, edge.pVertex_end)

    def test_edge_tojson(self):
        """Test edge JSON serialization."""
        edge = pyedge(self.v1, self.v2)
        edge.set_edge_id(45)
        json_str = edge.tojson()
        data = json.loads(json_str)
        self.assertEqual(data["lEdgeID"], 45)


class TestPyFlowline(unittest.TestCase):
    """Test suite for pyflowline class."""

    def setUp(self):
        self.v1 = pyvertex({"dLongitude_degree": -77.0, "dLatitude_degree": 38.0, "lVertexID": 1})
        self.v2 = pyvertex({"dLongitude_degree": -77.1, "dLatitude_degree": 38.1, "lVertexID": 2})
        self.v3 = pyvertex({"dLongitude_degree": -77.2, "dLatitude_degree": 38.2, "lVertexID": 3})
        self.e1 = pyedge(self.v1, self.v2)
        self.e2 = pyedge(self.v2, self.v3)

    def test_flowline_initialization_single_edge(self):
        """Test flowline creation with a single edge."""
        fl = pyflowline([self.e1])
        self.assertEqual(fl.nEdge, 1)
        self.assertEqual(fl.nVertex, 2)
        self.assertEqual(fl.pVertex_start, self.v1)
        self.assertEqual(fl.pVertex_end, self.v2)
        self.assertAlmostEqual(fl.dLength, self.e1.dLength)

    def test_flowline_initialization_multiple_edges(self):
        """Test flowline creation with multiple edges."""
        fl = pyflowline([self.e1, self.e2])
        self.assertEqual(fl.nEdge, 2)
        self.assertEqual(fl.nVertex, 3)
        self.assertEqual(fl.pVertex_start, self.v1)
        self.assertEqual(fl.pVertex_end, self.v3)
        expected_len = self.e1.dLength + self.e2.dLength
        self.assertAlmostEqual(fl.dLength, expected_len)

    def test_flowline_empty_or_invalid_input(self):
        """Test error handling on invalid edge inputs."""
        with self.assertRaises(ValueError):
            pyflowline([])
        with self.assertRaises(TypeError):
            pyflowline("invalid")
        with self.assertRaises(ValueError):
            pyflowline([None])

    def test_flowline_properties_and_mutators(self):
        """Test flowline properties and flag manipulation."""
        fl = pyflowline([self.e1])
        fl.set_flowline_id(101)
        self.assertEqual(fl.lFlowlineID, 101)
        self.assertEqual(fl.iFlag_keep, 1)

        fl.iStream_order = 2
        self.assertEqual(fl.iStream_order, 2)

    def test_flowline_tojson(self):
        """Test flowline JSON export."""
        fl = pyflowline([self.e1])
        fl.set_flowline_id(88)
        json_str = fl.tojson()
        data = json.loads(json_str)
        self.assertEqual(data["lFlowlineID"], 88)


class TestPyConfluence(unittest.TestCase):
    """Test suite for pyconfluence class."""

    def setUp(self):
        # Confluence vertex at (0, 0)
        self.v_center = pyvertex({"dLongitude_degree": 0.0, "dLatitude_degree": 0.0, "lVertexID": 0})
        # Two upstream branches
        # Branch 1 starts at (-1, 0) and flows into (0, 0)
        v_up1 = pyvertex({"dLongitude_degree": -1.0, "dLatitude_degree": 0.0, "lVertexID": 1})
        e_up1 = pyedge(v_up1, self.v_center)
        self.fl_up1 = pyflowline([e_up1])

        # Branch 2 starts at (0, 1) and flows into (0, 0)
        v_up2 = pyvertex({"dLongitude_degree": 0.0, "dLatitude_degree": 1.0, "lVertexID": 2})
        e_up2 = pyedge(v_up2, self.v_center)
        self.fl_up2 = pyflowline([e_up2])

        # Downstream branch starts at (0, 0) and flows to (1, 0)
        v_down = pyvertex({"dLongitude_degree": 1.0, "dLatitude_degree": 0.0, "lVertexID": 3})
        e_down = pyedge(self.v_center, v_down)
        self.fl_down = pyflowline([e_down])

    def test_confluence_initialization(self):
        """Test confluence initialization and upstream/downstream counts."""
        conf = pyconfluence(self.v_center, [self.fl_up1, self.fl_up2], [self.fl_down])
        self.assertEqual(conf.pVertex_confluence, self.v_center)
        self.assertEqual(conf.nUpstream, 2)
        self.assertEqual(conf.nDownstream, 1)
        self.assertEqual(conf.lConfluenceID, -1)

    def test_confluence_branching_angle(self):
        """Test calculation of branching angle between upstream flowlines."""
        conf = pyconfluence(self.v_center, [self.fl_up1, self.fl_up2], [self.fl_down])
        angle = conf.calculate_branching_angle()
        # Branch 1 is from (-1, 0) to (0, 0) -> vector (-1, 0)
        # Branch 2 is from (0, 1) to (0, 0) -> vector (0, 1)
        # Angle between (-1, 0) and (0, 1) is 90 degrees
        self.assertAlmostEqual(angle, 90.0, places=1)
        self.assertAlmostEqual(conf.dAngle_upstream, 90.0, places=1)

    def test_confluence_tojson(self):
        """Test confluence JSON serialization."""
        conf = pyconfluence(self.v_center, [self.fl_up1, self.fl_up2], [self.fl_down])
        conf.lConfluenceID = 7
        json_str = conf.tojson()
        data = json.loads(json_str)
        self.assertEqual(data["lConfluenceID"], 7)


if __name__ == "__main__":
    unittest.main()

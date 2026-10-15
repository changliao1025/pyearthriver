"""
Unit tests for format loading and vector I/O in pyearthriver.formats.
"""

import unittest
import tempfile
import os
import json
from pyearthriver.formats.load_flowline import read_flowlines_from_geojson
from pyearthriver.classes.vertex import pyvertex
from pyearthriver.classes.edge import pyedge
from pyearthriver.classes.flowline import pyflowline


class TestLoadFlowline(unittest.TestCase):
    """Test reading flowlines from GeoJSON sources."""

    def setUp(self):
        self.sample_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": 101,
                    "properties": {
                        "lineid": 1,
                        "drainage_area": 1250.5,
                        "stream_order": 2,
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [-77.0, 41.0, 100.0],
                            [-77.1, 41.1, 95.0],
                            [-77.2, 41.2, 90.0],
                        ],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {
                        "lineid": 2,
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [-77.2, 41.2],
                            [-77.3, 41.3],
                        ],
                    },
                },
            ],
        }

    def test_read_from_dict(self):
        """Test reading flowlines from a pre-parsed GeoJSON dictionary."""
        flowlines = read_flowlines_from_geojson(self.sample_geojson)
        self.assertEqual(len(flowlines), 2)

        # Check first flowline
        f1 = flowlines[0]
        self.assertEqual(f1.lFlowlineID, 1)
        self.assertEqual(len(f1.aEdge), 2)
        self.assertEqual(len(f1.aVertex), 3)
        self.assertAlmostEqual(f1.aVertex[0].dLongitude_degree, -77.0)
        self.assertAlmostEqual(f1.aVertex[0].dLatitude_degree, 41.0)
        self.assertAlmostEqual(f1.aVertex[0].dElevation, 100.0)
        self.assertAlmostEqual(f1.dDrainage_area, 1250.5)
        self.assertEqual(f1.iStream_order, 2)

        # Check second flowline
        f2 = flowlines[1]
        self.assertEqual(f2.lFlowlineID, 2)
        self.assertEqual(len(f2.aEdge), 1)

    def test_read_from_file(self):
        """Test reading flowlines from a temporary GeoJSON file."""
        with tempfile.NamedTemporaryFile("w", suffix=".geojson", delete=False) as tmp:
            json.dump(self.sample_geojson, tmp)
            tmp_path = tmp.name

        try:
            flowlines = read_flowlines_from_geojson(tmp_path)
            self.assertEqual(len(flowlines), 2)
            self.assertEqual(flowlines[0].lFlowlineID, 1)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_file_not_found_raises(self):
        """Test FileNotFoundError on non-existent path."""
        with self.assertRaises(FileNotFoundError):
            read_flowlines_from_geojson("/path/to/non_existent_file.geojson")

    def test_invalid_type_raises(self):
        """Test TypeError when passing invalid type."""
        with self.assertRaises(TypeError):
            read_flowlines_from_geojson(12345)  # type: ignore


if __name__ == "__main__":
    unittest.main()

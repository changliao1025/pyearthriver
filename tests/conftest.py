"""
Test environment configuration and compatibility shim.

If optional external runtime dependencies (numpy, pyearth, rtree, osgeo) are
not pre-installed in the active Python environment, this module configures minimal,
faithful stubs so that unit tests can execute cleanly under standard unittest / pytest.
If real dependencies are already present, they are preserved and used as-is.
"""

import sys
import types
import math

def setup_test_environment():
    # 1. numpy compatibility stub if not installed
    try:
        import numpy as np
    except ImportError:
        np = types.ModuleType("numpy")
        class _NumpyInteger(int): pass
        class _NumpyFloat(float): pass
        class _NumpyArray(list):
            def __getitem__(self, item):
                if isinstance(item, (list, tuple, _NumpyArray)):
                    return _NumpyArray([super(_NumpyArray, self).__getitem__(i) for i in item])
                return super().__getitem__(item)

        np.integer = (int, _NumpyInteger)
        np.float32 = (float, _NumpyFloat)
        np.float64 = (float, _NumpyFloat)
        np.ndarray = _NumpyArray
        np.min = min
        np.max = max
        np.array = lambda x, **kwargs: _NumpyArray(x)
        np.arange = lambda *args: list(range(*args))
        np.argsort = lambda a: _NumpyArray(sorted(range(len(a)), key=lambda i: a[i]))
        np.isnan = lambda x: False
        sys.modules["numpy"] = np

    # 2. rtree compatibility stub if not installed
    try:
        import rtree
    except ImportError:
        rtree = types.ModuleType("rtree")
        rtree_index = types.ModuleType("rtree.index")
        class RTreeIndex:
            def __init__(self):
                self.items = []
            def insert(self, item_id, bound):
                self.items.append((item_id, bound))
            def intersection(self, bound):
                left, bottom, right, top = bound
                return [
                    i for i, (il, ibot, ir, itop) in self.items
                    if not (right < il or left > ir or top < ibot or bottom > itop)
                ]
            def search_surrounding(self, pt):
                x, y = pt
                delta = 1e-4
                return self.intersection((x - delta, y - delta, x + delta, y + delta))
        rtree_index.Index = RTreeIndex
        rtree.index = rtree_index
        sys.modules["rtree"] = rtree
        sys.modules["rtree.index"] = rtree_index

    # 3. pyearth compatibility stub if not installed
    try:
        import pyearth
    except ImportError:
        pyearth = types.ModuleType("pyearth")
        pyearth.__path__ = []
        toolbox = types.ModuleType("pyearth.toolbox")
        toolbox.__path__ = []
        mesh = types.ModuleType("pyearth.toolbox.mesh")
        mesh.__path__ = []

        mesh_point = types.ModuleType("pyearth.toolbox.mesh.point")
        mesh_line = types.ModuleType("pyearth.toolbox.mesh.line")
        mesh_polyline = types.ModuleType("pyearth.toolbox.mesh.polyline")

        class pypoint:
            def __init__(self, aParameter):
                if not isinstance(aParameter, dict):
                    raise TypeError(f"aParameter must be a dictionary, got {type(aParameter)}")
                self.dLongitude_degree = float(aParameter.get("dLongitude_degree", 0.0))
                self.dLatitude_degree = float(aParameter.get("dLatitude_degree", 0.0))
                self.dX_meter = float(aParameter.get("x", aParameter.get("dX_meter", 0.0)))
                self.dY_meter = float(aParameter.get("y", aParameter.get("dY_meter", 0.0)))
                self.dZ_meter = float(aParameter.get("z", aParameter.get("dZ_meter", 0.0)))
                self.dElevation = float(aParameter.get("dElevation", 0.0))
                self.dLongitude_radian = math.radians(self.dLongitude_degree)
                self.dLatitude_radian = math.radians(self.dLatitude_degree)

            def __eq__(self, other):
                if not hasattr(other, "dLongitude_degree") or not hasattr(other, "dLatitude_degree"):
                    return False
                return (abs(self.dLongitude_degree - other.dLongitude_degree) < 1e-6 and
                        abs(self.dLatitude_degree - other.dLatitude_degree) < 1e-6)

            def __hash__(self):
                return hash((round(self.dLongitude_degree, 6), round(self.dLatitude_degree, 6)))

            def calculate_distance(self, other):
                lat1 = math.radians(self.dLatitude_degree)
                lat2 = math.radians(other.dLatitude_degree)
                dlat = lat2 - lat1
                dlon = math.radians(other.dLongitude_degree - self.dLongitude_degree)
                a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
                c = 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))
                return 6378137.0 * c

        class pyline:
            def __init__(self, pPoint_start, pPoint_end):
                self.pPoint_start = pPoint_start
                self.pPoint_end = pPoint_end
                self.dLength = pPoint_start.calculate_distance(pPoint_end)
                self.dSlope = 0.0

            def __eq__(self, other):
                if not hasattr(other, "pPoint_start") or not hasattr(other, "pPoint_end"):
                    return False
                return self.pPoint_start == other.pPoint_start and self.pPoint_end == other.pPoint_end

            def __hash__(self):
                return hash((self.pPoint_start, self.pPoint_end))

            def check_vertex_on_edge(self, pVertex):
                d_start = self.pPoint_start.calculate_distance(pVertex)
                d_end = pVertex.calculate_distance(self.pPoint_end)
                diff = abs((d_start + d_end) - self.dLength)
                # If sum of distances matches edge length within tolerance
                if diff < 1.0 or (self.dLength > 0 and diff / self.dLength < 1e-4):
                    return 1, d_start, diff
                return 0, d_start, diff

        class pypolyline:
            def __init__(self, aLine):
                self.aLine = aLine
                self.pPoint_start = aLine[0].pPoint_start
                self.pPoint_end = aLine[-1].pPoint_end
                points = [aLine[0].pPoint_start]
                for line in aLine:
                    points.append(line.pPoint_end)
                self.aPoint = points
                self.dLength = sum(l.dLength for l in aLine)

            def __eq__(self, other):
                if not hasattr(other, "aLine"):
                    return False
                return len(self.aLine) == len(other.aLine) and all(
                    l1 == l2 for l1, l2 in zip(self.aLine, other.aLine)
                )

        mesh_point.pypoint = pypoint
        mesh_line.pyline = pyline
        mesh_polyline.pypolyline = pypolyline

        mesh.point = mesh_point
        mesh.line = mesh_line
        mesh.polyline = mesh_polyline
        toolbox.mesh = mesh
        pyearth.toolbox = toolbox

        sys.modules["pyearth"] = pyearth
        sys.modules["pyearth.toolbox"] = toolbox
        sys.modules["pyearth.toolbox.mesh"] = mesh
        sys.modules["pyearth.toolbox.mesh.point"] = mesh_point
        sys.modules["pyearth.toolbox.mesh.line"] = mesh_line
        sys.modules["pyearth.toolbox.mesh.polyline"] = mesh_polyline

        gis = types.ModuleType("pyearth.gis")
        gis.__path__ = []
        gis_geom = types.ModuleType("pyearth.gis.geometry")
        gis_geom.__path__ = []
        geom_kernel = types.ModuleType("pyearth.gis.geometry.kernel")
        geom_calc = types.ModuleType("pyearth.gis.geometry.calculate_angle_between_point")

        def calc_angle(x1, y1, x2, y2, x3, y3):
            v1_x, v1_y = x1 - x2, y1 - y2
            v2_x, v2_y = x3 - x2, y3 - y2
            dot = v1_x * v2_x + v1_y * v2_y
            mag1 = math.hypot(v1_x, v1_y)
            mag2 = math.hypot(v2_x, v2_y)
            if mag1 == 0 or mag2 == 0:
                return 0.0
            return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2)))))

        geom_kernel.calculate_angle_between_point = calc_angle
        geom_calc.calculate_angle_between_point = calc_angle
        gis_geom.kernel = geom_kernel
        gis_geom.calculate_angle_between_point = geom_calc
        gis.geometry = gis_geom
        pyearth.gis = gis

        sys.modules["pyearth.gis"] = gis
        sys.modules["pyearth.gis.geometry"] = gis_geom
        sys.modules["pyearth.gis.geometry.kernel"] = geom_kernel
        sys.modules["pyearth.gis.geometry.calculate_angle_between_point"] = geom_calc

        gis_gdal = types.ModuleType("pyearth.gis.gdal")
        gis_gdal.__path__ = []
        gdal_vector = types.ModuleType("pyearth.gis.gdal.gdal_vector_format_support")
        gdal_vector.get_vector_driver_from_filename = lambda fn: "GeoJSON"
        gis_gdal.gdal_vector_format_support = gdal_vector
        gis.gdal = gis_gdal
        sys.modules["pyearth.gis.gdal"] = gis_gdal
        sys.modules["pyearth.gis.gdal.gdal_vector_format_support"] = gdal_vector

    # 4. osgeo compatibility stub if not installed
    try:
        from osgeo import ogr, osr
    except ImportError:
        osgeo = types.ModuleType("osgeo")
        ogr = types.ModuleType("osgeo.ogr")
        osr = types.ModuleType("osgeo.osr")
        class SpatialReference: pass
        osr.SpatialReference = SpatialReference
        osgeo.ogr = ogr
        osgeo.osr = osr
        sys.modules["osgeo"] = osgeo
        sys.modules["osgeo.ogr"] = ogr
        sys.modules["osgeo.osr"] = osr


setup_test_environment()

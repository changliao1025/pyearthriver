"""Module for loading river network flowlines from vector formats.

Supports Shapefile (.shp), GeoJSON (.geojson, .json), GeoPackage (.gpkg),
and other vector formats supported by GDAL/OGR and pyearth.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from osgeo import gdal, ogr, osr
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_driver_from_filename

from pyearthriver.classes.edge import pyedge
from pyearthriver.classes.flowline import pyflowline
from pyearthriver.classes.vertex import pyvertex

PathLike = Union[str, os.PathLike]


def _read_flowlines_from_geojson_dict(
    data: Dict[str, Any],
    id_field: Optional[str] = None,
) -> List[pyflowline]:
    """Parse flowline objects from an in-memory GeoJSON dictionary.

    Args:
        data: Parsed GeoJSON dictionary.
        id_field: Optional field name to extract flowline ID from properties.

    Returns:
        List of constructed pyflowline objects.
    """
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    elif data.get("type") == "Feature":
        features = [data]
    elif data.get("type") in ("LineString", "MultiLineString"):
        features = [{"type": "Feature", "geometry": data, "properties": {}}]
    else:
        raise ValueError(
            "Invalid GeoJSON: must be FeatureCollection, Feature, or LineString geometry"
        )

    flowlines: List[pyflowline] = []
    global_vertex_id = 1
    flowline_counter = 1
    flowline_index = 0

    for feature in features:
        geom = feature.get("geometry")
        if not geom:
            continue

        geom_type = geom.get("type")
        coords = geom.get("coordinates", [])
        props = feature.get("properties") or {}

        resolved_id = None
        if id_field and id_field in props:
            resolved_id = props[id_field]
        else:
            for candidate in (
                "lineid",
                "line_id",
                "id",
                "flowline_id",
                "ID",
                "LineID",
                "arcid",
                "subbasin",
            ):
                if candidate in props:
                    resolved_id = props[candidate]
                    break
        if resolved_id is None:
            resolved_id = feature.get("id", flowline_counter)

        try:
            resolved_id = int(resolved_id)
        except (ValueError, TypeError):
            resolved_id = flowline_counter

        lines_coords = []
        if geom_type == "LineString":
            lines_coords.append(coords)
        elif geom_type == "MultiLineString":
            lines_coords.extend(coords)
        else:
            continue

        for line_coord in lines_coords:
            unique_pts = []
            for pt in line_coord:
                if not unique_pts or (pt[0] != unique_pts[-1][0] or pt[1] != unique_pts[-1][1]):
                    unique_pts.append(pt)
            if len(unique_pts) < 2:
                continue

            vertices: List[pyvertex] = []
            for pt in unique_pts:
                lon = float(pt[0])
                lat = float(pt[1])
                elev = float(pt[2]) if len(pt) > 2 else 0.0
                v = pyvertex(
                    {
                        "lVertexID": global_vertex_id,
                        "dLongitude_degree": lon,
                        "dLatitude_degree": lat,
                        "dElevation": elev,
                    }
                )
                vertices.append(v)
                global_vertex_id += 1

            edges: List[pyedge] = []
            for i in range(len(vertices) - 1):
                edges.append(pyedge(vertices[i], vertices[i + 1]))

            if not edges:
                continue

            flowline = pyflowline(edges)
            flowline.set_flowline_id(resolved_id)
            flowline.lFlowlineIndex = flowline_index

            if "drainage_area" in props or "area" in props:
                flowline.dDrainage_area = float(props.get("drainage_area", props.get("area", 0.0)))
            if "stream_order" in props or "order" in props:
                flowline.iStream_order = int(props.get("stream_order", props.get("order", -1)))
            if "stream_segment" in props or "segment" in props:
                flowline.iStream_segment = int(
                    props.get("stream_segment", props.get("segment", -1))
                )
            if (
                "nhdplus_id" in props
                or "NHDPlusID" in props
                or "lNHDPlusID" in props
                or "ComID" in props
            ):
                flowline.lNHDPlusID = int(
                    props.get(
                        "nhdplus_id",
                        props.get(
                            "NHDPlusID",
                            props.get("lNHDPlusID", props.get("ComID", -1)),
                        ),
                    )
                )

            flowlines.append(flowline)
            flowline_index += 1
            flowline_counter += 1

    return flowlines


def read_flowlines_from_vector(
    source: Union[PathLike, Dict[str, Any]],
    id_field: Optional[str] = None,
    return_spatial_ref: bool = False,
) -> Union[List[pyflowline], Tuple[List[pyflowline], Optional[str]]]:
    """Read flowline objects from any vector format supported by GDAL/OGR or GeoJSON dictionary.

    Supports Shapefile (.shp), GeoJSON (.geojson, .json), GeoPackage (.gpkg),
    and other vector formats supported by pyearth and GDAL. Automatically
    reprojects coordinates to WGS84 (EPSG:4326) if necessary.

    Args:
        source: File path to a vector file or an already-parsed GeoJSON dictionary.
        id_field: Field name to extract flowline ID from properties.
        return_spatial_ref: If True, returns a tuple of (flowlines, spatial_ref_wkt).
            Defaults to False.

    Returns:
        List of constructed pyflowline objects, or (flowlines, spatial_ref_wkt)
        if return_spatial_ref is True.

    Raises:
        FileNotFoundError: If a file path is provided and does not exist.
        RuntimeError: If GDAL/OGR cannot open the vector file.
    """
    if isinstance(source, dict):
        flowlines = _read_flowlines_from_geojson_dict(source, id_field=id_field)
        if return_spatial_ref:
            return flowlines, None
        return flowlines

    filepath = os.fspath(source)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Vector file not found: {filepath}")

    # Open dataset using pyearth driver resolver or fallback to ogr.Open
    pDataset = None
    try:
        pDriver = get_vector_driver_from_filename(filepath)
        if pDriver is not None:
            pDataset = pDriver.Open(filepath, gdal.GA_ReadOnly)
    except Exception:
        pDataset = None

    if pDataset is None:
        pDataset = ogr.Open(filepath, 0)

    if pDataset is None:
        raise RuntimeError(f"Unable to open vector dataset: {filepath}")

    try:
        pLayer = pDataset.GetLayer(0)
        if pLayer is None:
            raise RuntimeError(f"No vector layers found in {filepath}")

        pSpatialRef = pLayer.GetSpatialRef()
        spatial_ref_wkt = pSpatialRef.ExportToWkt() if pSpatialRef is not None else None

        pSpatial_reference_gcs = osr.SpatialReference()
        pSpatial_reference_gcs.ImportFromEPSG(4326)
        pSpatial_reference_gcs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        transform = None
        if pSpatialRef is not None and not pSpatialRef.IsSame(pSpatial_reference_gcs):
            transform = osr.CoordinateTransformation(pSpatialRef, pSpatial_reference_gcs)

        # Inspect fields available on layer
        ldefn = pLayer.GetLayerDefn()
        field_names = [ldefn.GetFieldDefn(i).GetName() for i in range(ldefn.GetFieldCount())]
        field_lookup = {name.lower(): name for name in field_names}

        def _find_field(candidates: List[str], preferred: Optional[str] = None) -> Optional[str]:
            if preferred and preferred in field_names:
                return preferred
            if preferred and preferred.lower() in field_lookup:
                return field_lookup[preferred.lower()]
            for cand in candidates:
                if cand.lower() in field_lookup:
                    return field_lookup[cand.lower()]
            return None

        field_id = _find_field(
            ["lineid", "line_id", "flowline_id", "id", "arcid", "subbasin"],
            preferred=id_field,
        )
        field_segment = _find_field(["stream_segment", "stream_seg", "segment"])
        field_order = _find_field(["stream_order", "stream_ord", "order"])
        field_nhdplus = _find_field(["nhdplusid", "nhdplus_id", "comid"])
        field_area = _find_field(["drainage_area", "drainage_a", "area"])

        flowlines: List[pyflowline] = []
        global_vertex_id = 1
        flowline_counter = 1
        lFlowlineIndex = 0

        pLayer.ResetReading()
        for pFeature in pLayer:
            if pFeature is None:
                continue

            pGeometry = pFeature.GetGeometryRef()
            if pGeometry is None:
                continue

            if transform is not None:
                pGeometry.Transform(transform)

            resolved_id = None
            if field_id:
                val = pFeature.GetField(field_id)
                if val is not None:
                    try:
                        resolved_id = int(val)
                    except (ValueError, TypeError):
                        resolved_id = flowline_counter
            if resolved_id is None:
                resolved_id = flowline_counter

            val_seg = pFeature.GetField(field_segment) if field_segment else None
            iStream_segment = int(val_seg) if val_seg is not None else -1

            val_order = pFeature.GetField(field_order) if field_order else None
            iStream_order = int(val_order) if val_order is not None else -1

            val_nhd = pFeature.GetField(field_nhdplus) if field_nhdplus else None
            lNHDPlusID = int(val_nhd) if val_nhd is not None else -1

            val_area = pFeature.GetField(field_area) if field_area else None
            dDrainage_area = float(val_area) if val_area is not None else 0.0

            sGeomType = pGeometry.GetGeometryName().upper()
            lines_coords = []
            if sGeomType == "LINESTRING":
                lines_coords.append(pGeometry.GetPoints())
            elif sGeomType == "MULTILINESTRING":
                for i in range(pGeometry.GetGeometryCount()):
                    subline = pGeometry.GetGeometryRef(i)
                    if subline is not None:
                        lines_coords.append(subline.GetPoints())
            else:
                continue

            for raw_pts in lines_coords:
                unique_pts = []
                for pt in raw_pts:
                    if not unique_pts or (pt[0] != unique_pts[-1][0] or pt[1] != unique_pts[-1][1]):
                        unique_pts.append(pt)
                if len(unique_pts) < 2:
                    continue

                vertices: List[pyvertex] = []
                for pt in unique_pts:
                    lon = float(pt[0])
                    lat = float(pt[1])
                    elev = float(pt[2]) if len(pt) > 2 else 0.0
                    v = pyvertex(
                        {
                            "lVertexID": global_vertex_id,
                            "dLongitude_degree": lon,
                            "dLatitude_degree": lat,
                            "dElevation": elev,
                        }
                    )
                    vertices.append(v)
                    global_vertex_id += 1

                edges: List[pyedge] = []
                for k in range(len(vertices) - 1):
                    edges.append(pyedge(vertices[k], vertices[k + 1]))

                if not edges:
                    continue

                flowline = pyflowline(edges)
                flowline.set_flowline_id(resolved_id)
                flowline.lFlowlineIndex = lFlowlineIndex
                flowline.iStream_segment = iStream_segment
                flowline.iStream_order = iStream_order
                flowline.lNHDPlusID = lNHDPlusID
                flowline.dDrainage_area = dDrainage_area

                flowlines.append(flowline)
                lFlowlineIndex += 1
                flowline_counter += 1

        if return_spatial_ref:
            return flowlines, spatial_ref_wkt
        return flowlines

    finally:
        pLayer = None
        pDataset = None

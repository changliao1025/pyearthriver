"""
Module for loading river network flowlines from vector formats like GeoJSON.
"""

import json
import os
from typing import List, Dict, Any, Union, Optional
from pyearthriver.classes.vertex import pyvertex
from pyearthriver.classes.edge import pyedge
from pyearthriver.classes.flowline import pyflowline


def read_flowlines_from_geojson(
    source: Union[str, Dict[str, Any]],
    id_field: Optional[str] = None
) -> List[pyflowline]:
    """
    Read flowline objects from a GeoJSON file or dictionary.

    Supports both 'LineString' and 'MultiLineString' geometries within
    a GeoJSON FeatureCollection or a single Feature.

    Args:
        source (Union[str, Dict[str, Any]]): File path to a .geojson file or
            an already-parsed GeoJSON dictionary.
        id_field (Optional[str]): Field name to extract flowline ID from properties.
            Defaults to inspecting 'lineid', 'id', 'flowline_id', or feature index.

    Returns:
        List[pyflowline]: List of constructed pyflowline objects.

    Raises:
        FileNotFoundError: If a file path is provided and does not exist.
        ValueError: If the source format is not valid GeoJSON or contains no line features.
    """
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"GeoJSON file not found: {source}")
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(source, dict):
        data = source
    else:
        raise TypeError(f"source must be a file path string or dict, got {type(source)}")

    # Extract features list
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    elif data.get("type") == "Feature":
        features = [data]
    elif data.get("type") in ("LineString", "MultiLineString"):
        features = [{"type": "Feature", "geometry": data, "properties": {}}]
    else:
        raise ValueError("Invalid GeoJSON: must be FeatureCollection, Feature, or LineString geometry")

    flowlines: List[pyflowline] = []
    global_vertex_id = 1
    flowline_counter = 1

    for feature in features:
        geom = feature.get("geometry")
        if not geom:
            continue

        geom_type = geom.get("type")
        coords = geom.get("coordinates", [])
        props = feature.get("properties") or {}

        # Resolve flowline ID
        resolved_id = None
        if id_field and id_field in props:
            resolved_id = props[id_field]
        else:
            for candidate in ("lineid", "line_id", "id", "flowline_id", "ID", "LineID"):
                if candidate in props:
                    resolved_id = props[candidate]
                    break
        if resolved_id is None:
            resolved_id = feature.get("id", flowline_counter)

        try:
            resolved_id = int(resolved_id)
        except (ValueError, TypeError):
            resolved_id = flowline_counter

        # Handle LineString and MultiLineString
        lines_coords = []
        if geom_type == "LineString":
            lines_coords.append(coords)
        elif geom_type == "MultiLineString":
            lines_coords.extend(coords)
        else:
            continue

        for line_coord in lines_coords:
            if len(line_coord) < 2:
                continue

            vertices: List[pyvertex] = []
            for pt in line_coord:
                lon = float(pt[0])
                lat = float(pt[1])
                elev = float(pt[2]) if len(pt) > 2 else 0.0
                v = pyvertex({
                    "lVertexID": global_vertex_id,
                    "dLongitude_degree": lon,
                    "dLatitude_degree": lat,
                    "dElevation": elev,
                })
                vertices.append(v)
                global_vertex_id += 1

            edges: List[pyedge] = []
            for i in range(len(vertices) - 1):
                e = pyedge(vertices[i], vertices[i + 1])
                edges.append(e)

            flowline = pyflowline(edges)
            flowline.set_flowline_id(resolved_id)

            # Assign other optional properties if available
            if "drainage_area" in props:
                flowline.dDrainage_area = float(props["drainage_area"])
            if "stream_order" in props:
                flowline.iStream_order = int(props["stream_order"])
            if "nhdplus_id" in props or "lNHDPlusID" in props:
                flowline.lNHDPlusID = int(props.get("nhdplus_id", props.get("lNHDPlusID", -1)))

            flowlines.append(flowline)
            flowline_counter += 1

    return flowlines

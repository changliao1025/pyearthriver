import os
import json
from typing import List, Optional, Dict, Any
from osgeo import ogr, osr
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_driver_from_filename

def export_flowline_to_vector(aFlowline_in: List,
    sFilename_vector_out: str,
    iFlag_projected_in: Optional[int] = None,
    pSpatial_reference_in: Optional[osr.SpatialReference] = None,
    aAttribute_field: Optional[List[str]] = None,
    aAttribute_data: Optional[List[List[Any]]] = None,
    aAttribute_dtype: Optional[List[str]] = None) -> None:
    """
    Export flowline objects to a vector format file (GeoJSON, Shapefile, etc.).
    
    This function converts a list of flowline objects to various vector formats.

    Args:
        aFlowline_in (List): List of flowline objects to export
        sFilename_vector_out (str): Output file path with format extension (e.g., .geojson, .shp)
        iFlag_projected_in (Optional[int]): Flag indicating if coordinates are projected (1) or geographic (0). Defaults to None (treated as 0).
        pSpatial_reference_in (Optional[osr.SpatialReference]): Spatial reference system. Defaults to WGS84 (EPSG:4326).
        aAttribute_field (Optional[List[str]]): List of attribute field names. Defaults to None.
        aAttribute_data (Optional[List[List[Any]]]): List of attribute data lists. Defaults to None.
        aAttribute_dtype (Optional[List[str]]): List of attribute data types ('int' or 'float'). Defaults to None.

    Returns:
        None

    Raises:
        ValueError: If attribute data is inconsistent or file format is not supported.
    """
    try:
        # Remove existing file
        if os.path.exists(sFilename_vector_out):
            os.remove(sFilename_vector_out)

        nFlowline = len(aFlowline_in)
        is_projected = iFlag_projected_in == 1 if iFlag_projected_in is not None else False

        if pSpatial_reference_in is None:
            pSpatial_reference_in = osr.SpatialReference()
            pSpatial_reference_in.ImportFromEPSG(4326)  # WGS84 lat/lon

        # Validate attributes
        has_attributes = all([aAttribute_field, aAttribute_data, aAttribute_dtype])
        if has_attributes:
            nAttribute1, nAttribute2, nAttribute3 = len(aAttribute_field), len(aAttribute_data), len(aAttribute_dtype)
            if not (nAttribute1 == nAttribute2 == nAttribute3):
                raise ValueError('Attribute field, data, and dtype lists must have the same length')
            if nFlowline != len(aAttribute_data[0]):
                raise ValueError(f'Number of flowlines ({nFlowline}) does not match attribute data length ({len(aAttribute_data[0])})')

        # Get appropriate driver based on file extension
        pDriver = get_vector_driver_from_filename(sFilename_vector_out)
        pDataset = pDriver.CreateDataSource(sFilename_vector_out)
        pLayer = pDataset.CreateLayer('flowline', pSpatial_reference_in, ogr.wkbLineString)

        # Add ID field
        pLayer.CreateField(ogr.FieldDefn('lineid', ogr.OFTInteger64))

        # Add attribute fields
        dtype_to_ogr = {'int': ogr.OFTInteger64, 'float': ogr.OFTReal}
        if has_attributes:
            for field, dtype in zip(aAttribute_field, aAttribute_dtype):
                if dtype not in dtype_to_ogr:
                    raise ValueError(f'Unsupported data type: {dtype}. Must be "int" or "float"')
                pLayer.CreateField(ogr.FieldDefn(field, dtype_to_ogr[dtype]))

        pLayerDefn = pLayer.GetLayerDefn()
        coord_attrs = ('dx', 'dy') if is_projected else ('dLongitude_degree', 'dLatitude_degree')

        # Add features
        for lID in range(nFlowline):
            pFlowline = aFlowline_in[lID]
            pLine = ogr.Geometry(ogr.wkbLineString)
            
            for vertex in pFlowline.aVertex:
                pLine.AddPoint(*(getattr(vertex, attr) for attr in coord_attrs))

            pLine.FlattenTo2D()
            
            pFeature = ogr.Feature(pLayerDefn)
            pFeature.SetGeometry(pLine)
            pFeature.SetField("lineid", lID + 1)

            if has_attributes:
                for sField, dtype, data_list in zip(aAttribute_field, aAttribute_dtype, aAttribute_data):
                    value = int(data_list[lID]) if dtype == 'int' else float(data_list[lID])
                    pFeature.SetField(sField, value)

            pLayer.CreateFeature(pFeature)
            pFeature = None

        # Flush and cleanup
        pDataset.FlushCache()
        pDataset = None

    except Exception as e:
        raise RuntimeError(f"Error exporting flowlines to {sFilename_vector_out}: {str(e)}")





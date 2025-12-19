import os
from typing import List, Optional
from osgeo import ogr, osr
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_driver_from_filename

def export_vertex_to_vector(aVertex_in: List,
        sFilename_vector_out: str,
        iFlag_projected_in: Optional[int] = None,
        pSpatial_reference_in: Optional[osr.SpatialReference] = None,
        aAttribute_data: Optional[List[int]] = None) -> None:
    """
    Export vertices to a vector format file (shapefile, GeoJSON, etc.).

    Args:
        aVertex_in (list): List of vertex objects to export
        sFilename_vector_out (str): Output file path
        iFlag_projected_in (int, optional): Flag indicating if coordinates are projected (1) or geographic (0). Defaults to None (treated as 0).
        pSpatial_reference_in (osr.SpatialReference, optional): Spatial reference system. Defaults to WGS84 (EPSG:4326).
        aAttribute_data (list, optional): List of attribute values corresponding to vertices. Defaults to None.
    
    Returns:
        None
    """
    try:
        pDriver = get_vector_driver_from_filename(sFilename_vector_out)
        if os.path.exists(sFilename_vector_out):
            os.remove(sFilename_vector_out)

        # Set default values
        is_projected = iFlag_projected_in == 1 if iFlag_projected_in is not None else False
        if pSpatial_reference_in is None:
            pSpatial_reference_in = osr.SpatialReference()
            pSpatial_reference_in.ImportFromEPSG(4326)  # WGS84 lat/lon

        has_attributes = aAttribute_data is not None and len(aAttribute_data) > 0

        # Create datasource and layer
        pDataset = pDriver.CreateDataSource(sFilename_vector_out)
        pLayer = pDataset.CreateLayer('vertex', pSpatial_reference_in, ogr.wkbPoint)
        
        # Add fields
        pLayer.CreateField(ogr.FieldDefn('pointid', ogr.OFTInteger64))
        if has_attributes:
            pLayer.CreateField(ogr.FieldDefn('connectivity', ogr.OFTInteger64))

        pLayerDefn = pLayer.GetLayerDefn()

        # Add features
        for lID, pVertex in enumerate(aVertex_in):
            pPoint = ogr.Geometry(ogr.wkbPoint)
            if is_projected:
                pPoint.AddPoint(pVertex.dx, pVertex.dy)
            else:
                pPoint.AddPoint(pVertex.dLongitude_degree, pVertex.dLatitude_degree)

            pFeature = ogr.Feature(pLayerDefn)
            pFeature.SetGeometry(pPoint)
            pFeature.SetField("pointid", lID + 1)

            if has_attributes:
                pFeature.SetField("connectivity", int(aAttribute_data[lID]))

            pLayer.CreateFeature(pFeature)
            pFeature = None

        # Flush and cleanup
        pDataset.FlushCache()
        pDataset = None
        
    except Exception as e:
        raise RuntimeError(f"Error exporting vertices to {sFilename_vector_out}: {str(e)}")

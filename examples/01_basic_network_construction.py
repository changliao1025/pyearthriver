#!/usr/bin/env python3
"""
Example 01: Basic River Network Construction and Exploration.

Demonstrates:
1. Loading river flowlines from a GeoJSON file using read_flowlines_from_geojson.
2. Building a directed pyrivergraph.
3. Querying river network topology (vertices, headwaters/sources, outlets/sinks).
4. Exporting the processed flowlines.
"""

import os
from pyearthriver import (
    pyrivergraph,
    read_flowlines_from_geojson,
    export_flowline_to_vector,
)


def main():
    # 1. Locate sample dataset
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    geojson_path = os.path.join(data_dir, "linear_network.geojson")

    print(f"Loading river flowlines from: {geojson_path}")
    flowlines = read_flowlines_from_geojson(geojson_path)
    print(f"Loaded {len(flowlines)} flowline reach(es).")

    for f in flowlines:
        print(f"  Reach ID {f.lFlowlineID}: length={f.dLength:.1f}m, vertices={len(f.aVertex)}")

    # 2. Build the river network graph
    outlet_vertex = flowlines[-1].pVertex_end
    print(f"\nBuilding pyrivergraph with outlet at: ({outlet_vertex.dLongitude_degree:.3f}, {outlet_vertex.dLatitude_degree:.3f})")
    graph = pyrivergraph(flowlines, outlet_vertex)

    # 3. Query network graph metrics
    sources = graph.get_sources()
    sinks = graph.get_sinks()
    total_vertices = graph.get_vertex_count()

    print(f"Network Summary:")
    print(f"  Total Vertices: {total_vertices}")
    print(f"  Headwater Nodes (Sources): {sources}")
    print(f"  Mouth/Outlet Nodes (Sinks): {sinks}")

    # 4. Export flowlines to a vector file
    output_path = "output_linear_network.geojson"
    export_flowline_to_vector(graph.aFlowline, output_path)
    print(f"\nExported network to {output_path}")

    # Cleanup temporary output
    if os.path.exists(output_path):
        os.remove(output_path)
        print("Cleaned up temporary export file.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example 04: End-to-End River Network Conditioning Pipeline (pyflowline integration).

Demonstrates:
1. Loading raw river network hydrography data from GeoJSON.
2. Invoking the high-level condition_river_network() pipeline function.
3. Automatically executing duplicate removal, disconnected channel pruning,
   braided river simplification, cycle detection, confluence identification,
   and Strahler stream ordering.
4. Exporting the topologically conditioned network ready for hydrologic modeling.
"""

import os
from pyearthriver import (
    condition_river_network,
    read_flowlines_from_geojson,
    export_flowline_to_vector,
)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    geojson_path = os.path.join(data_dir, "braided_network.geojson")

    print("=================================================================")
    print("PyEarthRiver: End-to-End River Conditioning Pipeline")
    print("=================================================================")

    # 1. Load raw hydrography flowlines
    print(f"\n1. Ingesting raw vector flowlines from: {geojson_path}")
    raw_flowlines = read_flowlines_from_geojson(geojson_path)
    print(f"   Raw reaches count: {len(raw_flowlines)}")

    # 2. Execute the complete conditioning pipeline
    print("\n2. Executing condition_river_network pipeline...")
    graph, conditioned_flowlines = condition_river_network(
        raw_flowlines,
        outlet_vertex=raw_flowlines[-1].pVertex_end,
        remove_duplicates=True,
        remove_disconnected=True,
        remove_braids=True,
        remove_parallels=True,
        remove_cycles=True,
        define_topology=True,
        compute_stream_order=True,
    )

    print("\n3. Conditioning Pipeline Completed Successfully!")
    print(f"   Conditioned flowline reaches count: {len(conditioned_flowlines)}")
    for f in conditioned_flowlines:
        print(f"   - Reach ID {f.lFlowlineID}: Order={f.iStream_order}, Length={f.dLength:.1f}m")

    # 4. Export conditioned flowlines for downstream hydrologic modeling
    out_vector = "conditioned_network.geojson"
    print(f"\n4. Exporting conditioned hydrography to: {out_vector}")
    export_flowline_to_vector(conditioned_flowlines, out_vector)

    if os.path.exists(out_vector):
        os.remove(out_vector)
        print("   (Cleaned up temporary export output)")

    print("\nThe river network is topologically conditioned, loop-free, and ready")
    print("for mesh generation (pyflowline) or routing engines (MOSART / MPAS-River).")


if __name__ == "__main__":
    main()

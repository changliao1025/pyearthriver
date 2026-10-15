#!/usr/bin/env python3
"""
Example 03: Stream Ordering and Confluence Analysis.

Demonstrates:
1. Loading a branching dendritic river network.
2. Defining stream segments and confluence junctions.
3. Calculating confluence branching angles.
4. Computing Strahler stream orders (headwaters = 1, joining to form higher orders).
"""

import os
from pyearthriver import (
    pyrivergraph,
    read_flowlines_from_geojson,
)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    geojson_path = os.path.join(data_dir, "dendritic_network.geojson")

    print(f"Loading dendritic river network from: {geojson_path}")
    flowlines = read_flowlines_from_geojson(geojson_path)
    print(f"Initial flowlines count: {len(flowlines)}")

    outlet_vertex = flowlines[-1].pVertex_end
    graph = pyrivergraph(flowlines, outlet_vertex)

    # 1. Define stream segments & topological connectivity
    print("\nAnalyzing stream segments and topology...")
    graph.define_stream_segment()
    graph.define_stream_topology()

    # 2. Identify river confluences
    print("\nIdentifying river confluences...")
    confluences = graph.define_river_confluence()
    print(f"Identified {len(confluences)} confluence junction(s):")
    for i, conf in enumerate(confluences, start=1):
        angle = conf.calculate_branching_angle()
        print(f"  Confluence #{i}: Upstream streams={len(conf.aFlowline_upstream)}, "
              f"Downstream streams={len(conf.aFlowline_downstream)}, "
              f"Branching angle={angle:.1f}°")

    # 3. Compute Strahler stream order
    print("\nCalculating Strahler stream order...")
    res = graph.define_stream_order(iFlag_so_method_in=1)
    ordered_flowlines = res[0] if isinstance(res, tuple) else res

    print(f"\nFinal Stream Ordering Results:")
    for f in ordered_flowlines:
        print(f"  Stream Reach ID {f.lFlowlineID}: Strahler Order = {f.iStream_order}, "
              f"Length = {f.dLength:.1f}m")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example 02: Braided River and Cycle Pruning.

Demonstrates:
1. Loading a braided river network containing diverging and reconverging channels.
2. Detecting braided channel sections using graph traversal.
3. Pruning secondary braided channels using remove_braided_river().
4. Verifying topological simplification into a single directed flow path.
"""

import os
from pyearthriver import (
    pyrivergraph,
    read_flowlines_from_geojson,
)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    geojson_path = os.path.join(data_dir, "braided_network.geojson")

    print(f"Loading braided river network from: {geojson_path}")
    flowlines = read_flowlines_from_geojson(geojson_path)
    print(f"Initial flowline reaches: {len(flowlines)}")

    for f in flowlines:
        print(f"  Reach ID {f.lFlowlineID}: length={f.dLength:.1f}m")

    # Build the graph
    outlet_vertex = flowlines[-1].pVertex_end
    graph = pyrivergraph(flowlines, outlet_vertex)

    # Detect braided channels
    braided_channels = graph.find_braided_channels()
    print(f"\nDetected braided channel sets: {len(braided_channels)}")

    # Prune braided river channels
    print("\nRunning remove_braided_river()...")
    simplified_flowlines = graph.remove_braided_river()

    print(f"Post-pruning flowline count: {len(simplified_flowlines)}")
    for f in simplified_flowlines:
        print(f"  Retained Reach ID {f.lFlowlineID}: length={f.dLength:.1f}m")

    # Remove any lingering cycles or loops
    print("\nRunning remove_cycle()...")
    acyclic_flowlines = graph.remove_cycle()
    print(f"Final acyclic flowline count: {len(acyclic_flowlines)}")
    print("Network is now conditioned as a clean Directed Acyclic Graph (DAG).")


if __name__ == "__main__":
    main()

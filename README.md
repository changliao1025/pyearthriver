# PyEarthRiver

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/pyearthriver.svg)](https://badge.fury.io/py/pyearthriver)

A Python library for analyzing and processing river network flowlines using graph-based algorithms. PyEarthRiver handles complex river networks with braided channels, loops, and parallel streams.

## Features

- **River Network Graph Analysis**: Build and analyze river networks as directed graphs
- **Braided Channel Detection**: Identify and process braided river channels
- **Cycle Detection**: Detect and handle loops in river networks
- **Confluence Analysis**: Analyze river confluence points and junctions
- **Flowline Processing**: Process and manipulate river flowline data
- **Topological Operations**: Perform various topological operations on river networks

## Installation

Install PyEarthRiver from PyPI:

```bash
pip install pyearthriver
```

Or install from source:

```bash
git clone https://github.com/changliao1025/pyearthriver.git
cd pyearthriver
pip install -e .
```

## Quick Start

### 1. High-Level River Network Conditioning Pipeline

Condition a raw river network in a single call:

```python
from pyearthriver import (
    condition_river_network,
    read_flowlines_from_geojson,
    export_flowline_to_vector,
)

# Load flowlines from GeoJSON
flowlines = read_flowlines_from_geojson("river_network.geojson")

# Execute end-to-end topological conditioning
graph, conditioned_flowlines = condition_river_network(
    flowlines,
    outlet_vertex=flowlines[-1].pVertex_end,
    remove_braids=True,
    remove_cycles=True,
    compute_stream_order=True,
)

# Export cleaned network for downstream hydrologic models (e.g. pyflowline, MOSART)
export_flowline_to_vector(conditioned_flowlines, "conditioned_network.geojson")
```

### 2. Fine-Grained Graph Analysis

```python
from pyearthriver import pyrivergraph, read_flowlines_from_geojson

flowlines = read_flowlines_from_geojson("river_network.geojson")
graph = pyrivergraph(flowlines)

# Detect and remove braided river channels
graph.remove_braided_river()

# Detect and break cycles/loops
graph.remove_cycle()

# Calculate Strahler stream orders
ordered_flowlines, orders = graph.define_stream_order(iFlag_so_method_in=1)
```

## Examples

Runnable examples with synthetic sample datasets are available in the [`examples/`](examples/) folder:
- **`01_basic_network_construction.py`**: Reading GeoJSON, querying sources/sinks, and exporting vector data.
- **`02_braided_river_and_cycle_removal.py`**: Detecting braided loops, pruning secondary branches, and breaking cycles.
- **`03_stream_ordering_and_confluences.py`**: Calculating confluence branching angles and Strahler stream orders.
- **`04_pyflowline_integration_pipeline.py`**: The complete end-to-end conditioning pipeline used by `pyflowline`.

## Main Classes

### `pyrivergraph`
Main class for river network analysis. Provides methods for:
- Building river network graphs
- Detecting and removing braided channels
- Identifying cycles and loops
- Analyzing network topology
- Exporting results

### `pyvertex`
Represents a vertex (node) in the river network with spatial coordinates.

### `pyflowline`
Represents a flowline (edge sequence) connecting vertices in the network.

### `pyedge`
Represents an edge connection between two vertices.

### `pyconfluence`
Represents a confluence point where multiple streams meet.

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0

## Development

To set up a development environment:

```bash
git clone https://github.com/changliao1025/pyearthriver.git
cd pyearthriver
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PyEarthRiver in your research, please cite:

```bibtex
@software{pyearthriver,
  author = {Liao, Chang},
  title = {PyEarthRiver: River Network Graph Analysis Library},
  year = {2025},
  url = {https://github.com/changliao1025/pyearthriver}
}
```

## Authors

- Chang Liao - Pacific Northwest National Laboratory

## Acknowledgments

This work was supported by the U.S. Department of Energy's Earth and Environmental System Sciences Division.

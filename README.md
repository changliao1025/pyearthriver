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

```python
from pyearthriver import pyrivergraph, pyvertex, pyflowline

# Create vertices
outlet_vertex = pyvertex(id=1, x=100.0, y=200.0)
inlet_vertex = pyvertex(id=2, x=110.0, y=210.0)

# Create flowlines
flowlines = [
    pyflowline(id=1, vertices=[outlet_vertex, inlet_vertex])
]

# Build river network graph
graph = pyrivergraph(flowlines, outlet_vertex)

# Analyze the network
graph.remove_braided_river()
graph.detect_cycles()

# Export results
graph.export_flowlines("output.geojson")
```

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

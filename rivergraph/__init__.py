"""
PyRivergraph - River Network Graph Analysis Library

A Python library for analyzing and processing river network flowlines using
graph-based algorithms. Handles complex river networks with braided channels,
loops, and parallel streams.

Main Classes:
    pyrivergraph: Main class for river network analysis (facade)
    pyvertex: Vertex representation in the network
    pyflowline: Flowline (edge sequence) representation
    pyedge: Edge representation between vertices
    pyconfluence: Confluence point representation

Example:
    >>> from rivergraph import pyrivergraph
    >>> graph = pyrivergraph(flowlines, outlet_vertex)
    >>> graph.remove_braided_river()
    >>> graph.detect_cycles()
"""

__version__ = "0.2.0"
__author__ = "Chang Liao"

# Import main classes for convenient access
# Updated to use new modular structure
from rivergraph.classes.vertex import pyvertex
from rivergraph.classes.flowline import pyflowline
from rivergraph.classes.edge import pyedge
from rivergraph.classes.confluence import pyconfluence
# Import from new modular location
from rivergraph.core.rivergraph import pyrivergraph

__all__ = [
    'pyrivergraph',
    'pyvertex',
    'pyflowline',
    'pyedge',
    'pyconfluence',
]
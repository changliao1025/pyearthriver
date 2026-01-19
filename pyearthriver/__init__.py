"""
PyEarthRiver - River Network Graph Analysis Library

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
    >>> from pyearthriver import pyrivergraph
    >>> graph = pyrivergraph(flowlines, outlet_vertex)
    >>> graph.remove_braided_river()
    >>> graph.detect_cycles()
"""

__version__ = "0.2.0"
__author__ = "Chang Liao"

# Import main classes for convenient access
from pyearthriver.classes.vertex import pyvertex
from pyearthriver.classes.flowline import pyflowline
from pyearthriver.classes.edge import pyedge
from pyearthriver.classes.confluence import pyconfluence
from pyearthriver.classes.rivergraph import pyrivergraph

__all__ = [
    'pyrivergraph',
    'pyvertex',
    'pyflowline',
    'pyedge',
    'pyconfluence',
]
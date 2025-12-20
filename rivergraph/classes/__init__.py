"""
Core data classes for river network representation.

This module contains the fundamental data structures used throughout
the rivergraph library.
"""

from .vertex import pyvertex
from .flowline import pyflowline
from .edge import pyedge
from .confluence import pyconfluence
# Import from new modular location for backward compatibility
from ..core.rivergraph import pyrivergraph

__all__ = [
    'pyvertex',
    'pyflowline',
    'pyedge',
    'pyconfluence',
    'pyrivergraph',
]
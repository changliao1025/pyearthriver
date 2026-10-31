"""
Stream ordering and headwater classification algorithms (Strahler and Shreve).
"""

from .strahler import (
    define_stream_order,
    identify_headwater_flowlines,
    process_confluences_iteratively,
    update_headwater_stream_order,
)

__all__ = [
    "define_stream_order",
    "identify_headwater_flowlines",
    "update_headwater_stream_order",
    "process_confluences_iteratively",
]

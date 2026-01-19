#!/usr/bin/env python
"""
Setup.py for backward compatibility.
Configuration is in pyproject.toml (PEP 517/518).
Cython extensions are handled by build_backend.py.
"""

from setuptools import setup
from build_backend import get_extensions

# Get Cython extensions
ext_modules = get_extensions()

# Configuration is in pyproject.toml
setup(ext_modules=ext_modules)

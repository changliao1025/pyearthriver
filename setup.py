#!/usr/bin/env python
"""
Setup.py for backward compatibility.
Configuration is in pyproject.toml (PEP 517/518).
Cython extensions are handled within this file.
"""

from setuptools import setup, Extension


def get_extensions():
    """Build list of extension modules with proper numpy include paths.

    Returns:
        list: List of Extension objects for Cython modules
    """
    import numpy

    # Try to import Cython, but don't fail if it's not available
    try:
        from Cython.Build import cythonize
        HAVE_CYTHON = True
    except ImportError:
        HAVE_CYTHON = False
        cythonize = None

    # Define Cython extension sources
    if HAVE_CYTHON:
        ext_sources = {
            "pyearthriver.algorithms.cython.kernel": ["pyearthriver/algorithms/cython/kernel.pyx"],
        }
    else:
        ext_sources = {
            "pyearthriver.algorithms.cython.kernel": ["pyearthriver/algorithms/cython/kernel.c"],
        }

    # Build extensions list
    extensions = [
        Extension(
            name,
            sources,
            include_dirs=[numpy.get_include()],
            libraries=[],
            library_dirs=[],
            language="c++",  # Specify C++ language for libcpp support
        )
        for name, sources in ext_sources.items()
    ]

    # Cythonize if Cython is available
    if HAVE_CYTHON:
        extensions = cythonize(
            extensions,
            compiler_directives={'language_level': "3"},
            force=False  # Only rebuild if source changed
        )

    return extensions


# Get Cython extensions
ext_modules = get_extensions()

# Configuration is in pyproject.toml
setup(ext_modules=ext_modules)

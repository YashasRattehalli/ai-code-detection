"""
AI Code Detector package.

This package provides tools and models to detect AI-generated code.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("code-detector")
except PackageNotFoundError:
    __version__ = "unknown" 
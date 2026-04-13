#!/usr/bin/env python3
"""Compatibility shim for setuptools-based workflows.

Canonical build metadata now lives in ``pyproject.toml``. This file remains so
legacy commands such as ``python setup.py --version`` and editable-install
entrypoints continue to work without maintaining duplicated packaging metadata.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()

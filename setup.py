#!/usr/bin/env python3
"""
setup.py

Editable/standard installs for the MY_SWAMP JAX port using a src/ layout.

Expected layout:
  repo_root/
    setup.py
    README.md
    src/
      my_swamp/
        __init__.py
        model.py
        spectral_transform.py
        ...
"""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


def _read_readme() -> str:
    root = Path(__file__).resolve().parent
    readme = root / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return "MY_SWAMP: JAX port of SWAMPE."


setup(
    # pip distribution name (this is what shows up in `pip list`)
    name="MY_SWAMP",
    version="0.1.0",
    description="MY_SWAMP: JAX port of SWAMPE",
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    # src/ layout
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "imageio>=2.31",
        # CPU-only JAX is fine: pip resolves the right jaxlib wheel
        "jax>=0.4",
        "jaxlib>=0.4",
    ],
    extras_require={
        "dev": [
            "pytest>=7",
        ],
    },
)

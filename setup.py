#!/usr/bin/env python3
"""
setup.py

Packaging for a src/ layout:

  repo_root/
    setup.py
    README.md
    pyproject.toml          (recommended; see steps below)
    src/
      my_swamp/
        __init__.py
        ...

Notes
- Distribution name (pip install ...) can differ from import name (import my_swamp).
- PyPI names are case-insensitive and normalize "_" to "-".
"""

from __future__ import annotations

import re
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent


def _read_readme() -> str:
    readme = ROOT / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return "MY_SWAMP: JAX port + reference implementation of SWAMPE."


def _read_version(default: str = "0.1.0") -> str:
    """
    If you later add src/my_swamp/_version.py containing __version__ = "x.y.z",
    this will pick it up without importing my_swamp (avoids importing JAX at build time).
    """
    version_file = ROOT / "src" / "my_swamp" / "_version.py"
    if not version_file.exists():
        return default

    text = version_file.read_text(encoding="utf-8")
    m = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", text)
    return m.group(1) if m else default


setup(
    # What users will `pip install ...` (choose something unique on PyPI).
    name="my-swamp",
    version=_read_version(),
    description="MY_SWAMP: JAX port + reference implementation of SWAMPE",
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    # src/ layout
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=False,
    exclude_package_data={"": ["*.pyc", "*.pyo", "*.pyd", "__pycache__/*", ".DS_Store", "__MACOSX/*"]},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "imageio>=2.31",
        # Do NOT pin jaxlib here; let users choose CPU/GPU via JAX's recommended installs.
        "jax>=0.4,<1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7",
            "build>=1.2",
            "twine>=5",
        ],
    },
    entry_points={
        "console_scripts": [
            # provides `my-swamp` command -> runs the argparse CLI in main_function.py
            "my-swamp=my_swamp.main_function:cli_main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
)

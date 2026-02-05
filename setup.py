from setuptools import setup, find_packages

setup(
    name="SWAMPE",
    version="1.0.0+jax",
    description="2D Shallow-Water General Circulation Model for Exoplanet Atmospheres (JAX rewrite)",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Ekaterina Landgren (original), JAX port contributors",
    license="BSD-3-Clause",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "jax",
        "jaxlib",
        "matplotlib",
        "imageio",
    ],
    extras_require={
        "scipy": ["scipy"],
    },
    zip_safe=False,
)

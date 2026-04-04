"""Compile waveid.crypto to a native C extension via Cython.

Usage:
    python setup_cython.py build_ext --inplace

Produces:
    src/waveid/crypto.cpython-<version>-<platform>.so
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="waveid.crypto",
        sources=["src/waveid/crypto.py"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    ),
)

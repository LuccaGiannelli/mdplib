import sys

from setuptools import setup, Extension
import pybind11
import numpy as np

# MSVC (used on Windows, e.g. by cibuildwheel's windows-latest runner) takes
# different flags than GCC/Clang: /std:c++17 instead of -std=c++17, and it
# already optimizes reasonably at /O2 (its highest conventional level; /Ox
# is not a general recommendation the way -O3 is for GCC/Clang).
if sys.platform == "win32":
    extra_compile_args = ["/std:c++17", "/O2"]
else:
    extra_compile_args = ["-std=c++17", "-O3"]  # O3 = max optimization

ext = Extension(
    name="mdp._core._mdp_core",
    sources=["mdp/_core/mdp_core.cpp"],
    include_dirs=[
        pybind11.get_include(),
        np.get_include(),
    ],
    language="c++",
    extra_compile_args=extra_compile_args,
)

setup(
    name="mdplib",
    ext_modules=[ext],
)

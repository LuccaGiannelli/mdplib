from setuptools import setup, Extension
import pybind11
import numpy as np

ext = Extension(
    name="mdp._core._mdp_core",
    sources=["mdp/_core/mdp_core.cpp"],
    include_dirs=[
        pybind11.get_include(),
        np.get_include(),
    ],
    language="c++",
    extra_compile_args=["-std=c++17", "-O3"],  # O3 = max optimization
)

setup(
    name="mdp-solver-ic",
    ext_modules=[ext],
)

# Import the compiled C++ module
try:
    from mdp._core import _mdp_core
except ImportError as e:
    raise ImportError(
        "The C++ module '_mdp_core' was not compiled. "
        "Run 'pip install .' from within the mdp-solver folder to compile it."
    ) from e

# Import the compiled C++ module
try:
    from mdp._core import _mdp_core
except ImportError as e:
    raise ImportError(
        "The C++ module '_mdp_core' was not compiled. This usually means no "
        "prebuilt wheel was available for your platform/Python version and "
        "the source build failed. Install a C++17 compiler and re-run "
        "'pip install mdplib' (or 'pip install .' from the repository root)."
    ) from e

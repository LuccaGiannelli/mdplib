# Contributing to mdplib

Thank you for your interest in mdplib. Contributions, bug reports and questions are welcome.

## Reporting bugs

Open an issue at <https://github.com/LuccaGiannelli/mdplib/issues> and include:

- the mdplib version (`pip show mdplib`), Python version and operating system;
- a minimal example that reproduces the problem;
- what you expected to happen and what happened instead, with the full traceback if there is one.

## Asking for help or suggesting a feature

Open an issue describing your use case. For a new algorithm, a reference to the paper or textbook that defines it helps a lot.

## Contributing code

1. Fork the repository and create a branch from `main`.
2. Install the library from source in editable mode. This compiles the C++ core, so a C++17 compiler is required:

   ```bash
   pip install -e .
   pip install pytest
   ```

3. Make your change. Keep each pull request focused on a single topic.
4. Add or update tests in `tests/` for any change in behaviour.
5. Run the test suite and make sure it passes:

   ```bash
   pytest tests/
   ```

6. Open a pull request describing what the change does and why.

The README explains how the library is organised and how to add a new algorithm.

## License

By contributing, you agree that your contributions will be licensed under the MIT License that covers this project.

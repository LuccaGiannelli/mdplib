# mdp-solver-ic

[![PyPI](https://img.shields.io/pypi/v/mdp-solver-ic.svg)](https://pypi.org/project/mdp-solver-ic/)
[![Python](https://img.shields.io/pypi/pyversions/mdp-solver-ic.svg)](https://pypi.org/project/mdp-solver-ic/)

A Python library of algorithms for solving infinite-horizon, discounted **Markov Decision Processes (MDPs)**, with a performance-critical core written in C++ (via [pybind11](https://github.com/pybind/pybind11)).

Implements:

- **Value Iteration** (Jacobi and Gauss-Seidel variants)
- **Policy Iteration**
- **Q-Learning** and **SARSA** (model-free reinforcement learning)

## Installation

```bash
pip install mdp-solver-ic
```

Building from source requires a C++17 compiler (the extension is compiled at install time):

```bash
git clone https://github.com/LuccaGiannelli/mdp-solver.git
cd mdp-solver
pip install .
```

## Basic usage

```python
from mdp import MDP

S = [0, 1, 2]   # states
A = [0, 1, 2]   # actions

def Pt(sn, a, sn1):
    # returns P(sn1 | sn, a)
    ...

def rt(sn, a, sn1):
    # returns the reward
    ...

problem = MDP(S, A, Pt, rt)

# Value Iteration
problem.VI(gamma=0.9, tol=1e-6)
problem.print_results()

# Policy Iteration
problem.PI(gamma=0.9, tol=1e-6)
problem.print_results()
```

See [`examples/basic_example.py`](examples/basic_example.py) for a complete example (a machine maintenance problem).

## Available algorithms

| Method | Description |
|--------|-------------|
| `VI`   | Value Iteration (Jacobi) |
| `VIGS` | Value Iteration with Gauss-Seidel |
| `PI`   | Policy Iteration |
| `QLearning` | Q-Learning (off-policy TD control) |
| `SARSA`     | SARSA (on-policy TD control) |

## Common parameters

| Parameter | Description |
|-----------|-------------|
| `gamma`     | Discount factor (0 ≤ γ < 1) |
| `tol`       | Convergence tolerance |
| `max_iter`  | Maximum number of iterations |
| `v0`        | Initial value vector (optional) |
| `verbose`   | Print progress on completion |
| `norm_type` | 0 = relative sup norm, 1 = sup norm, 2 = euclidean norm |
| `epsilon`   | Final epsilon for the epsilon-greedy policy (Q-Learning / SARSA) |
| `alpha`     | Learning rate (Q-Learning / SARSA) |

## How it works

You define the state space `S`, action space `A`, a transition probability function `Pt(sn, a, sn1)`, and a reward function `rt(sn, a, sn1)`. The constructor enumerates all feasible transitions into a CSV (`sn,a,sn1,prob,reward`) and loads it into a NumPy array; the algorithms then hand that array to the C++ core, which does the inner loops with direct memory access for performance.

After solving, results are available as:

| Attribute | Description |
|-----------|-------------|
| `value`        | optimal value for each state |
| `policy`       | optimal action for each state |
| `time`         | execution time (seconds) |
| `n_iterations` | number of iterations to convergence |

## Adding a new algorithm

1. Create `mdp/algorithms/new_algo.py` with a function `def NewAlgo(self, ...):`
2. Import it in `mdp/algorithms/__init__.py`
3. Attach it to the class in `mdp/core.py`: `NewAlgo = NewAlgo`

## Tests

```bash
pip install pytest
pytest tests/
```

## License

TBD.

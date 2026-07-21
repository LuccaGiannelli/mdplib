import time
from mdp._core import _mdp_core


#******************************************************************
# Pure Value Iteration
#******************************************************************

def VI(self, gamma, tol, v0=None, max_iter=1000, verbose=True, norm_type=0):
    """
    Value Iteration (Jacobi).

    Computes the optimal value vector and the optimal policy using
    synchronous updates. The inner loop runs in C++ via pybind11.

    Parameters
    ----------
    gamma : float
        Discount factor (0 <= gamma < 1).
    tol : float
        Convergence tolerance.
    v0 : list, optional
        Initial value vector. Default: zeros.
    max_iter : int, optional
        Maximum number of iterations. Default: 1000.
    verbose : bool, optional
        If True, prints the final result. Default: True.
    norm_type : int, optional
        0 = relative sup norm, 1 = sup norm, 2 = euclidean norm. Default: 0.

    Returns
    -------
    self
    """
    start_time = time.time()

    dimS = len(self.S)
    if v0 is None:
        v0 = [0.0] * dimS

    result = _mdp_core.vi_cpp(
        self.transition_matrix,
        self.transition_index,
        dimS,
        gamma,
        tol,
        v0,
        max_iter,
        norm_type,
    )

    self.value        = list(result["value"])
    self.policy       = [self.A[i] for i in result["policy_idx"]]
    self.n_iterations = result["n_iter"]
    self.time         = time.time() - start_time

    if verbose:
        print(f'Time: {self.time:.4f}s | Iterations: {self.n_iterations}')
    return self


#******************************************************************
# Value Iteration with Gauss-Seidel
#******************************************************************

def VIGS(self, gamma, tol, v0=None, max_iter=1000, verbose=True, norm_type=0):
    """
    Value Iteration with Gauss-Seidel.

    Same as VI, but with asynchronous updates. The inner loop runs in C++.

    Parameters
    ----------
    gamma : float
        Discount factor (0 <= gamma < 1).
    tol : float
        Convergence tolerance.
    v0 : list, optional
        Initial value vector. Default: zeros.
    max_iter : int, optional
        Maximum number of iterations. Default: 1000.
    verbose : bool, optional
        If True, prints the final result. Default: True.
    norm_type : int, optional
        0 = relative sup norm, 1 = sup norm, 2 = euclidean norm. Default: 0.

    Returns
    -------
    self
    """
    start_time = time.time()

    dimS = len(self.S)
    if v0 is None:
        v0 = [0.0] * dimS

    result = _mdp_core.vigs_cpp(
        self.transition_matrix,
        self.transition_index,
        dimS,
        gamma,
        tol,
        v0,
        max_iter,
        norm_type,
    )

    self.value        = list(result["value"])
    self.policy       = [self.A[i] for i in result["policy_idx"]]
    self.n_iterations = result["n_iter"]
    self.time         = time.time() - start_time

    if verbose:
        print(f'Time: {self.time:.4f}s | Iterations: {self.n_iterations}')
    return self

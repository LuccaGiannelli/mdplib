import time
import random
from mdp._core import _mdp_core


#******************************************************************
# Q-Learning
#******************************************************************

def QLearning(self, gamma, epsilon, alpha, s0=None, q0=None, max_iter=1000, verbose=True, seed=0):
    """
    Q-Learning (off-policy TD control).

    Learns Q(s,a) by interacting with the environment. Inner loop in C++.

    Parameters
    ----------
    gamma : float
        Discount factor (0 <= gamma < 1).
    epsilon : float
        Final epsilon of the epsilon-greedy policy.
    alpha : float
        Learning rate (0 < alpha <= 1).
    s0 : int, optional
        Initial state index. Default: random.
    q0 : ignored
        Reserved for future compatibility.
    max_iter : int, optional
        Number of simulation steps. Default: 1000.
    verbose : bool, optional
        If True, prints the final result. Default: True.
    seed : int, optional
        Seed of the random number generator, for reproducibility. Default: 0.

    Returns
    -------
    self
    """
    start_time = time.time()

    dimS = len(self.S)
    if s0 is None:
        s0 = random.randint(0, dimS - 1)

    result = _mdp_core.qlearning_cpp(
        self.transition_matrix,
        self.transition_index,
        dimS,
        gamma,
        epsilon,
        alpha,
        s0,
        max_iter,
        seed,
    )

    self.value        = list(result["value"])
    self.policy       = [self.A[i] for i in result["policy_idx"]]
    self.n_iterations = result["n_iter"]
    self.time         = time.time() - start_time

    if verbose:
        print(f'Time: {self.time:.4f}s | Iterations: {self.n_iterations}')
    return self


#******************************************************************
# SARSA
#******************************************************************

def SARSA(self, gamma, epsilon, alpha, s0=None, q0=None, max_iter=1000, verbose=True, seed=0):
    """
    SARSA (on-policy TD control).

    Same as Q-Learning but on-policy. Inner loop in C++.

    Parameters
    ----------
    gamma : float
        Discount factor (0 <= gamma < 1).
    epsilon : float
        Final epsilon of the epsilon-greedy policy.
    alpha : float
        Learning rate (0 < alpha <= 1).
    s0 : int, optional
        Initial state index. Default: random.
    q0 : ignored
        Reserved for future compatibility.
    max_iter : int, optional
        Number of simulation steps. Default: 1000.
    verbose : bool, optional
        If True, prints the final result. Default: True.
    seed : int, optional
        Seed of the random number generator, for reproducibility. Default: 0.

    Returns
    -------
    self
    """
    start_time = time.time()

    dimS = len(self.S)
    if s0 is None:
        s0 = random.randint(0, dimS - 1)

    result = _mdp_core.sarsa_cpp(
        self.transition_matrix,
        self.transition_index,
        dimS,
        gamma,
        epsilon,
        alpha,
        s0,
        max_iter,
        seed,
    )

    self.value        = list(result["value"])
    self.policy       = [self.A[i] for i in result["policy_idx"]]
    self.n_iterations = result["n_iter"]
    self.time         = time.time() - start_time

    if verbose:
        print(f'Time: {self.time:.4f}s | Iterations: {self.n_iterations}')
    return self

import numpy as np

from mdp.utils import (
    build_transition_matrix,
    print_results as _print_results,
    print_partial_results as _print_partial_results,
    save_results as _save_results,
)
from mdp._core import _mdp_core
from mdp.algorithms.vi import VI, VIGS
from mdp.algorithms.pi import PI
from mdp.algorithms.rl import QLearning, SARSA


#******************************************************************
# MDP OBJECT
#******************************************************************

class MDP(object):
    """
    Markov Decision Process with infinite horizon and discount factor.

    The computational core (VI, VIGS, PI, Q-Learning, SARSA) runs in C++
    via pybind11. The Python interface stays the same.

    Parameters
    ----------
    S : list
        List of all possible states.
    A : list
        List of all possible actions.
    Pt : callable
        Function (sn, a, sn1) -> float with P(sn1 | sn, a).
    rt : callable
        Function (sn, a, sn1) -> float with the reward.
    build_transition_file : bool, optional
        If True, (re)generates the transition matrix CSV. Default: True.
    transition_file : str, optional
        Path to the CSV. Default: "MTrans.csv".
    S1a : list, optional
        For factored models.
    S1 : list, optional
        For factored models.

    Attributes after optimization
    ------------------------------
    value : list
    policy : list
    time : float
    n_iterations : int
    """

    def __init__(self, S, A, Pt, rt, build_transition_file=True, transition_file="MTrans.csv", S1a=None, S1=None):
        self.S = S
        self.A = A
        self.Pt = Pt
        self.rt = rt

        S1aVI = None
        if S1a is not None:
            S1aVI = list(S1a)
            S1a = [[i, S1a[i][0], S1a[i][1]] for i in range(len(S1a))]
            S1a.sort(key=lambda x: x[2], reverse=True)
            cum_pct = 0
            for i in range(len(S1a)):
                cum_pct += S1a[i][2]
                S1a[i].append(cum_pct)

        self.S1aVI = S1aVI
        self.S1a   = S1a
        self.S1    = S1

        if build_transition_file:
            build_transition_matrix(self.S, self.A, self.Pt, self.rt, transition_file, S1=self.S1)

        with open(transition_file, 'r') as f:
            lines = f.read().split('\n')[1:-1]
        transition_matrix = [[int(c[0]), int(c[1]), int(c[2]), float(c[3]), float(c[4])]
                       for row in lines for c in [row.split(',')]]

        # numpy array — base for the C++ core
        self.transition_matrix = np.array(transition_matrix, dtype=np.float64)

        # index built in C++
        self.transition_index = _mdp_core.build_transition_index(self.transition_matrix)


    #******************************************************************
    # Algorithms
    #******************************************************************

    VI        = VI
    VIGS      = VIGS
    PI        = PI
    QLearning = QLearning
    SARSA     = SARSA


    #****************************************************
    # PRINTING FUNCTIONS
    #****************************************************

    def print_results(self, n_iter=True):
        """Print optimal values and optimal policy for all states."""
        _print_results(self, n_iter)

    def print_partial_results(self, s_n, scenario_cols, n_iter=True):
        """Print values and policy filtered by a specific scenario."""
        _print_partial_results(self, s_n, scenario_cols, n_iter)

    def save_results(self, results_file, value_file, n_iter=True):
        """Save results to two CSV files."""
        _save_results(self, results_file, value_file, n_iter=True)

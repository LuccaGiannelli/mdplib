import time
import pandas as pd


#******************************************************************
# RESULT PRINTING AND SAVING FUNCTIONS
#******************************************************************

def print_results(mdp, n_iter=True):
    """Print optimal values and optimal policy for all states."""
    print("Optimal values:")
    for i in range(len(mdp.S)):
        print('s:', mdp.S[i], ' - value:', mdp.value[i])

    print("Optimal policy:")
    for i in range(len(mdp.S)):
        print('s:', mdp.S[i], ' - action:', mdp.policy[i])

    print("Execution time: ", mdp.time)

    if n_iter:
        print("n_iterations: ", mdp.n_iterations)

    print("")


def print_partial_results(mdp, s_n, scenario_cols, n_iter=True):
    """Print values and policy filtered by the states that match s_n,
    except in the dimensions listed in scenario_cols."""
    print("Optimal values:")
    for i in range(len(mdp.S)):
        show = True
        for j in range(len(s_n)):
            if j not in scenario_cols:
                if mdp.S[i][j] != s_n[j]:
                    show = False
        if show:
            print('s:', mdp.S[i], ' - value:', mdp.value[i])

    print("Optimal policy:")
    for i in range(len(mdp.S)):
        show = True
        for j in range(len(s_n)):
            if j not in scenario_cols:
                if mdp.S[i][j] != s_n[j]:
                    show = False
        if show:
            print('s:', mdp.S[i], ' - action:', mdp.policy[i])

    print("Execution time: ", mdp.time)

    if n_iter:
        print("n_iterations: ", mdp.n_iterations)

    print("")


def save_results(mdp, results_file, value_file, n_iter=True):
    """Save results to two CSV files: one with state/action/value, one with values only."""
    f = open(results_file, 'w')
    f.write('s,a,value\n')
    for i in range(len(mdp.S)):
        f.write(str(mdp.S[i]) + ',' + str(mdp.policy[i]) + ',' + str(mdp.value[i]) + '\n')

    f.write("Execution time: " + str(mdp.time) + "\n")
    if n_iter:
        f.write("n_iterations: " + str(mdp.n_iterations) + "\n")

    f.close()

    df_value = pd.DataFrame([[v] for v in mdp.value], columns=['Value'])
    df_value.to_csv(value_file, index=False, encoding='utf_8', sep=';', decimal=',')


#******************************************************************
# Build a list with only the feasible transitions
#******************************************************************

def build_transition_matrix(S, A, Pt, rt, transition_file="MTrans.csv", S1=None):
    """
    Write to CSV all feasible transitions (prob > 0).

    Iterates over every combination (sn, a, sn1) and writes the rows where
    Pt(sn, a, sn1) > 0. This file is read by MDP.__init__.

    Parameters
    ----------
    S : list
        List of origin states.
    A : list
        List of actions.
    Pt : callable
        Function (sn, a, sn1) -> float, the transition probability.
    rt : callable
        Function (sn, a, sn1) -> float, the reward.
    transition_file : str, optional
        Path to the output CSV file. Default: "MTrans.csv".
    S1 : list, optional
        List of destination states. If None, uses S. Default: None.
    """
    if S1 is None:
        S1 = S

    f = open(transition_file, 'w')
    f.write('sn,a,sn1,prob,reward\n')
    start_time = time.time()
    len_s = len(S)
    n_states = 0
    n_rows = 0

    for s_n in range(len(S)):
        for a in range(len(A)):
            for s_n1 in range(len(S1)):
                prob = Pt(S[s_n], A[a], S1[s_n1])
                if prob > 0:
                    f.write(str(s_n) + ',' + str(a) + ',' + str(s_n1) + ',' + str(prob) + ',' + str(rt(S[s_n], A[a], S1[s_n1])) + '\n')
                    n_rows += 1
        print("Done: ", (n_states + 1), " of ", len_s, " // ", time.time() - start_time, " secs // ", n_rows, " rows")
        n_states += 1
    f.close()


#******************************************************************
# Start and end row for each (s,a) pair in the feasible Transition Matrix
#******************************************************************

def build_state_action_index(transition_matrix):
    """
    Build a row index for each (state, action) pair in the transition matrix.

    Returns a list where state_index[s][a] = [start_row, end_row],
    indicating the slice of the transition matrix for the (s, a) pair. Used
    by the algorithms to directly access the transitions of each pair
    without scanning the whole table.

    Parameters
    ----------
    transition_matrix : list
        Transition matrix in the format [[sn, a, sn1, prob, reward], ...].

    Returns
    -------
    list
        state_index[s] = list of [start, end] for each action of state s.
    """
    state_index = []
    action_index = []

    n_rows = len(transition_matrix)
    n_rows_1 = n_rows - 1
    start_row = 0

    for row in range(n_rows_1):

        # state change
        if transition_matrix[row][0] != transition_matrix[row + 1][0]:
            end_row = row + 1
            action_index.append([start_row, end_row])
            start_row = end_row
            state_index.append(action_index)
            action_index = []
        # action change
        elif transition_matrix[row][1] != transition_matrix[row + 1][1]:
            end_row = row + 1
            action_index.append([start_row, end_row])
            start_row = end_row

    end_row = n_rows
    action_index.append([start_row, end_row])
    state_index.append(action_index)

    return state_index


#******************************************************************
# Norm calculation functions
#******************************************************************

def euclidean_norm(v1, v0):
    """Euclidean norm between two vectors: sqrt(sum((v1-v0)^2))."""
    return sum([(v1[i] - v0[i]) ** 2 for i in range(len(v0))]) ** (1 / 2)

def sup_norm(v1, v0):
    """Sup norm between two vectors: max(|v1-v0|)."""
    return max([abs(v1[i] - v0[i]) for i in range(len(v0))])

def relative_sup_norm(v1, v0):
    """Relative (percentage) sup norm between two vectors: max(|v1-v0| / |v0|)."""
    max_delta = 0
    for i in range(len(v0)):
        if v0[i] == 0 and v1[i] == 0:
            delta = 0
        elif v0[i] == 0:
            delta = 1000
        else:
            delta = abs((v1[i] - v0[i]) / v0[i])
        if delta > max_delta:
            max_delta = delta
    return max_delta

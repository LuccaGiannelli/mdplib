from mdp import MDP

#******************************************************************
# Example: machine maintenance problem
#
# States: degradation level [0, 1, 2, ..., 9]
#   0 = new machine,  9 = broken machine
#
# Actions:
#   0 = do nothing
#   1 = light maintenance  (reduces degradation by 1)
#   2 = heavy maintenance  (reduces degradation by 3)
#******************************************************************

S = list(range(10))
A = [0, 1, 2]

def Pt(sn, a, sn1):
    if sn == 9:
        return 1.0 if sn1 == 8 else 0.0
    pos = max(0, sn - [0,1,3][a])
    if sn1 == min(pos + 1, 9): return 0.7
    if sn1 == pos:              return 0.3
    return 0.0

def rt(sn, a, sn1):
    if sn == 9: return -20.0
    return -1.0 + [0, -3, -7][a]

problem = MDP(S, A, Pt, rt, transition_file="MTrans_maintenance.csv")

print("========== Value Iteration ==========")
problem.VI(gamma=0.95, tol=1e-6, verbose=False)
problem.print_results()

print("========== Value Iteration Gauss-Seidel ==========")
problem.VIGS(gamma=0.95, tol=1e-6, verbose=False)
problem.print_results()

print("========== Policy Iteration ==========")
problem.PI(gamma=0.95, tol=1e-6, verbose=False)
problem.print_results()

print("========== Q-Learning ==========")
problem.QLearning(gamma=0.95, epsilon=0.05, alpha=0.1, max_iter=200000, verbose=False)
problem.print_results()

print("========== SARSA ==========")
problem.SARSA(gamma=0.95, epsilon=0.05, alpha=0.1, max_iter=200000, verbose=False)
problem.print_results()

import pytest
from mdp import MDP


#******************************************************************
# Small MDP shared between tests
# (same problem as basic_example.py)
#******************************************************************

@pytest.fixture(scope="session")
def mdp_inventory(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mtrans")
    transition_file = str(tmp / "MTrans_test.csv")

    S = [0, 1, 2]
    A = [0, 1, 2]

    def Pt(sn, a, sn1):
        stock_after_order = min(sn + a, 2)
        demand = stock_after_order - sn1
        if demand == 0:
            return 0.3
        elif demand == 1:
            return 0.7
        else:
            return 0.0

    def rt(sn, a, sn1):
        stock_after_order = min(sn + a, 2)
        demand = stock_after_order - sn1
        order_cost   = a * 1
        shortage_cost = max(0, demand - stock_after_order) * 3
        return -(order_cost + shortage_cost)

    return MDP(S, A, Pt, rt, build_transition_file=True, transition_file=transition_file)

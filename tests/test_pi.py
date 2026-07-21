from mdp import MDP


#******************************************************************
# PI tests
#******************************************************************

def test_pi_runs_without_error(mdp_inventory):
    mdp_inventory.PI(gamma=0.9, tol=1e-6, verbose=False)
    assert mdp_inventory.value  is not None
    assert mdp_inventory.policy is not None


def test_pi_output_size(mdp_inventory):
    mdp_inventory.PI(gamma=0.9, tol=1e-6, verbose=False)
    assert len(mdp_inventory.value)  == len(mdp_inventory.S)
    assert len(mdp_inventory.policy) == len(mdp_inventory.S)


def test_pi_valid_policy(mdp_inventory):
    mdp_inventory.PI(gamma=0.9, tol=1e-6, verbose=False)
    for a in mdp_inventory.policy:
        assert a in mdp_inventory.A


def test_pi_vi_same_policy(mdp_inventory):
    # PI and VI should converge to the same policy
    mdp_inventory.VI(gamma=0.9, tol=1e-6, verbose=False)
    policy_vi = mdp_inventory.policy[:]

    mdp_inventory.PI(gamma=0.9, tol=1e-6, verbose=False)
    policy_pi = mdp_inventory.policy[:]

    assert policy_vi == policy_pi

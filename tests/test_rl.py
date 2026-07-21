#******************************************************************
# Q-Learning and SARSA tests
#******************************************************************

def test_qlearning_runs_without_error(mdp_inventory):
    mdp_inventory.QLearning(gamma=0.9, epsilon=0.05, alpha=0.1, max_iter=50000, verbose=False)
    assert mdp_inventory.value  is not None
    assert mdp_inventory.policy is not None


def test_qlearning_output_size(mdp_inventory):
    mdp_inventory.QLearning(gamma=0.9, epsilon=0.05, alpha=0.1, max_iter=50000, verbose=False)
    assert len(mdp_inventory.value)  == len(mdp_inventory.S)
    assert len(mdp_inventory.policy) == len(mdp_inventory.S)


def test_qlearning_valid_policy(mdp_inventory):
    mdp_inventory.QLearning(gamma=0.9, epsilon=0.05, alpha=0.1, max_iter=50000, verbose=False)
    for a in mdp_inventory.policy:
        assert a in mdp_inventory.A


def test_sarsa_runs_without_error(mdp_inventory):
    mdp_inventory.SARSA(gamma=0.9, epsilon=0.05, alpha=0.1, max_iter=50000, verbose=False)
    assert mdp_inventory.value  is not None
    assert mdp_inventory.policy is not None


def test_sarsa_output_size(mdp_inventory):
    mdp_inventory.SARSA(gamma=0.9, epsilon=0.05, alpha=0.1, max_iter=50000, verbose=False)
    assert len(mdp_inventory.value)  == len(mdp_inventory.S)
    assert len(mdp_inventory.policy) == len(mdp_inventory.S)


def test_sarsa_valid_policy(mdp_inventory):
    mdp_inventory.SARSA(gamma=0.9, epsilon=0.05, alpha=0.1, max_iter=50000, verbose=False)
    for a in mdp_inventory.policy:
        assert a in mdp_inventory.A

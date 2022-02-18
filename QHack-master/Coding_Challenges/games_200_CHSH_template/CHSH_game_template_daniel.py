#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


dev = qml.device("default.qubit", wires=2)


def prepare_entangled(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # QHACK #
    norm = (alpha ** 2 + beta ** 2) ** (1/2)
    phi = (np.arccos(alpha / norm) + np.arcsin(beta / norm)) / 2
    qml.RY(2 * phi, wires=0)
    qml.CNOT(wires=(0, 1))
    # QHACK #


@qml.qnode(dev)
def chsh_circuit(theta_A0, theta_A1, theta_B0, theta_B1, x, y, alpha, beta):
    """Construct a circuit that implements Alice's and Bob's measurements in the rotated bases

    Args:
        - theta_A0 (float): angle that Alice chooses when she receives x=0
        - theta_A1 (float): angle that Alice chooses when she receives x=1
        - theta_B0 (float): angle that Bob chooses when he receives y=0
        - theta_B1 (float): angle that Bob chooses when he receives y=1
        - x (int): bit received by Alice
        - y (int): bit received by Bob
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (np.tensor): Probabilities of each basis state
    """

    prepare_entangled(alpha, beta)

    # QHACK #
    theta_alice = (1 - x) * theta_A0 + x * theta_A1
    qml.RY(- 2 * theta_alice, wires=0)

    theta_bob = (1 - y) * theta_B0 + y * theta_B1
    qml.RY(- 2 * theta_bob, wires=1)
    # QHACK #

    return qml.probs(wires=[0, 1])
    

def winning_prob(params, alpha, beta):
    """Define a function that returns the probability of Alice and Bob winning the game.

    Args:
        - params (list(float)): List containing [theta_A0,theta_A1,theta_B0,theta_B1]
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning the game
    """

    # QHACK #
    p_win = 0.0
    for (x, y) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        p_xy = chsh_circuit(theta_A0=params[0], theta_A1=params[1], theta_B0=params[2], theta_B1=params[3],
                            x=x, y=y, alpha=alpha, beta=beta)

        # p_alice_choose_0 = p_xy[0] + p_xy[1]
        # p_alice_choose_1 = p_xy[2] + p_xy[3]
        #
        # p_bob_choose_0 = p_xy[0] + p_xy[2]
        # p_bob_choose_1 = p_xy[1] + p_xy[3]
        #
        # p_a_plus_b_is_0 = (p_alice_choose_0 * p_bob_choose_0 + p_alice_choose_1 * p_bob_choose_1)
        # p_a_plus_b_is_1 = (p_alice_choose_0 * p_bob_choose_1 + p_alice_choose_1 * p_bob_choose_0)

        p_a_plus_b_is_0 = p_xy[0] + p_xy[3]
        p_a_plus_b_is_1 = p_xy[1] + p_xy[2]

        p_win_given_current_x_and_y = (1 - x * y) * p_a_plus_b_is_0 + (x * y) * p_a_plus_b_is_1
        p_current_x_and_y = 1 / 4
        p_win = p_win + p_win_given_current_x_and_y * p_current_x_and_y
    return p_win
    # QHACK #
    

def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning
    """

    def cost(parameters):
        """Define a cost function that only depends on params, given alpha and beta fixed"""
        current_p_win = winning_prob(parameters, alpha, beta)
        return 0.5 * ((1.0 - current_p_win) ** 2)
    # QHACK #

    # Initialize parameters, choose an optimization method and number of steps
    opt = qml.AdamOptimizer()
    init_params = np.random.random(4) * 2 * np.pi
    steps = 1000

    # QHACK #
    
    # set the initial parameter values
    params = init_params

    cost_wrs = []

    for i in range(steps):
        # update the circuit parameters 
        # QHACK #
        params, _cost = opt.step_and_cost(cost, params)
        cost_wrs.append(_cost)
        print("Step {}: cost = {}".format(i + 1, cost_wrs[-1]))
        # QHACK #
    return winning_prob(params, alpha, beta)


if __name__ == '__main__':
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")

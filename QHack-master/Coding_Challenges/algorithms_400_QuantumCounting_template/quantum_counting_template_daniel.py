#! /usr/bin/python3

import sys

import numpy
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import QuantumPhaseEstimation


dev = qml.device("default.qubit", wires=8)


def oracle_matrix(indices):
    """Return the oracle matrix for a secret combination.

    Args:
        - indices (list(int)): A list of bit indices (e.g. [0,3]) representing the elements that are map to 1.

    Returns:
        - (np.ndarray): The matrix representation of the oracle
    """

    # QHACK #
    diag = []
    for idx in range(4 ** 2):
        if idx in indices:
            diag.append(-1.0)
        else:
            diag.append(+1.0)
    my_array = np.diag(diag)
    # QHACK #

    return my_array


def diffusion_matrix():

    # DO NOT MODIFY anything in this code block

    psi_piece = (1 / 2 ** 4) * np.ones(2 ** 4)
    ident_piece = np.eye(2 ** 4)
    return 2 * psi_piece - ident_piece


def grover_operator(indices):

    # DO NOT MODIFY anything in this code block

    return np.dot(diffusion_matrix(), oracle_matrix(indices))


def grover_power(grover_mat, powers):
    power_grover_mat = None
    for time in range(powers):
        if power_grover_mat is None:
            power_grover_mat = grover_mat
        else:
            power_grover_mat = power_grover_mat @ grover_mat

    return power_grover_mat


dev = qml.device("default.qubit", wires=8)

@qml.qnode(dev)
def circuit(indices):
    """Return the probabilities of each basis state after applying QPE to the Grover operator

    Args:
        - indices (list(int)): A list of bits representing the elements that map to 1.

    Returns:
        - (np.tensor): Probabilities of measuring each computational basis state
    """

    # QHACK #

    target_wires = range(0, 4)

    estimation_wires = range(4, 8)

    wires = range(0, 8)

    # Build your circuit here
    for wire in wires:
        qml.Hadamard(wire)

    grover_matrix = grover_operator(indices)
    for wire in estimation_wires:
        wire_in_seq = wire - min(estimation_wires) + 1
        current_power_of_grover = 2 ** (4 - wire_in_seq)
        current_grover_power = grover_power(grover_matrix, current_power_of_grover)
        qml.ControlledQubitUnitary(current_grover_power, control_wires=wire, wires=target_wires)

    qml.QFT(wires=estimation_wires).inv()
    # QHACK #

    return qml.probs(estimation_wires)

def number_of_solutions(indices):
    """Implement the formula given in the problem statement to find the number of solutions from the output of your circuit

    Args:
        - indices (list(int)): A list of bits representing the elements that map to 1.

    Returns:
        - (float): number of elements as estimated by the quantum counting algorithm
    """

    # QHACK #
    max_idx = np.argmax(circuit(indices))
    theta = max_idx * (np.pi / 8)
    M = 4 * np.sin(theta / 2)
    return M ** 2

    # QHACK #

def relative_error(indices):
    """Calculate the relative error of the quantum counting estimation

    Args:
        - indices (list(int)): A list of bits representing the elements that map to 1.

    Returns: 
        - (float): relative error
    """

    # QHACK #
    quantum_counting_estimation = number_of_solutions(indices)
    true_number_of_elements = len(indices)

    rel_err = ((quantum_counting_estimation / true_number_of_elements) - 1) * 100

    # QHACK #

    return rel_err

if __name__ == '__main__':
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    lst=[int(i) for i in inputs]
    output = relative_error(lst)
    print(f"{output}")

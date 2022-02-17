#! /usr/bin/python3

import sys
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
    my_array = np.diag([-1 if i in indices else 1 for i in range(16)])
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

    target_wires = [0, 1, 2, 3]

    estimation_wires = [4, 5, 6, 7]

    # Build your circuit here
    for i in target_wires + estimation_wires:
        qml.Hadamard(wires=[i])
    g = grover_operator(indices)
    for i, e in zip(reversed(range(4)), estimation_wires):
        pow = 2 ** i
        gi = np.linalg.matrix_power(g, pow)
        qml.ControlledQubitUnitary(gi, control_wires=e, wires=target_wires)
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
    res = circuit(indices)
    decimal_value = np.argmax(res)
    theta = decimal_value * (np.pi / 8)
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
    sol_num = number_of_solutions(indices)
    rel_err = 100 * ((sol_num - len(indices)) / len(indices))
    # QHACK #

    return rel_err


if __name__ == '__main__':
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    lst = [int(i) for i in inputs]
    output = relative_error(lst)
    print(f"{output}")
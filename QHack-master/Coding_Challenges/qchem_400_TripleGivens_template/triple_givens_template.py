import sys
import pennylane as qml
from pennylane import numpy as np

NUM_WIRES = 6


def triple_excitation_matrix(gamma):
    """The matrix representation of a triple-excitation Givens rotation.

    Args:
        - gamma (float): The angle of rotation

    Returns:
        - (np.ndarray): The matrix representation of a triple-excitation
    """

    # QHACK #

    c = qml.math.cos(gamma / 2)
    s = qml.math.sin(gamma / 2)

    mat = np.zeros((64, 64))

    for i in range(64):
        mat[i, i] = 1

    mat[7, 7], mat[56, 56] = c, c
    mat[7, 56], mat[56, 7] = -s, s

    # mat = qml.math.diag([1.0] * 7 + [c] + [1.0] * 48 + [c] + [1.0] * 7)
    # mat = qml.math.scatter_element_add(mat, (7, 56), -s)
    # mat = qml.math.scatter_element_add(mat, (56, 7), s)

    return mat
    # QHACK #


dev = qml.device("default.qubit", wires=6)


@qml.qnode(dev)
def circuit(angles):
    """Prepares the quantum state in the problem statement and returns qml.probs

    Args:
        - angles (list(float)): The relevant angles in the problem statement in this order:
        [alpha, beta, gamma]

    Returns:
        - (np.tensor): The probability of each computational basis state
    """

    # QHACK #
    # print("sin(a/2)={}".format(np.sin(angles[0] / 2) ** 2))
    # print("cos(a/2)sin(beta/2)={}".format((np.cos(angles[0] / 2) * np.sin(angles[1] / 2)) ** 2))
    # print("cos(a/2)cos(beta/2)cos(gamma/2)={}".format(
    #     (np.cos(angles[0] / 2) * np.cos(angles[1] / 2) * np.cos(angles[2] / 2)) ** 2))
    #
    # print("cos(a/2)cos(beta/2)sin(gamma/2)={}".format(
    #     (np.cos(angles[0] / 2) * np.cos(angles[1] / 2) * np.sin(angles[2] / 2)) ** 2))
    qml.PauliX(0)
    qml.PauliX(1)
    qml.PauliX(2)

    qml.SingleExcitation(angles[0], wires=[5, 0])
    qml.DoubleExcitation(angles[1], wires=[4, 5, 0, 1])
    qml.QubitUnitary(triple_excitation_matrix(angles[2]), wires=[3, 4, 5, 0, 1, 2])
    # QHACK #

    return qml.probs(wires=range(NUM_WIRES))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = np.array(sys.stdin.read().split(","), dtype=float)
    probs = circuit(inputs).round(6)
    # print(np.argwhere(probs != 0))
    print(*probs, sep=",")

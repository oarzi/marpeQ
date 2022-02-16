#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np

# np.set_printoptions(threshold=100)
dev = qml.device("default.mixed", wires=3)

def error_wire_ofir(circuit_output):
    """Function that returns an error readout.

    Args:
        - circuit_output (?): the output of the `circuit` function.

    Returns:
        - (np.ndarray): a length-4 array that reveals the statistics of the
        error channel. It should display your algorithm's statistical prediction for
        whether an error occurred on wire `k` (k in {1,2,3}). The zeroth element represents
        the probability that a bitflip error does not occur.

        e.g., [0.28, 0.0, 0.72, 0.0] means a 28% chance no bitflip error occurs, but if one
        does occur it occurs on qubit #2 with a 72% chance.
    """

    # QHACK #
    # process the circuit output here and return which qubit was the victim of a bitflip error!
    p0 = np.matmul(P0(wires=[0, 1, 2]), circuit_output).trace().real

    p1 = np.matmul(P1(wires=[0, 1, 2]), circuit_output).trace().real

    p2 = np.matmul(P2(wires=[0, 1, 2]), circuit_output).trace().real

    p3 = np.matmul(P3(wires=[0, 1, 2]), circuit_output).trace().real

    norm = p0 + p1 + p2 + p3
    p0, p1, p2, p3 = p0/norm, p1/norm, p2/norm, p3/norm
    return p0, p1, p2, p3

    # QHACK #

@qml.qnode(dev)
def circuit_ofir(p, alpha, tampered_wire):
    """A quantum circuit that will be able to identify bitflip errors.

    DO NOT MODIFY any already-written lines in this function.

    Args:
        p (float): The bit flip probability
        alpha (float): The parameter used to calculate `density_matrix(alpha)`
        tampered_wire (int): The wire that may or may not be flipped (zero-index)

    Returns:
        Some expectation value, state, probs, ... you decide!
    """

    qml.QubitDensityMatrix(density_matrix(alpha), wires=[0, 1, 2])

    # QHACK #
    # put any input processing gates here
    qml.CNOT(wires=(0, 1))
    qml.CNOT(wires=(0, 2))
    qml.BitFlip(p, wires=int(tampered_wire))
    # put any gates here after the bitflip error has occurred

    # return something!
    # QHACK #
    return qml.density_matrix(wires=[0, 1, 2])


def P0_mat():
    down = np.array([1, 0])
    up = np.array([0, 1])
    m1 = np.kron(down, np.kron(down, down))
    m2 = np.kron(up, np.kron(up, up))
    p0 = np.outer(m1 + m2, np.conj(m1 + m2))

    return p0


def P0(wires):
    p0 = P0_mat()

    # return qml.Hermitian(p0, wires=wires)
    return p0


def P1(wires):
    p0 = P0_mat()
    X = np.array([[0, 1], [1, 0]])
    I = np.array([[1, 0], [0, 1]])
    X1 = np.kron(np.kron(X, I), I)

    p1 = np.matmul(X1, np.matmul(p0, X1))

    # return qml.Hermitian(p1, wires=wires)
    return p1


def P2(wires):
    p0 = P0_mat()
    X = np.array([[0, 1], [1, 0]])
    I = np.array([[1, 0], [0, 1]])
    X2 = np.kron(np.kron(I, X), I)

    p2 = np.matmul(X2, np.matmul(p0, X2))

    # return qml.Hermitian(p2, wires=wires)
    return p2


def P3(wires):
    p0 = P0_mat()
    X = np.array([[0, 1], [1, 0]])
    I = np.array([[1, 0], [0, 1]])
    X3 = np.kron(np.kron(I, I), X)

    p3 = np.matmul(X3, np.matmul(p0, X3))

    # return qml.Hermitian(p3, wires=wires)
    return p3


def error_wire(circuit_output):
    """Function that returns an error readout.

    Args:
        - circuit_output (?): the output of the `circuit` function.

    Returns:
        - (np.ndarray): a length-4 array that reveals the statistics of the
        error channel. It should display your algorithm's statistical prediction for
        whether an error occurred on wire `k` (k in {1,2,3}). The zeroth element represents
        the probability that a bitflip error does not occur.

        e.g., [0.28, 0.0, 0.72, 0.0] means a 28% chance no bitflip error occurs, but if one
        does occur it occurs on qubit #2 with a 72% chance.
    """

    # QHACK #
    # process the circuit output here and return which qubit was the victim of a bitflip error!
    res = np.diag(circuit_output).real
    return np.stack([res[0], res[3], res[2], res[1]])
    # QHACK #




@qml.qnode(dev)
def circuit(p, alpha, tampered_wire):
    """A quantum circuit that will be able to identify bitflip errors.

    DO NOT MODIFY any already-written lines in this function.

    Args:
        p (float): The bit flip probability
        alpha (float): The parameter used to calculate `density_matrix(alpha)`
        tampered_wire (int): The wire that may or may not be flipped (zero-index)

    Returns:
        Some expectation value, state, probs, ... you decide!
    """

    qml.QubitDensityMatrix(density_matrix(alpha), wires=[0, 1, 2])

    # QHACK #

    # put any input processing gates here
    qml.CNOT(wires=(0, 1))
    qml.CNOT(wires=(0, 2))
    qml.BitFlip(p, wires=int(tampered_wire))
    # put any gates here after the bitflip error has occurred
    qml.CNOT(wires=(0, 1))
    qml.CNOT(wires=(0, 2))
    qml.Toffoli(wires=(1, 2, 0))

    # return something!
    return qml.density_matrix(wires=[1, 2])
    # QHACK #


def density_matrix(alpha):
    """Creates a density matrix from a pure state."""
    # DO NOT MODIFY anything in this code block
    psi = alpha * np.array([1, 0], dtype=float) + np.sqrt(1 - alpha**2) * np.array(
        [0, 1], dtype=float
    )
    psi = np.kron(psi, np.array([1, 0, 0, 0], dtype=float))
    return np.outer(psi, np.conj(psi))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = np.array(sys.stdin.read().split(","), dtype=float)
    p, alpha, tampered_wire = inputs[0], inputs[1], int(inputs[2])

    error_readout = np.zeros(4, dtype=float)
    circuit_output = circuit(p, alpha, tampered_wire)
    error_readout = error_wire(circuit_output)

    print(*error_readout, sep=",")

import sys
import pennylane as qml
from pennylane import numpy as np


def second_renyi_entropy(rho):
    """Computes the second Renyi entropy of a given density matrix."""
    # DO NOT MODIFY anything in this code block
    rho_diag_2 = np.diagonal(rho) ** 2.0
    return -np.real(np.log(np.sum(rho_diag_2)))


def circuit(wires, theta):
    qml.Hadamard(wires=1)
    qml.ControlledQubitUnitary(np.array([[0, 1], [1, 0]]), control_wires=[wires[1]], wires=wires[0], control_values='0')
    SRX = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [np.sin(theta / 2), 1j * np.cos(theta / 2)]])
    qml.ControlledQubitUnitary(SRX, control_wires=wires[1], wires=wires[2])
    qml.CNOT(wires=[wires[2], wires[1]])

    return qml.density_matrix(wires=1)


def compute_entanglement(theta):
    """Computes the second Renyi entropy of circuits with and without a tardigrade present.

    Args:
        - theta (float): the angle that defines the state psi_ABT

    Returns:
        - (float): The entanglement entropy of qubit B with no tardigrade
        initially present
        - (float): The entanglement entropy of qubit B where the tardigrade
        was initially present
    """

    dev = qml.device("default.qubit", wires=3)

    # QHACK #
    tardigrade = qml.QNode(circuit, dev)
    dense_t = tardigrade([0, 1, 2], theta)
    entropy_t = second_renyi_entropy(dense_t)

    no_tardigrade = qml.QNode(circuit, dev)
    dense_not = no_tardigrade([0, 1, 2], 0)
    entropy_not = second_renyi_entropy(dense_not)
    # QHACK #
    return entropy_not, entropy_t


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    theta = np.array(sys.stdin.read(), dtype=float)

    S2_without_tardigrade, S2_with_tardigrade = compute_entanglement(theta)
    print(*[S2_without_tardigrade, S2_with_tardigrade], sep=",")

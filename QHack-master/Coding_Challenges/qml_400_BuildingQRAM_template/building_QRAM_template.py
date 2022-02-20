#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qRAM(thetas):
    """Function that generates the superposition state explained above given the thetas angles.

    Args:
        - thetas (list(float)): list of angles to apply in the rotations.

    Returns:
        - (list(complex)): final state.
    """

    # QHACK #

    # Use this space to create auxiliary functions if you need it.

    # QHACK #

    dev = qml.device("default.qubit", wires=range(4))
    y = qml.PauliY.matrix

    @qml.qnode(dev)
    def circuit():

        # QHACK #
        qml.Hadamard(0)
        qml.Hadamard(1)
        qml.Hadamard(2)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    control = str(k) + str(j) + str(i)
                    # m = qml.RY(thetas[4 * i + 2 * j + k], wires=3).matrix
                    thetai = thetas[4 * k + 2 * j + i]
                    # m = np.exp(1j * (thetai/2) * y)
                    # print(m)
                    qml.ControlledQubitUnitary(qml.RY(thetai, wires=3).matrix, control_wires=[0, 1, 2], wires=3,
                                               control_values=control)
        # Create your circuit: the first three qubits will refer to the index, the fourth to the RY rotation.

        # QHACK #

        return qml.state()

    return circuit()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    thetas = np.array(inputs, dtype=float)

    output = qRAM(thetas)
    output = [float(i.real.round(6)) for i in output]
    print(*output, sep=",")

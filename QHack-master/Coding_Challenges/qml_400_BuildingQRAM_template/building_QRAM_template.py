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
        c, s = lambda x: np.cos(x / 2), lambda x: np.sin(x / 2)
        m = lambda x: np.array([[c(x), -s(x)],
                                [s(x), c(x)]])
        qml.Hadamard(0)
        qml.Hadamard(1)
        qml.Hadamard(2)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    control = str(k) + str(j) + str(i)
                    thetai = thetas[4 * k + 2 * j + i]

                    qml.ControlledQubitUnitary(m(thetai), control_wires=[0, 1, 2], wires=3,
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

#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def deutsch(f, wires):
    qml.PauliX(wires[0])
    qml.PauliX(wires[1])
    qml.PauliX(wires[2])
    qml.Hadamard(wires[0])
    qml.Hadamard(wires[1])
    qml.Hadamard(wires[2])
    # print(f)
    f(wires)
    # print(help(f))
    # print(np.array(f()))
    # qml.QubitUnitary(f, wires=wires)

    qml.Hadamard(wires[0])
    qml.Hadamard(wires[1])


def deutsch_jozsa(fs):
    """Function that determines whether four given functions are all of the same type or not.

    Args:
        - fs (list(function)): A list of 4 quantum functions. Each of them will accept a 'wires' parameter.
        The first two wires refer to the input and the third to the output of the function.

    Returns:
        - (str) : "4 same" or "2 and 2"
    """

    # QHACK #
    dev = qml.device('default.qubit', wires=7, shots=1)

    def oracle(funcs):
        deutsch(funcs[0], wires=[0, 1, 2])
        qml.Hadamard(wires=2)
        qml.Toffoli(wires=[0, 2, 1])
        qml.PauliX(wires=2)

        deutsch(funcs[1], wires=[1, 2, 3])
        qml.Hadamard(wires=3)
        qml.Toffoli(wires=[1, 3, 2])
        qml.PauliX(wires=3)

        deutsch(funcs[2], wires=[2, 3, 4])
        qml.Hadamard(wires=4)
        qml.Toffoli(wires=[2, 4, 3])
        qml.PauliX(wires=4)

        deutsch(funcs[3], wires=[3, 4, 5])

        return [qml.expval(qml.PauliZ(i)) for i in [0, 1, 2, 3]]
        # QHACK #

    qnode = qml.QNode(oracle, dev)
    res = int(sum(qnode(fs)))
    res_dic = {4: "4 same", -4: "4 same", 0: "2 and 2"}

    return res_dic[res]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    # Definition of the four oracles we will work with.


    def f1(wires):
        qml.CNOT(wires=[wires[numbers[0]], wires[2]])
        qml.CNOT(wires=[wires[numbers[1]], wires[2]])


    def f2(wires):
        qml.CNOT(wires=[wires[numbers[2]], wires[2]])
        qml.CNOT(wires=[wires[numbers[3]], wires[2]])


    def f3(wires):
        qml.CNOT(wires=[wires[numbers[4]], wires[2]])
        qml.CNOT(wires=[wires[numbers[5]], wires[2]])
        qml.PauliX(wires=wires[2])


    def f4(wires):
        qml.CNOT(wires=[wires[numbers[6]], wires[2]])
        qml.CNOT(wires=[wires[numbers[7]], wires[2]])
        qml.PauliX(wires=wires[2])


    output = deutsch_jozsa([f1, f2, f3, f4])
    print(f"{output}")

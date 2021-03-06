#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


def circuit(gates, gatewires, mes, meswires):
    def ret_circ(gateparams):
        for gate, params, wire in zip(gates, gateparams, gatewires):
            gate(params, wires=wire)
        return qml.expval(mes(meswires))

    return ret_circ


def compare_circuits_ofir(angles):
    dev = qml.device("default.qubit", wires=1)
    c1 = circuit([qml.RX, qml.RY], [0, 0], qml.PauliX, 0)
    c2 = circuit([qml.RY, qml.RX], [0, 0], qml.PauliX, 0)

    qnode1 = qml.QNode(c1, dev)(angles)
    qnode2 = qml.QNode(c2, dev)(list(reversed(angles)))

    return np.absolute(qnode1 - qnode2)
    # QHACK #


def xy_qfunc(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliX(0))


def yx_qfunc(params):
    qml.RY(params[1], wires=0)
    qml.RX(params[0], wires=0)
    return qml.expval(qml.PauliX(0))


def compare_circuits(angles):
    """Given two angles, compare two circuit outputs that have their order of operations flipped: RX then RY VERSUS RY then RX.

    Args:
        - angles (np.ndarray): Two angles

    Returns:
        - (float): | < \sigma^x >_1 - < \sigma^x >_2 |
    """

    # QHACK #
    # define a device and quantum functions/circuits here
    dev = qml.device("default.qubit", wires=1)

    # define an executable quantum node as a circuit applying RX followed by RY and vice versa
    xy_qnode = qml.QNode(func=xy_qfunc, device=dev)
    yx_qnode = qml.QNode(func=yx_qfunc, device=dev)

    res = xy_qnode(angles) - yx_qnode(angles)
    return qml.numpy.absolute(res)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    angles = np.array(sys.stdin.read().split(","), dtype=float)
    output = compare_circuits(angles)
    print(f"{output:.6f}")

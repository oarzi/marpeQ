#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def distance(A, B):
    """Function that returns the distance between two vectors.

    Args:
        - A (list[int]): person's information: [age, minutes spent watching TV].
        - B (list[int]): person's information: [age, minutes spent watching TV].

    Returns:
        - (float): distance between the two feature vectors.
    """

    # QHACK #

    # The Swap test is a method that allows you to calculate |<A|B>|^2 , you could use it to help you.
    # The qml.AmplitudeEmbedding operator could help you too.

    def SWAP_prep1(A, B):
        qml.AmplitudeEmbedding(
            [A[0] * B[0], A[0] * B[1], A[1] * B[0], A[1] * B[1]], wires=[1, 2], normalize=True)

        return

    def SWAP_prep2(A, B):
        phi_a = np.arctan(A[0] / A[1]) if A[1] != 0 else 0
        phi_b = np.arctan(B[0] / B[1]) if B[1] != 0 else 0

        qml.RY(2 * (np.pi / 2 - phi_a), wires=1)
        qml.RY(2 * (np.pi / 2 - phi_b), wires=2)

        return

    dev = qml.device("default.qubit", wires=[0, 1, 2], shots=200)

    @qml.qnode(dev)
    def SWAP_test(A, B, SWAP_prep):
        SWAP_prep(A, B)
        qml.Hadamard(0)
        qml.CSWAP(wires=[0, 1, 2])
        qml.Hadamard(0)

        return qml.probs(wires=[0])

    # return (2 * SWAP_test(A, B, SWAP_prep1)[0] - 1) ** 0.5
    return np.linalg.norm(np.array(A)/np.linalg.norm(A)-np.array(B)/np.linalg.norm(B))
    # QHACK #


def predict(dataset, new, k):
    """Function that given a dataset, determines if a new person do like Beatles or not.
    Args:
        - dataset (list): List with the age, minutes that different people watch TV, and if they like Beatles.
        - new (list(int)): Age and TV minutes of the person we want to classify.
        - k (int): number of nearby neighbors to be taken into account.
    Returns:
        - (str): "YES" if they like Beatles, "NO" otherwise.
    """

    # DO NOT MODIFY anything in this code block

    def k_nearest_classes():
        """Function that returns a list of k near neighbors."""
        distances = []
        for data in dataset:
            distances.append(distance(data[0], new))
        nearest = []
        for _ in range(k):
            indx = np.argmin(distances)
            nearest.append(indx)
            distances[indx] += 2

        return [dataset[i][1] for i in nearest]

    output = k_nearest_classes()

    return (
        "YES" if len([i for i in output if i == "YES"]) > len(output) / 2 else "NO",
        float(distance(dataset[0][0], new)),
    )

if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    dataset = []
    new = [int(inputs[0]), int(inputs[1])]
    k = int(inputs[2])
    for i in range(3, len(inputs), 3):
        dataset.append([[int(inputs[i + 0]), int(inputs[i + 1])], str(inputs[i + 2])])

    output = predict(dataset, new, k)
    sol = 0 if output[0] == "YES" else 1
    print(f"{sol},{output[1]}")

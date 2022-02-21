import sys
import pennylane as qml
from pennylane import numpy as np
import pennylane.optimize as optimize

DATA_SIZE = 250


def square_loss(labels, predictions):
    """Computes the standard square loss between model predictions and true labels.

    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)

    Returns:
        - loss (float): the square loss
    """

    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def accuracy(labels, predictions):
    """Computes the accuracy of the model's predictions against the true labels.

    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)

    Returns:
        - acc (float): The accuracy.
    """

    acc = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            acc = acc + 1
    acc = acc / len(labels)

    return acc


def classify_ising_data(ising_configs, labels):
    """Learn the phases of the classical Ising model.

    Args:
        - ising_configs (np.ndarray): 250 rows of binary (0 and 1) Ising model configurations
        - labels (np.ndarray): 250 rows of labels (1 or -1)

    Returns:
        - predictions (list(int)): Your final model predictions

    Feel free to add any other functions than `cost` and `circuit` within the "# QHACK #" markers
    that you might need.
    """

    # for idx in range(len(labels)):
    #     if labels[idx] == 1:
    #         print("{}: Ordered".format(ising_configs[idx]))
    #     else:
    #         print("{}: Disordered".format(ising_configs[idx]))

    # QHACK #

    num_wires = ising_configs.shape[1]
    dev = qml.device("default.qubit", wires=num_wires)

    # Define a variational circuit below with your needed arguments and return something meaningful
    @qml.qnode(dev)
    def circuit(params, config):
        qml.BasisState(config, wires=range(num_wires))
        obs_list = [qml.PauliZ(wires=i) @ qml.PauliZ(wires=(i+1)) for i in range(num_wires - 1)]
        ising_hamiltonian = qml.Hamiltonian(params, obs_list)
        return qml.expval(ising_hamiltonian)

    # Define a cost function below with your needed arguments
    def cost():
        pass
        # QHACK #

        # Insert an expression for your model predictions here
        predictions = 1

        # QHACK #

        return square_loss(Y, predictions) # DO NOT MODIFY this line

    # optimize your circuit here
    opt = qml.GradientDescentOptimizer(0.2)
    params = np.array([-1] * (num_wires - 1), requires_grad=True)

    energy = [cost(params)]
    angle = [theta]

    max_iterations = 100
    conv_tol = 1e-7

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(avg_energy_fn, theta)

        energy.append(avg_energy_fn(theta))
        angle.append(theta)

        conv = np.abs(energy[-1] - prev_energy)
        # print(f"Step = {n},  Energy = {energy[-1]:.19f} Ha")

        if conv <= conv_tol:
            break

    # QHACK #

    return predictions


if __name__ == "__main__":
    inputs = np.array(
        sys.stdin.read().split(","), dtype=int, requires_grad=False
    ).reshape(DATA_SIZE, -1)
    ising_configs = inputs[:, :-1]
    labels = inputs[:, -1]
    predictions = classify_ising_data(ising_configs, labels)
    print(*predictions, sep=",")

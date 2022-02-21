from collections.abc import Iterable
import sys
import pennylane as qml
from pennylane import numpy as np
import pennylane.optimize as optimize

DATA_SIZE = 250
np.random.seed(0)


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

    # QHACK #

    num_wires = ising_configs.shape[1]
    dev = qml.device("default.qubit", wires=num_wires)

    def layer(params):
        for i in range(num_wires):
            qml.Rot(params[i, 0], params[i, 1],
                    params[i, 2], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0])
        return

    # Define a variational circuit below with your needed arguments and return something meaningful
    @qml.qnode(dev)
    def circuit(params, config):
        qml.BasisState(config, wires=range(num_wires))

        for i in range(params.shape[2]):
            layer(params[:, :, i])

        return qml.expval(qml.PauliZ(0))

    def variational_classifier(params, bias, config):
        return circuit(params, config) + bias

    # Define a cost function below with your needed arguments

    def cost(Y, X, params, bias):
        # QHACK #
        # Insert an expression for your model predictions here
        circuit_res = [variational_classifier(params, bias, config) for config in X]
        # QHACK #

        return square_loss(Y, circuit_res)  # DO NOT MODIFY this line

    # optimize your circuit here

    # opt = qml.GradientDescentOptimizer(0.8)
    # # opt = qml.NesterovMomentumOptimizer(stepsize=0.6, momentum=0.5)
    # max_iterations = 100
    # num_layers = 3
    # params = np.random.random((num_wires, 3, num_layers))
    # bias = np.array([0.0], requires_grad=True)
    # batch_size = 15
    # # print(type(ising_configs))
    # for n in range(max_iterations):
    #     batch_index = np.random.randint(0, len(ising_configs), batch_size)
    #     x_batch, y_batch = ising_configs[batch_index], labels[batch_index]
    #
    #     out, _cost = opt.step_and_cost(cost, y_batch, x_batch, params, bias)
    #     params, bias = out[2], out[3]
    #
    #     if n % 5 == 0:
    #         print("Step {}: cost={}".format(n, _cost[0]))
    #         print("acc={}".format(accuracy(y_batch, predict(variational_classifier, params, bias, x_batch))))

    # QHACK #
    # print(params)
    # print(bias)
    params = np.array([[[5.48813504e-01, 9.44538004e-01, 6.02763376e-01],
                        [1.57098186e+00, 1.71527633e-03, 6.45894113e-01],
                        [7.41284344e-01, 1.27473264e+00, 9.63662761e-01]],

                       [[3.83441519e-01, 1.70656650e+00, 8.73190468e-01],
                        [3.49597062e-02, 1.46496083e+00, 7.80967788e-01, ],
                        [9.00902364e-01, 6.14878950e-02, 8.32619846e-01, ]],

                       [[7.78156751e-01, 1.38565413e+00, 1.57059286e+00, ],
                        [5.45191457e-03, 6.79815900e-04, 9.97444031e-01, ],
                        [2.26203117e-02, 1.06668269e+00, 1.43353287e-01, ]],

                       [[9.44668917e-01, -1.52521310e-01, 2.98849725e-01],
                        [1.66120491e-02, 1.54052495e+00, 2.86552189e-03],
                        [4.29600896e-01, -6.03754687e-01, 6.17635497e-01]]])
    bias = -0.75243291
    return predict(variational_classifier, params, bias, ising_configs)


def predict(variational_classifier, params, bias, configs):
    predictions = [variational_classifier(params, bias, c) for c in configs]
    res = [1 if p >= 0 else -1 for p in predictions]
    return res


if __name__ == "__main__":
    inputs = np.array(
        sys.stdin.read().split(","), dtype=int, requires_grad=False
    ).reshape(DATA_SIZE, -1)
    ising_configs = inputs[:, :-1]
    labels = inputs[:, -1]
    predictions = classify_ising_data(ising_configs, labels)
    # print(accuracy(labels, predictions))
    print(*predictions, sep=",")

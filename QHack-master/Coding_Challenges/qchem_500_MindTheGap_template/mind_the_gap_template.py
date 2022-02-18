import sys
import pennylane as qml
from pennylane import numpy as np
from pennylane import hf


def ground_state_VQE(H):
    """Perform VQE to find the ground state of the H2 Hamiltonian.

    Args:
        - H (qml.Hamiltonian): The Hydrogen (H2) Hamiltonian

    Returns:
        - (float): The ground state energy
        - (np.ndarray): The ground state calculated through your optimization routine
    """

    # QHACK #
    dev = qml.device("default.qubit", wires=4)

    def prep_trial_state(param):
        initial_hf_state = np.array([1, 1, 0, 0])
        qml.BasisState(initial_hf_state, wires=range(4))
        qml.DoubleExcitation(param, wires=[0, 1, 2, 3])

    @qml.qnode(dev)
    def avg_energy_fn(param):
        prep_trial_state(param=param)
        return qml.expval(H)

    @qml.qnode(dev)
    def get_ground_state(ideal_param):
        prep_trial_state(param=ideal_param)
        return qml.state()

    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0, requires_grad=True)

    energy = [avg_energy_fn(theta)]
    angle = [theta]

    max_iterations = 100
    conv_tol = 1e-19

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(avg_energy_fn, theta)

        energy.append(avg_energy_fn(theta))
        angle.append(theta)

        conv = np.abs(energy[-1] - prev_energy)

        print(f"Step = {n},  Energy = {energy[-1]:.19f} Ha")

        if conv <= conv_tol:
            break

    return energy[-1], get_ground_state(angle[-1])
    # QHACK #


def create_H1(ground_state, beta, H):
    """Create the H1 matrix, then use `qml.Hermitian(matrix)` to return an observable-form of H1.

    Args:
        - ground_state (np.ndarray): from the ground state VQE calculation
        - beta (float): the prefactor for the ground state projector term
        - H (qml.Hamiltonian): the result of hf.generate_hamiltonian(mol)()

    Returns:
        - (qml.Observable): The result of qml.Hermitian(H1_matrix)
    """

    # QHACK #

    # QHACK #


def excited_state_VQE(H1):
    """Perform VQE using the "excited state" Hamiltonian.

    Args:
        - H1 (qml.Observable): result of create_H1

    Returns:
        - (float): The excited state energy
    """

    # QHACK #

    # QHACK #


if __name__ == "__main__":
    coord = float(sys.stdin.read())
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, -coord], [0.0, 0.0, coord]], requires_grad=False)
    mol = hf.Molecule(symbols, geometry)

    H = hf.generate_hamiltonian(mol)()
    E0, ground_state = ground_state_VQE(H)

    beta = 15.0
    H1 = create_H1(ground_state, beta, H)
    E1 = excited_state_VQE(H1)

    answer = [np.real(E0), E1]
    print(*answer, sep=",")

"""
Text the ansatz
"""
import numpy as np
import cudaq
from src.vqe_cudaq_qnp import VQE
from src.vqe_cudaq_qnp import get_cudaq_hamiltonian, convert_state_big_endian
import pickle
from itertools import product
import time
import pandas as pd
import os
from datetime import datetime

from openfermion.linalg import get_sparse_operator
from openfermion.hamiltonians import s_squared_operator, sz_operator, number_operator



def kernel(input_state: List[complex]):
    qubits = cudaq.qvector(input_state)

    y.ctrl(qubits[0], qubits[1])

@cudaq.kernel
def vqe_ansatz(params, input_state):
    thetas = params
    n_qubits = int(np.log2(input_state.size))
    n_layers = 1
    number_of_blocks = n_qubits // 2 - 1

    qubits = cudaq.qvector(input_state)

    count_params = 0
    for idx_layer in range(n_layers):
        for starting_block_num in [0, 1]:
            for idx_block in range(starting_block_num, number_of_blocks, 2):
                qubit_list = [qubits[2 * idx_block + j] for j in range(4)]

                # PX gates decomposed in terms of standard gates
                # and NO controlled Y rotations.
                # See Appendix E1 of Anselmetti et al New J. Phys. 23 (2021) 113010

                a, b, c, d = qubit_list
                cx(d, b)
                cx(d, a)
                rz(parameter=-np.pi / 2, target=a)
                s(b)
                h(d)
                cx(d, c)
                cx(b, a)
                ry(parameter=(1 / 8) * thetas[count_params], target=c)
                ry(parameter=(-1 / 8) * thetas[count_params], target=d)
                rz(parameter=+np.pi / 2, target=a)
                cz(a, d)
                cx(a, c)
                ry(parameter=(-1 / 8) * thetas[count_params], target=d)
                ry(parameter=(+1 / 8) * thetas[count_params], target=c)
                cx(b, c)
                cx(b, d)
                rz(parameter=+np.pi / 2, target=b)
                ry(parameter=(-1 / 8) * thetas[count_params], target=c)
                ry(parameter=(+1 / 8) * thetas[count_params], target=d)
                cx(a, c)
                cz(a, d)
                ry(parameter=(-1 / 8) * thetas[count_params], target=c)
                ry(parameter=(1 / 8) * thetas[count_params], target=d)
                cx(d, c)
                h(d)
                cx(d, b)
                s(d)
                rz(parameter=-np.pi / 2, target=b)
                cx(b, a)
                count_params += 1

                # Orbital rotation
                fermionic_swap(np.pi, b, c)
                givens_rotation((-1 / 2) * thetas[count_params], a, b)
                givens_rotation((-1 / 2) * thetas[count_params], c, d)
                fermionic_swap(np.pi, b, c)
                count_params += 1

    # return kernel, thetas


def get_unitary(param_list, num_qubits):  # cudaq.kernel, num_qubits: int) -> np.ndarray:
    """Return the unitary matrix of a `cudaq.kernel`. Currently relies on simulation, could change in future releases
    of cudaq."""

    N = 2 ** num_qubits
    U = np.zeros((N, N), dtype=np.complex128)

    for j in range(N):
        state_j = np.zeros((N), dtype=np.complex128)
        state_j[j] = 1.0
        state_ansatz = convert_state_big_endian(cudaq.get_state(vqe_ansatz, param_list, state_j))

        U[:, j] = np.array(state_ansatz, copy=False)

    return U


def main():
    n_qubits = 4
    num_act_orbitals = n_qubits // 2
    num_active_electrons = 2
    spin = 0

    vqe = VQE(n_qubits=n_qubits,
              num_active_electrons=num_active_electrons,
              spin=spin,
              options={})

    param_list = np.random.rand(vqe.num_params)
    U = get_unitary(param_list, n_qubits)

    spin_s_square_sparse = get_sparse_operator(s_squared_operator(num_act_orbitals))
    spin_s_z_sparse = get_sparse_operator(sz_operator(num_act_orbitals))
    num_operator_sparse = get_sparse_operator(number_operator(2 * num_act_orbitals))

    correlator_1 = U @ spin_s_square_sparse - spin_s_square_sparse @ U

    correlator_2 = U @ spin_s_z_sparse - spin_s_z_sparse @ U

    correlator_3 = U @ num_operator_sparse - num_operator_sparse @ U

    print(not np.any(correlator_1))

    print(not np.any(correlator_2))

    print(not np.any(correlator_3))


if __name__ == "__main__":
    main()

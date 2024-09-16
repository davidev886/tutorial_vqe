"""
Text the ansatz
"""
import numpy as np
import cudaq
from src.vqe_cudaq_qnp import VQE
from src.vqe_cudaq_qnp import get_cudaq_hamiltonian
import pickle
from itertools import product
import time
import pandas as pd
import os
from datetime import datetime

from openfermion.linalg import get_sparse_operator
from openfermion.hamiltonians import s_squared_operator, sz_operator, number_operator


def get_unitary(kernel, param_list, num_qubits): # cudaq.kernel, num_qubits: int) -> np.ndarray:
    """Return the unitary matrix of a `cudaq.kernel`. Currently relies on simulation, could change in future releases
    of cudaq."""

    N = 2 ** num_qubits
    U = np.zeros((N, N), dtype=np.complex128)

    for j in range(N):
        state_j = np.zeros((N), dtype=np.complex128)
        state_j[j] = 1.0

        U[:, j] = np.array(cudaq.get_state(kernel, param_list, state_j), copy=False)

    return U

def main():
    n_qubits = 4
    num_act_orbitals = n_qubits // 2
    num_active_electrons = 2
    spin = 0
    start_t = time.time()
    vqe = VQE(n_qubits=n_qubits,
              num_active_electrons=num_active_electrons,
              spin=spin,
              options={})

    kernel, thetas = vqe.layers()
    param_list = np.random.rand(vqe.num_params)
    U = get_unitary(kernel, param_list, n_qubits)

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

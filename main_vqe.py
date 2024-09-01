"""
Contains the main file for running a complete VQE with cudaquantum
"""
from collections import defaultdict

import numpy as np
from src.vqe_cudaq_qnp import get_molecular_hamiltonian
from src.vqe_cudaq_qnp import VQE
from src.vqe_cudaq_qnp import get_cudaq_hamiltonian
import pickle
from itertools import product
import time
import pandas as pd
import os
from datetime import datetime

str_date = datetime.today().strftime('%Y%m%d_%H%M%S')
os.makedirs(str_date, exist_ok=True)

num_active_orbitals = 5
num_active_electrons = 5
spin = 1
geometry = "systems/geo_o3.xyz"
basis = 'cc-pvtz'

jw_hamiltonian_file = f"fenta/ham_fenta_cc-pvtz_{num_active_electrons}e_{num_active_orbitals}o.pickle"

if jw_hamiltonian_file:
    time_s = time.time()
    with open(jw_hamiltonian_file, "rb") as hamiltonian_file:
        jw_hamiltonian = pickle.load(hamiltonian_file)
    hamiltonian, constant_term = get_cudaq_hamiltonian(jw_hamiltonian)
    time_e = time.time()
    print("Time for converting openfermion to cudaquantum spinop", time_e - time_s)
else:
    hamiltonian, constant_term = get_molecular_hamiltonian(geometry=geometry,
                                                           num_active_electrons=num_active_electrons,
                                                           num_active_orbitals=num_active_orbitals)

MINIMIZE_METHODS = ['nelder-mead', 'powell', 'cobyla']

best_parameters = None
result_final_energy = defaultdict(list)

for optimizer_type, num_layers in product(MINIMIZE_METHODS, range(1, 10)):
    print(f"# {optimizer_type}, {num_layers}")
    if optimizer_type == 'nelder-mead':
        best_parameters = None

    options = {'n_vqe_layers': num_layers,
               'maxiter': 10000,
               'energy_core': constant_term,
               'return_final_state_vec': False,
               'optimizer': optimizer_type,
               'target': 'nvidia',
               'target_option': 'mqpu'}

    n_qubits = 2 * num_active_orbitals
    start_t = time.time()
    vqe = VQE(n_qubits=n_qubits,
              num_active_electrons=num_active_electrons,
              spin=spin,
              options=options)

    if num_layers == 1:
        vqe.options['initial_parameters'] = np.random.rand(vqe.num_params)
    else:
        # use as starting parameters the best from previous VQE
        vqe.options['initial_parameters'] = best_parameters

    results = vqe.execute(hamiltonian)

    # Best energy from VQE
    optimized_energy = results['energy_optimized']
    callback_energies = results['callback_energies']
    best_parameters = results['best_parameters']

    end_t = time.time()
    print("# Time for VQE [min]", (end_t - start_t) / 60)
    result_final_energy["num_layers"].append(num_layers)
    result_final_energy["optimized_energy"].append(optimized_energy)
    result_final_energy["optimizer_type"].append(optimizer_type)
    result_final_energy["time_vqe [s]"].append((end_t - start_t))

    if len(result_final_energy["num_layers"]) > 1:
        df = pd.DataFrame(result_final_energy)
    else:
        df = pd.DataFrame(result_final_energy, index=[0])

    df.to_csv(f'{str_date}/energies.csv', index=False)

    fname = f"callback_energies_fenta_{basis}_"\
            f"cas_{num_active_electrons}e_{num_active_orbitals}o_"\
            f"layer_{num_layers}_opt_{optimizer_type}.dat"

    np.savetxt(os.path.join(str_date, fname),
               callback_energies)
    print()

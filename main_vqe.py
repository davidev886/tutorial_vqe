"""
Contains the main file for running a complete VQE with cudaquantum
"""
import numpy as np
from src.vqe_cudaq_qnp import get_molecular_hamiltonian
from src.vqe_cudaq_qnp import VQE
from src.vqe_cudaq_qnp import get_cudaq_hamiltonian
import pickle

jw_hamiltonian_file = ""
geometry = "systems/geo_o3.xyz"

num_active_orbitals = 12
num_active_electrons = 9
spin = 1
if jw_hamiltonian_file:
    with open(r"jw_hamiltonian_file", "rb") as hamiltonian_file:
        jw_hamiltonian = pickle.load(hamiltonian_file)
    hamiltonian, constant_term = get_cudaq_hamiltonian(jw_hamiltonian)
else:
    hamiltonian, constant_term = get_molecular_hamiltonian(geometry=geometry,
                                                           num_active_electrons=num_active_electrons,
                                                           num_active_orbitals=num_active_orbitals)

MINIMIZE_METHODS_NEW_CB = ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                           'l-bfgs-b', 'trust-constr', 'dogleg', 'trust-ncg',
                           'trust-exact', 'trust-krylov', 'cobyqa']

# Define some options for the VQE
options = {'n_vqe_layers': 1,
           'maxiter': 1000,
           'energy_core': constant_term,
           'return_final_state_vec': False,
           'optimizer': 'nelder-mead'}

n_qubits = 2 * num_active_orbitals

vqe = VQE(n_qubits=n_qubits,
          num_active_electrons=num_active_electrons,
          spin=spin,
          options=options)

results = vqe.execute(hamiltonian)

# Best energy from VQE
optimized_energy = results['energy_optimized']

energy_optimized = results['callback_energies']

np.savetxt("energies.dat", energy_optimized)

"""
Contains the main file for running a complete VQE with cudaquantum
"""

from src.vqe_cudaq_qnp import get_molecular_hamiltonian
from src.vqe_cudaq_qnp import VQE

geometry = "systems/geo_o3.xyz"

num_active_orbitals = 6
num_active_electrons = 8
spin = 0
hamiltonian_o3, constant_term = get_molecular_hamiltonian(geometry=geometry,
                                                          num_active_electrons=num_active_electrons,
                                                          num_active_orbitals=num_active_orbitals)

MINIMIZE_METHODS_NEW_CB = ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                           'l-bfgs-b', 'trust-constr', 'dogleg', 'trust-ncg',
                           'trust-exact', 'trust-krylov', 'cobyqa']

# Define some options for the VQE
options = {'n_vqe_layers': 1,
           'maxiter': 100,
           'energy_core': constant_term,
           'return_final_state_vec': True,
           'optimizer': 'nelder-mead'}

n_qubits = 2 * num_active_orbitals

vqe = VQE(n_qubits=n_qubits,
          num_active_electrons=num_active_electrons,
          spin=spin,
          options=options)

results = vqe.execute(hamiltonian_o3)

# Best energy from VQE
optimized_energy = results['energy_optimized']

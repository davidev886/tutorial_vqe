"""
Contains the main file for running a complete VQE with cudaquantum
"""

from src.vqe_cudaq_qnp import get_molecular_hamiltonian

geometry = "systems/geo_o3.xyz"

num_active_orbitals = 6
num_active_electrons = 8

hamiltonian_o3 = get_molecular_hamiltonian(geometry=geometry,
                                           num_active_electrons=num_active_electrons,
                                           num_active_orbitals=num_active_orbitals)

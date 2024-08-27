"""
Contains the main file for running a complete VQE with cudaquantum
"""

from src.utils_pyscf import System

geometry = "systems/geo_o3.xyz"

num_active_orbitals = 6
num_active_electrons = 8

system_o3 = System(geometry=geometry,
                   num_active_electrons=num_active_electrons,
                   num_active_orbitals=num_active_orbitals)


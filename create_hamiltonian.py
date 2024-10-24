from src.utils_ipie import get_molecular_hamiltonian

num_active_orbitals = 30
num_active_electrons = 17
spin = 1
geometry = "systems/geo_fenta.xyz"

basis = "cc-pVDZ"

data_hamiltonian = get_molecular_hamiltonian(geometry=geometry,
                                             basis=basis,
                                             spin=spin,
                                             num_active_electrons=num_active_electrons,
                                             num_active_orbitals=num_active_orbitals,
                                             create_cudaq_ham=True,
                                             verbose=1,
                                             label_molecule="FeNTA",
                                             dir_save_hamiltonian="./"
                                             )

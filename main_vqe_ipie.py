"""
Contains the main file for running a complete VQE with cudaquantum  plus AFMQC with ipie
"""

import numpy as np
from src.vqe_cudaq_qnp import get_molecular_hamiltonian
from src.vqe_cudaq_qnp import VQE

# AFQMC
from ipie.config import config

config.update_option("use_gpu", False)
from ipie.qmc.afqmc import AFQMC
from ipie.analysis.extraction import extract_observable
from src.utils_ipie import get_afqmc_data

num_active_orbitals = 5
num_active_electrons = 5
spin = 1
geometry = "systems/geo_fenta.xyz"
basis = "sto-3g"
num_vqe_layers = 2
random_seed = 1

n_qubits = 2 * num_active_orbitals
hamiltonian, pyscf_data = get_molecular_hamiltonian(geometry=geometry,
                                                    basis=basis,
                                                    num_active_electrons=num_active_electrons,
                                                    num_active_orbitals=num_active_orbitals)

MINIMIZE_METHODS = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']

optimizer_type = 'Nelder-Mead'
np.random.seed(random_seed)
print(f"# {optimizer_type}, {num_vqe_layers}")

options = {'n_vqe_layers': num_vqe_layers,
           'maxiter': 10000,
           'energy_core': pyscf_data["energy_core"],
           'return_final_state_vec': True,
           'optimizer': optimizer_type,
           'target': 'nvidia',
           'target_option': 'mqpu'}

vqe = VQE(n_qubits=n_qubits,
          num_active_electrons=num_active_electrons,
          spin=spin,
          options=options)

vqe.options['initial_parameters'] = np.random.rand(vqe.num_params)

result = vqe.execute(hamiltonian)

# Best energy from VQE
optimized_energy = result['energy_optimized']
# Final state vector from VQE
final_state_vector = result["state_vec"]

afqmc_hamiltonian, trial_wavefunction = get_afqmc_data(pyscf_data, final_state_vector)

# Setup the AFQMC parameters
afqmc_msd = AFQMC.build(
    pyscf_data["mol"].nelec,
    afqmc_hamiltonian,
    trial_wavefunction,
    num_walkers=100,
    num_steps_per_block=25,
    num_blocks=10,
    timestep=0.005,
    stabilize_freq=5,
    seed=random_seed,
    pop_control_freq=5,
    verbose=False)

# Run the AFQMC
afqmc_msd.run()
afqmc_msd.finalise(verbose=False)

# Extract the energies
qmc_data = extract_observable(afqmc_msd.estimators.filename, "energy")
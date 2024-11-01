"""
Contains the main file for running AFMQC with ipie
"""
import os
import numpy as np
from src.utils_ipie import get_molecular_hamiltonian

from ipie.config import config
config.update_option("use_gpu", True)

from ipie.qmc.afqmc import AFQMC
from ipie.analysis.extraction import extract_observable
from src.utils_ipie import get_afqmc_data

import matplotlib.pyplot as plt

num_active_orbitals = 5
num_active_electrons = 5
spin = 1
geometry = "systems/geo_fenta.xyz"
basis = "cc-pvtz"
num_vqe_layers = 1
random_seed = 1

n_qubits = 2 * num_active_orbitals
data_hamiltonian = get_molecular_hamiltonian(geometry=geometry,
                                             basis=basis,
                                             spin=spin,
                                             num_active_electrons=num_active_electrons,
                                             num_active_orbitals=num_active_orbitals)

pyscf_data = data_hamiltonian["scf_data"]

list_wfname = ["wf_fenta_cc-pvtz_cas_5e_5o_layer_11_opt_COBYLA.npy",
               "wf_fenta_cc-pvtz_cas_5e_5o_layer_11_opt_Powell.npy"]

list_fig_names = ["plot_energy_fenta_cc-pvtz_cas_5e_5o_layer_11_opt_COBYLA.pdf",
                  "plot_energy_fenta_cc-pvtz_cas_5e_5o_layer_11_opt_Powell.pdf"]

list_energy_names = ["energy_fenta_cc-pvtz_cas_5e_5o_layer_11_opt_COBYLA.dat",
                     "energy_fenta_cc-pvtz_cas_5e_5o_layer_11_opt_Powell.dat"]

wfname = list_wfname[0]
name_fig = list_fig_names[0]
energy_fname = list_energy_names[0]

final_state_vector = np.load(os.path.join("best_params", wfname))

afqmc_hamiltonian, trial_wavefunction = get_afqmc_data(pyscf_data, final_state_vector)

# Setup the AFQMC parameters
afqmc_msd = AFQMC.build(
    pyscf_data["mol"].nelec,
    afqmc_hamiltonian,
    trial_wavefunction,
    num_walkers=1000,
    num_steps_per_block=25,
    num_blocks=50,
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

# vqe_y = vqe_energies
# vqe_x = list(range(len(vqe_y)))
# plt.plot(vqe_x, vqe_y, label="VQE")

afqmc_y = list(qmc_data["ETotal"])
plt.plot(afqmc_y, label="AFQMC")

np.savetxt(energy_fname, afqmc_y)

plt.xlabel("Optimization steps")
plt.ylabel("Energy [Ha]")
plt.legend()

plt.savefig(name_fig)

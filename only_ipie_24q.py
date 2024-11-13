from ipie.qmc.afqmc import AFQMC
from ipie.analysis.extraction import extract_observable

import numpy as np

from src.utils_ipie import get_molecular_hamiltonian
from src.utils_ipie import get_afqmc_data

import matplotlib.pyplot as plt

num_active_orbitals = 5
num_active_electrons = 5
spin = 1
num_vqe_layers = 10
random_seed = 1
n_qubits = 2 * num_active_orbitals
chkptfile_rohf = "chkfiles/scf_fenta_sd_converged.chk"
chkptfile_cas = f"chkfiles/{n_qubits}q/mcscf_fenta_converged_{n_qubits}q.chk"

data_hamiltonian = get_molecular_hamiltonian(chkptfile_rohf=chkptfile_rohf,
                                             chkptfile_cas=chkptfile_cas,
                                             num_active_electrons=num_active_electrons,
                                             num_active_orbitals=num_active_orbitals,
                                             create_cudaq_ham=False,
                                             )

pyscf_data = data_hamiltonian["scf_data"]

np.random.seed(random_seed)

final_state_vector = np.load(f'data/state_vec_fenta_cas_{n_qubits}q_layer_10_opt_COBYLA.dat.npy')
vqe_energies = np.loadtxt(f'data/callback_energies_fenta_cas_{n_qubits}q_layer_10_opt_COBYLA.dat')

afqmc_hamiltonian, trial_wavefunction = get_afqmc_data(pyscf_data, final_state_vector)

afqmc_msd = AFQMC.build(
    pyscf_data["mol"].nelec,
    afqmc_hamiltonian,
    trial_wavefunction,
    num_walkers=200,
    num_steps_per_block=10,
    num_blocks=1000,
    timestep=0.005,
    stabilize_freq=5,
    seed=random_seed,
    pop_control_freq=5,
    verbose=True)

afqmc_msd.run(estimator_filename=f"afqmc_data_{n_qubits}q.h5")

afqmc_msd.finalise(verbose=False)

qmc_data = extract_observable(afqmc_msd.estimators.filename, "energy")
np.savetxt(f"{n_qubits}q_vqe_energy.dat", vqe_energies)
np.savetxt(f"{n_qubits}q_afqmc_energy.dat", list(qmc_data["ETotal"]))

vqe_y = vqe_energies
vqe_x = list(range(len(vqe_y)))
plt.plot(vqe_x, vqe_y, label="VQE")

afqmc_y = list(qmc_data["ETotal"])
afqmc_x = [i + vqe_x[-1] for i in list(range(len(afqmc_y)))]
plt.plot(afqmc_x, afqmc_y, label="AFQMC")

plt.xlabel("Optimization steps")
plt.ylabel("Energy [Ha]")
plt.legend()

plt.savefig(f'vqe_afqmc_plot_{n_qubits}q.png')

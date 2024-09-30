"""
Contains the main file for running a complete VQE with cudaquantum  plus AFMQC with ipie
"""

import numpy as np
from src.vqe_cudaq_qnp import get_molecular_hamiltonian
from src.vqe_cudaq_qnp import VQE

# AFQMC
from ipie.config import config
config.update_option("use_gpu", False)
# from ipie.qmc.afqmc import AFQMC
# from ipie.analysis.extraction import extract_observable
# from src.utils_ipie import get_afqmc_data

import matplotlib.pyplot as plt

num_active_orbitals = 5
num_active_electrons = 8
spin = 0
geometry = "systems/geo_o3.xyz"
basis = "sto-3g"
num_vqe_layers = 1
random_seed = 1

n_qubits = 2 * num_active_orbitals
hamiltonian, pyscf_data, chkfile, jw_hamiltonian = get_molecular_hamiltonian(geometry=geometry,
                                                             basis=basis,
                                                             num_active_electrons=num_active_electrons,
                                                             num_active_orbitals=num_active_orbitals)

MINIMIZE_METHODS = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']

optimizer_type = 'Nelder-Mead'
np.random.seed(random_seed)
print(f"# {optimizer_type}, {num_vqe_layers}")

options = {'n_vqe_layers': num_vqe_layers,
           'maxiter': 100,
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
vqe_energies = result["callback_energies"]
# Final state vector from VQE
final_state_vector = result["state_vec"]

from openfermion.linalg import get_sparse_operator

hamiltonian_array = get_sparse_operator(jw_hamiltonian, n_qubits).toarray()

energy_computed = final_state_vector.conj().T @ hamiltonian_array @ final_state_vector

print("energy from openfermion", energy_computed)

# AFQMC
####

from src.utils_ipie import get_coeff_wf, gen_ipie_input_from_pyscf_chk

from ipie.config import config

config.update_option("use_gpu", False)
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.particle_hole import ParticleHole
from ipie.analysis.extraction import extract_observable

# Generate the input Hamiltonian for ipie from the checkpoint file from pyscf
ipie_hamiltonian = gen_ipie_input_from_pyscf_chk(chkfile,
                                                 mcscf=True,
                                                 chol_cut=1e-5)

h1e, cholesky_vectors, e0 = ipie_hamiltonian

num_basis = cholesky_vectors.shape[1]
num_chol = cholesky_vectors.shape[0]

system = Generic(nelec=pyscf_data["mol"].nelec)

afqmc_hamiltonian = HamGeneric(
    np.array([h1e, h1e]),
    cholesky_vectors.transpose((1, 2, 0)).reshape((num_basis * num_basis, num_chol)),
    e0,
)

wavefunction = get_coeff_wf(final_state_vector,
                            n_active_elec=num_active_electrons,
                            spin=spin
                            )

trial = ParticleHole(
    wavefunction,
    pyscf_data["mol"].nelec,
    num_basis,
    num_dets_for_props=len(wavefunction[0]),
    verbose=True)

trial.compute_trial_energy = True
trial.build()
trial.half_rotate(afqmc_hamiltonian)

# Setup the AFQMC parameters
afqmc_msd = AFQMC.build(
    pyscf_data["mol"].nelec,
    afqmc_hamiltonian,
    trial,
    num_walkers=100,
    num_steps_per_block=25,
    num_blocks=10,
    timestep=0.005,
    stabilize_freq=5,
    seed=96264512,
    pop_control_freq=5,
    verbose=True)

# Run the AFQMC
afqmc_msd.run()
afqmc_msd.finalise(verbose=True)

# Extract the energies
qmc_data = extract_observable(afqmc_msd.estimators.filename, "energy")

exit()
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

vqe_y = vqe_energies
vqe_x = list(range(len(vqe_y)))
plt.plot(vqe_x, vqe_y, label="VQE")

afqmc_y = list(qmc_data["ETotal"])
afqmc_x = [i + vqe_x[-1] for i in list(range(len(afqmc_y)))]
plt.plot(afqmc_x, afqmc_y, label="AFQMC")

plt.xlabel("Optimization steps")
plt.ylabel("Energy [Ha]")
plt.legend()

plt.savefig('vqe_afqmc_plot.png')

import os
import h5py
import json
import pickle

from ipie.config import config
config.update_option("use_gpu", True)

# os.environ['IPIE_USE_GPU'] = "1"
# config.update_option("use_gpu", True)

from ipie.qmc.afqmc import AFQMC
from ipie.analysis.extraction import extract_observable

import numpy as np

from src.utils_ipie import get_molecular_hamiltonian
# from src.vqe_cudaq_qnp import VQE
from src.utils_ipie import get_afqmc_data

from src.vqe_cudaq_qnp import get_cudaq_hamiltonian

import matplotlib.pyplot as plt

# !pip install -r requirements.txt
# !echo cuda-quantum | sudo -S apt-get install -y cuda-toolkit-11.8 && python -m pip install cupy
# ! sudo -S apt-get install -y cuda-toolkit-12.6


system = '10q'
# system = '24q'
# system = '40q'
# system = '72q'

geometry = "chkfiles/FeNTA_spin5_QZVPP/geo.xyz"
chkptfile_rohf = "chkfiles/FeNTA_spin5_QZVPP/rohf_orbs.chk"
basis = "def2-qzvpp"
spin = 5

if system == '10q':

    num_active_orbitals = 5
    num_active_electrons = 5
    num_walkers = 2000
    chkptfile_cas = "chkfiles/FeNTA_spin5_QZVPP/5_5_CASCI/mcscf.chk"
    jw_hamiltonian_name = "hamiltonians/ham_FeNTA_def2-QZVPP_5e_5o.pickle"

elif system == '24q':

    num_active_orbitals = 12
    num_active_electrons = 9
    num_walkers = 2000
    chkptfile_cas = "chkfiles/FeNTA_spin5_QZVPP/9_12_CASCI/mcscf.chk"
    jw_hamiltonian_name = "hamiltonians/ham_FeNTA_def2-QZVPP_9e_12o.pickle"

elif system == '40q':

    num_active_orbitals = 20
    num_active_electrons = 17
    num_walkers = 2000
    chkptfile_cas = "chkfiles/FeNTA_spin5_QZVPP/17_20_DMRGCASCI/mcscf.chk"
    jw_hamiltonian_name = "hamiltonians/ham_FeNTA_def2-QZVPP_17e_20o.pickle"

elif system == '72q':
    num_active_orbitals = 36
    num_active_electrons = 49
    num_walkers = 2000
    chkptfile_cas = "chkfiles/FeNTA_spin5_QZVPP/49_36_DMRGCASCI/mcscf.chk"
    jw_hamiltonian_name = "hamiltonians/ham_FeNTA_def2-QZVPP_49e_36o.pickle"

else:
    print("system not defined")
    exit()

with open(jw_hamiltonian_name, 'rb') as filehandler:
    jw_hamiltonian = pickle.load(filehandler)

hamiltonian_cudaq, energy_core_cudaq_ham = get_cudaq_hamiltonian(jw_hamiltonian)

num_vqe_layers = 10
optimizer_type = "Powell"
maxiter_vqe = 2000
random_seed = 1
n_qubits = 2 * num_active_orbitals

# Get the molecular Hamiltonian and molecular data from pyscf
data_hamiltonian = get_molecular_hamiltonian(chkptfile_rohf=chkptfile_rohf,
                                             chkptfile_cas=chkptfile_cas,
                                             num_active_electrons=num_active_electrons,
                                             num_active_orbitals=num_active_orbitals,
                                             create_cudaq_ham=False,
                                             )

hamiltonian = data_hamiltonian["hamiltonian"]
pyscf_data = data_hamiltonian["scf_data"]

# MINIMIZE_METHODS = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']

# # Define optimization methods for VQE
# np.random.seed(random_seed)

# # Define options for the VQE algorithm
# options = {'n_vqe_layers': num_vqe_layers,
#            'maxiter': maxiter_vqe,
#            'energy_core': pyscf_data["energy_core_cudaq_ham"],
#            'return_final_state_vec': True,
#            'optimizer': optimizer_type,
#            'target': 'nvidia',
#            'target_option': 'mqpu'}


# # Initialize the VQE algorithm
# vqe = VQE(n_qubits=n_qubits,
#           num_active_electrons=num_active_electrons,
#           spin=spin,
#           options=options)

# # Set initial parameters for the VQE algorithm
# vqe.options['initial_parameters'] = np.random.rand(vqe.num_params)

# # Execute the VQE algorithm
# result = vqe.execute(hamiltonian)

# # Extract results from the VQE execution
# optimized_energy = result['energy_optimized']
# vqe_energies = result["callback_energies"]
# final_state_vector = result["state_vec"]
# best_parameters = result["best_parameters"]

# np.save('final_state_vector_' + system + '.npy', final_state_vector)

final_state_vector = np.load('final_state_vector_' + system + '.npy')

# Get AFQMC data (hamiltonian and trial wave function) in ipie format
# using the molecular data from pyscf and the final state vector from VQE
afqmc_hamiltonian, trial_wavefunction = get_afqmc_data(pyscf_data, final_state_vector)

# Initialize AFQMC
afqmc_msd = AFQMC.build(
    pyscf_data["mol"].nelec,
    afqmc_hamiltonian,
    trial_wavefunction,
    num_walkers=num_walkers,
    num_steps_per_block=25,
    num_blocks=10,
    timestep=0.001,
    stabilize_freq=5,
    seed=random_seed,
    pop_control_freq=5,
    verbose=True)

# Run the AFQMC simulation and save data to .h5 file
afqmc_msd.run(estimator_filename='afqmc_data_' + system + '.h5')

afqmc_msd.finalise(verbose=False)

# Extract and plot results
qmc_data = extract_observable(afqmc_msd.estimators.filename, "energy")
# np.savetxt(system + '_vqe_energy.dat', vqe_energies)
np.savetxt(system + '_afqmc_energy.dat', list(qmc_data["ETotal"]))

# vqe_y = vqe_energies
# vqe_x = list(range(len(vqe_y)))
# plt.plot(vqe_x, vqe_y, label="VQE")

# afqmc_y = list(qmc_data["ETotal"])
# afqmc_x = [i + vqe_x[-1] for i in list(range(len(afqmc_y)))]
# plt.plot(afqmc_x, afqmc_y, label="AFQMC")

# plt.xlabel("Optimization steps")
# plt.ylabel("Energy [Ha]")
# plt.legend()

# plt.savefig('vqe_afqmc_'+system+'_plot.png')

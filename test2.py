import os
os.environ['IPIE_USE_GPU'] = "1"

from ipie.config import config
config.update_option("use_gpu", True)

from ipie.qmc.afqmc import AFQMC
from ipie.analysis.extraction import extract_observable

import numpy as np
import cupy as cp 

from src.utils_ipie import get_molecular_hamiltonian
from src.vqe_cudaq_qnp import VQE
from src.utils_ipie import get_afqmc_data

import matplotlib.pyplot as plt

# !pip install -r requirements.txt 
# !echo cuda-quantum | sudo -S apt-get install -y cuda-toolkit-11.8 && python -m pip install cupy

# num_active_orbitals = 6
# num_active_electrons = 8
# spin = 0
# geometry = "systems/geo_o3.xyz"
# basis = "sto-3g"
# # basis = "cc-pVDZ"
# num_vqe_layers = 1
# random_seed = 1
# n_qubits = 2 * num_active_orbitals

num_active_orbitals = 5
num_active_electrons = 5
spin = 1
geometry = "systems/geo_fenta.xyz"
# basis = "sto-3g"
# basis = "cc-pVDZ"
basis = "cc-pVTZ"
num_vqe_layers = 13
random_seed = 1
n_qubits = 2 * num_active_orbitals


data_hamiltonian = get_molecular_hamiltonian(geometry=geometry,
                                             basis=basis,
                                             spin=spin,
                                             num_active_electrons=num_active_electrons,
                                             num_active_orbitals=num_active_orbitals,
                                             create_cudaq_ham=True,
                                             )

hamiltonian = data_hamiltonian["hamiltonian"]
pyscf_data = data_hamiltonian["scf_data"]

MINIMIZE_METHODS = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']

optimizer_type = 'COBYLA'
np.random.seed(random_seed)

options = {'n_vqe_layers': num_vqe_layers,
           'maxiter': 3000,
           'energy_core': pyscf_data["energy_core_cudaq_ham"],
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

optimized_energy = result['energy_optimized']
vqe_energies = result["callback_energies"]
final_state_vector = result["state_vec"]


# np.save('final_state_vector.npy', final_state_vector)

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
cp.cuda.Device(rank).use()

# final_state_vector = np.load('final_state_vector.npy')
    
afqmc_hamiltonian, trial_wavefunction = get_afqmc_data(pyscf_data, final_state_vector)

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


afqmc_msd.run()


afqmc_msd.finalise(verbose=False)


if rank == 0: 
    
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

comm.Barrier()

MPI.Finalize()

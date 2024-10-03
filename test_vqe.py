import numpy as np
from src.utils_ipie import get_molecular_hamiltonian
from src.vqe_cudaq_qnp import VQE
import matplotlib.pyplot as plt

num_active_orbitals = 5
num_active_electrons = 8
spin = 0
geometry = "systems/geo_o3.xyz"
basis = "sto-3g"
num_vqe_layers = 1
random_seed = 1

n_qubits = 2 * num_active_orbitals
data_hamiltonian = get_molecular_hamiltonian(geometry=geometry,
                                             basis=basis,
                                             num_active_electrons=num_active_electrons,
                                             num_active_orbitals=num_active_orbitals)

hamiltonian = data_hamiltonian["hamiltonian"]
pyscf_data = data_hamiltonian["scf_data"]

MINIMIZE_METHODS = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']

optimizer_type = 'Nelder-Mead'
np.random.seed(random_seed)
print(f"# {optimizer_type}, {num_vqe_layers}")

options = {'n_vqe_layers': num_vqe_layers,
           'maxiter': 100,
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

# Best energy from VQE
optimized_energy = result['energy_optimized']
vqe_energies = result["callback_energies"]
# Final state vector from VQE
final_state_vector = result["state_vec"]

np.save("final_state_vector.npy", final_state_vector)

vqe_y = vqe_energies
vqe_x = list(range(len(vqe_y)))
plt.plot(vqe_x, vqe_y, label="VQE")

plt.savefig('vqe_plot.png')

import numpy as np
from ipie.analysis.extraction import extract_observable
import matplotlib.pyplot as plt


file_energy = "afqmc_data_10q.h5"
vqe_energy_file = "./data/callback_energies_fenta_cas_10q_layer_10_opt_COBYLA.dat"

vqe_energies = np.loadtxt(vqe_energy_file)
qmc_data = extract_observable(file_energy, "energy")

afqmc_energies = list(qmc_data["ETotal"])

vqe_y = vqe_energies
vqe_x = list(range(len(vqe_y)))
plt.plot(vqe_x, vqe_y, label="VQE")

afqmc_y = list(qmc_data["ETotal"])
afqmc_x = [i + vqe_x[-1] for i in list(range(len(afqmc_y)))]
plt.plot(afqmc_x, afqmc_y, label="AFQMC")

plt.xlabel("Optimization steps")
plt.ylabel("Energy [Ha]")
plt.legend()
# plt.show()
plt.savefig(f"vqe_afqmc_plot_{len(afqmc_energies)}.png")

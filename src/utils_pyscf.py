"""
Contains the class for creating the cudaquantum hamiltonian from its geometry, basis etc. via pyscf
"""

import os
import numpy as np
from pyscf import gto, scf, ao2mo, mcscf
import time


class System(object):
    def __init__(self,
                 geometry,
                 num_active_orbitals,
                 num_active_electrons,
                 basis="cc-pVDZ",
                 spin=0,
                 charge=0,
                 verbose=0):

        self.molecule = gto.M(
            atom=geometry,
            spin=spin,
            basis=basis,
            charge=charge,
            verbose=verbose
        )
        print('# Start Hartree-Fock computation')
        hartee_fock = scf.ROHF(self.molecule)
        # Run Hartree-Fock
        hartee_fock.kernel()

        from openfermion.transforms import jordan_wigner
        from openfermion import generate_hamiltonian
        from src.vqe_cudaq_qnp import get_cudaq_hamiltonian

        my_casci = mcscf.CASCI(hartee_fock, num_active_orbitals, num_active_electrons)
        ss = (self.molecule.spin / 2 * (self.molecule.spin / 2 + 1))
        my_casci.fix_spin_(ss=ss)

        print('# Start CAS computation')
        e_tot, e_cas, fcivec, mo_output, mo_energy = my_casci.kernel()

        h1, energy_core = my_casci.get_h1eff()
        h2 = my_casci.get_h2eff()
        h2_no_symmetry = ao2mo.restore('1', h2, num_active_orbitals)
        tbi = np.asarray(h2_no_symmetry.transpose(0, 2, 3, 1), order='C')

        mol_ham = generate_hamiltonian(h1, tbi, energy_core.item())
        jw_hamiltonian = jordan_wigner(mol_ham)
        print("# Preparing the cudaq Hamiltonian")
        start = time.time()
        self.hamiltonian_cudaq, self.energy_core = get_cudaq_hamiltonian(jw_hamiltonian)
        end = time.time()
        print("# Time for preparing the cudaq Hamiltonian:", end - start)

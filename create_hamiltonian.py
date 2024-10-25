import numpy as np
import time
import os
from pyscf import gto, scf, ao2mo, mcscf
import pickle
from openfermion.transforms import jordan_wigner
from openfermion import generate_hamiltonian
from pyscf import dmrgscf
from pyscf import lib


def save_jw_molecular_hamiltonian(
        geometry,
        num_active_orbitals,
        num_active_electrons,
        basis="cc-pVDZ",
        spin=0,
        charge=0,
        verbose=0,
        label_molecule="molecule",
        dir_save_hamiltonian="./",
        dmrg=False) -> dict:
    """
     Compute the molecular Hamiltonian for a given molecule using Hartree-Fock and CASCI methods.

     :param str geometry: Atomic coordinates of the molecule in the format required by PySCF.
     :param int num_active_orbitals: Number of active orbitals for the CASCI calculation.
     :param int num_active_electrons: Number of active electrons for the CASCI calculation.
     :param str basis: Basis set to be used for the calculation. Default is 'cc-pVDZ'.
     :param int spin: Spin multiplicity of the molecule. Default is 0.
     :param int charge: Charge of the molecule. Default is 0.
     :param int verbose: Verbosity level of the calculation. Default is 0.
     :param str label_molecule: optional label for saving the hamiltonian file
     :param str dir_save_hamiltonian: optional directory name for saving the hamiltonian file
     :param bool dmrg: True if dmrg casci  should be performed

    :return: An InteractionOperatro containing the Hamiltonian

     """
    molecule = gto.M(
        atom=geometry,
        spin=spin,
        basis=basis,
        charge=charge,
        verbose=verbose
    )
    start_t = time.time()
    print('# Start Hartree-Fock computation')
    hartee_fock = scf.ROHF(molecule)
    # Run Hartree-Fock
    hartee_fock.kernel()
    print("Time for HF:",  time.time() - start_t)
    hcore = scf.hf.get_hcore(molecule)
    s1e = molecule.intor("int1e_ovlp_sph")

    my_casci = mcscf.CASCI(hartee_fock, num_active_orbitals, num_active_electrons)
    if dmrg:
        dmrg_states = 500

        my_casci.fcisolver = dmrgscf.DMRGCI(molecule, maxM=dmrg_states, tol=1E-10)
        my_casci.fcisolver.threads = 16
        my_casci.fcisolver.memory = int(molecule.max_memory / 1000)  # mem in GB
        my_casci.fcisolver.conv_tol = 1e-14
        if verbose:
            print(f"# using dmrg pyscf with {dmrg_states} states")
            print(f"# using dmrg pyscf in {my_casci.fcisolver.runtimeDir} runtimeDir")
            print(f"# using dmrg pyscf in {my_casci.fcisolver.scratchDirectory} scratchDirectory")
    else:
        ss = (molecule.spin / 2 * (molecule.spin / 2 + 1))
        my_casci.fix_spin_(ss=ss)

    print('# Start CAS computation')
    e_tot, e_cas, fcivec, mo_output, mo_energy = my_casci.kernel()
    print('# Energy CAS', e_tot)
    h1, energy_core = my_casci.get_h1eff()
    h2 = my_casci.get_h2eff()
    h2_no_symmetry = ao2mo.restore('1', h2, num_active_orbitals)
    tbi = np.asarray(h2_no_symmetry.transpose(0, 2, 3, 1), order='C')

    mol_ham = generate_hamiltonian(h1, tbi, energy_core.item())
    jw_hamiltonian = jordan_wigner(mol_ham)
    hamiltonian_fname = (f"ham_{label_molecule}_{basis.lower()}_"
                         f"{num_active_electrons}e_{num_active_orbitals}o.pickle")
    with open(os.path.join(dir_save_hamiltonian, hamiltonian_fname), 'wb') as filehandler:
        print(f"# saving hamiltonian pickle in {os.path.join(dir_save_hamiltonian, hamiltonian_fname)}")
        pickle.dump(jw_hamiltonian, filehandler)

    return jw_hamiltonian


def run():
    num_active_orbitals = 30
    num_active_electrons = 17
    spin = 1
    geometry = "systems/geo_fenta.xyz"

    basis = "cc-pVDZ"

    save_jw_molecular_hamiltonian(geometry=geometry,
                                  basis=basis,
                                  spin=spin,
                                  num_active_electrons=num_active_electrons,
                                  num_active_orbitals=num_active_orbitals,
                                  verbose=1,
                                  label_molecule="FeNTA",
                                  dir_save_hamiltonian="./",
                                  dmrg=True
                                  )

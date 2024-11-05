import os
import h5py
import numpy as np
from pyscf import fci, gto, mcscf, scf

from ipie.config import config

config.update_option("use_gpu", True)

from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.particle_hole import ParticleHoleNonChunked, ParticleHole
from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
from ipie.analysis.extraction import extract_observable
import shutil
from pyscf.lib import chkfile
from src.utils import get_coeff_wf
import json


def run():
    nocca = 3
    noccb = 2
    n_active_orbitals = 5
    n_active_electrons = nocca + noccb
    spin = nocca - noccb
    chkptfile_rohf = "../systems/FeNTA_spin_1/basis_cc-pVTZ/ROHF/scfref.chk"
    chkptfile_cas = "../systems/FeNTA_spin_1/basis_cc-pVTZ/CAS_5_5/mcscf.chk"
    basis = "ccpvtz"

    with h5py.File(chkptfile_rohf, "r") as f:
        mol_bytes = f["mol"][()]
        mol = json.loads(mol_bytes.decode('utf-8'))
        geometry = mol["_atom"]

    mol = gto.M(
        atom=geometry,
        basis=basis,
        verbose=4,
        spin=spin,
        unit="Bohr",
    )
    mol_nelec = mol.nelec

    create_input_ipie = not (os.path.isfile("hamiltonian.h5") and os.path.isfile("wavefunction.h5"))

    if create_input_ipie:
        mf = scf.ROHF(mol)
        dm = mf.from_chk(chkptfile_rohf)
        mf.kernel(dm)

        mc = mcscf.CASCI(mf, n_active_orbitals, n_active_electrons)
        mo = chkfile.load(chkptfile_cas, 'mcscf/mo_coeff')
        e_tot, e_cas, fcivec, mo, mo_energy = mc.kernel(mo)

        shutil.copyfile(chkptfile_rohf, "./scf.chk")

        final_state_vector = np.load("../best_params/wf_fenta_cc-pvtz_cas_5e_5o_layer_11_opt_Powell.npy")
        coeff, occa, occb = get_coeff_wf(final_state_vector, (nocca, noccb), thres=1e-6)

        # Need to write wavefuncti on to checkpoint file.
        with h5py.File("scf.chk", "r+") as fh5:
            fh5["mcscf/ci_coeffs"] = coeff
            fh5["mcscf/occs_alpha"] = occa
            fh5["mcscf/occs_beta"] = occb

        gen_ipie_input_from_pyscf_chk("scf.chk", mcscf=True, chol_cut=1e-3, num_frozen_core=69)

    with h5py.File("hamiltonian.h5") as fa:
        chol = fa["LXmn"][()]
        h1e = fa["hcore"][()]
        e0 = fa["e0"][()]

    num_basis = chol.shape[1]
    system = Generic(nelec=mol_nelec)

    num_chol = chol.shape[0]
    ham = HamGeneric(
        np.array([h1e, h1e]),
        chol.transpose((1, 2, 0)).reshape((num_basis * num_basis, num_chol)),
        e0,
    )

    # Build trial wavefunction
    with h5py.File("wavefunction.h5", "r") as fh5:
        coeff = fh5["ci_coeffs"][:]
        occa = fh5["occ_alpha"][:]
        occb = fh5["occ_beta"][:]
    wavefunction = (coeff, occa, occb)
    trial = ParticleHole(
        wavefunction,
        mol_nelec,
        num_basis,
        num_dets_for_props=len(wavefunction[0]),
        verbose=True,
    )
    trial.compute_trial_energy = True
    trial.build()
    trial.half_rotate(ham)

    afqmc_msd = AFQMC.build(
        mol_nelec,
        ham,
        trial,
        num_walkers=1000,
        num_steps_per_block=25,
        num_blocks=400,
        timestep=0.005,
        stabilize_freq=5,
        seed=96264512,
        pop_control_freq=5,
        verbose=True,
    )
    afqmc_msd.run(estimator_filename="energy_powell.h5")
    afqmc_msd.finalise(verbose=True)

    qmc_data = extract_observable(afqmc_msd.estimators.filename, "energy")

    afqmc_y = list(qmc_data["ETotal"])
    np.savetxt("afqmc_energies.dat", afqmc_y)


if __name__ == "__main__":
    run()

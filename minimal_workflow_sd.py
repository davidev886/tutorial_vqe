import os
import sys
import numpy as np
from pyscf import cc, gto, scf
from ipie.config import MPI

try:
    import cupy
except ImportError:
    sys.exit(0)

mol = gto.M(
    atom=[("H", 1.6 * i, 0, 0) for i in range(0, 4)],
    basis="sto-6g",
    verbose=4,
    unit="Bohr",
)
mf = scf.UHF(mol)
mf.chkfile = "scf.chk"
mf.kernel()

# Need to flag that we want to use GPU before **any** ipie modules are imported
from ipie.config import config
config.update_option("use_gpu", True)

from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk

# Checkpoint integrals and wavefunction
# Running in serial but still need MPI World
gen_ipie_input_from_pyscf_chk(mf.chkfile, verbose=0)

comm = MPI.COMM_WORLD

from ipie.qmc.calc import build_afqmc_driver

afqmc = build_afqmc_driver(comm, nelec=mol.nelec)
afqmc.params.num_blocks = 20
afqmc.run()

from pyscf import gto, scf

spin = 1
geometry = "../systems/geo_fenta.xyz"
basis = "def2-TZVP"

molecule = gto.M(
    atom=geometry,
    spin=spin,
    basis=basis,
    charge=0,
    verbose=4
)
print('# Start Hartree-Fock computation')
hartee_fock = scf.ROHF(molecule)
hartee_fock.chkfile = 'scf_fenta_sd.chk'
hartee_fock.kernel()

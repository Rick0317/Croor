from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2 import tools
from block2 import SU2 as block2_SU2
from block2 import SZ as block2_SZ
from pathlib import Path
import numpy as np


def check_eigenstates(n, ne, s, obt, tbt, optimized_ket: str, mps_directory: str):
    symmetry_type = SymmetryTypes.SZ
    stack_mem = 10 * 1024 * 1024 * 1024
    stack_mem_ratio = 0.5
    num_threads = 3
    restart_dir = Path("./temp")

    mps_dir = Path(f"./{mps_directory}/mps_storage/{optimized_ket}")
    driver = DMRGDriver(
        stack_mem=stack_mem,
        scratch=str(mps_dir),
        clean_scratch=True,  # Default value
        restart_dir=str(restart_dir),
        n_threads=num_threads,
        # n_mkl_threads=n_mkl_threads,  # Default value is 1
        symm_type=symmetry_type,
        mpi=None,  # Default value
        stack_mem_ratio=stack_mem_ratio,  # Default value 0.4
        fp_codec_cutoff=1e-120,  # Default value 1e-16,
    )

    tools.init(block2_SZ)
    mps = tools.loadMPSfromDir(mps_info=None, mpsSaveDir=mps_dir)

    driver.initialize_system(n_sites=n, n_elec=ne,
                             spin=s, orb_sym=None)

    csfs, coeffs = driver.get_csf_coefficients(mps, cutoff=1e-308, iprint=0)

    mpo = driver.get_qc_mpo(h1e=obt, g2e=tbt,
                            ecore=0, iprint=0)
    ener = driver.expectation(mps, mpo, mps)

    bra = driver.get_random_mps(tag="GS5", bond_dim=250, nroots=1)
    applied_mps = driver.multiply(bra, mpo, mps)

    variance = np.sqrt(abs(applied_mps ** 2 - ener ** 2))

    return variance

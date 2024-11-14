from pyscf import gto, scf
import pyscf
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2 import tools
from basic_definitions import chemist_to_physicist, get_ferm_op, physicist_to_chemist
from pyblock2._pyscf.ao2mo import integrals as itg


class DMRGHamiltonians:
    def __init__(self, driver, mpo, mps):
        self._driver = driver
        self._mpo = mpo
        self._mps = mps

    def get_driver(self):
        return self._driver

    def get_mpo(self):
        return self._mpo

    def get_mps(self):
        return self._mps


def get_dmrg_hamiltonian(mol, obt, tbt):

    mf = scf.RHF(mol).run(conv_tol=1E-14)
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf,
                                                                         ncore=0,
                                                                         ncas=None,
                                                                         g2e_symm=8)

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ,
                        n_threads=4)
    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin)

    mpo = driver.get_qc_mpo(
        h1e=obt,
        g2e=tbt,
        ecore=0,
        para_type=None,  # Default value
        reorder=None,  # Reordering not done here
        cutoff=1e-120,  # Default value
        integral_cutoff=1e-120,  # Default value
        post_integral_cutoff=1e-120,  # Default value
        fast_cutoff=1e-120,  # Default value
        unpack_g2e=True,  # Default value
        algo_type=None,
        # Default value, MPOAlgorithmTypes.FastBipartite will be used
        normal_order_ref=None,  # Default value, normal ordering not done here
        normal_order_single_ref=None,
        # Default value, only used if normal_order_ref is not None
        normal_order_wick=True,
        # Default value, only used if normal_order_ref is not None
        symmetrize=False,
        # Only impacts if orb_sym  in initialize_system is not None
        # Set False to avoid unexpected behavior
        sum_mpo_mod=-1,  # Default value, no effect if algo_type=None
        compute_accurate_svd_error=True,
        # Default value, no effect if algo_type=None
        csvd_sparsity=0.0,  # Default value, no effect if algo_type=None
        csvd_eps=1e-10,  # Default value, no effect if algo_type=None
        csvd_max_iter=1000,  # Default value, no effect if algo_type=None
        disjoint_levels=None,  # Default value, no effect if algo_type=None
        disjoint_all_blocks=False,  # Default value, no effect if algo_type=None
        disjoint_multiplier=1.0,  # Default value, no effect if algo_type=None
        block_max_length=False,  # Default value, no effect if algo_type=None
        add_ident=True,
        # Default value, adds ecore*identity to the MPO for expectation values
        esptein_nesbet_partition=False,
        # Default value, only used for perturbative DMRG
        ancilla=False,  # Default value, don't add ancilla sites
        # reorder_imat=None,  # Default value, will not reorder the integrals
        # gaopt_opts=None,  # Default value, options for gaopt reordering
        iprint=0,
    )

    ket = driver.get_random_mps(tag="KET", bond_dim=250, nroots=1)

    return DMRGHamiltonians(driver, mpo, ket)


def dmrg_non_exact(mol, obt, tbt, ed_energy, energy_threshold):
    dmrg_hamil = get_dmrg_hamiltonian(mol, obt, tbt)

    # ed_energy = input_data.get_ed_energy()

    not_hitting_threshold = True
    bd = 1
    bd_max = 100  # (math.comb(L, L // 2)) ** 2
    estimated_bd = 0

    bd_list = []
    err_list = []

    bond_dims = [1] * 12
    noises = [1e-4] * 4 + [1e-5] * 4 + [0]
    thrds = [1e-20] * 20

    print(ed_energy)

    while bd < bd_max + 1 and not_hitting_threshold:
        bond_dims = [bd] * 16
        noises = [1e-4] * 4 + [1e-5] * 4 + [0]
        thrds = [1e-20] * 20

        input_data2 = get_dmrg_hamiltonian(mol, obt, tbt)

        driver2 = input_data2.get_driver()
        mpo2 = input_data2.get_mpo()
        mps2 = input_data2.get_mps()

        driver2.dmrg(
            mpo2,
            mps2,
            n_sweeps=28,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=0,
            twosite_to_onesite=None,
            cutoff=1e-120
        )


        ds, dws, eners = driver2.get_dmrg_results()
        # We set the convergence point as where all the energies in the output
        # being within the threshold
        isLess = True
        for i in range(len(eners)):
            print(eners[i][0])
            if eners[i][0] - ed_energy >= energy_threshold:
                isLess = False

        if isLess:
            # We store the maximum bond dimension being used
            result = max(estimated_bd, max(ds))
            print(f"Converged energies: {eners}")
            return result

        estimated_bd = max(estimated_bd, max(ds))

        bd_list.append(bd)
        err_list.append((min(eners) - ed_energy)[0])
        bd += 3

    return estimated_bd

import scipy
from dmrghandler.src.dmrghandler import dmrg_calc_prepare
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from dmrghandler.src.dmrghandler.qchem_dmrg_calc import single_qchem_dmrg_calc
from get_dmrg_config import  get_dmrg_config, get_dmrg_config_type1
from pathlib import Path

def get_convergence_bd(path, energy_threshold):
    """
    Verify the linear relasionship between the discarded weights
    and the energy difference
    :param L: Number of spatial sites
    :param Ne: Number of electrons
    :param t: Strength of the hopping term
    :param U: Strength of the onsite interaction term
    :param energy_threshold: The convergence threshold
    :return: (Discarded weights, energy differences)
    """

    # %%
    (
        one_body_tensor,
        two_body_tensor,
        nuc_rep_energy,
        num_orbitals,
        num_spin_orbitals,
        num_electrons,
        two_S,
        two_Sz,
        orb_sym,
        extra_attributes,
    ) = dmrg_calc_prepare.load_tensors_from_fcidump(
        data_file_path=path)


    dmrg_params_1 = get_dmrg_config_type1(
        num_orbitals, num_electrons, two_Sz,
        two_S)

    result = single_qchem_dmrg_calc(one_body_tensor, two_body_tensor, dmrg_params_1)
    ed_energy = result['dmrg_ground_state_energy']


    not_hitting_threshold = True
    bd = 1
    bd_max = 2 ** num_orbitals
    estimated_bd = 0

    bd_list = []
    err_list = []
    ds_min_list = []
    de_min_list = []

    while bd < bd_max and not_hitting_threshold:
        bond_dims = [bd] * 16
        noises = [1e-4] * 4 + [1e-5] * 4 + [1e-6] * 4 + [0]
        thrds = [1e-20] * 20
        driver2 = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ,
                            n_threads=4)
        driver2.initialize_system(n_sites=num_orbitals, n_elec=num_electrons, spin=two_S,
                                 orb_sym=orb_sym)

        ### Obtain MPO from tensors ###

        mpo2 = driver2.get_qc_mpo(one_body_tensor, two_body_tensor, iprint=0)
        mps2 = driver2.get_random_mps(tag="GS")

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

        # We set the convergence point as all the energies in the output
        # are within the threshold
        isLess = True
        for i in range(len(eners)):
            if eners[i][0] - ed_energy >= energy_threshold:
                isLess = False

        if isLess:
            # We store the maximum bond dimension being used
            result = max(estimated_bd, max(ds))
            print('BOND DIMS         = ', ds[3::2])
            print('Discarded Weights = ', dws[3::2])
            print('Energies          = ', eners[3::2, 0])
            reg = scipy.stats.linregress(dws[3::2], eners[3::2, 0])
            emin, emax = min(eners[3::2, 0]), max(eners[3::2, 0])
            print('DMRG energy (extrapolated) = %20.15f +/- %15.10f' %
                  (reg.intercept, abs(reg.intercept - emin) / 5))

            de = emax - emin


            return

        estimated_bd = max(estimated_bd, max(ds))

        bd_list.append(bd)
        ds_min_list.append(dws[-4])
        de_min_list.append(eners[-4] - ed_energy)
        err_list.append((min(eners) - ed_energy)[0])
        bd += 3

    print('BOND DIMS         = ', ds[3::2])
    print('Discarded Weights = ', dws[3::2])
    print('Energies          = ', eners[3::2, 0])
    reg = scipy.stats.linregress(dws[3::2], eners[3::2, 0])
    emin, emax = min(eners[3::2, 0]), max(eners[3::2, 0])
    print('DMRG energy (extrapolated) = %20.15f +/- %15.10f' %
          (reg.intercept, abs(reg.intercept - emin) / 5))

    de = emax - emin


    return ds_min_list[1:], de_min_list[1:]


if __name__ == '__main__':
    path = Path("fcidumps_directory/fcidump.N2_22_NO")
    energy_threshold = 1e-5
    get_convergence_bd(path, energy_threshold)

import numpy as np
import pyscf


def get_obt_combined_tbt(obt, tbt, num_orbitals, num_electrons):
    """
    Given a obt and a tbt, return the one body tensor combined TBT.
    :param obt: The original one body tensor
    :param tbt: The original two body tensor
    :param num_orbitals: The number of spatial orbitals
    :param num_electrons: The number of electrons
    :return:
    """
    absorbed_tbt = pyscf.fci.direct_nosym.absorb_h1e(np.array(obt),
                                                     np.array(tbt),
                                                     norb=num_orbitals,
                                                     nelec=num_electrons)
    h_in_tbt = pyscf.ao2mo.restore(1, absorbed_tbt.copy(), num_orbitals).astype(
        np.array(obt).dtype, copy=False)

    return h_in_tbt

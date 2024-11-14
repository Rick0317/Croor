import numpy as np
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from pyscf import gto, scf


class HamiltonianData:
    """
    The basic Hamiltonian data structure.
    """
    def __init__(self, ncas, n_elec, spin, h1e, g2e, orb_sym):
        self.ncas = ncas
        self.n_elec = n_elec
        self.spin = spin
        self.orb_sym = orb_sym
        self.h1e = h1e
        self.g2e = g2e


def get_n2(bond_length):
    """
    Get the Hamiltonian data for N2 molecule.
    :return:
    """
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {bond_length}", basis="sto3g", symmetry="d2h",
                verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1E-14)
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf,
                                                                         ncore=0,
                                                                         ncas=None,
                                                                         g2e_symm=8)

    return HamiltonianData(ncas, n_elec, spin, h1e, g2e, orb_sym)


def get_hf(bond_length):
    """
    Get the Hamiltonian data for N2 molecule.
    :return:
    """
    mol = gto.M(atom=f"H 0 0 0; F 0 0 {bond_length}", basis="sto3g", symmetry="d2h",
                verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1E-14)
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf,
                                                                         ncore=0,
                                                                         ncas=None,
                                                                         g2e_symm=8)

    return HamiltonianData(ncas, n_elec, spin, h1e, g2e, orb_sym)


def get_h2o(bond_length):
    """
    Get the Hamiltonian data for H2O molecule.
    :return:
    """

    mol = gto.M(
        atom=f"O 0 0 0; H  0 {bond_length} 0; H 0 0 {bond_length}",
        basis='sto-3g')

    mf = scf.RHF(mol).run(conv_tol=1E-14)
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf,
                                                                         ncore=0,
                                                                         ncas=None,
                                                                         g2e_symm=8)

    return HamiltonianData(ncas, n_elec, spin, h1e, g2e, orb_sym)


def get_he():
    mol = gto.M(atom="He 0 0 0", basis="sto-3g")
    mf = scf.RHF(mol).run(conv_tol=1E-14)
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf,
                                                                         ncore=0,
                                                                         ncas=None,
                                                                         g2e_symm=8)

    return HamiltonianData(ncas, n_elec, spin, h1e, g2e, orb_sym)


def get_co():
    mol = gto.M(atom = 'C 0 0 0; O 0 0 1.5', basis="sto-3g")
    mf = scf.RHF(mol).run(conv_tol=1E-14)
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf,
                                                                         ncore=0,
                                                                         ncas=None,
                                                                         g2e_symm=8)

    return HamiltonianData(ncas, n_elec, spin, h1e, g2e, orb_sym)


def get_h2():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.0', basis="sto-3g")
    mf = scf.RHF(mol).run(conv_tol=1E-14)
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf,
                                                                         ncore=0,
                                                                         ncas=None,
                                                                         g2e_symm=8)

    return HamiltonianData(ncas, n_elec, spin, h1e, g2e, orb_sym)


if __name__ == "__main__":
    helium = get_he()
    print(helium.n_elec)

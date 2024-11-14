import numpy as np
from openfermion import FermionOperator, hermitian_conjugated, normal_ordered

a_ = np.array([[0, 1],
               [0, 0]])
a_dag = np.array([[0, 0],
                   [1, 0]])

a_u_ = np.array([[0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])

a_u_dag = np.array([[0, 0, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 1, 0]])

a_d_ = np.array([[0, 0, 1, 0],
                 [0, 0, 0, -1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]])

a_d_dag = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, -1, 0, 0]])

z_ = np.array([[1, 0],
               [0, -1]])


def a_p(p: int, n: int):
    """
    Annihilation operator acting at p.
    :param p: 0 <= p < n
    :param n: Total number of sites
    :return:
    """
    tensor = 1
    for _ in range(p):
        tensor = np.kron(tensor, z_)

    tensor = np.kron(tensor, a_)

    for _ in range(n - p -1):
        tensor = np.kron(tensor, np.eye(2))

    return tensor


def a_dag_p(p: int, n: int):
    """
    Creation operator acting at p.
    :param p: 0 <= p < n
    :param n: Total number of sites
    :return:
    """
    tensor = 1
    for _ in range(p):
        tensor = np.kron(tensor, z_)

    tensor = np.kron(tensor, a_dag)

    for _ in range(n - p - 1):
        tensor = np.kron(tensor, np.eye(2))

    return tensor


def a_p_q_nf(p: int, q: int, n: int):
    tensor = np.zeros((n, n))
    tensor[p, q] = 1
    return tensor


def a_u_p(p: int, n: int):
    pass


def a_d_p(p: int, n: int):
    pass


def a_u_dag_p(p: int, n: int):
    pass


def a_d_dag_p(p: int, n: int):
    pass


def validate_unitary(U: np.ndarray):
    n = U.shape[0]
    identity = np.eye(n)
    U_dag = U.transpose().conj()
    assert np.allclose(U @ U_dag, identity), "input is not unitary"
    assert np.allclose(U_dag @ U, identity), "input is not unitary"
    print("Input is unitary")


def chemist_to_physicist_trunc(obt, tbt, trunc):
    n = obt.shape[0]
    obt_phy = obt.copy()
    tbt_copy = tbt.copy()
    for p in range(n):
        for q in range(n):
            obt_phy[p, q] += sum([tbt.copy()[p, r, r, q] for r in range(n)])
    for p in range(n):
        for q in range(n):
            if abs(obt_phy[p, q]) < trunc:
                obt_phy[p, q] = 0
            for r in range(n):
                for s in range(n):
                    if abs(tbt[p, q, r, s]) < trunc:
                        tbt_copy[p, q, r, s] = 0



    return obt_phy, 2 * tbt_copy

def chemist_to_physicist(obt, tbt):
    n = obt.shape[0]
    obt_phy = obt.copy()
    for p in range(n):
        for q in range(n):
            obt_phy[p, q] += sum([tbt.copy()[p, r, r, q] for r in range(n)])



    return obt_phy, 2 * tbt.copy()

def physicist_to_chemist(obt, tbt):
    n = obt.shape[0]
    obt_phy = obt.copy()
    for p in range(n):
        for q in range(n):
            obt_phy[p, q] -= 0.5 * sum([tbt.copy()[p, r, r, q] for r in range(n)])

    return obt_phy, 0.5 * tbt.copy()




def get_ferm_op_one(obt, spin_orb):
    '''
    Return the corresponding fermionic operators based on one body tensor
    '''
    n = obt.shape[0]
    op = FermionOperator.zero()
    for i in range(n):
        for j in range(n):
            if not spin_orb:
                for a in range(2):
                    op += FermionOperator(
                        term = (
                            (2*i+a, 1), (2*j+a, 0)
                        ), coefficient=obt[i, j]
                    )
            else:
                op += FermionOperator(
                    term = (
                        (i, 1), (j, 0)
                    ), coefficient=obt[i, j]
                )
    return op

def get_ferm_op_two(tbt, spin_orb):
    '''
    Return the corresponding fermionic operators based on tbt (two body tensor)
    This tensor can index over spin-orbtals or orbitals
    '''
    n = tbt.shape[0]
    op = FermionOperator.zero()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if not spin_orb:
                        for a in range(2):
                            for b in range(2):
                                op += FermionOperator(
                                    term = (
                                        (2*i+a, 1), (2*j+a, 0),
                                        (2*k+b, 1), (2*l+b, 0)
                                    ), coefficient=tbt[i, j, k, l]
                                )
                    else:
                        op += FermionOperator(
                            term=(
                                (i, 1), (j, 0),
                                (k, 1), (l, 0)
                            ), coefficient=tbt[i, j, k, l]
                        )
    return op

def get_ferm_op(tsr, spin_orb=False):
    '''
    Return the corresponding fermionic operators based on the tensor
    This tensor can index over spin-orbtals or orbitals
    '''
    if len(tsr.shape) == 4:
        return get_ferm_op_two(tsr, spin_orb)
    elif len(tsr.shape) == 2:
        return get_ferm_op_one(tsr, spin_orb)



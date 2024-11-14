import numpy as np
import scipy
import basic_definitions as base


def orbital_rotation_p_q(p: int, q: int, n: int,  theta: float) -> np.ndarray:
    """
    Orbital rotation given as e^{theta * (a^_p a_q - h.c.)}
    :param p: Creation operator index
    :param q: Annihilation operator index
    :param n: The total number of sites
    :param theta: The rotation angle
    :return:
    """
    generator = base.a_dag_p(p, n) @ base.a_p(q, n)
    return scipy.linalg.expm(theta * (generator - generator.transpose().conj()))


def orbital_rotation_p_q_nf(p: int, q: int, n: int, theta: float) -> np.ndarray:
    generator = base.a_p_q_nf(p, q, n)
    return scipy.linalg.expm(theta * (generator - generator.transpose().conj()))


if __name__ == '__main__':
    h1e_orig = np.array([[1, -1, 3],
                    [-1, 0, 4],
                    [3, 4, 2]])
    U_1 = orbital_rotation_p_q_nf(1, 2, 3, 0.1)
    U_2 = orbital_rotation_p_q(1, 1, 3, 0.2)

    p = np.einsum_path("pi, qj, ij -> pq", U_1, U_1,  h1e_orig)[0]

    h1e = np.einsum("pi, qj, ij -> pq", U_1, U_1, h1e_orig, optimize=p)

    print(h1e)
    base.validate_unitary(U_1)


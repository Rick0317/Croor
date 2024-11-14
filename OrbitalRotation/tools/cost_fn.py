import numpy as np


def cost_fn_diag(H, rotations: list[np.ndarray]):
    dag_rotations = []
    for rotation in rotations:
        dag_rotations.append(rotation.conj().transpose())

    H_eff = 1


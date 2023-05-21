"""
"""

import numpy as np


def convert_to_companion(coefs):
    """
    Return compansion matrix for the VAR(1) representation for a VAR(p) process
    (companion form)

    A = [A_1 A_2 ... A_p-1 A_p
         I_K 0       0     0
         0   I_K ... 0     0
         0 ...       I_K   0]
    """
    p, k1, k2 = coefs.shape
    if k1 != k2:
        raise ValueError('coefs must be 3-d with shape (p, k, k).')

    kp = k1 * p

    result = np.zeros((kp, kp))
    result[:k1] = np.concatenate(coefs, axis=1)

    # Set I_K matrices
    if p > 1:
        result[np.arange(k1, kp), np.arange(kp-k1)] = 1

    return result

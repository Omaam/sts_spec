"""
"""
import tensorflow as tf
import numpy as np

import utils


class VARSpectraAnalyzer:
    """
    """
    def __init__(self, coefficients, level_scales, freqs=None):
        coefficients = tf.convert_to_tensor(coefficients, name="coefficients")
        if len(coefficients.shape) < 3:
            raise ValueError(
                "len(coefficients) must be greater than 3. "
                f"len(coefficients)={len(coefficients)}")

        batch_shape = coefficients.shape[:-3]
        coef_shape = coefficients.shape[-3:]
        order = coefficients.shape[-3]
        vector_dim = coefficients.shape[-2]

        if freqs is None:
            freq_edges = tf.linspace(0., 0.5, 101)
            freqs = (freq_edges[1:] + freq_edges[:-1]) / 2
        freqs = tf.convert_to_tensor(value=freqs, name="frequencies")

        self.batch_shape = batch_shape
        self.coef_shape = coef_shape
        self.order = order
        self.vector_dim = vector_dim
        self.coefficients = coefficients
        self.level_scales = tf.convert_to_tensor(level_scales, dtype=float)
        self.freqs = freqs

        self._cross_spectra = None

    def compute_auto_cavariance(self):
        def vec(mat):
            return tf.reshape(mat, [-1])

        def unvec(v):
            k = int(np.sqrt(len(v)))
            assert k * k == len(v)
            return v.reshape((k, k), order="F")

        kron = tf.experimental.numpy.kron

        p = self.order
        k = self.vector_dim
        A = utils.convert_to_companion(self.coefficients)
        W = tf.broadcast_to(tf.linalg.diag(self.level_scales),
                            self.coefficients.shape)
        W = np.zeros((k * p, k * p))
        W[:k, :k] = tf.linalg.diag(self.level_scales)

        acov_vec = tf.linalg.solve(
                tf.eye((k * p) ** 2, dtype=tf.float64) - kron(A, A),
                tf.cast(vec(W), dtype=tf.float64)[:, tf.newaxis])
        acov = tf.reshape(acov_vec, (-1, k*p, k*p))

        vecACF = np.linalg.solve(np.eye((k * p) ** 2) - np.kron(A, A), vec(W))
        acf = unvec(vecACF)

        assert np.allclose(acov, acf)

        return acov

    def compute_cross_spectra(self):
        A = self._compute_coefficients_fourier()
        B = tf.linalg.inv(A)

        permutation_former = tf.range(B.ndim-2, dtype=tf.int32)
        permutation_latter = tf.range(B.ndim-1, B.ndim-3, -1, dtype=tf.int32)
        permutation = tf.concat([permutation_former, permutation_latter],
                                axis=0)
        B_H = tf.transpose(B, perm=permutation, conjugate=True)

        W = tf.broadcast_to(tf.linalg.diag(self.level_scales), B.shape)
        W = tf.cast(W, dtype=tf.complex64)
        cross_spectra = B @ W @ B_H
        self._cross_spectra = cross_spectra
        return cross_spectra

    def compute_power_spectra(self):
        cross_spectra = self.cross_spectra
        power_spectra = tf.linalg.diag_part(cross_spectra)
        return power_spectra

    def reset_frequency(self, freqs):
        self._cross_spectra = None
        self.freqs = freqs

    @property
    def frequency(self):
        return self.freqs

    @property
    def cross_spectra(self):
        if self._cross_spectra is None:
            self.compute_cross_spectra()
        return self._cross_spectra

    def _compute_coefficients_fourier(self):
        a0 = tf.linalg.diag(tf.repeat(-1., self.vector_dim))

        a0 = tf.broadcast_to(a0, self.coefficients.shape)[..., 0:1, :, :]
        coefficients_with_a0 = tf.concat([a0, self.coefficients], axis=-3)
        coefficients_with_a0 = tf.expand_dims(coefficients_with_a0, axis=-4)
        coefficients_with_a0 = tf.cast(coefficients_with_a0,
                                       dtype=tf.complex64)

        phases = tf.complex(
          0.,
          -2 * np.pi * self.freqs[:, tf.newaxis] * tf.range(self.order + 1.))
        fourier_coefficients = coefficients_with_a0 * tf.exp(
          phases)[tf.newaxis, ..., tf.newaxis, tf.newaxis]
        fourier_coefficients = tf.reduce_sum(fourier_coefficients, axis=-3)
        return fourier_coefficients

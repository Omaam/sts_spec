"""Spectra tests.
"""
import unittest

import tensorflow as tf
import numpy as np

from spectra import VARSpectraAnalyzer


class VarAnalyzer:
    def __init__(self, arcoef, W):
        self.arcoef = np.array(arcoef)
        self.arorder = arcoef.shape[0]
        self.num_series = arcoef.shape[1]

        self.W = W
        self.P = None

    def compute_cross_spectra(self, num_freqs=None):
        freqs = self._generate_frequency(num_freqs)
        A = self._compute_arcoef_fourier(freqs)
        B = np.linalg.inv(A)
        B_H = np.conjugate(np.transpose(B, axes=[0, 2, 1]))
        W = self.W

        P = B @ W @ B_H

        self.num_freqs = num_freqs
        self.freqs = freqs = freqs
        self.P = P

        return P

    @property
    def amplitude_spectra(self):
        self._maybe_compute_computation_cross_spectra()
        return np.abs(self.P)

    @property
    def coherency(self):
        self._maybe_compute_computation_cross_spectra()

        alpha_jk = self.amplitude_spectra**2
        p_jj = np.real(np.diagonal(self.cross_spectra, axis1=1, axis2=2))
        p_kk = np.real(np.diagonal(self.cross_spectra, axis1=1, axis2=2))
        coherency = alpha_jk / p_jj[:, :, None] / p_kk[:, None, :]

        return coherency

    @property
    def cross_spectra(self):
        if self.P is None:
            self.compute_cross_spectra()
        return self.P

    @property
    def decomposed_powerspectra(self):
        self._maybe_compute_computation_cross_spectra()

        W = self.W
        B = np.linalg.inv(self.A)

        decomp_pspec = np.abs(B)**2 * np.diag(np.abs(W))
        decomp_pspec = np.cumsum(decomp_pspec, axis=2)
        return decomp_pspec

    @property
    def frequency(self):
        self._maybe_compute_computation_cross_spectra()
        return self.freqs

    @property
    def relative_power_contribution(self):
        self._maybe_compute_computation_cross_spectra()
        decomp_pspec = self.decomposed_powerspectra
        rel_pcontrib = decomp_pspec / decomp_pspec[:, :, -1][:, :, None]
        return rel_pcontrib

    @property
    def phase_spectra(self):
        self._maybe_compute_computation_cross_spectra()
        return np.angle(self.cross_spectra)

    @property
    def power_spectra(self):
        self._maybe_compute_computation_cross_spectra()
        power_spectra = np.diagonal(self.cross_spectra, axis1=1, axis2=2).real
        return power_spectra

    def _maybe_compute_computation_cross_spectra(self):
        if self.P is None:
            self.compute_cross_spectra()

    def _compute_arcoef_fourier(self, freqs):
        a0 = np.diag(np.repeat(-1, self.num_series))
        arcoef = np.insert(self.arcoef, 0, a0, axis=0)

        phases = -2j * np.pi * freqs[:, None] * np.arange(self.arorder + 1)
        A = arcoef * np.exp(phases)[:, :, None, None]
        A = A.sum(axis=1)

        self.A = A
        return A

    def _generate_frequency(self, num_freqs=None):
        num_freqs = 101 if num_freqs is None else num_freqs
        freq_edges = np.linspace(0, 0.5, num_freqs, endpoint=True)
        freqs = (freq_edges[1:] + freq_edges[:-1]) / 2
        return freqs


class VARSpectraAnalyzerTest(unittest.TestCase):
    """Test class of `VARSpectraAnalyzer`.

    Most tests are comparithon with `VarAnalyzer`, which is not
    compatible with batch processing.
    """
    def setUp(self):
        order = 2
        vector_dim = 2
        coefficients = tf.constant(
            [[[0.5, 0.2],
              [-0.2, 0.1]],
             [[0.2, -0.1],
              [0.1, 0.1]]])
        level_scales = tf.constant([0.1, 0.1])
        var_spec_analyzer = VarAnalyzer(coefficients, np.diag(level_scales))
        var_spec_analyzer_batch = VARSpectraAnalyzer(
            coefficients, level_scales)

        self.order = order
        self.vector_dim = vector_dim
        self.coefficients = coefficients
        self.level_scales = level_scales
        self.var_spec_analyzer = var_spec_analyzer
        self.var_spec_analyzer_batch = var_spec_analyzer_batch

    def test_compute_cross_spectra_value(self):
        expected = self.var_spec_analyzer.compute_cross_spectra()
        actual = self.var_spec_analyzer_batch.compute_cross_spectra()
        is_correct = np.allclose(expected, actual[0])
        self.assertTrue(is_correct)

    def test_compute_cross_spectra_shape(self):
        expected = self.var_spec_analyzer.compute_cross_spectra()
        actual = self.var_spec_analyzer_batch.compute_cross_spectra()
        is_correct = np.allclose(expected.shape, actual.shape[1:])
        self.assertTrue(is_correct)

    def test_compute_power_spectra_value(self):
        expected = self.var_spec_analyzer.power_spectra
        actual = self.var_spec_analyzer_batch.compute_power_spectra()
        is_correct = np.allclose(expected, actual[0])
        self.assertTrue(is_correct)

    def test_compute_power_spectra_shape(self):
        expected = self.var_spec_analyzer.power_spectra
        actual = self.var_spec_analyzer_batch.compute_power_spectra()
        is_correct = np.allclose(expected.shape, actual.shape[1:])
        self.assertTrue(is_correct)


if __name__ == "__main__":
    unittest.main()

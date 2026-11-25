import unittest
from math import pi

import numpy as np

from gegenbauer_fir import (
    FilterSpec,
    design_lowpass,
    design_minimax,
    gegenbauer_basis_matrix,
    quantize_symmetric_taps,
    _gegenbauer_to_chebyshev,
    _taps_from_chebyshev,
)


class GegenbauerFIRTests(unittest.TestCase):
    def test_endpoint_normalization_and_parity(self):
        x = np.array([1.0, -1.0])
        for lam in (0.0, 0.5, 1.5, 4.0):
            B = gegenbauer_basis_matrix(x, 12, lam)
            np.testing.assert_allclose(B[0], np.ones(13), atol=2e-12)
            np.testing.assert_allclose(B[1], (-1.0) ** np.arange(13), atol=2e-12)

    def test_basis_conversion_matches_symmetric_fir(self):
        rng = np.random.default_rng(1234)
        for lam in (0.0, 0.25, 0.5, 1.5, 4.0):
            for order in (2, 8, 24):
                a = rng.normal(size=order // 2 + 1)
                b = _gegenbauer_to_chebyshev(a, lam)
                h = _taps_from_chebyshev(b)
                omega = np.linspace(0.0, pi, 257)
                A_basis = gegenbauer_basis_matrix(np.cos(omega), order // 2, lam) @ a
                H = np.sum(
                    h[None, :] * np.exp(-1j * omega[:, None] * np.arange(order + 1)),
                    axis=1,
                )
                A_taps = (H * np.exp(1j * (order // 2) * omega)).real
                np.testing.assert_allclose(A_basis, A_taps, atol=2e-10, rtol=2e-10)
                np.testing.assert_allclose(h, h[::-1], atol=0.0, rtol=0.0)

    def test_minimax_is_invariant_to_lambda(self):
        results = []
        for lam in (0.0, 0.5, 1.5, 5.0):
            spec = FilterSpec(
                order=24, wp=0.20 * pi, ws=0.27 * pi, lam=lam,
                regularization=0.0, quadrature_order=96, dc_gain=None,
            )
            results.append(design_minimax(spec, max_iterations=100).taps)
        for h in results[1:]:
            np.testing.assert_allclose(h, results[0], atol=3e-8, rtol=3e-8)

    def test_weighted_l2_enforces_exact_dc_when_requested(self):
        spec = FilterSpec(
            order=20, wp=0.20 * pi, ws=0.30 * pi, lam=1.5,
            quadrature_order=96, dc_gain=1.0,
        )
        result = design_lowpass(spec, diagnostic_grid=4097)
        self.assertAlmostEqual(float(np.sum(result.taps)), 1.0, places=10)
        np.testing.assert_allclose(result.taps, result.taps[::-1], atol=0.0, rtol=0.0)

    def test_quantization_preserves_symmetry(self):
        spec = FilterSpec(
            order=16, wp=0.20 * pi, ws=0.30 * pi, lam=0.5,
            quadrature_order=64, dc_gain=1.0,
        )
        result = design_lowpass(spec, diagnostic_grid=2049)
        q = quantize_symmetric_taps(result.taps, frac_bits=27, total_bits=32)
        np.testing.assert_array_equal(q, q[::-1])


if __name__ == "__main__":
    unittest.main()

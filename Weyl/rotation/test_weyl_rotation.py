import unittest
import numpy as np
from Weyl_Rotation_Consolidated import (
    ModularPhaseSpace,
    ModularWeylXOperator,
    ModularWeylPOperator,
    LemmaBasedCorrectionStrategy,
    WeylAlgebraBasedCorrectionStrategy,
    SymplecticCorrectionStrategy,
    WignerCorrectionStrategy,
    create_shape_matrix,
    rotate_matrix_around_center,
    apply_corrected_rotation,
    estimate_quantization_effect_modular,
    calculate_rotation_quality,
    discrete_wigner_function
)

class TestWeylRotation(unittest.TestCase):
    def setUp(self):
        self.dims = [16, 32]
        self.moduli = [16, 32]

    def test_wigner_function(self):
        N = 16
        matrix = np.zeros((N, N), dtype=complex)
        matrix[0, 0] = 1.0
        W = discrete_wigner_function(matrix)
        self.assertEqual(W.shape, (N, N))
        self.assertNotEqual(np.sum(np.abs(W)), 0)

    def test_weyl_operators_non_commutativity(self):
        for dim, mod in zip(self.dims, self.moduli):
            ps = ModularPhaseSpace(dim, mod)
            wx = ModularWeylXOperator(ps, 1)
            wp = ModularWeylPOperator(ps, 1)
            matrix = create_shape_matrix('circle', dim)
            q_effect = estimate_quantization_effect_modular(matrix, wx, wp)
            self.assertGreater(q_effect, 1e-10)

    def test_all_strategies(self):
        dim = 16
        ps = ModularPhaseSpace(dim, dim)
        wx = ModularWeylXOperator(ps, 1)
        wp = ModularWeylPOperator(ps, 1)
        matrix = create_shape_matrix('square', dim)
        angle = 45.0

        strategies = [
            LemmaBasedCorrectionStrategy(),
            WeylAlgebraBasedCorrectionStrategy(),
            SymplecticCorrectionStrategy(),
            WignerCorrectionStrategy()
        ]

        for strategy in strategies:
            c = strategy.get_correction_matrix(matrix, angle, wx, wp)
            self.assertEqual(c.shape, (dim, dim))
            rotated = apply_corrected_rotation(matrix, c, angle)
            self.assertEqual(rotated.shape, (dim, dim))

    def test_parameter_ranges(self):
        dim = 16
        ps = ModularPhaseSpace(dim, dim)
        for q in [0, dim-1, dim]:
            wx = ModularWeylXOperator(ps, q)
            self.assertEqual(wx.quantization_level, q % dim)

if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from Weyl_Rotation_Consolidated import (
    ModularPhaseSpace,
    ModularWeylXOperator,
    ModularWeylPOperator,
    LemmaBasedCorrectionStrategy,
    EnhancedLemmaBasedCorrectionStrategy,
    WeylAlgebraBasedCorrectionStrategy,
    create_shape_matrix,
    rotate_matrix_around_center,
    apply_corrected_rotation,
    estimate_quantization_effect_modular
)

class TestWeylRotation(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        self.modulus = 32
        self.phase_space = ModularPhaseSpace(self.dim, self.modulus)
        self.quant_level = 1
        self.weyl_x = ModularWeylXOperator(self.phase_space, self.quant_level)
        self.weyl_p = ModularWeylPOperator(self.phase_space, self.quant_level)

    def test_modular_phase_space(self):
        self.assertEqual(self.phase_space.get_index(35), 3)
        self.assertEqual(self.phase_space.shift_index(30, 5), 3)

    def test_weyl_x_operator(self):
        matrix = np.zeros((self.dim, self.dim))
        matrix[5, 5] = 1.0
        shifted = self.weyl_x(matrix)
        self.assertEqual(shifted[6, 5], 1.0)
        self.assertEqual(np.sum(shifted), 1.0)

    def test_weyl_p_operator(self):
        matrix = np.zeros((self.dim, self.dim))
        matrix[5, 5] = 1.0
        shifted = self.weyl_p(matrix)
        # Clock operator multiplies by exp(2pi * i * j * q / N)
        # At (5,5), j=5, q=1, N=32. phase = exp(2pi * i * 5 / 32)
        expected_phase = np.exp(2j * np.pi * 5 / 32)
        self.assertAlmostEqual(shifted[5, 5], expected_phase)
        self.assertEqual(np.sum(np.abs(shifted)), 1.0)

    def test_quantization_effect(self):
        # With the new phase operator, non-commutativity should be present
        matrix = create_shape_matrix('circle', self.dim)
        q = estimate_quantization_effect_modular(matrix, self.weyl_x, self.weyl_p)
        self.assertGreater(q, 1e-10)
        print(f"Quantization effect Q = {q}")

    def test_strategies(self):
        matrix = create_shape_matrix('square', self.dim)
        angle = 45.0
        strategies = [
            LemmaBasedCorrectionStrategy(0.01),
            EnhancedLemmaBasedCorrectionStrategy(0.01),
            WeylAlgebraBasedCorrectionStrategy(0.01)
        ]
        for strategy in strategies:
            c = strategy.get_correction_matrix(matrix, angle, self.weyl_x, self.weyl_p)
            self.assertEqual(c.shape, (self.dim, self.dim))
            # Since our image is real, the correction should also be real-valued for rotation
            self.assertTrue(np.isrealobj(c))
            rotated = apply_corrected_rotation(matrix, c, angle)
            self.assertEqual(rotated.shape, (self.dim, self.dim))

    def test_shapes(self):
        shapes = ['square', 'circle', 'triangle', 'line', 'cross']
        for s in shapes:
            mat = create_shape_matrix(s, self.dim)
            self.assertEqual(mat.shape, (self.dim, self.dim))
            self.assertGreater(np.sum(mat), 0)

if __name__ == '__main__':
    unittest.main()

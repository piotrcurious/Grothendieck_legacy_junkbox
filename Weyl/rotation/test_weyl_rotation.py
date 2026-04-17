import unittest
import numpy as np
from Weyl_Rotation_Consolidated import (
    ModularPhaseSpace,
    ModularWeylXOperator,
    ModularWeylPOperator,
    LemmaBasedCorrectionStrategy,
    EnhancedLemmaBasedCorrectionStrategy,
    WeylAlgebraBasedCorrectionStrategy,
    SymplecticCorrectionStrategy,
    create_shape_matrix,
    rotate_matrix_around_center,
    apply_corrected_rotation,
    estimate_quantization_effect_modular,
    calculate_rotation_quality
)

class TestWeylRotation(unittest.TestCase):
    def setUp(self):
        self.dims = [16, 32, 64]
        self.moduli = [16, 32, 64]
        self.quant_levels = [1, 2, 5]

    def test_modular_phase_space(self):
        for dim, mod in zip(self.dims, self.moduli):
            ps = ModularPhaseSpace(dim, mod)
            self.assertEqual(ps.get_index(mod + 3), 3)
            self.assertEqual(ps.shift_index(mod - 1, 2), 1)

    def test_weyl_operators_non_commutativity(self):
        for dim, mod in zip(self.dims, self.moduli):
            ps = ModularPhaseSpace(dim, mod)
            for q in self.quant_levels:
                if q >= mod: continue
                wx = ModularWeylXOperator(ps, q)
                wp = ModularWeylPOperator(ps, q)
                matrix = create_shape_matrix('circle', dim)

                # Check that [P, X] != 0 on a non-trivial matrix
                q_effect = estimate_quantization_effect_modular(matrix, wx, wp)
                self.assertGreater(q_effect, 1e-10, f"Operators should not commute for dim={dim}, mod={mod}, q={q}")

    def test_all_strategies(self):
        dim = 32
        mod = 32
        ps = ModularPhaseSpace(dim, mod)
        wx = ModularWeylXOperator(ps, 1)
        wp = ModularWeylPOperator(ps, 1)
        matrix = create_shape_matrix('square', dim)
        angle = 30.0

        strategies = [
            LemmaBasedCorrectionStrategy(0.01),
            EnhancedLemmaBasedCorrectionStrategy(0.01, 0.5),
            WeylAlgebraBasedCorrectionStrategy(0.01, 0.2),
            SymplecticCorrectionStrategy(0.01, 0.5)
        ]

        standard_rotated = rotate_matrix_around_center(matrix, angle)

        for strategy in strategies:
            c = strategy.get_correction_matrix(matrix, angle, wx, wp)
            self.assertEqual(c.shape, (dim, dim))
            self.assertEqual(c.dtype, complex)

            corrected_rotated = apply_corrected_rotation(matrix, c, angle)
            self.assertEqual(corrected_rotated.shape, (dim, dim))
            self.assertEqual(corrected_rotated.dtype, complex)

            metrics = calculate_rotation_quality(matrix, standard_rotated, corrected_rotated)
            self.assertIn('improvement_ratio', metrics)
            self.assertGreater(metrics['overlap_corrected'], 0)

    def test_shape_reconstruction(self):
        dim = 32
        shapes = ['square', 'circle', 'triangle', 'line', 'cross']
        for s in shapes:
            mat = create_shape_matrix(s, dim)
            self.assertEqual(mat.shape, (dim, dim))
            self.assertEqual(np.sum(mat > 0.5), np.count_nonzero(mat)) # Check binary nature

    def test_quantization_magnitude(self):
        dim = 32
        mod = 32
        ps = ModularPhaseSpace(dim, mod)
        # q = 1 should have non-zero non-commutativity
        wx = ModularWeylXOperator(ps, 1)
        wp = ModularWeylPOperator(ps, 1)
        matrix = create_shape_matrix('circle', dim)
        q_effect = estimate_quantization_effect_modular(matrix, wx, wp)
        self.assertGreater(q_effect, 0.1)

if __name__ == '__main__':
    unittest.main()

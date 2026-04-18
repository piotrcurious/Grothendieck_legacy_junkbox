import unittest
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the solver
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unified_solver import UnifiedAnnihilatorSolver

class TestUnifiedSolver(unittest.TestCase):
    def setUp(self):
        self.solver = UnifiedAnnihilatorSolver(dim=2)

    def test_dynamics_norm_conservation(self):
        # Hamiltonian must be Hermitian for norm conservation
        H = np.array([[1.0, 0.1], [0.1, 1.2]]) * 1e-18
        psi0 = np.array([1.0, 0.0], dtype=complex)
        t_span = (0, 1e-15)
        t_eval = [1e-15]
        sol = self.solver.solve_dynamics(H, psi0, t_span, t_eval)
        final_norm = np.linalg.norm(sol.y[:, -1])
        self.assertAlmostEqual(final_norm, 1.0, places=3)

    def test_groebner_basis(self):
        # Test if basis is computed and has expected properties
        basis = self.solver.compute_constraints(1.0, 2.0)
        self.assertTrue(len(basis) > 0)
        # Check if solutions satisfy original equations
        # x^2 + y^2 = 2, xy = 1 => (1,1) or (-1,-1)
        # Substitute y=1 into basis members
        from sympy import symbols
        x, y = symbols('x y')
        for b in basis:
            res = b.subs({x: 1, y: 1})
            self.assertAlmostEqual(float(res), 0, places=7)

    def test_parameter_estimation(self):
        mock_data = 0.5 * np.exp(-0.2 * np.arange(10))
        res = self.solver.estimate_parameters(mock_data, [0.4, 0.1])
        self.assertAlmostEqual(res.x[0], 0.5, places=2)
        self.assertAlmostEqual(res.x[1], 0.2, places=2)

    def test_orthonormality(self):
        s1 = np.array([1, 0])
        s2 = np.array([0, 1])
        overlap = self.solver.check_orthonormality([s1, s2])
        np.testing.assert_array_almost_equal(overlap, np.eye(2))

    def test_lindblad_trace_conservation(self):
        H = np.eye(2) * 1e-18
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        L_ops = [np.array([[0, 1], [0, 0]]) * 1e-9] # Decay
        sol = self.solver.solve_lindblad(H, rho0, L_ops, (0, 1e-6), [1e-6])
        final_rho = sol.y[:, -1].reshape(2, 2)
        trace = np.trace(final_rho)
        self.assertAlmostEqual(trace.real, 1.0, places=5)

    def test_entanglement_entropy(self):
        # Product state (Zero entropy)
        psi_prod = np.array([1, 0, 0, 0]) # |0>|0>
        entropy_prod = self.solver.calculate_entanglement_entropy(psi_prod, 2, 2)
        self.assertAlmostEqual(entropy_prod, 0.0, places=5)

        # Maximally entangled state (Bell state, Entropy = 1)
        # 1/sqrt(2) * (|00> + |11>) => [1, 0, 0, 1] / sqrt(2)
        psi_bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
        entropy_bell = self.solver.calculate_entanglement_entropy(psi_bell, 2, 2)
        self.assertAlmostEqual(entropy_bell, 1.0, places=5)

if __name__ == '__main__':
    unittest.main()

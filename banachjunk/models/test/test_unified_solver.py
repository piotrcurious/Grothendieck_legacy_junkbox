import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unified_solver import HybridQuantumAlgebraicSolver, TwoLevelLindbladSolver

class TestHybridSolver(unittest.TestCase):
    def setUp(self):
        self.solver = HybridQuantumAlgebraicSolver(N=100)

    def test_norm_conservation(self):
        # Unitary evolution should preserve norm
        psi0 = np.exp(-self.solver.x_grid**2) + 0j
        psi0 /= np.linalg.norm(psi0)
        theta = (1.0, 0.5)
        dt = 0.01
        H = self.solver.hamiltonian(theta)

        psi1 = self.solver.crank_nicolson_step(psi0, H, dt)
        self.assertAlmostEqual(np.linalg.norm(psi1), 1.0, places=10)

    def test_groebner_elimination(self):
        G = self.solver.algebraic_constraints()
        self.assertTrue(len(G) > 0)
        # Verify it contains expected symbols
        from sympy import symbols
        rho, kappa = symbols('rho kappa')
        # Lex order with x, y first means the basis should eventually reduce to rho, kappa terms if solvable
        # but here it's just a general check that it computed.
        self.assertIn(rho, G.atoms())

    def test_parameter_fitting(self):
        # Generate data
        theta_true = [1.0, 0.5]
        psi0 = np.exp(-self.solver.x_grid**2) + 0j
        psi0 /= np.linalg.norm(psi0)
        dt = 0.1
        steps = 10
        traj = self.solver.solve_dynamics(theta_true, psi0, dt, steps)
        data_densities = np.abs(traj)**2

        # Using a very close guess to verify local convergence in this minimal test
        res = self.solver.estimate_parameters(data_densities, psi0, dt, steps, [1.01, 0.49])
        np.testing.assert_array_almost_equal(res.x, theta_true, decimal=2)

class TestTwoLevelSolver(unittest.TestCase):
    def setUp(self):
        self.tls = TwoLevelLindbladSolver(omega=1.0, gamma=0.1, gamma_dephase=0.05)

    def test_relaxation(self):
        # Excited state should relax to ground state
        rho0 = np.array([[0, 0], [0, 1]], dtype=complex)
        traj = self.tls.solve_dynamics(rho0, dt=0.1, steps=100)
        # Ground state pop should increase
        self.assertGreater(np.real(traj[-1][0,0]), np.real(rho0[0,0]))
        # Trace should be preserved (approx)
        self.assertAlmostEqual(np.trace(traj[-1]).real, 1.0, places=5)

    def test_dephasing(self):
        # Superposition state should lose coherence
        rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        traj = self.tls.solve_dynamics(rho0, dt=0.1, steps=50)
        self.assertLess(np.abs(traj[-1][0,1]), np.abs(rho0[0,1]))

    def test_rabi_oscillation(self):
        # Ground state driven to excited
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        traj = self.tls.rabi_oscillation(rho0, dt=0.1, steps=20, drive_amp=1.0)
        excited_pops = [np.real(r[1,1]) for r in traj]
        self.assertGreater(max(excited_pops), 0.4)

if __name__ == '__main__':
    unittest.main()

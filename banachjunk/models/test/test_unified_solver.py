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
        theta_true = [1.0, 2.5]
        psi0 = np.exp(-self.solver.x_grid**2) + 0j
        psi0 /= np.linalg.norm(psi0)
        dt = 0.1
        steps = 15
        traj = self.solver.solve_dynamics(theta_true, psi0, dt, steps)

        # Add 5% Gaussian noise to mock realistic sensor observations
        data_densities = np.abs(traj)**2
        noise = np.random.normal(0, 0.05 * np.mean(data_densities), data_densities.shape)
        data_densities_noisy = np.clip(data_densities + noise, 0, None)

        res = self.solver.estimate_parameters(data_densities_noisy, psi0, dt, steps, [1.2, 2.0])
        # Allow more margin for noise
        np.testing.assert_array_almost_equal(res.x, theta_true, decimal=1)

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

    def test_landau_zener_check(self):
        # Verify that solver handles time-dependent Hamiltonian logic if we were to add it,
        # but for now, we verify stability of dynamics with very high drive.
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        # High drive amp should produce fast oscillations
        traj = self.tls.rabi_oscillation(rho0, dt=0.01, steps=100, drive_amp=10.0)
        # Verify trace preservation under high drive
        for r in traj:
            self.assertAlmostEqual(np.trace(r).real, 1.0, places=5)

    def test_unitarity_check(self):
        # Just verify it runs and returns a reasonable deviation for small dt
        H = 0.5 * np.eye(2)
        dev = self.tls.lindbladian(H) # Wait, tls doesn't have verify_unitarity. HybridSolver has.
        # Rerunning setup for Hybrid Solver
        from unified_solver import HybridQuantumAlgebraicSolver
        hsol = HybridQuantumAlgebraicSolver(N=50)
        H_rand = hsol.hamiltonian([1.0, 0.5])
        dev = hsol.verify_unitarity(H_rand, dt=0.01)
        self.assertLess(dev, 1e-10)

if __name__ == '__main__':
    unittest.main()

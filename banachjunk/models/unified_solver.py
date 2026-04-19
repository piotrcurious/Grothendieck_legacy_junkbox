import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize
from sympy import symbols, groebner
import scipy.constants as const
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class HybridQuantumAlgebraicSolver:
    def __init__(self, N=200, L=10.0, hbar=1.0, m=1.0):
        if N < 5:
            raise ValueError("Grid size N must be at least 5 for stable Laplacian.")
        self.N = N
        self.L = L
        self.dx = L / N
        self.hbar = hbar
        self.m = m
        self.x_grid = np.linspace(-L/2, L/2, N)

        # Laplacian (finite difference)
        diag = -2.0 * np.ones(N)
        off = np.ones(N-1)
        self.laplacian = (np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)) / self.dx**2
        self.T = -(self.hbar**2 / (2 * self.m)) * self.laplacian

    def potential(self, theta):
        """Effective potential based on parameters theta=(rho, kappa)."""
        rho, kappa = theta
        return rho * self.x_grid**2 + kappa * np.exp(-self.x_grid**2)

    def hamiltonian(self, theta):
        """Constructs the Hamiltonian matrix."""
        V = np.diag(self.potential(theta))
        return self.T + V

    def crank_nicolson_step(self, psi, H, dt):
        """Performs one time step using the Crank-Nicolson method."""
        I = np.eye(len(psi))
        A = I + 1j * dt / (2 * self.hbar) * H
        B = I - 1j * dt / (2 * self.hbar) * H
        return solve(A, B @ psi)

    def verify_unitarity(self, H, dt):
        """Computes deviation from unitarity for the Crank-Nicolson propagator."""
        I = np.eye(H.shape[0])
        A = I + 1j * dt / (2 * self.hbar) * H
        B = I - 1j * dt / (2 * self.hbar) * H
        U = solve(A, B)
        # Check U^dag * U approx I
        U_dag = U.conj().T
        deviation = np.linalg.norm(U_dag @ U - I)
        return deviation

    def solve_dynamics(self, theta, psi0, dt, steps):
        """Evolves the wavefunction over time."""
        H = self.hamiltonian(theta)
        psi = psi0.copy()
        trajectory = [psi]
        # Unitary propagator: U = solve(I + iHdt/2, I - iHdt/2)
        I = np.eye(len(psi))
        A = I + 1j * dt / (2 * self.hbar) * H
        B = I - 1j * dt / (2 * self.hbar) * H
        U = solve(A, B)

        for _ in range(steps):
            psi = U @ psi
            trajectory.append(psi)
        return np.array(trajectory)

    def calculate_ipr(self, psi):
        """Calculates the Inverse Participation Ratio (IPR) to quantify localization."""
        # IPR = sum(|psi|^4) / (sum(|psi|^2))^2
        prob_density = np.abs(psi)**2
        norm_sq = np.sum(prob_density)
        if norm_sq < 1e-12: return 0
        ipr = np.sum(prob_density**2) / (norm_sq**2)
        return ipr

    def algebraic_constraints(self):
        """Computes symbolic Groebner basis for structure resolution."""
        x, y, rho, kappa = symbols('x y rho kappa')
        f1 = x**2 + y**2 - rho * kappa
        f2 = x * y - rho
        # Example elimination of hidden variables x, y
        G = groebner([f1, f2], x, y, rho, kappa, order='lex')
        return G

    def inference_loss(self, theta, data_densities, psi0, dt, steps):
        """Loss function based on probability density observables."""
        try:
            traj = self.solve_dynamics(theta, psi0, dt, steps)
            sim_densities = np.abs(traj)**2
            return np.sum((sim_densities - data_densities)**2)
        except Exception:
            return 1e12

    def estimate_parameters(self, data_densities, psi0, dt, steps, initial_guess):
        """Fits parameters theta to observed densities."""
        res = minimize(self.inference_loss, initial_guess,
                       args=(data_densities, psi0, dt, steps),
                       method='L-BFGS-B',
                       bounds=[(0.01, 5), (0.01, 5)]) # Add bounds for physical parameters
        return res

    def calculate_sensitivity(self, theta, data_densities, psi0, dt, steps):
        """Computes numerical gradient of the loss function."""
        eps = 1e-4
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            t_plus = np.array(theta, dtype=float)
            t_plus[i] += eps
            t_minus = np.array(theta, dtype=float)
            t_minus[i] -= eps
            grad[i] = (self.inference_loss(t_plus, data_densities, psi0, dt, steps) -
                       self.inference_loss(t_minus, data_densities, psi0, dt, steps)) / (2 * eps)
        return grad

    def check_annihilator(self, theta):
        """Checks if the parameters theta satisfy the discovered annihilator (invariant)."""
        rho, kappa = theta
        # From Groebner basis: -rho**2 - y**4 + kappa*rho*y**2 = 0
        # We check if there exists real y such that this holds.
        # This requires kappa*rho >= 2*|rho|.
        # Since theta bounds are (0.01, 5), rho is positive.
        discriminant = (kappa * rho)**2 - 4 * rho**2
        if discriminant < -1e-7:
            return False, f"No real latent state y exists for theta={theta}. Discriminant={discriminant}"
        y2_plus = (kappa * rho + np.sqrt(max(0, discriminant))) / 2
        y2_minus = (kappa * rho - np.sqrt(max(0, discriminant))) / 2
        return True, f"Latent state y^2 solutions: {y2_plus}, {y2_minus}"

class TwoLevelLindbladSolver:
    def __init__(self, omega=1.0, gamma=0.1, gamma_dephase=0.0):
        self.omega = omega
        self.gamma = gamma
        self.gamma_dephase = gamma_dephase
        self.sigma_z = np.array([[1, 0], [0, -1]])
        self.sigma_x = np.array([[0, 1], [1, 0]])
        # |0> is ground [1,0]^T, |1> is excited [0,1]^T
        self.sigma_p = np.array([[0, 0], [1, 0]]) # Raising: |0> -> |1>
        self.sigma_m = np.array([[0, 1], [0, 0]]) # Lowering: |1> -> |0>

    def lindbladian(self, rho):
        """Computes the Lindbladian for a 2-level system with multi-channel collapse."""
        H = 0.5 * self.omega * self.sigma_z
        # Hamiltonian part: -i[H, rho]
        comm = -1j * (H @ rho - rho @ H)

        diss = np.zeros_like(rho, dtype=complex)
        # Channel 1: Relaxation (lowering)
        if self.gamma > 0:
            L1 = self.sigma_m
            L1d = L1.T.conj()
            diss += self.gamma * (L1 @ rho @ L1d - 0.5 * (L1d @ L1 @ rho + rho @ L1d @ L1))

        # Channel 2: Pure Dephasing (sigma_z)
        if self.gamma_dephase > 0:
            L2 = self.sigma_z
            L2d = L2.T.conj()
            diss += self.gamma_dephase * (L2 @ rho @ L2d - 0.5 * (L2d @ L2 @ rho + rho @ L2d @ L2))

        return comm + diss

    def solve_dynamics(self, rho0, dt, steps):
        """Evolves the density matrix using RK4."""
        rho = rho0.copy()
        trajectory = [rho]
        for s in range(steps):
            k1 = self.lindbladian(rho)
            k2 = self.lindbladian(rho + 0.5 * dt * k1)
            k3 = self.lindbladian(rho + 0.5 * dt * k2)
            k4 = self.lindbladian(rho + dt * k3)
            rho = rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            # Physical sanity check: eigenvalues should be in [0, 1]
            evals = np.linalg.eigvals(rho)
            if np.any(evals.real < -1e-6) or np.any(evals.real > 1.00001):
                logging.warning(f"Step {s}: Physicality violation in density matrix. Eigenvalues: {evals}")

            trajectory.append(rho)
        return np.array(trajectory)

    def rabi_oscillation(self, rho0, dt, steps, drive_amp=0.5):
        """Simulates Rabi oscillations with a coherent drive sigma_x."""
        original_omega = self.omega
        # Effective Hamiltonian in rotating frame (resonance)
        H_drive = 0.5 * drive_amp * self.sigma_x

        def driven_lindbladian(rho):
            comm = -1j * (H_drive @ rho - rho @ H_drive)
            diss = np.zeros_like(rho, dtype=complex)
            if self.gamma > 0:
                L = self.sigma_m
                diss += self.gamma * (L @ rho @ L.T.conj() - 0.5 * (L.T.conj() @ L @ rho + rho @ L.T.conj() @ L))
            return comm + diss

        rho = rho0.copy()
        trajectory = [rho]
        for _ in range(steps):
            k1 = driven_lindbladian(rho)
            k2 = driven_lindbladian(rho + 0.5 * dt * k1)
            k3 = driven_lindbladian(rho + 0.5 * dt * k2)
            k4 = driven_lindbladian(rho + dt * k3)
            rho = rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            trajectory.append(rho)
        return np.array(trajectory)

def main():
    solver = HybridQuantumAlgebraicSolver()

    logging.info("1. Structural Resolution (Groebner Basis)")
    G = solver.algebraic_constraints()
    logging.info(f"Groebner Basis: {G}")

    logging.info("2. Numerical Time Evolution (Crank-Nicolson)")
    psi0 = np.exp(-solver.x_grid**2) + 0j
    psi0 /= np.linalg.norm(psi0)
    theta_true = (1.0, 2.5) # rho=1.0, kappa=2.5 satisfies rho*kappa >= 2*rho
    dt = 0.05
    steps = 20

    traj = solver.solve_dynamics(theta_true, psi0, dt, steps)
    logging.info(f"Dynamics solved for {steps} steps. Final norm: {np.linalg.norm(traj[-1])}")

    ipr_val = solver.calculate_ipr(traj[-1])
    logging.info(f"Inverse Participation Ratio (Localization): {ipr_val:.4f} (1/N={1.0/solver.N:.4f})")

    H_test = solver.hamiltonian(theta_true)
    dev = solver.verify_unitarity(H_test, dt)
    logging.info(f"Unitary Propagator Deviation (||U'U - I||): {dev:.2e}")

    logging.info("3. Parameter Inference")
    # Use generated trajectory as mock data
    data_densities = np.abs(traj)**2
    res = solver.estimate_parameters(data_densities, psi0, dt, steps, [0.8, 0.3])
    logging.info(f"Estimated parameters: {res.x} (True: {theta_true})")

    sens = solver.calculate_sensitivity(res.x, data_densities, psi0, dt, steps)
    logging.info(f"Parameter Sensitivity (Gradient): {sens}")

    logging.info("4. Annihilator Check")
    valid, msg = solver.check_annihilator(res.x)
    logging.info(f"Annihilator valid: {valid}. {msg}")

    logging.info("5. Two-Level System (Lindblad) Evolution (Multi-Channel)")
    tls = TwoLevelLindbladSolver(omega=2.0, gamma=0.5, gamma_dephase=0.2)
    rho0 = np.array([[0.5, 0.5], [0.5, 0.5]]) # Superposition
    steps_tls = 50
    dt_tls = 0.1
    traj_tls = tls.solve_dynamics(rho0, dt_tls, steps_tls)
    ground_pops = [np.real(r[0,0]) for r in traj_tls]
    coherences = [np.abs(r[0,1]) for r in traj_tls]
    logging.info(f"TLS Evolution: Ground state pop: {ground_pops[-1]:.4f}, Coherence: {coherences[-1]:.4f}")

    logging.info("6. Rabi Oscillation Simulation")
    traj_rabi = tls.rabi_oscillation(rho0, dt=0.1, steps=100, drive_amp=1.0)
    excited_pops = [np.real(r[1,1]) for r in traj_rabi]
    logging.info(f"Rabi Simulation: Max excited pop: {max(excited_pops):.4f}")

if __name__ == "__main__":
    main()

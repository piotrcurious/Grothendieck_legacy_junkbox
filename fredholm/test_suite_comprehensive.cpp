#include "FredholmEngine.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace Fredholm;
void testConvergence();

void testLinearSolver() {
    std::cout << "Testing Linear Solver..." << std::endl;
    std::vector<std::vector<double>> A = {{3, 1}, {1, 2}};
    std::vector<double> B = {9, 8};
    std::vector<double> X;
    SolverStatus status = solveLinearSystem(A, B, X);
    assert(status == SolverStatus::SUCCESS);
    assert(std::abs(X[0] - 2.0) < 1e-9);
    assert(std::abs(X[1] - 3.0) < 1e-9);
    std::cout << "  Passed!" << std::endl;
}

void testVolterra() {
    std::cout << "Testing Volterra Solver (phi = 1 + integral[0,x] phi)..." << std::endl;
    auto K = [](double x, double y) { return 1.0; };
    auto f = [](double x) { return 1.0; };
    auto phi = Solver::solveVolterra(0, 1, 1.0, K, f, 100);
    // Exact solution is phi(x) = exp(x)
    double error = std::abs(phi.back() - std::exp(1.0));
    std::cout << "  exp(1) approx: " << phi.back() << ", error: " << error << std::endl;
    assert(error < 1e-3);
    std::cout << "  Passed!" << std::endl;
}

void testEigen() {
    std::cout << "Testing Eigen Solver (Symmetric 2x2)..." << std::endl;
    Fredholm::Matrix M(2, 2);
    M(0, 0) = 2; M(0, 1) = 1;
    M(1, 0) = 1; M(1, 1) = 2;
    std::vector<double> evals;
    std::vector<std::vector<double>> evecs;
    computeEigen(M, evals, evecs);
    // Eigenvalues should be 3 and 1
    std::cout << "  Evals: " << evals[0] << ", " << evals[1] << std::endl;
    assert((std::abs(evals[0] - 3.0) < 1e-7 && std::abs(evals[1] - 1.0) < 1e-7) ||
           (std::abs(evals[0] - 1.0) < 1e-7 && std::abs(evals[1] - 3.0) < 1e-7));
    std::cout << "  Passed!" << std::endl;
}

void testFredholm() {
    std::cout << "Testing Fredholm Solver..." << std::endl;
    // phi(x) = 1 + 0.5 * integral[0,1] xy phi(y) dy
    auto K = [](double x, double y) { return x * y; };
    auto f = [](double x) { return 1.0; };
    auto nodes = Solver::solveFredholm(0, 1, 0.5, K, f, 16);
    // Analytical solution: phi(x) = 1 + (3/10)x
    double val = Solver::interpolateFredholm(0.5, 0, 1, 0.5, K, f, nodes, 16);
    double expected = 1.0 + 0.3 * 0.5;
    std::cout << "  phi(0.5) approx: " << val << ", expected: " << expected << std::endl;
    assert(std::abs(val - expected) < 1e-7);
    std::cout << "  Passed!" << std::endl;
}

void testSVD() {
    std::cout << "Testing SVD (Diagonal Matrix)..." << std::endl;
    Fredholm::Matrix M(2, 2);
    M(0, 0) = 5.0; M(1, 1) = 3.0;
    Fredholm::Matrix U(2, 2), V(2, 2); std::vector<double> S;
    computeSVD(M, U, S, V);
    assert(std::abs(S[0] - 5.0) < 1e-7);
    assert(std::abs(S[1] - 3.0) < 1e-7);
    std::cout << "  Passed!" << std::endl;
}

void testEdgeCases() {
    std::cout << "Testing Edge Cases (Numerical Stability)..." << std::endl;

    // Nearly singular system
    auto K_sing = [](double x, double y) { return 1.0; }; // Rank-1 kernel
    auto f = [](double x) { return 1.0; };
    auto nodes = Solver::solveFredholm(0, 1, 1.0, K_sing, f, 16);
    std::cout << "  Singular Fredholm (lambda=1, K=1) nodes size: " << nodes.size() << " (Expected empty or unstable)" << std::endl;

    // Zero-width kernel
    auto K_zero = [](double x, double y) { return (x == y) ? 1.0 : 0.0; };
    auto nodes_zero = Solver::solveFredholm(0, 1, 0.5, K_zero, f, 16);
    assert(!nodes_zero.empty());
    std::cout << "  Delta-like Kernel Passed." << std::endl;

    // High lambda oscillation
    auto nodes_high = Solver::solveFredholm(0, 1, 100.0, [](double x, double y){ return std::sin(x*y); }, f, 16);
    std::cout << "  High Lambda (100.0) Nodes Size: " << nodes_high.size() << std::endl;

    std::cout << "  Passed!" << std::endl;
}

void testKernelFactory() {
    std::cout << "Testing KernelFactory..." << std::endl;
    for(int i=0; i<5; i++) {
        auto K = KernelFactory::create(i, 0.2);
        double val = K(0.5, 0.5);
        assert(!std::isnan(val));
        std::cout << "  Kernel " << i << " (" << KernelFactory::getName(i) << ") valid." << std::endl;
    }
    std::cout << "  Passed!" << std::endl;
}

void testConditionNumber() {
    std::cout << "Testing Condition Number..." << std::endl;
    // Rank-1 kernel should be very ill-conditioned
    auto K = [](double x, double y) { return 1.0; };
    double cond = Solver::estimateConditionNumber(0, 1, 1.0, K, 8);
    std::cout << "  Cond(Identity - K) with K=1: " << cond << std::endl;
    assert(cond > 1e10 || std::isinf(cond));
    std::cout << "  Passed!" << std::endl;
}

int main() {
    try {
        testLinearSolver();
        testVolterra();
        testEigen();
        testFredholm();
        testSVD();
        testEdgeCases();
        testKernelFactory();
        testConditionNumber(); testConvergence();
        std::cout << "\nAll Engine Tests Passed Successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

void testConvergence() {
    std::cout << "Testing Convergence (Nystrom vs Galerkin)..." << std::endl;
    auto K = [](double x, double y) { return std::exp(-(x-y)*(x-y)); };
    auto f = [](double x) { return std::sin(M_PI * x); };
    double lambda = 0.5;

    auto phi_nystrom = Fredholm::Solver::solveFredholm(0, 1, lambda, K, f, 32);
    auto coeffs_galerkin = Fredholm::Solver::solveGalerkinOptimized(0, 1, lambda, K, f, 8);

    double x = 0.5;
    double val_nystrom = Fredholm::Solver::interpolateFredholm(x, 0, 1, lambda, K, f, phi_nystrom, 32);
    double val_galerkin = 0;
    for(int i=0; i<(int)coeffs_galerkin.size(); i++) val_galerkin += coeffs_galerkin[i] * Fredholm::Solver::legendreP(i, 2.0*x - 1.0);

    double diff = std::abs(val_nystrom - val_galerkin);
    std::cout << "  Nystrom: " << val_nystrom << ", Galerkin: " << val_galerkin << ", Diff: " << diff << std::endl;
    if (diff < 2e-4) std::cout << "  Passed!" << std::endl;
    else { std::cout << "  Failed! Diff too large." << std::endl; exit(1); }
}

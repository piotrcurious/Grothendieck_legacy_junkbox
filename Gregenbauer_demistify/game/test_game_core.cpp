#include <iostream>
#include <cmath>
#include <cassert>
#include "GegenbauerCore.h"
#include "GLCanvas.h"

void test_gegenbauer_core_properties() {
    std::cout << "[Test 1] Testing Gegenbauer Normalization and Derivative Anchors..." << std::endl;
    GegenbauerCore core(4, 10, 0.2); // d=4 -> lambda=1.0

    // phi_n(1) must be 1.0 exactly
    double phi_1 = core.eval_phi(10, 1.0);
    assert(std::abs(phi_1 - 1.0) < 1e-12);

    // Parity: phi_10(-x) == phi_10(x) for even degree
    double phi_pos = core.eval_phi(10, 0.4);
    double phi_neg = core.eval_phi(10, -0.4);
    assert(std::abs(phi_pos - phi_neg) < 1e-12);

    // Derivative anchor phi_n'(1) = n(n+2*lambda) / (2*lambda+1)
    // For n=10, lambda=1.0: 10 * 12 / 3 = 40.0
    double phi_p1 = core.eval_phi_prime(10, 1.0);
    assert(std::abs(phi_p1 - 40.0) < 1e-10);
    std::cout << "  -> phi_10(1) = " << phi_1 << ", phi_10'(1) = " << phi_p1 << " [PASSED]" << std::endl;
}

void test_golub_welsch_spectrum() {
    std::cout << "[Test 2] Testing Golub-Welsch Jacobi Eigensolver..." << std::endl;
    GegenbauerCore core(5, 8, 0.8); // d=5 -> lambda=1.5
    int m = 12;
    GolubWelschResult gw = core.compute_golub_welsch(m);

    assert(gw.m == m);
    assert(std::abs(gw.weight_sum - gw.expected_mu0) < 1e-10);
    assert(gw.eigenpair_residual < 1e-8);

    // All eigenvalues must be strictly within (-1, 1)
    for (double x_k : gw.eigenvalues) {
        assert(x_k > -1.0 && x_k < 1.0);
    }
    std::cout << "  -> m=" << m << " Eigenvalues inside (-1, 1), Weight Sum Residual = "
              << std::abs(gw.weight_sum - gw.expected_mu0) << " [PASSED]" << std::endl;
}

void test_boundary_layer_and_asymptotics() {
    std::cout << "[Test 3] Testing Boundary Layers & Composite Asymptotics..." << std::endl;
    GegenbauerCore core(3, 100, 0.01); // North pole theta=0.01, N_n = 100.5
    double f_north = core.eval_north_bessel(0.01);
    double f_comp = core.eval_composite_asymptotics(0.01);
    double f_exact = core.eval_phi(100, std::cos(0.01));

    std::cout << "  -> n=100, theta=0.01: exact=" << f_exact << ", north_bessel=" << f_north
              << ", composite=" << f_comp << std::endl;
    assert(std::abs(f_comp - f_exact) < 0.03); // Asymptotic approximation O(1/N_n)
    std::cout << "  -> Composite Asymptotic Error Bound [PASSED]" << std::endl;
}

void test_residuals_and_backends() {
    std::cout << "[Test 4] Testing Multi-Backend Residual Taxonomy..." << std::endl;
    GegenbauerCore core(3, 15, 1.2);

    ResidualState res_f64 = core.compute_residuals(BackendType::FLOAT64);
    assert(res_f64.is_certified);
    assert(res_f64.r_rec < 1e-8);
    assert(res_f64.r_ode < 1e-8);

    ResidualState res_rat = core.compute_residuals(BackendType::EXACT_RATIONAL);
    assert(res_rat.truth_class == TruthClass::ALGEBRAIC_EXACT);

    ResidualState res_rns = core.compute_residuals(BackendType::MODULAR_RNS);
    assert(res_rns.truth_class == TruthClass::ARITHMETIC_EXACT);

    std::cout << "  -> FLOAT64 R_rec=" << res_f64.r_rec << ", R_ODE=" << res_f64.r_ode << " [PASSED]" << std::endl;
}

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << "      VIII-LAYER GAME CORE MATHEMATICAL TEST SUITE    " << std::endl;
    std::cout << "====================================================" << std::endl;

    test_gegenbauer_core_properties();
    test_golub_welsch_spectrum();
    test_boundary_layer_and_asymptotics();
    test_residuals_and_backends();

    std::cout << "====================================================" << std::endl;
    std::cout << "       ALL MATHEMATICAL CORE TESTS PASSED (4/4)!     " << std::endl;
    std::cout << "====================================================" << std::endl;
    return 0;
}

#include <iostream>
#include <cmath>
#include <cassert>
#include "GegenbauerCore.h"
#include "GLCanvas.h"

void test_gegenbauer_core_properties() {
    std::cout << "[Test 1] Testing Gegenbauer Normalization and Derivative Anchors..." << std::endl;
    GegenbauerCore core(4, 10, 0.2); // d=4 -> lambda=1.0

    double phi_1 = core.eval_phi(10, 1.0);
    assert(std::abs(phi_1 - 1.0) < 1e-12);

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
    assert(std::abs(f_comp - f_exact) < 0.03);
    std::cout << "  -> Composite Asymptotic Error Bound [PASSED]" << std::endl;
}

void test_4_axis_certification_and_snapshot() {
    std::cout << "[Test 4] Testing 4-Axis Certification & Representation Snapshots..." << std::endl;
    GegenbauerCore core(3, 15, 1.2);

    CertificationStatus cert_f64 = core.compute_certification(BackendType::FLOAT64);
    assert(cert_f64.numerical_valid);
    assert(cert_f64.structural_residual < 1e-8);

    CertificationStatus cert_rat = core.compute_certification(BackendType::EXACT_RATIONAL);
    assert(cert_rat.algebraic_exact);

    CertificationStatus cert_rns = core.compute_certification(BackendType::MODULAR_RNS);
    assert(cert_rns.arithmetic_exact);

    // Test Representation Snapshot & Auto Router AI
    RepresentationSnapshot snap_auto = core.get_snapshot(LayerType::LAYER_I_HARMONIC_GEOMETRY, BackendType::FLOAT64, true);
    assert(snap_auto.auto_router);
    assert(snap_auto.regime == RegimeType::INTERIOR_WKB);

    // Endpoint North Pole routing test
    GegenbauerCore core_north(3, 100, 0.01);
    RepresentationSnapshot snap_north = core_north.get_snapshot(LayerType::LAYER_I_HARMONIC_GEOMETRY, BackendType::FLOAT64, true);
    assert(snap_north.regime == RegimeType::NORTH_ENDPOINT_BESSEL);
    assert(snap_north.layer == LayerType::LAYER_V_TWO_POLE_COORDS);

    std::cout << "  -> 4-Axis Certification & Auto Router Routing [PASSED]" << std::endl;
}

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << "      VIII-LAYER GAME CORE MATHEMATICAL TEST SUITE    " << std::endl;
    std::cout << "====================================================" << std::endl;

    test_gegenbauer_core_properties();
    test_golub_welsch_spectrum();
    test_boundary_layer_and_asymptotics();
    test_4_axis_certification_and_snapshot();

    std::cout << "====================================================" << std::endl;
    std::cout << "       ALL MATHEMATICAL CORE TESTS PASSED (4/4)!     " << std::endl;
    std::cout << "====================================================" << std::endl;
    return 0;
}

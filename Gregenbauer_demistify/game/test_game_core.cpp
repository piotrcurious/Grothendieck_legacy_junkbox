#include <iostream>
#include <cmath>
#include <cassert>
#include "GegenbauerCore.h"
#include "GLCanvas.h"
#include "GameUI.h"

void test_gegenbauer_core_properties() {
    std::cout << "[Test 1] Testing Gegenbauer Normalization and Derivative Anchors..." << std::endl;
    GegenbauerCore core;

    double phi_1 = core.eval_phi(10, 1.0, 1.0); // n=10, lambda=1.0, x=1.0
    assert(std::abs(phi_1 - 1.0) < 1e-12);

    double phi_pos = core.eval_phi(10, 1.0, 0.4);
    double phi_neg = core.eval_phi(10, 1.0, -0.4);
    assert(std::abs(phi_pos - phi_neg) < 1e-12);

    // Derivative anchor phi_n'(1) = n(n+2*lambda) / (2*lambda+1)
    // For n=10, lambda=1.0: 10 * 12 / 3 = 40.0
    double phi_p1 = core.eval_phi_prime(10, 1.0, 1.0);
    assert(std::abs(phi_p1 - 40.0) < 1e-10);
    std::cout << "  -> phi_10(1) = " << phi_1 << ", phi_10'(1) = " << phi_p1 << " [PASSED]" << std::endl;
}

void test_golub_welsch_spectrum() {
    std::cout << "[Test 2] Testing Golub-Welsch Jacobi Eigensolver..." << std::endl;
    GegenbauerCore core;
    int m = 12;
    GolubWelschResult gw = core.compute_golub_welsch(m, 1.5); // lambda=1.5

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
    GegenbauerCore core;
    double f_north = core.eval_north_bessel(100, 0.5, 0.01);
    double f_comp = core.eval_composite_asymptotics(100, 0.5, 0.01);
    double f_exact = core.eval_phi(100, 0.5, std::cos(0.01));

    std::cout << "  -> n=100, theta=0.01: exact=" << f_exact << ", north_bessel=" << f_north
              << ", composite=" << f_comp << std::endl;
    assert(std::abs(f_comp - f_exact) < 0.03);
    std::cout << "  -> Composite Asymptotic Error Bound [PASSED]" << std::endl;
}

void test_4_axis_certification_and_snapshot() {
    std::cout << "[Test 4] Testing GameState Evaluation & 4-Axis Certification..." << std::endl;
    GegenbauerCore core;

    GameState state;
    state.params.d = 3;
    state.params.n = 15;
    state.params.theta = 1.2;
    state.backend = BackendType::FLOAT64;

    RepresentationSnapshot snap_f64 = core.evaluate(state);
    assert(snap_f64.cert.numerical_approx);
    assert(snap_f64.r_rec < 1e-8);
    assert(snap_f64.r_jacobi < 1e-8);

    state.backend = BackendType::EXACT_RATIONAL;
    RepresentationSnapshot snap_rat = core.evaluate(state);
    assert(snap_rat.cert.algebraic_exact);

    state.backend = BackendType::MODULAR_RNS;
    RepresentationSnapshot snap_rns = core.evaluate(state);
    assert(snap_rns.cert.arithmetic_exact);

    // Test Representation Router AI
    state.auto_router = true;
    state.params.theta = 0.01; // North Pole
    RepresentationSnapshot snap_auto = core.evaluate(state);
    assert(snap_auto.regime == RegimeType::NORTH_ENDPOINT_BESSEL);
    assert(snap_auto.effective_layer == LayerType::LAYER_V_TWO_POLE_COORDS);

    std::cout << "  -> GameState Evaluation & Router Decision [PASSED]" << std::endl;
}

void test_game_ui_cache_and_transition() {
    std::cout << "[Test 5] Testing GameUI Cache & Transition Lifecycle..." << std::endl;
    GameUI ui(800, 600);

    // 1. Initial state check
    assert(ui.current_valid);
    assert(ui.target_valid);

    // 2. Cache miss & update in get_or_evaluate_snapshot
    SnapshotKey miss_key = SnapshotKey{
        ui.state.params.d, ui.state.params.n, ui.state.params.theta,
        ui.state.params.asymptotic_K, ui.state.params.jacobi_m,
        ui.state.params.error_target_idx, ui.state.backend,
        LayerType::LAYER_IV_JACOBI_CITY, ui.state.auto_router
    };

    ui.target_valid = false;
    RepresentationSnapshot snap = ui.get_or_evaluate_snapshot(miss_key, ui.state, LayerType::LAYER_IV_JACOBI_CITY);
    assert(ui.target_valid);
    assert(ui.target_key == miss_key);
    assert(snap.params.d == ui.state.params.d);

    // 3. Cache hit
    RepresentationSnapshot cached_snap = ui.get_or_evaluate_snapshot(miss_key, ui.state, LayerType::LAYER_IV_JACOBI_CITY);
    assert(cached_snap.params.d == snap.params.d);

    // 4. commit_layer_transition fallback when target_valid is false
    ui.state.target_layer = LayerType::LAYER_III_DIFFERENTIAL_WAVE;
    ui.target_valid = false;
    ui.commit_layer_transition();
    assert(ui.current_valid);
    assert(ui.target_valid);
    assert(ui.state.current_layer == LayerType::LAYER_III_DIFFERENTIAL_WAVE);

    // 5. Direct T=1 set_transition commit
    ui.state.target_layer = LayerType::LAYER_VI_COMPOSITE_ASYMPTOTICS;
    ui.target_valid = false;
    ui.set_transition(1.0);
    assert(ui.current_valid);
    assert(ui.target_valid);
    assert(ui.state.current_layer == LayerType::LAYER_VI_COMPOSITE_ASYMPTOTICS);
    assert(ui.state.transition == 1.0);

    std::cout << "  -> GameUI Cache & Transition Lifecycle [PASSED]" << std::endl;
}

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << "      VIII-LAYER GAME CORE MATHEMATICAL TEST SUITE    " << std::endl;
    std::cout << "====================================================" << std::endl;

    test_gegenbauer_core_properties();
    test_golub_welsch_spectrum();
    test_boundary_layer_and_asymptotics();
    test_4_axis_certification_and_snapshot();
    test_game_ui_cache_and_transition();

    std::cout << "====================================================" << std::endl;
    std::cout << "       ALL MATHEMATICAL CORE TESTS PASSED (5/5)!     " << std::endl;
    std::cout << "====================================================" << std::endl;
    return 0;
}

#include <iostream>
#include <cstring>
#include <FL/Fl.H>
#include "GegenbauerCore.h"
#include "GameUI.h"

int run_headless_tests() {
    std::cout << "=== Running EIGHTH LAYER Headless Verification Tests ===" << std::endl;

    GegenbauerCore core;

    // Test 1: d=3, n=0 -> phi_0(x) = 1.0
    double p0 = core.eval_phi(0, 0.5, 0.5);
    std::cout << "Test 1 [Degree 0 Constant]: phi_0(0.5) = " << p0 << std::endl;
    if (std::abs(p0 - 1.0) > 1e-12) {
        std::cerr << "FAILED Test 1" << std::endl;
        return 1;
    }

    // Test 2: d=3 (lambda=0.5, Legendre P_n), n=1 -> phi_1(x) = x
    double p1 = core.eval_phi(1, 0.5, 0.7);
    std::cout << "Test 2 [Legendre P_1]: phi_1(0.7) = " << p1 << " (expected 0.7)" << std::endl;
    if (std::abs(p1 - 0.7) > 1e-12) {
        std::cerr << "FAILED Test 2" << std::endl;
        return 1;
    }

    // Test 3: Jacobi subdiagonal alpha_0 for lambda=0.5
    double alpha0 = GegenbauerCore::get_jacobi_alpha(0, 0.5);
    std::cout << "Test 3 [Jacobi Alpha_0]: alpha_0 = " << alpha0 << std::endl;
    if (std::abs(alpha0 - 1.0 / std::sqrt(3.0)) > 1e-10) {
        std::cerr << "FAILED Test 3" << std::endl;
        return 1;
    }

    // Test 4: Golub-Welsch Eigensolver for m=5
    GolubWelschResult gw = core.compute_golub_welsch(5, 0.5);
    std::cout << "Test 4 [Golub-Welsch m=5]: Weight sum = " << gw.weight_sum
              << " (expected mu0 = " << gw.expected_mu0 << ")" << std::endl;
    std::cout << "        Max Eigenpair Residual R_J = " << gw.eigenpair_residual << std::endl;
    if (std::abs(gw.weight_sum - gw.expected_mu0) > 1e-10 || gw.eigenpair_residual > 1e-8) {
        std::cerr << "FAILED Test 4" << std::endl;
        return 1;
    }

    // Test 5: Representation Snapshot & 4-Axis Certification
    GameState state;
    state.params.d = 3;
    state.params.n = 5;
    state.params.theta = 0.5;
    state.backend = BackendType::FLOAT64;
    state.auto_router = true;

    RepresentationSnapshot snap = core.evaluate(state);
    std::cout << "Test 5 [Snapshot & Certification]: N = " << snap.N
              << ", Regime = " << snap.regime_name
              << ", ALG=" << snap.cert.algebraic_exact
              << ", NUM=" << snap.cert.numerical_approx
              << ", R_J=" << snap.r_jacobi << std::endl;
    if (!snap.cert.numerical_approx) {
        std::cerr << "FAILED Test 5" << std::endl;
        return 1;
    }

    std::cout << "=== All Headless Verification Tests Passed Successfully! ===" << std::endl;
    return 0;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--headless-test") == 0) {
            return run_headless_tests();
        }
    }

    Fl::scheme("gtk+");
    GameUI ui(1100, 820);
    ui.show();
    return Fl::run();
}

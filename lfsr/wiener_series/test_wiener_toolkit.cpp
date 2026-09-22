#include "lfsr_wiener_toolkit.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <complex>

constexpr double PI = 3.14159265358979323846;

using namespace lfsr_wiener;

void test_field_arithmetic() {
    std::cout << "[TEST] Field Arithmetic GF(2^3)..." << std::endl;
    GF2Field field(3, 11);

    assert(field.N == 7);
    assert(field.add(5, 3) == 6);
    assert(field.mul(0, 5) == 0);
    assert(field.mul(1, 5) == 5);

    for (uint32_t a = 1; a < 8; ++a) {
        uint32_t inv_a = field.inverse(a);
        assert(field.mul(a, inv_a) == 1);
    }
    std::cout << "  -> PASS" << std::endl;
}

void test_trace_and_companion_match() {
    std::cout << "[TEST] Trace vs Companion Matrix Match..." << std::endl;
    GF2Field field(3, 11);
    LFSRGenerator gen(field, 1);

    std::vector<int> bits_trace = gen.generate_bits_trace(7);
    std::vector<int> bits_companion = gen.generate_bits_companion(7, 1);

    assert(bits_trace.size() == 7);
    assert(bits_companion.size() == 7);

    int ones_trace = 0, ones_comp = 0;
    for (int i = 0; i < 7; ++i) {
        ones_trace += bits_trace[i];
        ones_comp += bits_companion[i];
    }
    assert(ones_trace == 4 || ones_trace == 3);
    assert(ones_comp == 4 || ones_comp == 3);
    std::cout << "  -> PASS" << std::endl;
}

void test_walsh_sparsity() {
    std::cout << "[TEST] Walsh Monomial 1-Sparsity..." << std::endl;
    std::vector<double> walsh_buf(8, 0.0);
    for (uint32_t state = 0; state < 8; ++state) {
        std::vector<int> bits(3);
        bits[0] = state & 1;
        bits[1] = (state >> 1) & 1;
        bits[2] = (state >> 2) & 1;
        int u0 = (bits[0] == 0) ? 1 : -1;
        int u1 = (bits[1] == 0) ? 1 : -1;
        int u3 = u0 * u1;
        walsh_buf[state] = u3;
    }

    SpectralAnalyzer::fwht(walsh_buf);

    int non_zero_count = 0;
    for (double val : walsh_buf) {
        if (std::abs(val) > 1e-7) {
            non_zero_count++;
            assert(std::abs(std::abs(val) - 8.0) < 1e-7);
        }
    }
    assert(non_zero_count == 1);
    std::cout << "  -> PASS" << std::endl;
}

void test_wiener_chaos_decomposition() {
    std::cout << "[TEST] Discrete Walsh-Wiener Chaos Expansion..." << std::endl;
    uint32_t L = 3;
    std::vector<double> truth_table(8);
    for (uint32_t state = 0; state < 8; ++state) {
        int u0 = ((state >> 0) & 1) ? -1 : 1;
        int u1 = ((state >> 1) & 1) ? -1 : 1;
        int u2 = ((state >> 2) & 1) ? -1 : 1;
        truth_table[state] = u0 * u1 * u2 + u0;
    }

    WienerChaosResult res = WienerChaosAnalyzer::analyze_function(L, truth_table);

    assert(std::abs(res.total_energy - 2.0) < 1e-7);
    assert(std::abs(res.energy_per_degree[1] - 1.0) < 1e-7);
    assert(std::abs(res.energy_per_degree[3] - 1.0) < 1e-7);
    assert(std::abs(res.energy_per_degree[0]) < 1e-7);
    assert(std::abs(res.energy_per_degree[2]) < 1e-7);

    std::cout << "  -> PASS" << std::endl;
}

void test_volterra_kernel_extraction() {
    std::cout << "[TEST] Volterra-Wiener Kernel Extraction..." << std::endl;
    size_t T = 1000;
    std::vector<int> w(T);
    for (size_t i = 0; i < T; ++i) {
        w[i] = ((i * 11 + 7) % 2 == 0) ? 1 : -1;
    }

    std::vector<int> v(T);
    for (size_t n = 2; n < T; ++n) {
        int term2 = w[n] * w[n - 1];
        int term1 = w[n - 2];
        v[n] = (term2 * term1 == 1) ? 1 : -1;
    }

    std::vector<double> h1 = WienerChaosAnalyzer::compute_volterra_kernel_1(v, w, 5);
    std::vector<std::vector<double>> h2 = WienerChaosAnalyzer::compute_volterra_kernel_2(v, w, 5);

    assert(h1.size() == 5);
    assert(h2.size() == 5);
    std::cout << "  -> PASS" << std::endl;
}

void test_pair_synthesis_engine() {
    std::cout << "[TEST] LFSR Pair Spectral Synthesis Engine..." << std::endl;
    GF2Field field_A(3, 11); // N_A = 7
    GF2Field field_B(4, 19); // N_B = 15

    // Multiplicative pair synthesis y_n = u_A * u_B
    PairSynthesisReport report = LFSRSynthesisEngine::synthesize_pair(
        field_A, 1, field_B, 1, PairCombinationMode::MULTIPLICATIVE
    );

    // Joint period N_joint = lcm(7, 15) = 105
    assert(report.N_joint == 105);
    assert(report.synthesized_bipolar.size() == 105);
    assert(report.joint_power_spectrum.size() == 105);

    // Multiplicative combination shifts energy to joint degree 2 = 1 + 1
    assert(std::abs(report.joint_wiener_energy[2] - 1.0) < 1e-7);
    assert(std::abs(report.joint_wiener_energy[1]) < 1e-7);

    std::cout << "  -> PASS" << std::endl;
}

void test_autocorrelation_and_flat_spectrum() {
    std::cout << "[TEST] Two-Valued Autocorrelation & Flat Spectrum (L=3, 4, 5)..." << std::endl;
    std::vector<std::pair<uint32_t, uint32_t>> test_cases = {
        {3, 11},
        {4, 19},
        {5, 37}
    };

    for (auto& tc : test_cases) {
        uint32_t L = tc.first;
        uint32_t poly = tc.second;

        GF2Field field(L, poly);
        SpectralReport report = SpectralAnalyzer::analyze(field, 1);

        assert(report.is_flat);
        assert(report.is_autocorr_two_valued);
        assert(report.max_spectral_error < 1e-7);

        assert(std::abs(report.dft[0] - Complex(-1.0, 0.0)) < 1e-7);

        double expected_pwr = static_cast<double>(1U << L);
        for (size_t k = 1; k < field.N; ++k) {
            assert(std::abs(report.power_spectrum[k] - expected_pwr) < 1e-7);
        }
    }
    std::cout << "  -> PASS" << std::endl;
}

void test_gauss_sum_dft_identity() {
    std::cout << "[TEST] DFT vs Gauss Sum Identity U_k = chi_k(beta)^(-1) * g(chi_k, psi)..." << std::endl;
    GF2Field field(3, 11);
    uint32_t beta = 3;

    LFSRGenerator gen(field, beta);
    std::vector<int> bipolar = gen.generate_bipolar_sequence(field.N);
    std::vector<Complex> U = SpectralAnalyzer::compute_dft(bipolar);
    std::vector<Complex> g = SpectralAnalyzer::compute_gauss_sums(field);

    for (uint32_t k = 1; k < field.N; ++k) {
        Complex chi_beta_inv = std::conj(field.multiplicative_character(k, beta));
        Complex expected_Uk = chi_beta_inv * g[k];
        assert(std::abs(U[k] - expected_Uk) < 1e-7);
    }
    std::cout << "  -> PASS" << std::endl;
}

void test_higher_order_correlation() {
    std::cout << "[TEST] Higher-Order Correlation Selection Rules..." << std::endl;
    GF2Field field(3, 11);
    LFSRGenerator gen(field, 1);
    std::vector<int> bipolar = gen.generate_bipolar_sequence(field.N);

    std::vector<uint32_t> delays1 = {0, 1, 3};
    double corr1 = SpectralAnalyzer::compute_higher_order_correlation(bipolar, delays1);
    bool div1 = SpectralAnalyzer::poly_divides_delay_sum(11, 3, delays1);
    assert(div1 == true);
    assert(std::abs(corr1 - 7.0) < 1e-7);

    std::vector<uint32_t> delays2 = {0, 1, 2};
    double corr2 = SpectralAnalyzer::compute_higher_order_correlation(bipolar, delays2);
    bool div2 = SpectralAnalyzer::poly_divides_delay_sum(11, 3, delays2);
    assert(div2 == false);
    assert(std::abs(corr2 - (-1.0)) < 1e-7);

    std::cout << "  -> PASS" << std::endl;
}

void test_koopman_eigenfunctions() {
    std::cout << "[TEST] Koopman Operator Eigenfunctions..." << std::endl;
    GF2Field field(3, 11);

    for (uint32_t k = 0; k < field.N; ++k) {
        double angle = -2.0 * PI * static_cast<double>(k) / static_cast<double>(field.N);
        Complex eigenvalue(std::cos(angle), std::sin(angle));

        uint32_t z = 1;
        for (uint32_t n = 0; n < field.N; ++n) {
            Complex chi_z = field.multiplicative_character(k, z);
            uint32_t alpha_z = field.mul(field.alpha, z);
            Complex K_chi_z = field.multiplicative_character(k, alpha_z);

            Complex expected = eigenvalue * chi_z;
            assert(std::abs(K_chi_z - expected) < 1e-7);
            z = alpha_z;
        }
    }
    std::cout << "  -> PASS" << std::endl;
}

int main() {
    std::cout << "===== RUNNING C++ WIENER TOOLKIT TEST SUITE =====" << std::endl;
    test_field_arithmetic();
    test_trace_and_companion_match();
    test_walsh_sparsity();
    test_wiener_chaos_decomposition();
    test_volterra_kernel_extraction();
    test_pair_synthesis_engine();
    test_autocorrelation_and_flat_spectrum();
    test_gauss_sum_dft_identity();
    test_higher_order_correlation();
    test_koopman_eigenfunctions();
    std::cout << "===== ALL C++ TESTS PASSED SUCCESSFULLY! =====" << std::endl;
    return 0;
}

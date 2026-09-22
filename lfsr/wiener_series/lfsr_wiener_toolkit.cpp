#include "lfsr_wiener_toolkit.hpp"
#include <iostream>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

namespace lfsr_wiener {

constexpr double PI = 3.14159265358979323846;

GF2Field::GF2Field(uint32_t degree, uint32_t primitive_poly, uint32_t prim_elem)
    : L(degree), poly(primitive_poly), alpha(prim_elem) {
    N = (1U << L) - 1;
}

uint32_t GF2Field::mul(uint32_t a, uint32_t b) const {
    if (a == 0 || b == 0) return 0;
    uint32_t result = 0;
    for (uint32_t i = 0; i < L; ++i) {
        if ((b >> i) & 1) {
            result ^= a;
        }
        bool msb = (a >> (L - 1)) & 1;
        a = (a << 1) & N;
        if (msb) {
            a ^= (poly & N);
        }
    }
    return result;
}

uint32_t GF2Field::power(uint32_t base, uint64_t exp) const {
    if (base == 0) return 0;
    exp %= N;
    uint32_t res = 1;
    uint32_t cur = base;
    while (exp > 0) {
        if (exp & 1) res = mul(res, cur);
        cur = mul(cur, cur);
        exp >>= 1;
    }
    return res;
}

uint32_t GF2Field::inverse(uint32_t a) const {
    if (a == 0) throw std::invalid_argument("Cannot invert 0 in GF(2^L)");
    return power(a, N - 1);
}

uint32_t GF2Field::trace(uint32_t z) const {
    uint32_t tr = 0;
    uint32_t cur = z;
    for (uint32_t i = 0; i < L; ++i) {
        tr ^= (cur & 1);
        cur = mul(cur, cur);
    }
    return tr & 1;
}

int GF2Field::additive_character(uint32_t z) const {
    return (trace(z) == 0) ? 1 : -1;
}

Complex GF2Field::multiplicative_character(uint32_t k, uint32_t z) const {
    if (z == 0) return 0.0;
    uint32_t n = 0;
    uint32_t cur = 1;
    bool found = false;
    for (uint32_t i = 0; i < N; ++i) {
        if (cur == z) {
            n = i;
            found = true;
            break;
        }
        cur = mul(cur, alpha);
    }
    if (!found) throw std::runtime_error("Element not in GF(2^L)*");
    double angle = -2.0 * PI * static_cast<double>((k % N) * n) / static_cast<double>(N);
    return Complex(std::cos(angle), std::sin(angle));
}

std::vector<int> GF2Field::element_to_vector(uint32_t z) const {
    std::vector<int> vec(L);
    for (uint32_t i = 0; i < L; ++i) {
        vec[i] = (z >> i) & 1;
    }
    return vec;
}

uint32_t GF2Field::vector_to_element(const std::vector<int>& vec) const {
    uint32_t z = 0;
    for (uint32_t i = 0; i < L && i < vec.size(); ++i) {
        if (vec[i] & 1) z |= (1U << i);
    }
    return z;
}

std::vector<int> LFSRGenerator::generate_bits_companion(size_t num_bits, uint32_t initial_state) const {
    std::vector<int> bits(num_bits);
    uint32_t state = initial_state & field.N;
    if (state == 0) state = 1;

    for (size_t i = 0; i < num_bits; ++i) {
        bits[i] = state & 1;
        bool feedback = false;
        for (uint32_t j = 0; j < field.L; ++j) {
            if ((field.poly >> j) & 1) {
                feedback ^= ((state >> j) & 1);
            }
        }
        state = (state >> 1) | (static_cast<uint32_t>(feedback) << (field.L - 1));
    }
    return bits;
}

std::vector<int> LFSRGenerator::generate_bits_trace(size_t num_bits) const {
    std::vector<int> bits(num_bits);
    uint32_t z = beta;
    for (size_t i = 0; i < num_bits; ++i) {
        bits[i] = field.trace(z);
        z = field.mul(z, field.alpha);
    }
    return bits;
}

std::vector<int> LFSRGenerator::generate_bipolar_sequence(size_t num_bits) const {
    std::vector<int> bits = generate_bits_trace(num_bits);
    std::vector<int> bipolar(num_bits);
    for (size_t i = 0; i < num_bits; ++i) {
        bipolar[i] = (bits[i] == 0) ? 1 : -1;
    }
    return bipolar;
}

WienerChaosResult WienerChaosAnalyzer::analyze_function(uint32_t L, const std::vector<double>& truth_table) {
    size_t num_states = 1U << L;
    if (truth_table.size() != num_states) {
        throw std::invalid_argument("Truth table size must match 2^L");
    }

    std::vector<double> buf = truth_table;
    SpectralAnalyzer::fwht(buf);

    WienerChaosResult result;
    result.L = L;
    result.walsh_coefficients.resize(num_states);
    result.energy_per_degree.assign(L + 1, 0.0);
    result.total_energy = 0.0;

    double max_lin_coeff = 0.0;

    for (size_t mask = 0; mask < num_states; ++mask) {
        double coeff = buf[mask] / static_cast<double>(num_states);
        result.walsh_coefficients[mask] = coeff;

        double energy = coeff * coeff;
        result.total_energy += energy;

        uint32_t degree = 0;
        for (uint32_t j = 0; j < L; ++j) {
            if ((mask >> j) & 1) degree++;
        }
        result.energy_per_degree[degree] += energy;

        if (degree == 1) {
            if (std::abs(coeff) > max_lin_coeff) {
                max_lin_coeff = std::abs(coeff);
            }
        }
    }

    result.nonlinearity = (1.0 - max_lin_coeff) / 2.0;
    return result;
}

double WienerChaosAnalyzer::compute_volterra_kernel_0(const std::vector<int>& v) {
    if (v.empty()) return 0.0;
    double sum = 0.0;
    for (int val : v) sum += val;
    return sum / static_cast<double>(v.size());
}

std::vector<double> WienerChaosAnalyzer::compute_volterra_kernel_1(const std::vector<int>& v, const std::vector<int>& w, size_t max_lag) {
    size_t T = std::min(v.size(), w.size());
    std::vector<double> h1(max_lag, 0.0);
    for (size_t k = 0; k < max_lag; ++k) {
        double sum = 0.0;
        size_t count = 0;
        for (size_t n = k; n < T; ++n) {
            sum += static_cast<double>(v[n] * w[n - k]);
            count++;
        }
        h1[k] = (count > 0) ? (sum / static_cast<double>(count)) : 0.0;
    }
    return h1;
}

std::vector<std::vector<double>> WienerChaosAnalyzer::compute_volterra_kernel_2(const std::vector<int>& v, const std::vector<int>& w, size_t max_lag) {
    size_t T = std::min(v.size(), w.size());
    std::vector<std::vector<double>> h2(max_lag, std::vector<double>(max_lag, 0.0));
    for (size_t k1 = 0; k1 < max_lag; ++k1) {
        for (size_t k2 = k1 + 1; k2 < max_lag; ++k2) {
            double sum = 0.0;
            size_t count = 0;
            size_t start_n = std::max(k1, k2);
            for (size_t n = start_n; n < T; ++n) {
                sum += static_cast<double>(v[n] * w[n - k1] * w[n - k2]);
                count++;
            }
            double val = (count > 0) ? (sum / static_cast<double>(count)) : 0.0;
            h2[k1][k2] = val;
            h2[k2][k1] = val;
        }
    }
    return h2;
}

uint64_t LFSRSynthesisEngine::gcd(uint64_t a, uint64_t b) {
    while (b > 0) {
        uint64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

uint64_t LFSRSynthesisEngine::lcm(uint64_t a, uint64_t b) {
    if (a == 0 || b == 0) return 0;
    return (a / gcd(a, b)) * b;
}

PairSynthesisReport LFSRSynthesisEngine::synthesize_pair(
    const GF2Field& field_A, uint32_t beta_A,
    const GF2Field& field_B, uint32_t beta_B,
    PairCombinationMode mode,
    double weight_A, double weight_B
) {
    PairSynthesisReport report;
    report.L_A = field_A.L;
    report.L_B = field_B.L;
    report.poly_A = field_A.poly;
    report.poly_B = field_B.poly;
    report.beta_A = beta_A;
    report.beta_B = beta_B;
    report.N_A = field_A.N;
    report.N_B = field_B.N;
    report.N_joint = static_cast<uint32_t>(lcm(field_A.N, field_B.N));
    report.mode = mode;

    LFSRGenerator gen_A(field_A, beta_A);
    LFSRGenerator gen_B(field_B, beta_B);

    std::vector<int> u_A = gen_A.generate_bipolar_sequence(report.N_joint);
    std::vector<int> u_B = gen_B.generate_bipolar_sequence(report.N_joint);

    report.synthesized_bipolar.resize(report.N_joint);
    for (size_t n = 0; n < report.N_joint; ++n) {
        if (mode == PairCombinationMode::MULTIPLICATIVE) {
            report.synthesized_bipolar[n] = u_A[n] * u_B[n];
        } else if (mode == PairCombinationMode::ADDITIVE) {
            report.synthesized_bipolar[n] = static_cast<int>(std::round(weight_A * u_A[n] + weight_B * u_B[n]));
        } else { // MULTIPLEXED
            report.synthesized_bipolar[n] = (n % 2 == 0) ? u_A[n] : u_B[n];
        }
    }

    report.joint_dft = SpectralAnalyzer::compute_dft(report.synthesized_bipolar);
    report.joint_power_spectrum.resize(report.N_joint);
    for (size_t k = 0; k < report.N_joint; ++k) {
        double mag = std::abs(report.joint_dft[k]);
        report.joint_power_spectrum[k] = mag * mag;
    }

    report.joint_autocorrelation = SpectralAnalyzer::compute_autocorrelation(report.synthesized_bipolar);

    // Compute joint Wiener chaos energy distribution
    size_t joint_L = field_A.L + field_B.L;
    size_t joint_num_states = 1U << joint_L;
    std::vector<double> joint_truth_table(joint_num_states);
    for (size_t state = 0; state < joint_num_states; ++state) {
        uint32_t state_A = state & field_A.N;
        uint32_t state_B = (state >> field_A.L) & field_B.N;

        int val_A = field_A.additive_character(field_A.mul(beta_A, state_A));
        int val_B = field_B.additive_character(field_B.mul(beta_B, state_B));

        if (mode == PairCombinationMode::MULTIPLICATIVE) {
            joint_truth_table[state] = val_A * val_B;
        } else if (mode == PairCombinationMode::ADDITIVE) {
            joint_truth_table[state] = weight_A * val_A + weight_B * val_B;
        } else {
            joint_truth_table[state] = (state % 2 == 0) ? val_A : val_B;
        }
    }

    WienerChaosResult chaos_res = WienerChaosAnalyzer::analyze_function(joint_L, joint_truth_table);
    report.joint_wiener_energy = chaos_res.energy_per_degree;

    return report;
}

std::string LFSRSynthesisEngine::export_pair_json(const PairSynthesisReport& report) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\n";
    ss << "  \"L_A\": " << report.L_A << ",\n";
    ss << "  \"L_B\": " << report.L_B << ",\n";
    ss << "  \"poly_A\": " << report.poly_A << ",\n";
    ss << "  \"poly_B\": " << report.poly_B << ",\n";
    ss << "  \"N_A\": " << report.N_A << ",\n";
    ss << "  \"N_B\": " << report.N_B << ",\n";
    ss << "  \"N_joint\": " << report.N_joint << ",\n";
    ss << "  \"mode\": " << (report.mode == PairCombinationMode::MULTIPLICATIVE ? "\"MULTIPLICATIVE\"" : (report.mode == PairCombinationMode::ADDITIVE ? "\"ADDITIVE\"" : "\"MULTIPLEXED\"")) << ",\n";

    ss << "  \"synthesized_bipolar\": [";
    for (size_t i = 0; i < report.synthesized_bipolar.size(); ++i) {
        ss << report.synthesized_bipolar[i] << (i + 1 < report.synthesized_bipolar.size() ? ", " : "");
    }
    ss << "],\n";

    ss << "  \"joint_power_spectrum\": [";
    for (size_t i = 0; i < report.joint_power_spectrum.size(); ++i) {
        ss << report.joint_power_spectrum[i] << (i + 1 < report.joint_power_spectrum.size() ? ", " : "");
    }
    ss << "],\n";

    ss << "  \"joint_autocorrelation\": [";
    for (size_t i = 0; i < report.joint_autocorrelation.size(); ++i) {
        ss << report.joint_autocorrelation[i] << (i + 1 < report.joint_autocorrelation.size() ? ", " : "");
    }
    ss << "],\n";

    ss << "  \"joint_wiener_energy\": [";
    for (size_t i = 0; i < report.joint_wiener_energy.size(); ++i) {
        ss << report.joint_wiener_energy[i] << (i + 1 < report.joint_wiener_energy.size() ? ", " : "");
    }
    ss << "]\n";

    ss << "}\n";
    return ss.str();
}

void SpectralAnalyzer::fwht(std::vector<double>& a) {
    size_t n = a.size();
    for (size_t len = 1; 2 * len <= n; len <<= 1) {
        for (size_t i = 0; i < n; i += 2 * len) {
            for (size_t j = 0; j < len; ++j) {
                double u = a[i + j];
                double v = a[i + len + j];
                a[i + j] = u + v;
                a[i + len + j] = u - v;
            }
        }
    }
}

std::vector<Complex> SpectralAnalyzer::compute_dft(const std::vector<int>& u) {
    size_t N = u.size();
    std::vector<Complex> U(N);
    for (size_t k = 0; k < N; ++k) {
        Complex sum = 0.0;
        for (size_t n = 0; n < N; ++n) {
            double angle = -2.0 * PI * static_cast<double>(k * n) / static_cast<double>(N);
            sum += static_cast<double>(u[n]) * Complex(std::cos(angle), std::sin(angle));
        }
        U[k] = sum;
    }
    return U;
}

std::vector<Complex> SpectralAnalyzer::compute_gauss_sums(const GF2Field& field) {
    uint32_t N = field.N;
    std::vector<Complex> g(N);

    for (uint32_t k = 0; k < N; ++k) {
        Complex sum = 0.0;
        uint32_t z = 1;
        for (uint32_t n = 0; n < N; ++n) {
            int psi_z = field.additive_character(z);
            Complex chi_z = field.multiplicative_character(k, z);
            sum += chi_z * static_cast<double>(psi_z);
            z = field.mul(z, field.alpha);
        }
        g[k] = sum;
    }
    return g;
}

std::vector<double> SpectralAnalyzer::compute_autocorrelation(const std::vector<int>& u) {
    size_t N = u.size();
    std::vector<double> R(N);
    for (size_t d = 0; d < N; ++d) {
        double sum = 0.0;
        for (size_t n = 0; n < N; ++n) {
            sum += u[n] * u[(n + d) % N];
        }
        R[d] = sum;
    }
    return R;
}

double SpectralAnalyzer::compute_higher_order_correlation(const std::vector<int>& u, const std::vector<uint32_t>& delays) {
    size_t N = u.size();
    double sum = 0.0;
    for (size_t n = 0; n < N; ++n) {
        double prod = 1.0;
        for (uint32_t d : delays) {
            prod *= u[(n + d) % N];
        }
        sum += prod;
    }
    return sum;
}

bool SpectralAnalyzer::poly_divides_delay_sum(uint32_t poly, uint32_t L, const std::vector<uint32_t>& delays) {
    uint32_t max_d = 0;
    for (uint32_t d : delays) {
        if (d > max_d) max_d = d;
    }
    std::vector<int> Q(max_d + 1, 0);
    for (uint32_t d : delays) {
        Q[d] ^= 1;
    }

    uint32_t p_deg = L;
    std::vector<int> P(p_deg + 1, 0);
    P[p_deg] = 1;
    for (uint32_t j = 0; j < L; ++j) {
        P[j] = (poly >> j) & 1;
    }

    int cur_deg = max_d;
    while (cur_deg >= static_cast<int>(p_deg)) {
        if (Q[cur_deg]) {
            for (size_t j = 0; j <= p_deg; ++j) {
                Q[cur_deg - p_deg + j] ^= P[j];
            }
        }
        cur_deg--;
    }

    for (int coef : Q) {
        if (coef != 0) return false;
    }
    return true;
}

SpectralReport SpectralAnalyzer::analyze(const GF2Field& field, uint32_t beta) {
    SpectralReport report;
    report.L = field.L;
    report.N = field.N;
    report.poly = field.poly;
    report.beta = beta;

    LFSRGenerator gen(field, beta);
    report.bits = gen.generate_bits_trace(field.N);
    report.bipolar = gen.generate_bipolar_sequence(field.N);

    report.dft = compute_dft(report.bipolar);
    report.power_spectrum.resize(field.N);
    double target_magnitude = std::sqrt(static_cast<double>(field.N + 1));

    report.is_flat = true;
    report.max_spectral_error = 0.0;

    for (size_t k = 0; k < field.N; ++k) {
        double mag = std::abs(report.dft[k]);
        report.power_spectrum[k] = mag * mag;
        if (k > 0) {
            double err = std::abs(mag - target_magnitude);
            if (err > report.max_spectral_error) {
                report.max_spectral_error = err;
            }
            if (err > 1e-7) {
                report.is_flat = false;
            }
        }
    }

    report.gauss_sums = compute_gauss_sums(field);
    report.autocorrelation = compute_autocorrelation(report.bipolar);

    report.is_autocorr_two_valued = true;
    for (size_t d = 0; d < field.N; ++d) {
        double expected = (d == 0) ? static_cast<double>(field.N) : -1.0;
        if (std::abs(report.autocorrelation[d] - expected) > 1e-7) {
            report.is_autocorr_two_valued = false;
        }
    }

    size_t num_states = 1U << field.L;
    std::vector<double> truth_table(num_states);
    for (size_t state = 0; state < num_states; ++state) {
        truth_table[state] = field.additive_character(field.mul(beta, state));
    }
    WienerChaosResult chaos_res = WienerChaosAnalyzer::analyze_function(field.L, truth_table);
    report.wiener_energy_distribution = chaos_res.energy_per_degree;

    return report;
}

std::string SpectralAnalyzer::export_json(const SpectralReport& report) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\n";
    ss << "  \"L\": " << report.L << ",\n";
    ss << "  \"N\": " << report.N << ",\n";
    ss << "  \"poly\": " << report.poly << ",\n";
    ss << "  \"beta\": " << report.beta << ",\n";
    ss << "  \"is_flat\": " << (report.is_flat ? "true" : "false") << ",\n";
    ss << "  \"is_autocorr_two_valued\": " << (report.is_autocorr_two_valued ? "true" : "false") << ",\n";
    ss << "  \"max_spectral_error\": " << report.max_spectral_error << ",\n";

    ss << "  \"bipolar\": [";
    for (size_t i = 0; i < report.bipolar.size(); ++i) {
        ss << report.bipolar[i] << (i + 1 < report.bipolar.size() ? ", " : "");
    }
    ss << "],\n";

    ss << "  \"power_spectrum\": [";
    for (size_t i = 0; i < report.power_spectrum.size(); ++i) {
        ss << report.power_spectrum[i] << (i + 1 < report.power_spectrum.size() ? ", " : "");
    }
    ss << "],\n";

    ss << "  \"autocorrelation\": [";
    for (size_t i = 0; i < report.autocorrelation.size(); ++i) {
        ss << report.autocorrelation[i] << (i + 1 < report.autocorrelation.size() ? ", " : "");
    }
    ss << "],\n";

    ss << "  \"wiener_energy_distribution\": [";
    for (size_t i = 0; i < report.wiener_energy_distribution.size(); ++i) {
        ss << report.wiener_energy_distribution[i] << (i + 1 < report.wiener_energy_distribution.size() ? ", " : "");
    }
    ss << "]\n";

    ss << "}\n";
    return ss.str();
}

} // namespace lfsr_wiener

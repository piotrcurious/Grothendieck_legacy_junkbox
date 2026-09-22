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
        tr ^= (cur & 1); // Tr(z) in F2 is sum of LSBs of Galois field elements or parity
        cur = mul(cur, cur);
    }
    // Alternatively, for canonical polynomial basis representation, Tr(z) is linear.
    // To ensure exact match with F2 trace: sum_{i=0}^{L-1} z^{2^i}
    // In characteristic 2, the result is in F2 (either 0 or 1).
    return tr & 1;
}

int GF2Field::additive_character(uint32_t z) const {
    return (trace(z) == 0) ? 1 : -1;
}

Complex GF2Field::multiplicative_character(uint32_t k, uint32_t z) const {
    if (z == 0) return 0.0;
    // Find logarithm of z base alpha: z = alpha^n
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
        // Companion matrix shift according to feedback poly
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

    // g(chi_k, psi) = sum_{z in GF(2^L)*} chi_k(z) * psi(z)
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
    // Q(t) = sum_{d in delays} t^d in F2[t]
    // Polynomial division over F2
    uint32_t max_d = 0;
    for (uint32_t d : delays) {
        if (d > max_d) max_d = d;
    }
    std::vector<int> Q(max_d + 1, 0);
    for (uint32_t d : delays) {
        Q[d] ^= 1;
    }

    // p(t) coefficients
    uint32_t p_deg = L; // degree of primitive poly is L
    std::vector<int> P(p_deg + 1, 0);
    P[p_deg] = 1;
    for (uint32_t j = 0; j < L; ++j) {
        P[j] = (poly >> j) & 1;
    }

    // Long division Q(t) by P(t) over F2
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
    double target_magnitude = std::sqrt(static_cast<double>(field.N + 1)); // 2^(L/2)

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
    ss << "]\n";

    ss << "}\n";
    return ss.str();
}

} // namespace lfsr_wiener

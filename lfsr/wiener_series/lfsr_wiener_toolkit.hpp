#ifndef LFSR_WIENER_TOOLKIT_HPP
#define LFSR_WIENER_TOOLKIT_HPP

#include <vector>
#include <complex>
#include <string>
#include <cstdint>
#include <map>

namespace lfsr_wiener {

using Complex = std::complex<double>;

// Represents a finite field GF(2^L) constructed via irreducible/primitive polynomial poly
class GF2Field {
public:
    uint32_t L;          // Field degree
    uint32_t poly;       // Primitive polynomial bitmask (e.g., t^3 + t + 1 -> 0b1011 = 11)
    uint32_t N;          // Multiplicative group order 2^L - 1
    uint32_t alpha;      // Primitive element (default: 2 representing 't')

    GF2Field(uint32_t degree, uint32_t primitive_poly, uint32_t prim_elem = 2);

    uint32_t add(uint32_t a, uint32_t b) const { return a ^ b; }
    uint32_t mul(uint32_t a, uint32_t b) const;
    uint32_t power(uint32_t base, uint64_t exp) const;
    uint32_t inverse(uint32_t a) const;

    // Field trace Tr_{K/F2}(z) = sum_{i=0}^{L-1} z^{2^i}
    uint32_t trace(uint32_t z) const;

    // Additive character psi(z) = (-1)^Tr(z)
    int additive_character(uint32_t z) const;

    // Multiplicative character chi_k(alpha^n) = exp(-2*pi*i*k*n / N)
    Complex multiplicative_character(uint32_t k, uint32_t z) const;

    // Convert integer state z to binary vector representation
    std::vector<int> element_to_vector(uint32_t z) const;
    uint32_t vector_to_element(const std::vector<int>& vec) const;
};

// Generator for LFSR sequence using companion matrix & field trace representations
class LFSRGenerator {
public:
    const GF2Field& field;
    uint32_t beta; // Initial seed parameter in GF(2^L)

    LFSRGenerator(const GF2Field& f, uint32_t seed_beta = 1) : field(f), beta(seed_beta) {}

    // Generate x_n in F2 via Companion matrix state evolution
    std::vector<int> generate_bits_companion(size_t num_bits, uint32_t initial_state) const;

    // Generate x_n in F2 via Trace formula: x_n = Tr(beta * alpha^n)
    std::vector<int> generate_bits_trace(size_t num_bits) const;

    // Generate bipolar sequence u_n = (-1)^x_n = psi(beta * alpha^n)
    std::vector<int> generate_bipolar_sequence(size_t num_bits) const;
};

// Comprehensive spectral and Wiener series analysis result
struct SpectralReport {
    uint32_t L;
    uint32_t N;
    uint32_t poly;
    uint32_t beta;
    std::vector<int> bits;
    std::vector<int> bipolar;
    std::vector<Complex> dft;
    std::vector<double> power_spectrum;
    std::vector<Complex> gauss_sums;
    std::vector<double> autocorrelation;
    bool is_flat;
    bool is_autocorr_two_valued;
    double max_spectral_error;
};

// Analysis toolkit class
class SpectralAnalyzer {
public:
    // Fast Walsh-Hadamard Transform on 2^L length array
    static void fwht(std::vector<double>& a);

    // Compute DFT over Z_N
    static std::vector<Complex> compute_dft(const std::vector<int>& u);

    // Compute exact Gauss sum g(chi_k, psi) = sum_{z in GF(2^L)*} chi_k(z) * psi(z)
    static std::vector<Complex> compute_gauss_sums(const GF2Field& field);

    // Compute periodic autocorrelation R(d) = sum_n u_n * u_{n+d}
    static std::vector<double> compute_autocorrelation(const std::vector<int>& u);

    // Compute higher-order correlation sum sum_n prod_{d in D} u_{n+d}
    static double compute_higher_order_correlation(const std::vector<int>& u, const std::vector<uint32_t>& delays);

    // Check if primitive polynomial p(t) divides Q(t) = sum_{d in D} t^d in F2[t]
    static bool poly_divides_delay_sum(uint32_t poly, uint32_t L, const std::vector<uint32_t>& delays);

    // Perform full spectral report generation for an LFSR
    static SpectralReport analyze(const GF2Field& field, uint32_t beta);

    // Export report to JSON string
    static std::string export_json(const SpectralReport& report);
};

} // namespace lfsr_wiener

#endif // LFSR_WIENER_TOOLKIT_HPP

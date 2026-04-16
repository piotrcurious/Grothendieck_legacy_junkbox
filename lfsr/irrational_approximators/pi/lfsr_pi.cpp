// lfsr_pi.cpp — LFSR-based π approximation via deterministic Monte Carlo
//
// Theory: Two maximal-length LFSRs with coprime periods generate a 2D point
// set covering (0,1)² with joint period T_A × T_B = 8,589,803,985.
// Geometric probability (quarter-circle test) estimates π.
//
// Polynomials (primitive over GF(2), from standard tables):
//   LFSR-A: x^16 + x^15 + x^13 + x^4 + 1   T_A = 65535
//   LFSR-B: x^17 + x^3  + 1                  T_B = 131071 (Mersenne prime)
//   gcd(T_A, T_B) = 1  →  joint period = T_A × T_B
//
// Galois mask derivation for p(x) = x^n + x^{e1} + x^{e2} + ... + 1:
//   mask = bit(n−1) | bit(e1−1) | bit(e2−1) | ...
//   LFSR-A: mask = bit(15)|bit(14)|bit(12)|bit(3) = 0xD008
//   LFSR-B: mask = bit(16)|bit(2)                 = 0x10004

#include <iostream>
#include <cstdint>
#include <cmath>
#include <iomanip>

// ─────────────────────────────────────────────────────────────────────────────
struct LFSR16 {
    uint16_t s;
    explicit LFSR16(uint16_t seed = 0xACE1u) : s(seed ? seed : 1u) {}

    void step() {
        uint16_t lsb = s & 1u;
        s >>= 1;
        if (lsb) s ^= 0xD008u;
    }

    double coord() const { return static_cast<double>(s) / 65536.0; }
};

// ─────────────────────────────────────────────────────────────────────────────
struct LFSR17 {
    uint32_t s;
    explicit LFSR17(uint32_t seed = 0x1ACE1u)
        : s((seed & 0x1FFFFu) ? (seed & 0x1FFFFu) : 1u) {}

    void step() {
        uint32_t lsb = s & 1u;
        s >>= 1;
        if (lsb) s ^= 0x10004u;
    }

    double coord() const { return static_cast<double>(s) / 131072.0; }
};

// ─────────────────────────────────────────────────────────────────────────────
double lfsr_pi(uint64_t N, uint16_t seedA = 0xABCDu, uint32_t seedB = 0x12345u) {
    LFSR16 A(seedA);
    LFSR17 B(seedB);
    uint64_t hits = 0;
    for (uint64_t i = 0; i < N; ++i) {
        A.step();
        B.step();
        double x = A.coord();
        double y = B.coord();
        if (x * x + y * y < 1.0) ++hits;
    }
    return 4.0 * static_cast<double>(hits) / static_cast<double>(N);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    constexpr double PI_REF = 3.14159265358979323846;

    std::cout << "LFSR-based pi approximation via deterministic Monte Carlo\n"
              << "LFSR-A: 16-bit  p(x)=x^16+x^15+x^13+x^4+1  mask=0xD008   T=65535\n"
              << "LFSR-B: 17-bit  p(x)=x^17+x^3+1             mask=0x10004  T=131071\n"
              << "Joint period: 65535 x 131071 = 8,589,803,985\n\n"
              << std::fixed << std::setprecision(8);

    std::cout << std::setw(12) << "N"
              << std::setw(14) << "pi estimate"
              << std::setw(14) << "|error|"
              << std::setw(14) << "1/sqrt(N)\n"
              << std::string(54, '-') << "\n";

    for (uint64_t N = 1000; N <= 100'000'000ULL; N *= 10) {
        double est   = lfsr_pi(N);
        double err   = std::abs(est - PI_REF);
        double bound = 1.0 / std::sqrt(static_cast<double>(N));
        std::cout << std::setw(12) << N
                  << std::setw(14) << est
                  << std::setw(14) << err
                  << std::setw(14) << bound << "\n";
    }
    return 0;
}

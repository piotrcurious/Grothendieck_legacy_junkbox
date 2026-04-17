#include <iostream>
#include <cstdint>
#include <cmath>
#include <iomanip>

// ─────────────────────────────────────────────────────────────────────────────
// LFSR16 — 16-bit Galois LFSR
//
// Primitive polynomial: p(x) = x^16 + x^15 + x^13 + x^4 + 1
// Galois mask derivation:
//   middle exponents {15, 13, 4} → mask = bit(14)|bit(12)|bit(3) | bit(15)
//   = 0x4000 | 0x1000 | 0x0008 | 0x8000 = 0xD008
// Period: T_A = 2^16 − 1 = 65535
// ─────────────────────────────────────────────────────────────────────────────
struct LFSR16 {
    uint16_t s;
    explicit LFSR16(uint16_t seed = 0xACE1u) : s(seed ? seed : 1u) {}

    void step() {
        uint16_t lsb = s & 1u;
        s >>= 1;
        if (lsb) s ^= 0xD008u;
    }

    // Coordinate in (0, 1): s / 2^16
    double coord() const { return static_cast<double>(s) / 65536.0; }
};

// ─────────────────────────────────────────────────────────────────────────────
// LFSR17 — 17-bit Galois LFSR
//
// Primitive polynomial: p(x) = x^17 + x^3 + 1
// Galois mask derivation:
//   middle exponent {3} → mask = bit(16) | bit(2) = 0x10000 | 0x4 = 0x10004
// Period: T_B = 2^17 − 1 = 131071  (Mersenne prime M_17)
//
// Coprimality: gcd(65535, 131071) = 1  (131071 is prime, 65535 < 131071)
// Joint period: T_A × T_B = 65535 × 131071 = 8,589,803,985
// ─────────────────────────────────────────────────────────────────────────────
struct LFSR17 {
    uint32_t s;  // 17-bit state, upper 15 bits always zero
    explicit LFSR17(uint32_t seed = 0x1ACE1u)
        : s((seed & 0x1FFFFu) ? (seed & 0x1FFFFu) : 1u) {}

    void step() {
        uint32_t lsb = s & 1u;
        s >>= 1;
        if (lsb) s ^= 0x10004u;
    }

    // Coordinate in (0, 1): s / 2^17
    double coord() const { return static_cast<double>(s) / 131072.0; }
};

// ─────────────────────────────────────────────────────────────────────────────
// lfsr_pi — estimate π using two coprime LFSRs via geometric probability
//
// Points (x, y) = (A.coord(), B.coord()) cover (0,1)² with joint period
// T_A × T_B ≈ 8.6 × 10^9. Over the full period every lattice pair (a, b) is
// visited exactly once: the estimate is a deterministic numerical integration
// of the quarter-circle indicator with error O(1/√N) in practice.
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

int main() {
    constexpr double PI_REF = 3.14159265358979323846;

    std::cout << "LFSR-based π approximation via deterministic Monte Carlo\n"
              << "LFSR-A: 16-bit, p(x)=x^16+x^15+x^13+x^4+1  mask=0xD008  T=65535\n"
              << "LFSR-B: 17-bit, p(x)=x^17+x^3+1             mask=0x10004 T=131071\n"
              << "Joint period: 65535 × 131071 = 8,589,803,985\n\n"
              << std::fixed << std::setprecision(8);

    std::cout << std::setw(12) << "N"
              << std::setw(14) << "π estimate"
              << std::setw(14) << "|error|"
              << std::setw(14) << "1/√N\n"
              << std::string(54, '─') << "\n";

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

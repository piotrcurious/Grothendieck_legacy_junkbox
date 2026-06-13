// lfsr_shift_register.cpp
// Genuine shift-register LFSR rewrite of the earlier finite-field demo.
//
// Build: g++ -std=c++20 -O2 lfsr_shift_register.cpp -o lfsr_shift_register
//
// Notes:
// - The LFSR is a real shift-register update:
//     state <- (state << 1) ^ taps    when the outgoing top bit is 1
// - The polynomial search is self-contained and does not require NTL.
// - Exact traversal of [0, max_val] is handled by a separate permutation,
//   because a raw LFSR cannot enumerate an arbitrary interval exactly once.

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using u64 = std::uint64_t;
using u128 = unsigned __int128;

// -----------------------------------------------------------------------------
// Polynomial helpers over GF(2), represented as bitmasks.
// Bit i = coefficient of x^i.
// -----------------------------------------------------------------------------

static int deg_u64(u64 x) {
    if (x == 0) return -1;
    return 63 - __builtin_clzll(x);
}

static int deg_u128(u128 x) {
    if (x == 0) return -1;
    u64 hi = static_cast<u64>(x >> 64);
    if (hi) return 64 + (63 - __builtin_clzll(hi));
    u64 lo = static_cast<u64>(x);
    return 63 - __builtin_clzll(lo);
}

static std::vector<u64> unique_prime_factors(u64 n) {
    std::vector<u64> f;
    for (u64 p = 2; p * p <= n; ++p) {
        if (n % p == 0) {
            f.push_back(p);
            while (n % p == 0) n /= p;
        }
    }
    if (n > 1) f.push_back(n);
    return f;
}

static u64 poly_rem(u128 a, u64 mod) {
    if (mod == 0) throw std::runtime_error("poly_rem: zero modulus");
    int dm = deg_u64(mod);
    while (a && deg_u128(a) >= dm) {
        int shift = deg_u128(a) - dm;
        a ^= (u128(mod) << shift);
    }
    return static_cast<u64>(a);
}

static u64 poly_mul_mod(u64 a, u64 b, u64 mod) {
    u128 prod = 0;
    while (b) {
        if (b & 1ULL) prod ^= (u128(a));
        a <<= 1;
        b >>= 1;
        // keep prod un-reduced for simplicity; reduction happens at the end
        // and degrees stay within 128 bits for our n <= 62 use case.
    }
    return poly_rem(prod, mod);
}

// Faster direct multiplication with explicit shifts.
static u64 poly_mul_mod_shift(u64 a, u64 b, u64 mod) {
    u128 prod = 0;
    for (int i = 0; i < 64; ++i) {
        if ((b >> i) & 1ULL) prod ^= (u128(a) << i);
    }
    return poly_rem(prod, mod);
}

static u64 poly_pow_mod(u64 base, u64 exp, u64 mod) {
    u64 r = 1;
    while (exp) {
        if (exp & 1ULL) r = poly_mul_mod_shift(r, base, mod);
        base = poly_mul_mod_shift(base, base, mod);
        exp >>= 1;
    }
    return r;
}

static u64 poly_gcd(u64 a, u64 b) {
    while (b) {
        u64 r = poly_rem((u128)a, b);
        a = b;
        b = r;
    }
    return a;
}

// Rabin irreducibility test over GF(2).
static bool is_irreducible(u64 p) {
    int n = deg_u64(p);
    if (n <= 0) return false;
    if ((p & 1ULL) == 0) return false;       // constant term must be 1 for a primitive poly
    if (n == 1) return true;                 // x + 1

    u64 x = 2ULL;                            // polynomial x
    u64 t = x;

    // Check x^(2^n) = x mod p
    for (int i = 0; i < n; ++i) {
        t = poly_mul_mod_shift(t, t, p);
    }
    if (t != x) return false;

    // Check gcd(p, x^(2^(n/q)) - x) = 1 for each prime divisor q of n
    for (u64 q : unique_prime_factors((u64)n)) {
        int k = n / (int)q;
        t = x;
        for (int i = 0; i < k; ++i) {
            t = poly_mul_mod_shift(t, t, p);
        }
        if (poly_gcd(p, t ^ x) != 1ULL) return false;
    }

    return true;
}

static bool is_primitive_poly(u64 p) {
    int n = deg_u64(p);
    if (n < 2 || n >= 63) return false;
    if (!is_irreducible(p)) return false;

    u64 order = (1ULL << n) - 1ULL;
    u64 x = 2ULL; // polynomial x

    for (u64 q : unique_prime_factors(order)) {
        u64 e = order / q;
        if (poly_pow_mod(x, e, p) == 1ULL) return false;
    }
    return true;
}

static u64 width_seed(long n) {
    return 0x9E3779B97F4A7C15ULL ^ (u64(n) * 0xD1B54A32D192ED03ULL);
}

// Random search for a primitive polynomial of degree n.
// Works well for the modest widths used in the demo.
static u64 find_primitive_polynomial(long n, u64 seed = 0) {
    if (n < 2 || n >= 63) throw std::runtime_error("find_primitive_polynomial: invalid n");

    std::mt19937_64 gen(seed ? seed : width_seed(n));
    const u64 top = 1ULL << n;
    const u64 interior_mask = (top - 1ULL) & ~1ULL; // bits 1..n-1

    for (long attempt = 0; attempt < 250000; ++attempt) {
        // Dense random candidate, with degree n and constant term 1.
        u64 p = top | 1ULL | (gen() & interior_mask);
        if (is_primitive_poly(p)) return p;
    }

    throw std::runtime_error("failed to find a primitive polynomial");
}

// -----------------------------------------------------------------------------
// Genuine shift-register LFSR
// Left-shift Galois form:
//   carry = top bit
//   state <<= 1
//   if (carry) state ^= lower_bits_of_primitive_polynomial
// -----------------------------------------------------------------------------

class GaloisLfsr {
    long n_;
    u64 poly_;
    u64 tap_mask_;
    u64 state_;
    u64 state_mask_;

public:
    GaloisLfsr(long n, u64 primitive_poly, u64 seed_state = 1)
        : n_(n),
          poly_(primitive_poly),
          tap_mask_(primitive_poly & ((1ULL << n) - 1ULL)),
          state_mask_((1ULL << n) - 1ULL),
          state_(seed_state & state_mask_) {
        if (n_ < 2 || n_ >= 63) throw std::runtime_error("GaloisLfsr: invalid n");
        if (((poly_ >> n_) & 1ULL) == 0 || (poly_ & 1ULL) == 0) {
            throw std::runtime_error("GaloisLfsr: polynomial must be monic and have constant term 1");
        }
        if (state_ == 0) state_ = 1;
    }

    long width() const { return n_; }
    u64 polynomial() const { return poly_; }
    u64 state() const { return state_; }

    // Advance one step and return the new state.
    u64 step() {
        const bool carry = ((state_ >> (n_ - 1)) & 1ULL) != 0;
        state_ = (state_ << 1) & state_mask_;
        if (carry) state_ ^= tap_mask_;
        if (state_ == 0) {
            // This should not happen for a primitive polynomial and nonzero seed.
            throw std::runtime_error("LFSR entered zero state");
        }
        return state_;
    }

    // Convenience: produce current state then step.
    u64 next() {
        u64 out = state_;
        step();
        return out;
    }
};

// -----------------------------------------------------------------------------
// Orbit
// -----------------------------------------------------------------------------

struct Orbit {
    long n = 0;
    u64 primitive_poly = 0;
    std::vector<u64> states;
};

static Orbit build_orbit(long n) {
    if (n < 2 || n >= 63) throw std::runtime_error("invalid degree n for orbit");

    Orbit out;
    out.n = n;
    out.primitive_poly = find_primitive_polynomial(n);

    GaloisLfsr lfsr(n, out.primitive_poly, 1);
    const u64 length = (1ULL << n) - 1ULL;
    out.states.reserve((size_t)length);

    for (u64 i = 0; i < length; ++i) {
        out.states.push_back(lfsr.next());
    }
    return out;
}

// -----------------------------------------------------------------------------
// Exact interval traversal (separate from the LFSR)
// This is a true duplicate-free traversal of [0, max_val].
// A raw LFSR cannot do this exactly, so we use an affine permutation.
// -----------------------------------------------------------------------------

class ExactRangeTraverser {
    u64 N_;
    u64 a_;
    u64 b_;
    u64 i_;

public:
    explicit ExactRangeTraverser(u64 max_val, u64 seed = 0)
        : N_(max_val + 1), a_(1), b_(0), i_(0) {
        std::mt19937_64 gen(seed ? seed : 0xC0FFEE123456789ULL);

        if (N_ == 0) throw std::runtime_error("ExactRangeTraverser: overflow");

        do {
            a_ = gen();
            a_ |= 1ULL; // make it odd
            a_ %= N_;
            if (a_ == 0) a_ = 1;
        } while (std::gcd(a_, N_) != 1);

        b_ = gen() % N_;
    }

    u64 next() {
        u64 out = (a_ * i_ + b_) % N_;
        ++i_;
        return out;
    }

    void reset() { i_ = 0; }
};

// -----------------------------------------------------------------------------
// Inference layer
// Demo recognizer: it checks whether a prefix matches the LFSR sequence that
// this program would generate for a given width.
// -----------------------------------------------------------------------------

struct Observation {
    std::vector<std::string> prefix;
};

struct InferenceResult {
    std::vector<long> candidates;
};

using Recognizer = std::function<bool(const Observation&, long candidate_width)>;

static std::string as_string(u64 x) {
    return std::to_string(x);
}

static u64 string_to_u64(const std::string& s) {
    return static_cast<u64>(std::stoull(s));
}

static Recognizer prefix_recognizer = [](const Observation& obs, long w) -> bool {
    try {
        if (obs.prefix.size() < 2) return false;

        // Deterministic search rule so the demo is reproducible.
        u64 poly = find_primitive_polynomial(w, width_seed(w));
        GaloisLfsr lfsr(w, poly, 1);

        for (const auto& expected : obs.prefix) {
            u64 x = lfsr.next();
            if (as_string(x) != expected) return false;
        }
        return true;
    } catch (...) {
        return false;
    }
};

static InferenceResult ran_extend(const std::vector<Recognizer>& recognizers,
                                  const Observation& obs) {
    InferenceResult res;
    for (long w = 2; w <= 32; ++w) {
        bool ok = true;
        for (const auto& r : recognizers) {
            if (!r(obs, w)) {
                ok = false;
                break;
            }
        }
        if (ok) res.candidates.push_back(w);
    }
    return res;
}

// -----------------------------------------------------------------------------
// Tests / demos
// -----------------------------------------------------------------------------

static void test_inference() {
    std::cout << "--- 1. Inference Test ---\n";
    long n = 8;

    u64 poly = find_primitive_polynomial(n, width_seed(n));
    GaloisLfsr lfsr(n, poly, 1);

    Observation obs;
    for (int i = 0; i < 5; ++i) {
        obs.prefix.push_back(as_string(lfsr.next()));
    }

    auto inf = ran_extend({prefix_recognizer}, obs);
    std::cout << "Inferred width(s):";
    for (long w : inf.candidates) std::cout << " " << w;
    std::cout << "\n";
    std::cout << "Primitive polynomial (hex): 0x" << std::hex << poly << std::dec << "\n\n";
}

static void test_lfsr_orbit(long n) {
    std::cout << "--- 2. LFSR Orbit Generation (n=" << n << ") ---\n";
    Orbit orb = build_orbit(n);

    u64 expected_len = (1ULL << n) - 1ULL;
    std::cout << "Generated orbit length: " << orb.states.size()
              << " (Expected: " << expected_len << ")\n";

    std::set<u64> seen(orb.states.begin(), orb.states.end());
    bool nonzero = (seen.find(0) == seen.end());
    bool unique = (seen.size() == orb.states.size());

    std::cout << "Verification: all states unique and nonzero? "
              << ((unique && nonzero) ? "YES" : "NO") << "\n";
    std::cout << "Primitive polynomial (hex): 0x" << std::hex << orb.primitive_poly << std::dec << "\n\n";
}

static void test_exact_range(u64 max_val, u64 seed = 12345ULL) {
    std::cout << "--- 3. Exact Range Traversal (max_val=" << max_val << ") ---\n";
    ExactRangeTraverser traverser(max_val, seed);

    std::vector<u64> seq;
    seq.reserve((size_t)max_val + 1);

    for (u64 i = 0; i <= max_val; ++i) {
        seq.push_back(traverser.next());
    }

    std::set<u64> seen(seq.begin(), seq.end());
    bool all_in_range = true;
    for (u64 x : seq) {
        if (x > max_val) all_in_range = false;
    }

    std::cout << "Sequence size: " << seq.size() << "\n";
    std::cout << "Verification: all numbers in [0, " << max_val << "] visited exactly once? "
              << ((seen.size() == max_val + 1 && all_in_range) ? "YES" : "NO") << "\n\n";
}

static void test_reproducibility() {
    std::cout << "--- 4. Reproducibility Test ---\n";
    long n = 8;
    u64 seed = 777;

    u64 poly1 = find_primitive_polynomial(n, width_seed(n));
    u64 poly2 = find_primitive_polynomial(n, width_seed(n));

    bool match_poly = (poly1 == poly2);

    GaloisLfsr t1(n, poly1, 1);
    GaloisLfsr t2(n, poly2, 1);

    bool match_seq = true;
    for (int i = 0; i < 20; ++i) {
        if (t1.next() != t2.next()) {
            match_seq = false;
            break;
        }
    }

    std::cout << "Deterministic polynomial search: " << (match_poly ? "PASS" : "FAIL") << "\n";
    std::cout << "Sequence reproducibility: " << (match_seq ? "PASS" : "FAIL") << "\n\n";
    (void)seed;
}

int main() {
    try {
        std::cout << "=== Genuine Shift-Register LFSR Suite ===\n\n";

        test_inference();
        test_lfsr_orbit(4);
        test_lfsr_orbit(8);
        test_lfsr_orbit(12);

        // Exact interval traversal is separate from the LFSR.
        test_exact_range(20, 12345ULL);
        test_exact_range(1000, 999ULL);
        test_exact_range(65535, 42ULL);

        test_reproducibility();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

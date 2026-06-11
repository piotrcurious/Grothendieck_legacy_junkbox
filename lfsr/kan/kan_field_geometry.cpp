// kan_field_geometry.cpp - Final Improved Version
// A geometric suite for LFSRs and algebraic recurrences.
// This implements the Kan extension ideas described in Readme.md.

#include <algorithm>
#include <bit>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using u64 = std::uint64_t;

// -----------------------------------------------------------------------------
// GF(2)[x] Polynomial Math
// -----------------------------------------------------------------------------

static unsigned degree(u64 p) {
    if (p == 0) throw std::invalid_argument("zero polynomial has no degree");
    return 63u - std::countl_zero(p);
}

static u64 mask_width(unsigned n) {
    if (n >= 63) return ~0ULL;
    return (u64{1} << n) - 1;
}

static std::string poly_to_string(u64 p) {
    if (p == 0) return "0";
    std::string s;
    bool first = true;
    for (int i = static_cast<int>(degree(p)); i >= 0; --i) {
        if (((p >> i) & 1ULL) == 0) continue;
        if (!first) s += " + ";
        if (i == 0) s += "1";
        else if (i == 1) s += "x";
        else s += "x^" + std::to_string(i);
        first = false;
    }
    return s;
}

static u64 poly_mod(u64 a, u64 b) {
    if (b == 0) throw std::invalid_argument("division by zero");
    unsigned db = degree(b);
    while (a && degree(a) >= db) {
        a ^= (b << (degree(a) - db));
    }
    return a;
}

static u64 poly_gcd(u64 a, u64 b) {
    while (b != 0) {
        u64 r = poly_mod(a, b);
        a = b; b = r;
    }
    return a;
}

static u64 poly_mul_mod(u64 a, u64 b, u64 mod) {
    unsigned n = degree(mod);
    u64 red = mod ^ (u64{1} << n);
    u64 mask = mask_width(n);
    a &= mask; b &= mask;
    u64 res = 0;
    while (b) {
        if (b & 1ULL) res ^= a;
        b >>= 1;
        bool carry = (a & (u64{1} << (n - 1))) != 0;
        a <<= 1; a &= mask;
        if (carry) a ^= red;
    }
    return res;
}

static u64 poly_pow_mod(u64 base, u64 exp, u64 mod) {
    u64 res = 1;
    while (exp) {
        if (exp & 1ULL) res = poly_mul_mod(res, base, mod);
        base = poly_mul_mod(base, base, mod);
        exp >>= 1;
    }
    return res;
}

static u64 calculate_trace(u64 a, u64 mod) {
    unsigned n = degree(mod);
    u64 sum = a;
    u64 term = a;
    for (unsigned i = 1; i < n; ++i) {
        term = poly_mul_mod(term, term, mod);
        sum ^= term;
    }
    return sum & 1ULL;
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

static bool is_irreducible(u64 p) {
    unsigned n = degree(p);
    if ((p & 1ULL) == 0) return false;
    u64 x = poly_mod(2, p);
    // x^(2^n) == x mod p
    u64 t = x;
    for (unsigned i = 0; i < n; ++i) t = poly_mul_mod(t, t, p);
    if (t != x) return false;
    for (u64 q : unique_prime_factors(n)) {
        u64 tq = x;
        for (unsigned i = 0; i < n/q; ++i) tq = poly_mul_mod(tq, tq, p);
        if (poly_gcd(tq ^ x, p) != 1) return false;
    }
    return true;
}

static bool is_primitive(u64 p) {
    unsigned n = degree(p);
    if (!is_irreducible(p)) return false;
    u64 x = poly_mod(2, p);
    u64 order = (u64{1} << n) - 1;
    for (u64 q : unique_prime_factors(order)) {
        if (poly_pow_mod(x, order / q, p) == 1) return false;
    }
    return true;
}

static std::vector<u64> primitive_locus(unsigned n) {
    std::vector<u64> out;
    u64 lo = (u64{1} << n) | 1ULL;
    u64 hi = (u64{1} << (n + 1));
    for (u64 p = lo; p < hi; p += 2ULL) {
        if (is_primitive(p)) out.push_back(p);
    }
    return out;
}

// -----------------------------------------------------------------------------
// Algebraic Machine
// -----------------------------------------------------------------------------

class AlgebraicLFSR {
public:
    unsigned n;
    u64 modulus;
    u64 state;

    AlgebraicLFSR(unsigned n_, u64 mod_, u64 seed = 1)
        : n(n_), modulus(mod_?mod_:((1ULL<<n_)|1)), state(seed & mask_width(n_)) {
        if (state == 0) state = 1;
    }

    void step() {
        state = poly_mul_mod(state, 2, modulus);
    }
};

// -----------------------------------------------------------------------------
// Kan Extension Infrastructure
// -----------------------------------------------------------------------------

struct Fragment {
    std::string chart_name;
    std::string coordinate_desc;
    AlgebraicLFSR machine;
    std::function<bool(u64, u64)> projection;

    std::string sample(std::size_t bits) {
        std::string s;
        u64 saved_state = machine.state;
        for (std::size_t i = 0; i < bits; ++i) {
            s.push_back(projection(machine.state, machine.modulus) ? '1' : '0');
            machine.step();
        }
        machine.state = saved_state; // preserve for other uses
        return s;
    }
};

using ChartFunctor = std::function<std::optional<Fragment>(u64 poly)>;

static std::vector<Fragment> lan_extend(const std::vector<ChartFunctor>& atlas, u64 poly) {
    std::vector<Fragment> fragments;
    for (auto& f : atlas) {
        auto res = f(poly);
        if (res) fragments.push_back(std::move(*res));
    }
    return fragments;
}

// Concrete Charts
static ChartFunctor companion_chart = [](u64 p) -> std::optional<Fragment> {
    unsigned n = degree(p);
    return Fragment{"Companion", "Polynomial basis bit 0", AlgebraicLFSR(n, p),
                    [](u64 s, u64) { return (s & 1) != 0; }};
};

static ChartFunctor trace_chart = [](u64 p) -> std::optional<Fragment> {
    unsigned n = degree(p);
    return Fragment{"Trace", "Field trace to GF(2)", AlgebraicLFSR(n, p),
                    [](u64 s, u64 m) { return calculate_trace(s, m) != 0; }};
};

static ChartFunctor reciprocal_chart = [](u64 p) -> std::optional<Fragment> {
    unsigned n = degree(p);
    u64 rp = 0;
    for (unsigned i = 0; i <= n; ++i) if ((p >> i) & 1) rp |= (1ULL << (n-i));
    return Fragment{"Reciprocal", "Dual basis / reversed orbit", AlgebraicLFSR(n, rp),
                    [](u64 s, u64) { return (s & 1) != 0; }};
};

// Inference (Ran extension)
struct Observation {
    unsigned n;
    std::string bits;
};

using Recognizer = std::function<std::vector<u64>(const Observation&)>;

static Recognizer direct_recognizer = [](const Observation& obs) {
    std::vector<u64> candidates;
    for (u64 p : primitive_locus(obs.n)) {
        AlgebraicLFSR m(obs.n, p, 1);
        bool match = true;
        for (char c : obs.bits) {
            if ((m.state & 1) != (c == '1')) { match = false; break; }
            m.step();
        }
        if (match) candidates.push_back(p);
    }
    return candidates;
};

static Recognizer dual_recognizer = [](const Observation& obs) {
    std::vector<u64> candidates;
    std::string rev = obs.bits;
    std::reverse(rev.begin(), rev.end());
    for (u64 p : primitive_locus(obs.n)) {
        u64 rp = 0; unsigned n = degree(p);
        for (unsigned i = 0; i <= n; ++i) if ((p >> i) & 1) rp |= (1ULL << (n-i));
        bool found = false;
        for (u64 s = 1; s < (1ULL << obs.n); ++s) {
            AlgebraicLFSR m(obs.n, rp, s);
            bool match = true;
            for (char c : rev) {
                if ((m.state & 1) != (c == '1')) { match = false; break; }
                m.step();
            }
            if (match) { found = true; break; }
        }
        if (found) candidates.push_back(p);
    }
    return candidates;
};

// -----------------------------------------------------------------------------
// Main Demo
// -----------------------------------------------------------------------------

#ifndef NO_MAIN
int main() {
    try {
        unsigned n = 6;
        auto locus = primitive_locus(n);
        if (locus.empty()) return 1;
        u64 poly = locus.front();

        std::cout << "--- Kan LFSR Geometry Suite ---\n";
        std::cout << "Global Object (Polynomial): " << poly_to_string(poly) << "\n\n";

        std::vector<ChartFunctor> atlas = {companion_chart, trace_chart, reciprocal_chart};
        auto fragments = lan_extend(atlas, poly);

        std::cout << "Left Kan Extension (Local implementation fragments):\n";
        for (auto& f : fragments) {
            std::cout << "  [" << std::left << std::setw(10) << f.chart_name << "] "
                      << std::setw(25) << f.coordinate_desc << " | Sample: " << f.sample(32) << "\n";
        }

        std::string prefix = fragments[0].sample(12);
        Observation obs{n, prefix};
        std::cout << "\nRight Kan Extension (Inference from observation '" << prefix << "'):\n";

        auto c1 = direct_recognizer(obs);
        auto c2 = dual_recognizer(obs);

        std::vector<u64> common;
        std::set_intersection(c1.begin(), c1.end(), c2.begin(), c2.end(), std::back_inserter(common));

        for (u64 p : common) {
            std::cout << "  Candidate: " << poly_to_string(p) << "\n";
        }

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
#endif

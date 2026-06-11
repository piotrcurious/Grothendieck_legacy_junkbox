#include <bit>
#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

using u64 = std::uint64_t;

// ---------- basic GF(2)[x] polynomial utilities ----------
// Polynomials are encoded as bitmasks.
// Bit i = coefficient of x^i.
// Example: x^5 + x^2 + 1  ->  0b100101

static unsigned degree(u64 p) {
    if (p == 0) throw std::invalid_argument("zero polynomial has no degree");
    return 63u - std::countl_zero(p);
}

static std::string poly_to_string(u64 p) {
    if (p == 0) return "0";
    std::string s;
    bool first = true;
    for (int i = static_cast<int>(degree(p)); i >= 0; --i) {
        if (!(p & (u64{1} << i))) continue;
        if (!first) s += " + ";
        if (i == 0) s += "1";
        else if (i == 1) s += "x";
        else s += "x^" + std::to_string(i);
        first = false;
    }
    return s;
}

static u64 poly_mod(u64 a, u64 b) {
    if (b == 0) throw std::invalid_argument("division by zero polynomial");
    unsigned db = degree(b);
    while (a && degree(a) >= db) {
        unsigned shift = degree(a) - db;
        a ^= (b << shift);
    }
    return a;
}

static bool poly_divides(u64 a, u64 b) {
    return poly_mod(a, b) == 0;
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

static u64 reciprocal_poly(u64 p) {
    unsigned n = degree(p);
    u64 r = (u64{1} << n); // leading term
    for (unsigned i = 0; i < n; ++i) {
        if (p & (u64{1} << i)) {
            r |= (u64{1} << (n - 1 - i));
        }
    }
    return r;
}

static u64 reverse_low_bits(u64 x, unsigned n) {
    u64 r = 0;
    for (unsigned i = 0; i < n; ++i) {
        if (x & (u64{1} << i)) {
            r |= (u64{1} << (n - 1 - i));
        }
    }
    return r;
}

static u64 mul_mod(u64 a, u64 b, u64 mod) {
    unsigned n = degree(mod);
    u64 red = mod ^ (u64{1} << n); // remove the leading x^n term
    u64 mask = (n == 63) ? ~u64{0} : ((u64{1} << n) - 1);

    a &= mask;
    b &= mask;

    u64 res = 0;
    while (b) {
        if (b & 1) res ^= a;
        b >>= 1;

        bool carry = (a & (u64{1} << (n - 1))) != 0;
        a <<= 1;
        a &= mask;
        if (carry) a ^= red;
    }
    return res & mask;
}

static u64 pow_mod(u64 base, u64 exp, u64 mod) {
    u64 res = 1;
    while (exp) {
        if (exp & 1) res = mul_mod(res, base, mod);
        base = mul_mod(base, base, mod);
        exp >>= 1;
    }
    return res;
}

// Brute-force irreducibility for small n.
// Good enough for a demonstrator, not for production.
static bool is_irreducible(u64 p) {
    unsigned n = degree(p);
    if ((p & 1ULL) == 0) return false; // constant term must be 1 for our family

    for (unsigned d = 1; d <= n / 2; ++d) {
        u64 lo = (u64{1} << d);
        u64 hi = (u64{1} << (d + 1));
        for (u64 q = lo; q < hi; ++q) {
            if ((q & 1ULL) == 0) continue; // constant term must be 1
            if (poly_divides(p, q)) return false;
        }
    }
    return true;
}

static bool is_primitive(u64 p) {
    unsigned n = degree(p);
    if (!is_irreducible(p)) return false;

    // Over GF(2), x mod p is represented by the polynomial "x" = 0b10.
    const u64 x = 0b10;
    const u64 order = (u64{1} << n) - 1;

    for (u64 q : unique_prime_factors(order)) {
        u64 e = order / q;
        if (pow_mod(x, e, p) == 1) return false;
    }
    return pow_mod(x, order, p) == 1;
}

static std::vector<u64> primitive_locus(unsigned n) {
    if (n == 0 || n >= 63) {
        throw std::invalid_argument("n must be in 1..62 for this demo");
    }

    std::vector<u64> out;
    u64 limit = (u64{1} << n); // lower n coefficients
    for (u64 lower = 1; lower < limit; lower += 2) { // constant term = 1
        u64 p = (u64{1} << n) | lower;               // monic degree n
        if (is_primitive(p)) out.push_back(p);
    }
    return out;
}

// ---------- LFSR as one local chart ----------
// Companion-form recurrence for p(x)=x^n + a_{n-1}x^{n-1} + ... + a_0.
// State is n bits; next bit is the parity of masked state.
struct LFSR {
    unsigned n;
    u64 feedback_mask; // lower n bits of the polynomial, excluding x^n
    u64 state;         // n-bit state, nonzero

    LFSR(unsigned n_, u64 poly, u64 seed)
        : n(n_), feedback_mask(poly & ((u64{1} << n_) - 1)), state(seed) {
        if (n_ == 0 || n_ >= 63) throw std::invalid_argument("bad width");
        if (state == 0) throw std::invalid_argument("seed must be nonzero");
        state &= ((u64{1} << n) - 1);
        if (state == 0) state = 1;
    }

    bool step() {
        bool out = state & 1ULL;
        bool fb = (std::popcount(state & feedback_mask) & 1U) != 0;
        state >>= 1;
        if (fb) state |= (u64{1} << (n - 1));
        return out;
    }
};

struct Fragment {
    std::string chart;
    std::string coordinates;
    LFSR machine;

    std::string sample(std::size_t bits) {
        std::string s;
        for (std::size_t i = 0; i < bits; ++i) {
            s.push_back(machine.step() ? '1' : '0');
        }
        return s;
    }
};

using Chart = std::function<std::optional<Fragment>(u64 target_poly)>;

struct Atlas {
    std::vector<Chart> charts;
};

// This is the toy "Lan_i F": extend local chart data to a global suite.
static std::vector<Fragment> lan_extend(const Atlas& atlas, u64 poly) {
    std::vector<Fragment> glued;
    for (const auto& chart : atlas.charts) {
        auto frag = chart(poly);
        if (frag) glued.push_back(std::move(*frag));
    }
    return glued;
}

// ---------- local charts ----------

static Chart companion_chart = [](u64 p) -> std::optional<Fragment> {
    unsigned n = degree(p);
    u64 seed = 1; // any nonzero seed works for a primitive polynomial
    return Fragment{
        "companion chart",
        "state as low-to-high bits",
        LFSR{n, p, seed}
    };
};

static Chart reciprocal_chart = [](u64 p) -> std::optional<Fragment> {
    unsigned n = degree(p);
    u64 rp = reciprocal_poly(p);
    u64 seed = 1;
    u64 rev_seed = reverse_low_bits(seed, n);
    if (rev_seed == 0) rev_seed = 1;

    return Fragment{
        "reciprocal chart",
        "reversed coordinates",
        LFSR{n, rp, rev_seed}
    };
};

// ---------- demo ----------

int main() {
    try {
        const unsigned n = 5;

        std::cout << "Primitive locus in A^" << (n - 1) << " over F2 for degree " << n << ":\n";
        auto prims = primitive_locus(n);
        for (u64 p : prims) {
            std::cout << "  " << poly_to_string(p) << "\n";
        }

        if (prims.empty()) {
            std::cout << "No primitive polynomials found.\n";
            return 0;
        }

        Atlas atlas;
        atlas.charts.push_back(companion_chart);
        atlas.charts.push_back(reciprocal_chart);

        u64 target = prims.front();
        std::cout << "\nChosen target polynomial:\n  " << poly_to_string(target) << "\n";

        auto suite = lan_extend(atlas, target);

        std::cout << "\nLocal fragments glued by the toy Kan-extension layer:\n";
        for (auto& frag : suite) {
            std::cout << "\n[" << frag.chart << "]\n";
            std::cout << "  coordinates: " << frag.coordinates << "\n";
            std::cout << "  sample bits:  " << frag.sample(24) << "\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

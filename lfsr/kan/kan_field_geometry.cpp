// g++ -std=c++20 -O2 lfsr_kan_geometry.cpp -o lfsr_kan_geometry

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
// GF(2)[x] polynomial utilities
// Polynomials are encoded as bitmasks:
// bit i = coefficient of x^i.
// -----------------------------------------------------------------------------

static unsigned degree(u64 p) {
    if (p == 0) throw std::invalid_argument("zero polynomial has no degree");
    return 63u - std::countl_zero(p);
}

static u64 mask_width(unsigned n) {
    if (n >= 63) throw std::invalid_argument("width too large for this demo");
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
    if (b == 0) throw std::invalid_argument("division by zero polynomial");
    unsigned db = degree(b);
    while (a && degree(a) >= db) {
        unsigned shift = degree(a) - db;
        a ^= (b << shift);
    }
    return a;
}

static u64 poly_gcd(u64 a, u64 b) {
    while (b != 0) {
        u64 r = poly_mod(a, b);
        a = b;
        b = r;
    }
    return a;
}

static u64 poly_mul_mod(u64 a, u64 b, u64 mod) {
    unsigned n = degree(mod);
    u64 red = mod ^ (u64{1} << n);          // remove leading x^n term
    u64 mask = (n == 62) ? ((u64{1} << 62) - 1) : ((u64{1} << n) - 1);

    a &= mask;
    b &= mask;

    u64 res = 0;
    while (b) {
        if (b & 1ULL) res ^= a;
        b >>= 1;

        bool carry = (a & (u64{1} << (n - 1))) != 0;
        a <<= 1;
        a &= mask;
        if (carry) a ^= red;
    }
    return res & mask;
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

static u64 frobenius_pow(u64 a, unsigned k, u64 mod) {
    for (unsigned i = 0; i < k; ++i) {
        a = poly_mul_mod(a, a, mod); // a <- a^2
    }
    return a; // a^(2^k)
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
    if ((p & 1ULL) == 0) return false; // constant term must be 1 in our family
    if (((p >> n) & 1ULL) == 0) return false;

    // Let x be the residue class of the polynomial x.
    u64 x = poly_mod(2, p);

    // Rabin test:
    // x^(2^n) == x mod p
    if (frobenius_pow(x, n, p) != x) return false;

    // gcd(x^(2^(n/q)) - x, p) == 1 for every prime q dividing n
    for (u64 q : unique_prime_factors(n)) {
        u64 t = frobenius_pow(x, static_cast<unsigned>(n / q), p);
        if (poly_gcd(t ^ x, p) != 1) return false;
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
    return poly_pow_mod(x, order, p) == 1;
}

static std::vector<u64> primitive_locus(unsigned n) {
    if (n == 0 || n >= 63) throw std::invalid_argument("n must be in 1..62");

    std::vector<u64> out;
    u64 lo = (u64{1} << n) | 1ULL;      // monic, constant term 1
    u64 hi = (u64{1} << (n + 1));       // next degree boundary

    for (u64 p = lo; p < hi; p += 2ULL) {
        if (is_primitive(p)) out.push_back(p);
    }
    return out;
}

static std::optional<u64> find_primitive_polynomial(unsigned n) {
    auto all = primitive_locus(n);
    if (all.empty()) return std::nullopt;
    return all.front();
}

// -----------------------------------------------------------------------------
// Algebraic "LFSR": state evolution is multiplication by x in GF(2)[x]/(p).
// This is the clean geometric version: one orbit of a point under an algebraic map.
// -----------------------------------------------------------------------------

class AlgebraicLFSR {
public:
    unsigned n;
    u64 modulus;
    u64 state;

    AlgebraicLFSR(unsigned n_, u64 modulus_, u64 seed = 1)
        : n(n_), modulus(modulus_), state(seed & mask_width(n_)) {
        if (n_ == 0 || n_ >= 63) throw std::invalid_argument("bad width");
        if (((modulus_ >> n_) & 1ULL) == 0) throw std::invalid_argument("modulus must be monic");
        if ((modulus_ & 1ULL) == 0) throw std::invalid_argument("constant term must be 1");
        if (state == 0) state = 1;
    }

    u64 step_state() {
        state = poly_mul_mod(state, 2, modulus); // multiply by x
        if (state == 0) state = 1;               // should not happen for primitive moduli
        return state;
    }

    bool step_bit() {
        bool out = (state & 1ULL) != 0;
        step_state();
        return out;
    }
};

struct PathSequence {
    unsigned n = 0;
    u64 modulus = 0;
    u64 seed = 1;
    std::vector<u64> states;
    std::string output_bits;
};

// Pick the smallest width n with 2^n - 1 >= length, then generate a prefix
// of one primitive orbit. No repeats inside the prefix.
static PathSequence construct_nonrepeating_path(std::size_t length) {
    if (length == 0) return {};

    unsigned n = 1;
    while (n < 62 && ((u64{1} << n) - 1ULL) < length) ++n;
    if (((u64{1} << n) - 1ULL) < length) {
        throw std::runtime_error("requested length too large for this demo");
    }

    auto mod = find_primitive_polynomial(n);
    if (!mod) throw std::runtime_error("no primitive polynomial found");

    PathSequence seq;
    seq.n = n;
    seq.modulus = *mod;
    seq.seed = 1;
    seq.states.reserve(length);
    seq.output_bits.reserve(length);

    AlgebraicLFSR gen(n, *mod, 1);
    for (std::size_t i = 0; i < length; ++i) {
        seq.states.push_back(gen.state);
        seq.output_bits.push_back(gen.step_bit() ? '1' : '0');
    }

    return seq;
}

// -----------------------------------------------------------------------------
// Kan-style layers: local charts and global glue.
// -----------------------------------------------------------------------------

struct Fragment {
    std::string chart;
    std::string coordinates;
    AlgebraicLFSR machine;

    std::string sample(std::size_t bits) {
        std::string s;
        s.reserve(bits);
        for (std::size_t i = 0; i < bits; ++i) {
            s.push_back(machine.step_bit() ? '1' : '0');
        }
        return s;
    }
};

using Chart = std::function<std::optional<Fragment>(u64 target_poly)>;

struct Atlas {
    std::vector<Chart> charts;
};

static std::vector<Fragment> lan_extend(const Atlas& atlas, u64 poly) {
    std::vector<Fragment> glued;
    for (const auto& chart : atlas.charts) {
        auto frag = chart(poly);
        if (frag) glued.push_back(std::move(*frag));
    }
    return glued;
}

struct Observation {
    unsigned n;
    std::string prefix;
};

struct Evidence {
    std::string chart;
    std::vector<u64> candidates;
};

using Recognizer = std::function<std::optional<Evidence>(const Observation&)>;

static std::string reverse_string(std::string s) {
    std::reverse(s.begin(), s.end());
    return s;
}

static std::vector<u64> intersect_candidate_sets(const std::vector<std::vector<u64>>& sets) {
    if (sets.empty()) return {};

    std::vector<u64> acc = sets.front();
    std::sort(acc.begin(), acc.end());

    for (std::size_t i = 1; i < sets.size(); ++i) {
        std::vector<u64> cur = sets[i];
        std::sort(cur.begin(), cur.end());

        std::vector<u64> next;
        std::set_intersection(
            acc.begin(), acc.end(),
            cur.begin(), cur.end(),
            std::back_inserter(next)
        );
        acc = std::move(next);
    }
    return acc;
}

static bool matches_prefix(u64 poly, unsigned n, const std::string& prefix, u64 seed = 1) {
    AlgebraicLFSR gen(n, poly, seed);
    for (char c : prefix) {
        char bit = gen.step_bit() ? '1' : '0';
        if (bit != c) return false;
    }
    return true;
}

static Recognizer direct_recognizer = [](const Observation& obs) -> std::optional<Evidence> {
    std::vector<u64> cand;
    for (u64 p : primitive_locus(obs.n)) {
        if (matches_prefix(p, obs.n, obs.prefix, 1)) {
            cand.push_back(p);
        }
    }
    return Evidence{"direct chart", std::move(cand)};
};

static Recognizer reciprocal_recognizer = [](const Observation& obs) -> std::optional<Evidence> {
    // Toy dual chart: try the reversed prefix against the reciprocal polynomial.
    std::vector<u64> cand;
    std::string rev = reverse_string(obs.prefix);

    for (u64 p : primitive_locus(obs.n)) {
        u64 rp = 0;
        unsigned n = degree(p);
        for (unsigned i = 0; i <= n; ++i) {
            if ((p >> i) & 1ULL) {
                rp |= (u64{1} << (n - i));
            }
        }
        if (matches_prefix(rp, obs.n, rev, 1)) {
            cand.push_back(p);
        }
    }
    return Evidence{"reciprocal chart", std::move(cand)};
};

struct GlobalInference {
    std::vector<u64> candidates;
};

static GlobalInference ran_extend(const std::vector<Recognizer>& recognizers, const Observation& obs) {
    std::vector<std::vector<u64>> all_sets;

    for (const auto& rec : recognizers) {
        auto ev = rec(obs);
        if (!ev) continue;
        all_sets.push_back(std::move(ev->candidates));
    }

    return GlobalInference{intersect_candidate_sets(all_sets)};
}

// -----------------------------------------------------------------------------
// Charts for the implementation layer.
// -----------------------------------------------------------------------------

static Chart companion_chart = [](u64 p) -> std::optional<Fragment> {
    unsigned n = degree(p);
    return Fragment{
        "companion chart",
        "multiplication by x modulo p",
        AlgebraicLFSR{n, p, 1}
    };
};

static Chart reciprocal_chart = [](u64 p) -> std::optional<Fragment> {
    unsigned n = degree(p);
    u64 rp = 0;
    for (unsigned i = 0; i <= n; ++i) {
        if ((p >> i) & 1ULL) {
            rp |= (u64{1} << (n - i));
        }
    }

    return Fragment{
        "reciprocal chart",
        "same orbit in dual coordinates",
        AlgebraicLFSR{n, rp, 1}
    };
};

// -----------------------------------------------------------------------------
// Demo
// -----------------------------------------------------------------------------

int main() {
    try {
        std::size_t target_length = 37;

        auto path = construct_nonrepeating_path(target_length);

        std::cout << "Requested nonrepeating length: " << target_length << "\n";
        std::cout << "Chosen width n: " << path.n << "\n";
        std::cout << "Primitive modulus: " << poly_to_string(path.modulus) << "\n\n";

        std::cout << "States visited in order:\n";
        for (std::size_t i = 0; i < path.states.size(); ++i) {
            std::cout << std::hex << std::showbase << path.states[i] << std::dec;
            if (i + 1 != path.states.size()) std::cout << " ";
        }
        std::cout << "\n\n";

        std::cout << "Output bits:\n" << path.output_bits << "\n\n";

        // Left Kan-style layer: glue local chart fragments.
        Atlas atlas;
        atlas.charts.push_back(companion_chart);
        atlas.charts.push_back(reciprocal_chart);

        auto fragments = lan_extend(atlas, path.modulus);

        std::cout << "Local fragments glued by the left-extension layer:\n";
        for (auto& frag : fragments) {
            std::cout << "  [" << frag.chart << "] ";
            std::cout << "sample bits: " << frag.sample(24) << "\n";
        }

        // Right Kan-style layer: infer candidates from an observed prefix.
        std::string observed_prefix = path.output_bits.substr(0, std::min<std::size_t>(12, path.output_bits.size()));
        Observation obs{path.n, observed_prefix};

        std::vector<Recognizer> recognizers{
            direct_recognizer,
            reciprocal_recognizer
        };

        auto inferred = ran_extend(recognizers, obs);

        std::cout << "\nObserved prefix: " << obs.prefix << "\n";
        std::cout << "Right-extension candidates:\n";
        if (inferred.candidates.empty()) {
            std::cout << "  <no compatible global object>\n";
        } else {
            for (u64 p : inferred.candidates) {
                std::cout << "  " << poly_to_string(p) << "\n";
            }
        }

        std::cout << "\nCheck: ";
        if (path.states.size() == ((u64{1} << path.n) - 1ULL)) {
            std::cout << "full nonzero orbit traversed exactly once.\n";
        } else {
            std::cout << "prefix of a maximal orbit, no repeats in the requested length.\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

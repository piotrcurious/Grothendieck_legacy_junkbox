// g++ -std=c++20 -O2 demo_ntl_lfsr_kan.cpp -lntl -lgmp -o demo_ntl_lfsr_kan
#ifdef USE_MOCK_NTL
#include "ntl_mock.h"
#else
#include <NTL/GF2X.h>
#include <NTL/GF2E.h>
#include <NTL/GF2XFactoring.h>
#endif

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <functional>
#include <iomanip>

using namespace NTL;
using u64 = std::uint64_t;

// -----------------------------------------------------------------------------
// Small integer utilities
// -----------------------------------------------------------------------------

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

static std::string gf2x_to_string(const GF2X& f) {
    std::ostringstream oss;
    oss << f;
    return oss.str();
}

static std::string gf2e_to_bits(const GF2E& a) {
    GF2X r = rep(a);
    long d = deg(GF2E::modulus());
    if (d < 0) return "0";

    std::string s;
    for (long i = d - 1; i >= 0; --i) {
        s.push_back(IsOne(coeff(r, i)) ? '1' : '0');
    }
    return s;
}

// -----------------------------------------------------------------------------
// Field construction
// -----------------------------------------------------------------------------

enum class ModulusPolicy {
    Sparse,
    Canonical
};

static GF2X build_modulus(long n, ModulusPolicy policy) {
    if (n <= 0) throw std::invalid_argument("degree must be positive");

    GF2X P;
    if (policy == ModulusPolicy::Sparse) {
        BuildSparseIrred(P, n);
    } else {
        BuildIrred(P, n);
    }
    return P;
}

static bool is_primitive_element(const GF2E& a, u64 field_order_minus_1) {
    for (u64 p : unique_prime_factors(field_order_minus_1)) {
        ZZ e = to_ZZ(field_order_minus_1 / p);
        GF2E t;
        power(t, a, e);
        if (IsOne(t)) return false;
    }
    GF2E full;
    power(full, a, to_ZZ(field_order_minus_1));
    return IsOne(full);
}

static std::optional<GF2E> find_primitive_element(u64 field_order_minus_1, long max_tries = 2000) {
    for (long i = 0; i < max_tries; ++i) {
        GF2E a;
        random(a);
        if (IsZero(a)) continue;
        if (is_primitive_element(a, field_order_minus_1)) return a;
    }
    return std::nullopt;
}

// -----------------------------------------------------------------------------
// Algebraic orbit generator
// -----------------------------------------------------------------------------

struct OrbitSequence {
    long n = 0;
    GF2X modulus;
    GF2E primitive;
    std::vector<GF2E> states;
};

static OrbitSequence build_orbit_sequence(std::size_t length,
                                          ModulusPolicy policy = ModulusPolicy::Sparse) {
    if (length == 0) return {};

    long n = 1;
    while (n < 62 && ((u64{1} << n) - 1ULL) < length) ++n;
    if (n >= 62) throw std::runtime_error("requested length too large for this demo");

    GF2X P = build_modulus(n, policy);
    GF2E::init(P);

    const u64 orbit_size = (u64{1} << n) - 1ULL;
    auto prim = find_primitive_element(orbit_size);
    if (!prim) {
        throw std::runtime_error("failed to find primitive element");
    }

    OrbitSequence seq;
    seq.n = n;
    seq.modulus = P;
    seq.primitive = *prim;
    seq.states.reserve(length);

    GF2E state(1);
    for (std::size_t i = 0; i < length; ++i) {
        seq.states.push_back(state);
        state *= seq.primitive;
    }

    return seq;
}

// -----------------------------------------------------------------------------
// Kan-style left extension: local charts -> global implementation fragments
// -----------------------------------------------------------------------------

struct Fragment {
    std::string chart;
    std::string coordinates;
    GF2E generator;

    std::string sample_bits(std::size_t bits) {
        std::string s;
        s.reserve(bits);
        GF2E x = generator;
        for (std::size_t i = 0; i < bits; ++i) {
            s.push_back(IsOne(coeff(rep(x), 0)) ? '1' : '0'); // low bit of polynomial basis rep
            x *= generator;
        }
        return s;
    }
};

using Chart = std::function<std::optional<Fragment>(const GF2X& modulus, const GF2E& primitive)>;

struct Atlas {
    std::vector<Chart> charts;
};

static std::vector<Fragment> lan_extend(const Atlas& atlas,
                                        const GF2X& modulus,
                                        const GF2E& primitive) {
    std::vector<Fragment> out;
    for (const auto& chart : atlas.charts) {
        auto frag = chart(modulus, primitive);
        if (frag) out.push_back(std::move(*frag));
    }
    return out;
}

static Chart polynomial_basis_chart = [](const GF2X& modulus, const GF2E& primitive) -> std::optional<Fragment> {
    (void)modulus;
    return Fragment{
        "polynomial-basis chart",
        "x^0..x^(n-1) coordinates",
        primitive
    };
};

static Chart jump_ahead_chart = [](const GF2X& modulus, const GF2E& primitive) -> std::optional<Fragment> {
    (void)modulus;
    GF2E g = primitive;
    ZZ jump = to_ZZ(8);
    power(g, g, jump);
    return Fragment{
        "jump-ahead chart",
        "8-step orbit map",
        g
    };
};

// -----------------------------------------------------------------------------
// Kan-style right extension: local observations -> global candidate set
// -----------------------------------------------------------------------------

struct Observation {
    long n;
    std::string prefix_bits;
};

struct Evidence {
    std::string chart;
    std::vector<std::size_t> candidate_lengths;
};

using Recognizer = std::function<std::optional<Evidence>(const Observation&)>;

static std::vector<std::size_t> intersect_sets(const std::vector<std::vector<std::size_t>>& sets) {
    if (sets.empty()) return {};
    std::vector<std::size_t> acc = sets.front();
    std::sort(acc.begin(), acc.end());

    for (std::size_t i = 1; i < sets.size(); ++i) {
        std::vector<std::size_t> cur = sets[i];
        std::sort(cur.begin(), cur.end());
        std::vector<std::size_t> next;
        std::set_intersection(acc.begin(), acc.end(),
                              cur.begin(), cur.end(),
                              std::back_inserter(next));
        acc = std::move(next);
    }
    return acc;
}

static Recognizer prefix_recognizer = [](const Observation& obs) -> std::optional<Evidence> {
    std::vector<std::size_t> candidates;
    for (std::size_t L = obs.prefix_bits.size(); L <= obs.prefix_bits.size() + 16; ++L) {
        try {
            auto seq = build_orbit_sequence(L);
            GF2E state(1);
            std::string got;
            for (std::size_t i = 0; i < obs.prefix_bits.size(); ++i) {
                got.push_back(IsOne(coeff(rep(state), 0)) ? '1' : '0');
                state *= seq.primitive;
            }
            if (got == obs.prefix_bits) candidates.push_back(L);
        } catch (...) {
        }
    }
    return Evidence{"prefix chart", std::move(candidates)};
};

static Recognizer reversed_prefix_recognizer = [](const Observation& obs) -> std::optional<Evidence> {
    std::string rev = obs.prefix_bits;
    std::reverse(rev.begin(), rev.end());

    std::vector<std::size_t> candidates;
    for (std::size_t L = rev.size(); L <= rev.size() + 16; ++L) {
        try {
            auto seq = build_orbit_sequence(L);
            GF2E state(1);
            std::string got;
            for (std::size_t i = 0; i < rev.size(); ++i) {
                got.push_back(IsOne(coeff(rep(state), 0)) ? '1' : '0');
                state *= seq.primitive;
            }
            if (got == rev) candidates.push_back(L);
        } catch (...) {
        }
    }
    return Evidence{"reversed chart", std::move(candidates)};
};

struct GlobalInference {
    std::vector<std::size_t> lengths;
};

static GlobalInference ran_extend(const std::vector<Recognizer>& recognizers, const Observation& obs) {
    std::vector<std::vector<std::size_t>> all;
    for (const auto& r : recognizers) {
        auto e = r(obs);
        if (e) all.push_back(std::move(e->candidate_lengths));
    }
    return GlobalInference{intersect_sets(all)};
}

// -----------------------------------------------------------------------------
// Demo
// -----------------------------------------------------------------------------

int main() {
    try {
        const std::size_t requested_length = 37;

        auto seq = build_orbit_sequence(requested_length, ModulusPolicy::Sparse);

        std::cout << "--- NTL Toy Orbit Demo ---\n";
        std::cout << "Chosen width n: " << seq.n << "\n";
        std::cout << "Modulus: " << gf2x_to_string(seq.modulus) << "\n";
        std::cout << "Primitive element found: yes\n\n";

        std::cout << "First " << requested_length << " orbit states:\n";
        for (std::size_t i = 0; i < seq.states.size(); ++i) {
            std::cout << "  " << std::setw(2) << i << ": " << gf2e_to_bits(seq.states[i]) << "\n";
        }

        Atlas atlas;
        atlas.charts.push_back(polynomial_basis_chart);
        atlas.charts.push_back(jump_ahead_chart);

        auto frags = lan_extend(atlas, seq.modulus, seq.primitive);
        std::cout << "\nLeft-extension fragments:\n";
        for (auto& f : frags) {
            std::cout << "  [" << f.chart << "] " << f.coordinates << "\n";
        }

        std::string prefix;
        {
            GF2E state(1);
            for (int i = 0; i < 12; ++i) {
                prefix.push_back(IsOne(coeff(rep(state), 0)) ? '1' : '0');
                state *= seq.primitive;
            }
        }

        Observation obs{seq.n, prefix};
        std::vector<Recognizer> recognizers{
            prefix_recognizer,
            reversed_prefix_recognizer
        };

        auto inf = ran_extend(recognizers, obs);
        std::cout << "\nObserved prefix: " << obs.prefix_bits << "\n";
        std::cout << "Right-extension inferred lengths:\n";
        if (inf.lengths.empty()) {
            std::cout << "  <none>\n";
        } else {
            for (auto L : inf.lengths) {
                std::cout << "  " << L << "\n";
            }
        }

        if (seq.states.size() == ((u64{1} << seq.n) - 1ULL)) {
            std::cout << "\nThis is a full nonzero orbit: every nonzero field state appears once.\n";
        } else {
            std::cout << "\nThis is a nonrepeating prefix of a primitive orbit.\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

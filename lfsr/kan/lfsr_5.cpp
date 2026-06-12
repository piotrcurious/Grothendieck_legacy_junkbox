// lfsr_5.cpp - Improved NTL Kan extension suite
#ifdef USE_MOCK_NTL
#include "ntl_mock.h"
#else
#include <NTL/GF2X.h>
#include <NTL/GF2E.h>
#include <NTL/GF2XFactoring.h>
#include <NTL/GF2EX.h>
#include <NTL/mat_GF2.h>
#endif

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <functional>
#include <iomanip>

using namespace NTL;
using u64 = std::uint64_t;

// -----------------------------------------------------------------------------
// Utilities
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

static std::string as_string(const GF2E& a) {
    std::ostringstream oss;
    oss << a;
    return oss.str();
}

static std::string as_string(const GF2EX& a) {
    std::ostringstream oss;
    oss << a;
    return oss.str();
}

// -----------------------------------------------------------------------------
// Field Model and Primitivity
// -----------------------------------------------------------------------------

struct Orbit {
    long n = 0;
    GF2X modulus;
    GF2E primitive;
    std::vector<GF2E> states;
};

static bool is_primitive_element(const GF2E& a, u64 order_minus_1) {
    if (IsZero(a)) return false;

    for (u64 p : unique_prime_factors(order_minus_1)) {
        GF2E t;
        power(t, a, to_ZZ(order_minus_1 / p));
        if (IsOne(t)) return false;
    }

    GF2E full;
    power(full, a, to_ZZ(order_minus_1));
    return IsOne(full);
}

static GF2E bits_to_gf2e(u64 bits) {
    GF2X f;
    for (int i = 0; i < 64; ++i) {
        if ((bits >> i) & 1) SetCoeff(f, i);
    }
    return conv<GF2E>(f);
}

static GF2E find_primitive_element(u64 order_minus_1, long tries = 5000) {
    GF2E a;
    for (long i = 0; i < tries; ++i) {
        random(a);
        if (is_primitive_element(a, order_minus_1)) return a;
    }
    throw std::runtime_error("failed to find primitive element");
}

static Orbit build_orbit(std::size_t length) {
    if (length == 0) return {};

    long n = 1;
    while (n < 62 && ((u64{1} << n) - 1ULL) < length) ++n;
    if (n >= 62) throw std::runtime_error("requested length too large");

    GF2X P;
    BuildSparseIrred(P, n);

    GF2EPush scope(P);

    const u64 order_minus_1 = (u64{1} << n) - 1ULL;
    GF2E alpha = find_primitive_element(order_minus_1);

    Orbit out;
    out.n = n;
    out.modulus = P;
    out.primitive = alpha;
    out.states.reserve(length);

    GF2E s(1);
    for (std::size_t i = 0; i < length; ++i) {
        out.states.push_back(s);
        s *= alpha;
    }
    return out;
}

// -----------------------------------------------------------------------------
// Kan Extension Infrastructure
// -----------------------------------------------------------------------------

struct Fragment {
    std::string chart_name;
    std::string note;
    std::string sample_output;
};

using Chart = std::function<std::optional<Fragment>(const Orbit&)>;

struct Atlas {
    std::vector<Chart> charts;
};

static std::vector<Fragment> lan_extend(const Atlas& atlas, const Orbit& orbit) {
    std::vector<Fragment> out;
    for (const auto& c : atlas.charts) {
        auto f = c(orbit);
        if (f) out.push_back(std::move(*f));
    }
    return out;
}

// Concrete Charts

static Chart basis_chart = [](const Orbit& orbit) -> std::optional<Fragment> {
    Fragment f;
    f.chart_name = "Polynomial Basis";
    f.note = "Standard orbit representation in GF(2)[x]/(p)";
    f.sample_output = as_string(orbit.states.front());
    return f;
};

static Chart trace_chart = [](const Orbit& orbit) -> std::optional<Fragment> {
    GF2EPush scope(orbit.modulus);
    Fragment f;
    f.chart_name = "Trace Map";
    f.note = "Projection via Tr: GF(2^n) -> GF(2)";
    std::string s;
    for (size_t i = 0; i < std::min<size_t>(orbit.states.size(), 24); ++i) {
        s += (IsOne(trace(orbit.states[i])) ? '1' : '0');
    }
    f.sample_output = s;
    return f;
};

static Chart jump_chart = [](const Orbit& orbit) -> std::optional<Fragment> {
    GF2EPush scope(orbit.modulus);
    // Precompute X^8 mod p in the extension field
    GF2EX fpoly;
    for (long i = 0; i <= orbit.n; ++i) {
        if (IsOne(coeff(orbit.modulus, i))) SetCoeff(fpoly, i, GF2E(1));
    }

    GF2EXModulus F(fpoly);
    GF2EX jump = PowerXMod(8, F);

    Fragment f;
    f.chart_name = "Jump-Ahead";
    f.note = "Preconditioned PowerXMod (step=8)";
    f.sample_output = as_string(jump);
    return f;
};

static Chart matrix_chart = [](const Orbit& orbit) -> std::optional<Fragment> {
    Fragment f;
    f.chart_name = "Matrix View";
    f.note = "Linear transformation in GF(2)^n";

    GF2EPush scope(orbit.modulus);
    mat_GF2 M;
    M.SetDims(orbit.n, orbit.n);
    // Matrix for multiplication by x in polynomial basis
    for (long j = 0; j < orbit.n; j++) {
        GF2E val = orbit.primitive * orbit.states[j];
        GF2X r = rep(val);
        for (long i = 0; i < orbit.n; i++) {
            if (IsOne(coeff(r, i))) M[i][j] = 1;
        }
    }

    std::ostringstream oss;
    oss << M;
    f.sample_output = oss.str();
    return f;
};

// -----------------------------------------------------------------------------
// Inference Layer (Right Kan Extension)
// -----------------------------------------------------------------------------

struct Observation {
    std::vector<std::string> prefix;
};

struct InferenceResult {
    std::vector<long> candidates;
};

using Recognizer = std::function<bool(const Observation&, long candidate_width)>;

static Recognizer prefix_recognizer = [](const Observation& obs, long w) -> bool {
    try {
        GF2X P;
        BuildSparseIrred(P, w);
        GF2EPush scope(P);

        u64 order_minus_1 = (1ULL << w) - 1;

        if (obs.prefix.size() < 2) return false;

        // The first state is always [1]. The second state is orbit[1] = 1 * alpha.
        // So alpha must be orbit[1].
        // We find val such that as_string(GF2E(val)) == obs.prefix[1].

        std::optional<GF2E> alpha;
        for (u64 val = 1; val < (1ULL << w); ++val) {
            GF2E cand = bits_to_gf2e(val);
            if (as_string(cand) == obs.prefix[1]) {
                alpha = cand;
                break;
            }
        }

        if (!alpha || !is_primitive_element(*alpha, order_minus_1)) return false;

        GF2E state(1);
        for (const auto& expected : obs.prefix) {
            if (as_string(state) != expected) return false;
            state *= (*alpha);
        }
        return true;
    } catch (...) {
        return false;
    }
};

static InferenceResult ran_extend(const std::vector<Recognizer>& recognizers,
                                 const Observation& obs) {
    InferenceResult res;
    // Search for compatible widths in a reasonable range
    for (long w = 2; w <= 16; ++w) {
        bool ok = true;
        for (auto& r : recognizers) {
            if (!r(obs, w)) { ok = false; break; }
        }
        if (ok) res.candidates.push_back(w);
    }
    return res;
}

// -----------------------------------------------------------------------------
// Main Demo
// -----------------------------------------------------------------------------

int main() {
    try {
        SetSeed(to_ZZ(42));
        std::size_t requested_length = 37;
        Orbit orbit = build_orbit(requested_length);

        std::cout << "=== Improved NTL Kan Extension Suite ===\n";
        std::cout << "Global Object: GF(2^" << orbit.n << ") / " << orbit.modulus << "\n";
        std::cout << "Primitive Element: " << orbit.primitive << "\n\n";

        std::cout << "Orbit states (prefix):\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(8, orbit.states.size()); ++i) {
            std::cout << "  " << i << ": " << orbit.states[i] << "\n";
        }
        std::cout << "  ...\n\n";

        Atlas atlas;
        atlas.charts.push_back(basis_chart);
        atlas.charts.push_back(trace_chart);
        atlas.charts.push_back(jump_chart);
        atlas.charts.push_back(matrix_chart);

        auto frags = lan_extend(atlas, orbit);
        std::cout << "Left Kan Extension (Glued Local Fragments):\n";
        for (const auto& f : frags) {
            std::cout << "  [" << std::left << std::setw(18) << f.chart_name << "] "
                      << std::setw(35) << f.note << " | Sample: " << f.sample_output << "\n";
        }

        Observation obs;
        for (std::size_t i = 0; i < 5; ++i) obs.prefix.push_back(as_string(orbit.states[i]));

        std::vector<Recognizer> recs = { prefix_recognizer };
        auto inf = ran_extend(recs, obs);

        std::cout << "\nRight Kan Extension (Global Object Inference):\n";
        std::cout << "  Observed " << obs.prefix.size() << " states.\n";
        std::cout << "  Compatible widths found: ";
        for (long w : inf.candidates) std::cout << w << " ";
        std::cout << "\n\n";

        if (orbit.states.size() == ((u64{1} << orbit.n) - 1ULL)) {
            std::cout << "Verification: Full cycle reached.\n";
        } else {
            std::cout << "Verification: Non-repeating primitive prefix.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

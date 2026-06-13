// lfsr_5.cpp - Improved NTL Kan Extension Suite
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
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

// as_string(GF2E) requires an active GF2EPush; callers must ensure that.
static std::string as_string(const GF2E& a) {
    std::ostringstream oss;
    oss << a;
    return oss.str();
}

// Kept for completeness; not used in current charts after jump_chart simplification.
static std::string as_string(const GF2EX& a) {
    std::ostringstream oss;
    oss << a;
    return oss.str();
}

// -----------------------------------------------------------------------------
// Field Model and Primitivity
// -----------------------------------------------------------------------------

// FIX (bug 2): Orbit stores states and the primitive element as GF2X coefficient
// vectors (via rep()) so they are valid independently of any GF2EPush scope.
// Charts that need GF2E arithmetic push orbit.modulus themselves, then convert
// via conv<GF2E>(orbit.states[i]).
struct Orbit {
    long n = 0;
    GF2X modulus;
    GF2X primitive;           // rep() of the primitive element
    std::vector<GF2X> states; // rep() of each orbit state
};

// Returns true iff a is a primitive element of the current GF2E context.
// order_minus_1 must equal 2^n - 1 for the current extension degree n.
//
// FIX (bug 5): The final a^(q-1) check from the original has been removed.
// By Fermat's little theorem every non-zero element of GF(2^n) satisfies
// a^(2^n-1) = 1, so the check was always true and cost a full exponentiation.
static bool is_primitive_element(const GF2E& a, u64 order_minus_1) {
    if (IsZero(a)) return false;
    for (u64 p : unique_prime_factors(order_minus_1)) {
        GF2E t;
        power(t, a, to_ZZ(order_minus_1 / p));
        if (IsOne(t)) return false;
    }
    return true;
}

// Convert a 64-bit integer bit-pattern to a GF2E element in the current context.
// Bit i of `bits` maps to the coefficient of x^i in the polynomial.
static GF2E bits_to_gf2e(u64 bits) {
    GF2X f;
    for (int i = 0; i < 64; ++i) {
        if ((bits >> i) & 1) SetCoeff(f, i);
    }
    return conv<GF2E>(f);
}

// Find a primitive element by random search in the current GF2E context.
static GF2E find_primitive_element(u64 order_minus_1, long tries = 5000) {
    GF2E a;
    for (long i = 0; i < tries; ++i) {
        random(a);
        if (is_primitive_element(a, order_minus_1)) return a;
    }
    throw std::runtime_error("failed to find primitive element");
}

// Build an orbit of the requested length.
// Selects the smallest n with 2^n - 1 >= length, constructs a sparse
// irreducible of degree n, finds a primitive element, and stores all elements
// as GF2X coefficient vectors so the result is context-independent.
//
// FIX (bug 2): GF2EPush is now contained entirely within this function.
// When the function returns the scope is destroyed, but the stored GF2X values
// in Orbit::states and Orbit::primitive are unaffected.
static Orbit build_orbit(std::size_t length) {
    if (length == 0) return {};

    long n = 1;
    while (n < 62 && ((u64{1} << n) - 1ULL) < length) ++n;
    if (n >= 62) throw std::runtime_error("requested length too large");

    GF2X P;
    BuildSparseIrred(P, n);

    const u64 order_minus_1 = (u64{1} << n) - 1ULL;

    GF2EPush scope(P); // context is active only inside this function

    GF2E alpha = find_primitive_element(order_minus_1);

    Orbit out;
    out.n         = n;
    out.modulus   = P;
    out.primitive = rep(alpha); // store as GF2X — context-free
    out.states.reserve(length);

    GF2E s(1);
    for (std::size_t i = 0; i < length; ++i) {
        out.states.push_back(rep(s)); // store as GF2X — context-free
        s *= alpha;
    }
    return out;
    // scope destroyed on return; GF2X values in `out` remain valid
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
// Each chart that needs GF2E arithmetic pushes orbit.modulus first, then
// converts stored GF2X values via conv<GF2E>().

// FIX (bug 8): guard against empty orbit before calling .front().
static Chart basis_chart = [](const Orbit& orbit) -> std::optional<Fragment> {
    if (orbit.states.empty()) return std::nullopt;
    GF2EPush scope(orbit.modulus);
    Fragment f;
    f.chart_name    = "Polynomial Basis";
    f.note          = "Standard orbit representation in GF(2)[x]/(p)";
    f.sample_output = as_string(conv<GF2E>(orbit.states.front()));
    return f;
};

static Chart trace_chart = [](const Orbit& orbit) -> std::optional<Fragment> {
    GF2EPush scope(orbit.modulus);
    Fragment f;
    f.chart_name = "Trace Map";
    f.note       = "Projection via Tr: GF(2^n) -> GF(2)";
    std::string s;
    for (size_t i = 0; i < std::min<size_t>(orbit.states.size(), 24); ++i) {
        GF2E elem = conv<GF2E>(orbit.states[i]);
        s += (IsOne(trace(elem)) ? '1' : '0');
    }
    f.sample_output = s;
    return f;
};

// FIX (bug 6): compute x^8 mod P directly in GF2X instead of lifting to
// GF2EX.  The result is the same (all coefficients live in {0,1} c GF2E)
// but GF2X arithmetic is cheaper and requires no GF2EPush.
static Chart jump_chart = [](const Orbit& orbit) -> std::optional<Fragment> {
    GF2X jump;
    PowerXMod(jump, 8, orbit.modulus);
    Fragment f;
    f.chart_name = "Jump-Ahead";
    f.note       = "x^8 mod P via PowerXMod (step=8)";
    std::ostringstream oss;
    oss << jump;
    f.sample_output = oss.str();
    return f;
};

// FIX (bug 3): compute the companion matrix — multiplication by x in the
// standard polynomial basis {1, x, x^2, ..., x^{n-1}} — not multiplication
// by alpha in the alpha-power basis as the original code did.
// Column j = GF(2) coefficients of (x * x^j) mod P = x^{j+1} mod P.
static Chart matrix_chart = [](const Orbit& orbit) -> std::optional<Fragment> {
    GF2EPush scope(orbit.modulus);

    GF2X xpoly; SetCoeff(xpoly, 1);      // polynomial x
    GF2E x_elem = conv<GF2E>(xpoly);     // x as GF2E in the current context

    mat_GF2 M;
    M.SetDims(orbit.n, orbit.n);

    for (long j = 0; j < orbit.n; j++) {
        GF2X basis_j; SetCoeff(basis_j, j);          // x^j (degree < n, no reduction)
        GF2E val = x_elem * conv<GF2E>(basis_j);     // x^{j+1} mod P
        GF2X r   = rep(val);
        for (long i = 0; i < orbit.n; i++) {
            if (IsOne(coeff(r, i))) M[i][j] = 1;
        }
    }

    Fragment f;
    f.chart_name = "Matrix View";
    f.note       = "Companion matrix: multiply by x in GF(2)^n";
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

// Attempts to identify whether obs is consistent with a GF(2^w) orbit whose
// modulus is the sparse irreducible returned by BuildSparseIrred(w).
//
// DESIGN NOTE: this implicitly assumes the observation was generated with
// exactly that polynomial.  If a different degree-w irreducible was used, string
// representations may diverge even for the correct w.  A production recognizer
// would iterate over all degree-w irreducibles or include the modulus itself in
// the observation.
static Recognizer prefix_recognizer = [](const Observation& obs, long w) -> bool {
    try {
        if (obs.prefix.size() < 2) return false;

        GF2X P;
        BuildSparseIrred(P, w);
        GF2EPush scope(P);

        const u64 order_minus_1 = (u64{1} << w) - 1;

        // FIX (bug 4): verify the first observed state is the multiplicative
        // identity before assuming the orbit starts at 1.
        {
            GF2E identity(1); // 1 mod 2 = 1 — always the unit element
            if (as_string(identity) != obs.prefix[0]) return false;
        }

        // Recover alpha from the second observed state by enumeration.
        // O(2^w) string comparisons; acceptable for w <= 20 in demo context.
        // A direct string -> GF2X parser would be O(w) and preferable in
        // production.
        std::optional<GF2E> alpha;
        for (u64 val = 1; val < (u64{1} << w); ++val) {
            GF2E cand = bits_to_gf2e(val);
            if (as_string(cand) == obs.prefix[1]) {
                alpha = cand;
                break;
            }
        }
        if (!alpha || !is_primitive_element(*alpha, order_minus_1)) return false;

        // Walk the orbit and compare each state to the observation.
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
    for (long w = 2; w <= 16; ++w) {
        bool ok = true;
        for (const auto& r : recognizers) {
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
        const long n = 4;
        GF2X P;
        SetCoeff(P, 4); SetCoeff(P, 1); SetCoeff(P, 0); // x^4 + x + 1
        GF2EPush scope(P);

        // FIX (bug 1): GF2E(2) evaluates to 0 because NTL reduces the integer
        // argument mod 2 (2 mod 2 = 0).  Construct x explicitly as a GF2X
        // polynomial with only the x^1 coefficient set, then embed into GF2E.
        GF2X x_poly;
        SetCoeff(x_poly, 1);               // polynomial x in GF2X
        GF2E x_elem = conv<GF2E>(x_poly);  // x as GF2E under the active scope

        // Verify x is indeed primitive for x^4 + x + 1.
        const u64 order_minus_1 = (u64{1} << n) - 1ULL; // 15
        if (!is_primitive_element(x_elem, order_minus_1))
            throw std::runtime_error("x is not primitive for the chosen modulus");

        Orbit orbit;
        orbit.n         = n;
        orbit.modulus   = P;
        orbit.primitive = x_poly;           // stored as GF2X — context-free

        orbit.states.reserve(15);
        GF2E s(1);
        for (int i = 0; i < 15; ++i) {
            orbit.states.push_back(rep(s)); // stored as GF2X — context-free
            s *= x_elem;
        }

        std::cout << "=== Improved NTL Kan Extension Suite ===\n";
        std::cout << "Global Object: GF(2^" << orbit.n << ") / " << orbit.modulus << "\n";
        std::cout << "Primitive Element: " << orbit.primitive << "\n\n";

        std::cout << "Orbit states (prefix):\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(8, orbit.states.size()); ++i) {
            // scope(P) is still active; safe to convert for display
            std::cout << "  " << i << ": "
                      << as_string(conv<GF2E>(orbit.states[i])) << "\n";
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
                      << std::setw(46) << f.note
                      << " | Sample: " << f.sample_output << "\n";
        }

        // Build observation strings while scope(P) is active so representations
        // are consistent with what prefix_recognizer will produce for width 4.
        Observation obs;
        for (std::size_t i = 0; i < 5; ++i)
            obs.prefix.push_back(as_string(conv<GF2E>(orbit.states[i])));

        std::vector<Recognizer> recs = { prefix_recognizer };
        auto inf = ran_extend(recs, obs);

        std::cout << "\nRight Kan Extension (Global Object Inference):\n";
        std::cout << "  Observed " << obs.prefix.size() << " states.\n";
        std::cout << "  Compatible widths found:";
        for (long w : inf.candidates) std::cout << " " << w;
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

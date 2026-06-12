// demo_ntl_kan.cpp
#ifdef USE_MOCK_NTL
#include "ntl_mock.h"
#else
#include <NTL/GF2X.h>
#include <NTL/GF2E.h>
#include <NTL/GF2XFactoring.h>
#include <NTL/GF2EX.h>
#endif

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <functional>

using namespace NTL;
using u64 = std::uint64_t;

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

static std::string show(const GF2E& a) {
    std::ostringstream oss;
    oss << a;
    return oss.str();
}

struct FieldModel {
    long n = 0;
    GF2X modulus;
    GF2E primitive;
};

static bool is_primitive_element(const GF2E& a, u64 order_minus_1) {
    if (IsZero(a)) return false;

    for (u64 p : unique_prime_factors(order_minus_1)) {
        GF2E t;
        power(t, a, static_cast<long>(order_minus_1 / p));
        if (IsOne(t)) return false;
    }

    GF2E full;
    power(full, a, static_cast<long>(order_minus_1));
    return IsOne(full);
}

static FieldModel build_field(long n) {
    if (n <= 0 || n >= 63) throw std::invalid_argument("n must be in 1..62");

    GF2X P;
    BuildSparseIrred(P, n);   // fast, NTL-optimized irreducible modulus
    GF2E::init(P);

    const u64 order_minus_1 = (u64{1} << n) - 1ULL;

    GF2E alpha;
    bool found = false;
    for (long tries = 0; tries < 5000; ++tries) {
        random(alpha);
        if (!IsZero(alpha) && is_primitive_element(alpha, order_minus_1)) {
            found = true;
            break;
        }
    }
    if (!found) throw std::runtime_error("no primitive element found");

    return FieldModel{n, P, alpha};
}

struct OrbitSequence {
    long n = 0;
    GF2X modulus;
    GF2E primitive;
    std::vector<GF2E> states;
};

static OrbitSequence construct_orbit(std::size_t length) {
    if (length == 0) return {};

    long n = 1;
    while (n < 62 && ((u64{1} << n) - 1ULL) < length) ++n;
    if (n >= 62) throw std::runtime_error("requested length too large");

    FieldModel F = build_field(n);

    OrbitSequence out;
    out.n = F.n;
    out.modulus = F.modulus;
    out.primitive = F.primitive;
    out.states.reserve(length);

    GF2E state(1);
    for (std::size_t i = 0; i < length; ++i) {
        out.states.push_back(state);
        state *= out.primitive;
    }
    return out;
}

struct Fragment {
    std::string chart;
    std::string description;
    std::vector<std::string> sample;
};

using Chart = std::function<std::optional<Fragment>(const FieldModel&)>;

struct Atlas {
    std::vector<Chart> charts;
};

static std::vector<Fragment> lan_extend(const Atlas& atlas, const FieldModel& F) {
    std::vector<Fragment> out;
    for (const auto& c : atlas.charts) {
        auto frag = c(F);
        if (frag) out.push_back(std::move(*frag));
    }
    return out;
}

static Chart polynomial_basis_chart = [](const FieldModel& F) -> std::optional<Fragment> {
    (void)F;
    Fragment frag;
    frag.chart = "polynomial-basis";
    frag.description = "orbit printed in the default GF2E coordinate chart";
    return frag;
};

static Chart jump_ahead_chart = [](const FieldModel& F) -> std::optional<Fragment> {
    Fragment frag;
    frag.chart = "jump-ahead";
    frag.description = "8-step chart using repeated field multiplication";
    GF2E s(1);
    for (int i = 0; i < 8; ++i) s *= F.primitive;
    frag.sample.push_back(show(s));
    return frag;
};

struct Observation {
    std::vector<std::string> prefix;
};

struct Evidence {
    std::string chart;
    bool compatible = false;
};

using Recognizer = std::function<std::optional<Evidence>(const Observation&, const FieldModel&)>;

static Recognizer direct_recognizer = [](const Observation& obs, const FieldModel& F) -> std::optional<Evidence> {
    GF2E state(1);
    for (const auto& s : obs.prefix) {
        if (show(state) != s) return Evidence{"direct", false};
        state *= F.primitive;
    }
    return Evidence{"direct", true};
};

static Recognizer reciprocal_recognizer = [](const Observation& obs, const FieldModel& F) -> std::optional<Evidence> {
    GF2E dual = inv(F.primitive);
    GF2E state(1);
    for (const auto& s : obs.prefix) {
        if (show(state) != s) return Evidence{"reciprocal", false};
        state *= dual;
    }
    return Evidence{"reciprocal", true};
};

struct GlobalInference {
    std::vector<std::string> compatible_charts;
};

static GlobalInference ran_extend(const std::vector<Recognizer>& recognizers,
                                  const Observation& obs,
                                  const FieldModel& F) {
    GlobalInference g;
    for (const auto& r : recognizers) {
        auto e = r(obs, F);
        if (e && e->compatible) g.compatible_charts.push_back(e->chart);
    }
    return g;
}

int main() {
    try {
        std::size_t requested_length = 37;
        auto seq = construct_orbit(requested_length);

        std::cout << "--- NTL Field Orbit Demo ---\n";
        std::cout << "Chosen width n: " << seq.n << "\n";
        std::cout << "Modulus: " << seq.modulus << "\n";
        std::cout << "Primitive element: " << seq.primitive << "\n\n";

        std::cout << "Orbit states:\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(10, seq.states.size()); ++i) {
            std::cout << std::setw(3) << i << ": " << seq.states[i] << "\n";
        }
        if (seq.states.size() > 10) std::cout << "  ...\n";

        FieldModel F{seq.n, seq.modulus, seq.primitive};

        Atlas atlas;
        atlas.charts.push_back(polynomial_basis_chart);
        atlas.charts.push_back(jump_ahead_chart);

        auto frags = lan_extend(atlas, F);
        std::cout << "\nLeft-extension fragments:\n";
        for (const auto& f : frags) {
            std::cout << "  [" << f.chart << "] " << f.description << "\n";
            for (const auto& s : f.sample) std::cout << "    sample: " << s << "\n";
        }

        Observation obs;
        for (std::size_t i = 0; i < std::min<std::size_t>(5, seq.states.size()); ++i) {
            obs.prefix.push_back(show(seq.states[i]));
        }

        std::vector<Recognizer> recognizers{direct_recognizer, reciprocal_recognizer};
        auto inf = ran_extend(recognizers, obs, F);

        std::cout << "\nRight-extension compatible charts:\n";
        for (const auto& c : inf.compatible_charts) {
            std::cout << "  " << c << "\n";
        }

        if (seq.states.size() == ((u64{1} << seq.n) - 1ULL)) {
            std::cout << "\nFull nonzero orbit: every nonzero state appears once.\n";
        } else {
            std::cout << "\nPrefix of a primitive orbit: no repeats in the requested length.\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

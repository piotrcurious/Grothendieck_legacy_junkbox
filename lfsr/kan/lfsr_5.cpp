// demo_ntl_kan_next.cpp
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

    GF2EPush scope(P); // RAII modulus guard

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

struct Fragment {
    std::string chart;
    std::string note;
    std::string sample;
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

static Chart basis_chart = [](const Orbit& orbit) -> std::optional<Fragment> {
    Fragment f;
    f.chart = "polynomial-basis";
    f.note = "field orbit in the default GF2E coordinate chart";
    f.sample = as_string(orbit.states.front());
    return f;
};

static Chart jump_chart = [](const Orbit& orbit) -> std::optional<Fragment> {
    // A toy GF2EX chart: precompute X^e mod f in GF2E[X].
    GF2EX fpoly;
    SetCoeff(fpoly, 8);  // X^8
    SetCoeff(fpoly, 1);  // + X
    SetCoeff(fpoly, 0);  // + 1

    GF2EXModulus F(fpoly);
    GF2EX jump = PowerXMod(64, F);

    Fragment f;
    f.chart = "jump-ahead";
    f.note = "GF2EX chart using PowerXMod on a preconditioned modulus";
    f.sample = as_string(jump);
    return f;
};

struct Observation {
    std::vector<std::string> prefix;
};

struct Inference {
    std::vector<long> compatible_widths;
};

using Recognizer = std::function<std::optional<long>(const Observation&, long candidate_width, const Orbit& reference)>;

static Recognizer prefix_recognizer = [](const Observation& obs, long candidate_width, const Orbit& ref) -> std::optional<long> {
    if (candidate_width != ref.n) return std::nullopt;
    if (obs.prefix.size() > ref.states.size()) return std::nullopt;

    for (std::size_t i = 0; i < obs.prefix.size(); ++i) {
        if (obs.prefix[i] != as_string(ref.states[i])) return std::nullopt;
    }
    return candidate_width;
};

static Inference ran_extend(const std::vector<Recognizer>& recognizers,
                            const Observation& obs,
                            const Orbit& ref) {
    Inference inf;
    for (long candidate = ref.n; candidate <= ref.n + 4; ++candidate) {
        bool ok = true;
        for (const auto& r : recognizers) {
            auto v = r(obs, candidate, ref);
            if (!v) {
                ok = false;
                break;
            }
        }
        if (ok) inf.compatible_widths.push_back(candidate);
    }
    return inf;
}

int main() {
    try {
        std::size_t requested_length = 37;
        Orbit orbit = build_orbit(requested_length);

        std::cout << "--- NTL Kan Orbit Demo ---\n";
        std::cout << "n = " << orbit.n << "\n";
        std::cout << "modulus = " << orbit.modulus << "\n";
        std::cout << "primitive element = " << orbit.primitive << "\n\n";

        std::cout << "orbit states:\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(10, orbit.states.size()); ++i) {
            std::cout << "  " << i << ": " << orbit.states[i] << "\n";
        }
        if (orbit.states.size() > 10) std::cout << "  ...\n";

        Atlas atlas;
        atlas.charts.push_back(basis_chart);
        atlas.charts.push_back(jump_chart);

        auto frags = lan_extend(atlas, orbit);
        std::cout << "\nleft-extension charts:\n";
        for (const auto& f : frags) {
            std::cout << "  [" << f.chart << "] " << f.note << "\n";
            std::cout << "    sample: " << f.sample << "\n";
        }

        Observation obs;
        for (std::size_t i = 0; i < std::min<std::size_t>(5, orbit.states.size()); ++i) {
            obs.prefix.push_back(as_string(orbit.states[i]));
        }

        std::vector<Recognizer> recognizers{prefix_recognizer};
        Inference inf = ran_extend(recognizers, obs, orbit);

        std::cout << "\nobserved prefix width inference:\n";
        for (long w : inf.compatible_widths) {
            std::cout << "  " << w << "\n";
        }

        if (orbit.states.size() == ((u64{1} << orbit.n) - 1ULL)) {
            std::cout << "\nfull nonzero orbit: every nonzero state appears once\n";
        } else {
            std::cout << "\nnonrepeating prefix of a primitive orbit\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

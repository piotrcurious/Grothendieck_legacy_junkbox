// lfsr_improved.cpp - Enhanced NTL LFSR Suite with Range Traversal
#include <NTL/GF2X.h>
#include <NTL/GF2E.h>
#include <NTL/GF2XFactoring.h>
#include <NTL/GF2EX.h>
#include <NTL/mat_GF2.h>

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
#include <random>
#include <set>

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

static GF2X string_to_gf2x(const std::string& s) {
    GF2X res;
    if (s.empty()) return res;
    
    std::string clean = s;
    clean.erase(std::remove(clean.begin(), clean.end(), '['), clean.end());
    clean.erase(std::remove(clean.begin(), clean.end(), ']'), clean.end());
    
    std::istringstream iss(clean);
    long coeff_val;
    long idx = 0;
    while (iss >> coeff_val) {
        if (coeff_val) SetCoeff(res, idx);
        idx++;
    }
    return res;
}

static GF2E bits_to_gf2e(u64 bits) {
    GF2X f;
    for (int i = 0; i < 64; ++i) {
        if ((bits >> i) & 1) SetCoeff(f, i);
    }
    return conv<GF2E>(f);
}

static u64 gf2x_to_bits(const GF2X& f) {
    u64 res = 0;
    for (long i = 0; i <= deg(f) && i < 64; ++i) {
        if (IsOne(coeff(f, i))) res |= (u64(1) << i);
    }
    return res;
}

// -----------------------------------------------------------------------------
// Field Model and Primitivity
// -----------------------------------------------------------------------------

struct Orbit {
    long n = 0;
    GF2X modulus;
    GF2X primitive;
    std::vector<GF2X> states;
};

static bool is_primitive_element(const GF2E& a, u64 order_minus_1) {
    if (IsZero(a)) return false;
    if (order_minus_1 == 0) return true;
    for (u64 p : unique_prime_factors(order_minus_1)) {
        GF2E t;
        power(t, a, to_ZZ(order_minus_1 / p));
        if (IsOne(t)) return false;
    }
    return true;
}

static GF2E find_primitive_element(u64 order_minus_1, long tries = 5000) {
    GF2E a;
    for (long i = 0; i < tries; ++i) {
        random(a);
        if (is_primitive_element(a, order_minus_1)) return a;
    }
    throw std::runtime_error("failed to find primitive element");
}

/**
 * build_orbit generates a full maximal length sequence (maximal orbit)
 * of an LFSR of degree n.
 */
static Orbit build_orbit(long n) {
    if (n <= 0 || n >= 62) throw std::runtime_error("invalid degree n for orbit");

    GF2X P;
    BuildSparseIrred(P, n);
    const u64 length = (u64{1} << n) - 1ULL;
    GF2EPush scope(P);
    GF2E alpha = find_primitive_element(length);

    Orbit out;
    out.n         = n;
    out.modulus   = P;
    out.primitive = rep(alpha);
    out.states.reserve(length);

    GF2E s(1);
    for (u64 i = 0; i < length; ++i) {
        out.states.push_back(rep(s));
        s *= alpha;
    }
    return out;
}

// -----------------------------------------------------------------------------
// Range Traversal (Practical Application)
// -----------------------------------------------------------------------------

class RangeTraverser {
    long n;
    GF2X modulus;
    GF2X primitive;
    u64 max_val;
    u64 current_state_bits;
    bool visited_zero;
    u64 seed;

public:
    RangeTraverser(u64 max_v, std::optional<u64> s = std::nullopt)
        : max_val(max_v), visited_zero(false)
    {
        if (s) seed = *s;
        else {
            std::random_device rd;
            seed = ((u64)rd() << 32) | rd();
        }

        SetSeed(to_ZZ(seed));

        if (max_val == 0) {
            n = 1;
            return;
        }
        
        n = 1;
        while (n < 62 && ((u64(1) << n) - 1) < max_val) n++;
        
        BuildSparseIrred(modulus, n);
        GF2EPush scope(modulus);
        primitive = rep(find_primitive_element((u64(1) << n) - 1));
        
        std::mt19937_64 gen(seed);
        std::uniform_int_distribution<u64> dis(1, (u64(1) << n) - 1);
        current_state_bits = dis(gen);
        
        while (current_state_bits > max_val) {
            step();
        }
    }

    void reset() {
        SetSeed(to_ZZ(seed));
        visited_zero = false;
        if (max_val == 0) return;
        std::mt19937_64 gen(seed);
        std::uniform_int_distribution<u64> dis(1, (u64(1) << n) - 1);
        current_state_bits = dis(gen);
        while (current_state_bits > max_val) {
            step();
        }
    }

    void step() {
        if (max_val == 0) return;
        GF2EPush scope(modulus);
        GF2E s = bits_to_gf2e(current_state_bits);
        GF2E alpha = conv<GF2E>(primitive);
        s *= alpha;
        current_state_bits = gf2x_to_bits(rep(s));
    }

    u64 next() {
        if (max_val == 0) return 0;
        if (!visited_zero) {
            visited_zero = true;
            return 0;
        }
        u64 val = current_state_bits;
        do {
            step();
        } while (current_state_bits > max_val);
        return val;
    }
};

// -----------------------------------------------------------------------------
// Inference Layer
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
        if (obs.prefix.size() < 2) return false;
        GF2X P;
        BuildSparseIrred(P, w);
        GF2EPush scope(P);
        const u64 order_minus_1 = (u64{1} << w) - 1;
        GF2X alpha_poly = string_to_gf2x(obs.prefix[1]);
        GF2E alpha = conv<GF2E>(alpha_poly);
        if (!is_primitive_element(alpha, order_minus_1)) return false;
        GF2E state(1);
        for (const auto& expected : obs.prefix) {
            if (as_string(state) != expected) return false;
            state *= alpha;
        }
        return true;
    } catch (...) {
        return false;
    }
};

static InferenceResult ran_extend(const std::vector<Recognizer>& recognizers, const Observation& obs) {
    InferenceResult res;
    for (long w = 2; w <= 32; ++w) {
        bool ok = true;
        for (const auto& r : recognizers) {
            if (!r(obs, w)) { ok = false; break; }
        }
        if (ok) res.candidates.push_back(w);
    }
    return res;
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

void test_inference() {
    std::cout << "--- 1. Inference Test ---\n";
    long n = 8;
    GF2X P; BuildSparseIrred(P, n);
    GF2EPush scope(P);
    SetSeed(to_ZZ(42));
    GF2E alpha = find_primitive_element((u64(1) << n) - 1);
    Observation obs;
    GF2E s(1);
    for(int i=0; i<5; ++i) {
        obs.prefix.push_back(as_string(s));
        s *= alpha;
    }
    auto inf = ran_extend({prefix_recognizer}, obs);
    std::cout << "Inferred width(s):";
    for(long w : inf.candidates) std::cout << " " << w;
    std::cout << "\n\n";
}

void test_range_traversal(u64 max_val, std::optional<u64> seed = std::nullopt) {
    std::cout << "--- 2. Range Traversal Demo (max_val=" << max_val << ") ---\n";
    RangeTraverser traverser(max_val, seed);
    std::vector<u64> sequence;
    for (u64 i = 0; i <= max_val; ++i) {
        sequence.push_back(traverser.next());
    }
    std::cout << "Sequence size: " << sequence.size() << "\n";
    std::set<u64> seen;
    bool all_in_range = true;
    for(u64 x : sequence) {
        if(x > max_val) all_in_range = false;
        seen.insert(x);
    }
    std::cout << "Verification: All numbers in [0, " << max_val << "] visited exactly once? "
              << (seen.size() == max_val + 1 && all_in_range ? "YES" : "NO") << "\n\n";
}

void test_build_orbit(long n) {
    std::cout << "--- 3. Full Orbit Generation (n=" << n << ") ---\n";
    Orbit orb = build_orbit(n);
    u64 expected_len = (u64(1) << n) - 1;
    std::cout << "Generated orbit length: " << orb.states.size() << " (Expected: " << expected_len << ")\n";
    std::set<u64> seen;
    for(const auto& s : orb.states) {
        seen.insert(gf2x_to_bits(s));
    }
    std::cout << "Verification: All states unique and non-zero? "
              << (seen.size() == expected_len && seen.find(0) == seen.end() ? "YES" : "NO") << "\n\n";
}

int main() {
    try {
        std::cout << "=== LFSR Suite Extended Testing ===\n\n";
        test_inference();
        test_range_traversal(20, 12345ULL);
        test_range_traversal(1000, 999ULL);
        test_range_traversal(65535, 42ULL); // Large range test
        test_build_orbit(4);
        test_build_orbit(8);
        test_build_orbit(12);

        std::cout << "--- 4. Reproducibility Test ---\n";
        u64 seed = 777;
        RangeTraverser t1(100, seed);
        RangeTraverser t2(100, seed);
        bool match = true;
        for(int i=0; i<=100; ++i) {
            if(t1.next() != t2.next()) { match = false; break; }
        }
        std::cout << "Reproducibility check: " << (match ? "PASS" : "FAIL") << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

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

// Improved: String to GF2X parser for O(n) recovery
// Handles formats like [1 0 1] or [0 1]
static GF2X string_to_gf2x(const std::string& s) {
    GF2X res;
    if (s.empty()) return res;
    
    std::string clean = s;
    // Remove brackets if present
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

// Convert GF2X back to bits
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

static bool is_primitive_element(const GF2E& a, u64 order_minus_1) {
    if (IsZero(a)) return false;
    if (order_minus_1 == 0) return true; // GF(2^1) case
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

// -----------------------------------------------------------------------------
// Range Traversal (Practical Application)
// -----------------------------------------------------------------------------

/**
 * RangeTraverser uses an LFSR to visit all numbers in [0, max_val] exactly once.
 * Since LFSR orbits of degree n cover [1, 2^n - 1], we map the range to fit.
 * If max_val is not 2^n - 1, we use "discarding" (re-stepping) to stay in range.
 */
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
        
        // Find smallest n such that 2^n - 1 >= max_val
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

        // Need to recreate the same sequence
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
        if (max_val == 0) {
            return 0;
        }

        if (!visited_zero) {
            visited_zero = true;
            return 0;
        }

        u64 val = current_state_bits;
        
        // Advance current_state_bits to the next valid state
        do {
            step();
        } while (current_state_bits > max_val);
        
        return val;
    }
};

// -----------------------------------------------------------------------------
// Inference Layer (Improved)
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

        // Recover alpha directly using the O(n) parser
        GF2X alpha_poly = string_to_gf2x(obs.prefix[1]);
        GF2E alpha = conv<GF2E>(alpha_poly);

        if (!is_primitive_element(alpha, order_minus_1)) return false;

        // Verify the sequence
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
// Main Demo & Tests
// -----------------------------------------------------------------------------

void test_inference() {
    std::cout << "--- 1. Improved Inference (O(n) Alpha Recovery) ---\n";
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
    std::cout << "Observed prefix: ";
    for(auto& str : obs.prefix) std::cout << str << " ";
    std::cout << "\nInferred width(s):";
    for(long w : inf.candidates) std::cout << " " << w;
    std::cout << "\n\n";
}

void test_range_traversal(u64 max_val, std::optional<u64> seed = std::nullopt) {
    std::cout << "--- Range Traversal Demo (max_val=" << max_val << ") ---\n";
    RangeTraverser traverser(max_val, seed);

    std::vector<u64> sequence;
    for (u64 i = 0; i <= max_val; ++i) {
        u64 val = traverser.next();
        sequence.push_back(val);
    }

    std::cout << "Sequence (first 20): ";
    for(size_t i=0; i<std::min<size_t>(sequence.size(), 20); ++i) std::cout << sequence[i] << " ";
    if (sequence.size() > 20) std::cout << "...";
    std::cout << "\n";

    // Verify uniqueness and completeness
    std::set<u64> seen;
    bool all_in_range = true;
    for(u64 x : sequence) {
        if(x > max_val) all_in_range = false;
        seen.insert(x);
    }

    std::cout << "Verification: All numbers in [0, " << max_val << "] visited exactly once? "
              << (seen.size() == max_val + 1 && all_in_range ? "YES" : "NO") << "\n\n";
}

int main() {
    try {
        std::cout << "=== Improved & Extended LFSR Suite ===\n\n";

        test_inference();

        test_range_traversal(20, 12345ULL);
        test_range_traversal(100, 54321ULL);
        test_range_traversal(1000, 999LL);
        test_range_traversal(0);

        std::cout << "--- 3. Reproducibility Test ---\n";
        u64 seed = 42;
        RangeTraverser t1(50, seed);
        RangeTraverser t2(50, seed);
        bool match = true;
        for(int i=0; i<=50; ++i) {
            u64 v1 = t1.next();
            u64 v2 = t2.next();
            if(v1 != v2) {
                match = false;
                break;
            }
        }
        std::cout << "Two traversers with same seed " << seed << " match? " << (match ? "YES" : "NO") << "\n\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

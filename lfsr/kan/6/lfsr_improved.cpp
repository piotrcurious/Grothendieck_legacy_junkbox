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
static GF2X string_to_gf2x(const std::string& s) {
    GF2X res;
    if (s.empty()) return res;
    
    // NTL format is usually [c0 c1 c2 ...]
    std::string clean = s;
    clean.erase(std::remove(clean.begin(), clean.end(), '['), clean.end());
    clean.erase(std::remove(clean.begin(), clean.end(), ']'), clean.end());
    
    std::istringstream iss(clean);
    long coeff;
    long idx = 0;
    while (iss >> coeff) {
        if (coeff) SetCoeff(res, idx);
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

struct Orbit {
    long n = 0;
    GF2X modulus;
    GF2X primitive;           
    std::vector<GF2X> states; 
};

static bool is_primitive_element(const GF2E& a, u64 order_minus_1) {
    if (IsZero(a)) return false;
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

static Orbit build_orbit(std::size_t length) {
    if (length == 0) return {};
    long n = 1;
    while (n < 62 && ((u64{1} << n) - 1ULL) < length) ++n;
    if (n >= 62) throw std::runtime_error("requested length too large");

    GF2X P;
    BuildSparseIrred(P, n);
    const u64 order_minus_1 = (u64{1} << n) - 1ULL;
    GF2EPush scope(P); 
    GF2E alpha = find_primitive_element(order_minus_1);

    Orbit out;
    out.n         = n;
    out.modulus   = P;
    out.primitive = rep(alpha); 
    out.states.reserve(length);

    GF2E s(1);
    for (std::size_t i = 0; i < length; ++i) {
        out.states.push_back(rep(s)); 
        s *= alpha;
    }
    return out;
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

public:
    RangeTraverser(u64 max_v) : max_val(max_v), visited_zero(false) {
        if (max_val == 0) throw std::runtime_error("Range must be at least [0, 0]");
        
        // Find smallest n such that 2^n - 1 >= max_val
        n = 1;
        while (n < 62 && ((u64(1) << n) - 1) < max_val) n++;
        
        BuildSparseIrred(modulus, n);
        GF2EPush scope(modulus);
        primitive = rep(find_primitive_element((u64(1) << n) - 1));
        
        // Start at a random state in [1, 2^n - 1]
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<u64> dis(1, (u64(1) << n) - 1);
        current_state_bits = dis(gen);
        
        // If the initial random state is out of range, step until it is in range
        while (current_state_bits > max_val) {
            step();
        }
    }

    void step() {
        GF2EPush scope(modulus);
        GF2E s = bits_to_gf2e(current_state_bits);
        GF2E alpha = conv<GF2E>(primitive);
        s *= alpha;
        current_state_bits = gf2x_to_bits(rep(s));
    }

    u64 next() {
        // Special case: handle 0 manually since LFSR never hits 0
        if (!visited_zero) {
            visited_zero = true;
            return 0;
        }

        u64 val = current_state_bits;
        
        // Advance to next valid state in range [1, max_val]
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

        // Verify identity
        GF2E identity(1);
        if (as_string(identity) != obs.prefix[0]) return false;

        // Improved: Recover alpha directly using the O(n) parser
        GF2X alpha_poly = string_to_gf2x(obs.prefix[1]);
        GF2E alpha = conv<GF2E>(alpha_poly);

        if (!is_primitive_element(alpha, order_minus_1)) return false;

        // Verify the rest of the sequence
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
        std::cout << "=== Improved & Extended LFSR Suite ===\n\n";

        // 1. Demonstrate Improved Inference
        std::cout << "--- 1. Improved Inference (O(n) Alpha Recovery) ---\n";
        long n = 6;
        GF2X P; BuildSparseIrred(P, n);
        GF2EPush scope(P);
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

        // 2. Demonstrate Range Traversal
        std::cout << "--- 2. Range Traversal (Traversing [0, 20] Pseudo-randomly) ---\n";
        u64 max_val = 20;
        RangeTraverser traverser(max_val);
        
        std::cout << "Sequence: ";
        std::vector<u64> sequence;
        for (u64 i = 0; i <= max_val; ++i) {
            u64 val = traverser.next();
            sequence.push_back(val);
            std::cout << val << " ";
        }
        std::cout << "\n";

        // Verify uniqueness
        std::sort(sequence.begin(), sequence.end());
        bool all_unique = true;
        for(size_t i=1; i<sequence.size(); ++i) {
            if(sequence[i] == sequence[i-1]) all_unique = false;
        }
        std::cout << "Verification: All numbers in [0, " << max_val << "] visited exactly once? " 
                  << (all_unique ? "YES" : "NO") << "\n\n";

        // 3. Large Range Demo
        std::cout << "--- 3. Large Range Traversal (Traversing [0, 1000] first 10 steps) ---\n";
        RangeTraverser large_traverser(1000);
        std::cout << "First 10 steps: ";
        for(int i=0; i<10; ++i) std::cout << large_traverser.next() << " ";
        std::cout << "...\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

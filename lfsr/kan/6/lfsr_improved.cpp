// lfsr_improved.cpp - Enhanced LFSR Suite with Quotient Geometry Traversal
#include <NTL/GF2X.h>
#include <NTL/GF2E.h>
#include <NTL/GF2XFactoring.h>
#include <NTL/GF2EX.h>

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
#include <numeric>

using namespace NTL;
using u64 = std::uint64_t;

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

static std::vector<u64> unique_prime_factors(u64 n) {
    std::vector<u64> f;
    u64 d = n;
    if (d == 0) return f;
    for (u64 p = 2; p * p <= d; ++p) {
        if (d % p == 0) {
            f.push_back(p);
            while (d % p == 0) d /= p;
        }
    }
    if (d > 1) f.push_back(d);
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

static bool poly_less(const GF2X& a, const GF2X& b) {
    long d1 = deg(a), d2 = deg(b);
    if (d1 != d2) return d1 < d2;
    for (long i = d1; i >= 0; --i) {
        bool c1 = IsOne(coeff(a, i));
        bool c2 = IsOne(coeff(b, i));
        if (c1 != c2) return c2;
    }
    return false;
}

// -----------------------------------------------------------------------------
// Geometric Traverser (The Algebraic Redesign)
// -----------------------------------------------------------------------------

/**
 * GeometricTraverser visits all numbers in [0, N-1] exactly once.
 * For odd N, it uses Quotient Geometry:
 * 1. Find n s.t. N | 2^n - 1.
 * 2. In GF(2^n), let alpha be primitive.
 * 3. zeta = alpha^((2^n-1)/N) has order N.
 * 4. Traversal is multiplication in the subgroup <zeta>.
 * 5. Output is a bijection from <zeta> to [0, N-1].
 *
 * For even N = 2^k * M, it combines a 2^k counter with a GeometricTraverser for M.
 */
class GeometricTraverser {
    u64 N_total;
    u64 k_pow2;
    u64 M_odd;

    // Algebraic state for M_odd
    long field_n;
    GF2X modulus;
    GF2E zeta_step;
    GF2E current_zeta;
    GF2E initial_zeta;

    struct RankPair {
        GF2X poly;
        u64 rank;
        bool operator<(const RankPair& other) const {
            return poly_less(poly, other.poly);
        }
    };
    std::vector<RankPair> rank_map;

    // State for 2^k
    u64 count_2k;
    u64 initial_count_2k;
    u64 seed;

    void init_odd_part(u64 M) {
        if (M <= 1) {
            field_n = 0;
            return;
        }

        // 1. Find smallest n s.t. M | 2^n - 1
        field_n = -1;
        for (long n = 1; n <= 1024; ++n) {
            ZZ res = PowerMod(to_ZZ(2), to_ZZ(n), to_ZZ(M));
            if (res == 1) {
                field_n = n;
                break;
            }
        }
        if (field_n == -1) {
            throw std::runtime_error("No suitable field degree found for N=" + std::to_string(N_total));
        }

        // 2. Build GF(2^field_n)
        BuildSparseIrred(modulus, field_n);
        GF2EPush scope(modulus);

        // 3. Find primitive element alpha
        SetSeed(to_ZZ(seed));
        ZZ order = (power_ZZ(2, field_n) - 1);
        auto factors = unique_prime_factors_ZZ(order);

        GF2E alpha;
        for (long i = 0; i < 5000; ++i) {
            random(alpha);
            if (IsZero(alpha)) continue;
            bool is_prim = true;
            for (const auto& p : factors) {
                GF2E t;
                power(t, alpha, order / p);
                if (IsOne(t)) { is_prim = false; break; }
            }
            if (is_prim) break;
        }

        // 4. zeta = alpha^(order/M)
        GF2E zeta;
        power(zeta, alpha, order / to_ZZ(M));

        // 5. Generate all M roots of unity and build rank map
        std::vector<GF2X> roots;
        roots.reserve(M);
        GF2E cur_z(1);
        for (u64 i = 0; i < M; ++i) {
            roots.push_back(rep(cur_z));
            cur_z *= zeta;
        }

        std::vector<GF2X> sorted_roots = roots;
        std::sort(sorted_roots.begin(), sorted_roots.end(), poly_less);

        rank_map.clear();
        for (u64 i = 0; i < M; ++i) {
            auto it = std::lower_bound(sorted_roots.begin(), sorted_roots.end(), roots[i], poly_less);
            rank_map.push_back({roots[i], (u64)std::distance(sorted_roots.begin(), it)});
        }
        std::sort(rank_map.begin(), rank_map.end());

        // 6. Setup step and initial state
        std::mt19937_64 gen(seed);
        u64 j = 1;
        if (M > 1) {
            do { j = (gen() % (M - 1)) + 1; } while (std::gcd(j, M) != 1);
        }
        power(zeta_step, zeta, to_ZZ(j));

        u64 start_pow = gen() % M;
        power(initial_zeta, zeta, to_ZZ(start_pow));
        current_zeta = initial_zeta;
    }

    u64 get_rank(const GF2E& s) {
        GF2X p = rep(s);
        auto it = std::lower_bound(rank_map.begin(), rank_map.end(), RankPair{p, 0});
        return it->rank;
    }

    std::vector<ZZ> unique_prime_factors_ZZ(ZZ n) {
        std::vector<ZZ> factors;
        ZZ d = n;
        ZZ p = to_ZZ(2);
        while (p * p <= d) {
            if (d % p == 0) {
                factors.push_back(p);
                while (d % p == 0) d /= p;
            }
            p++;
        }
        if (d > 1) factors.push_back(d);
        return factors;
    }

public:
    GeometricTraverser(u64 N, std::optional<u64> s = std::nullopt) : N_total(N) {
        if (N == 0) throw std::runtime_error("N must be > 0");

        if (s) seed = *s;
        else {
            std::random_device rd;
            seed = ((u64)rd() << 32) | rd();
        }

        k_pow2 = 0;
        M_odd = N;
        while (M_odd > 0 && M_odd % 2 == 0) {
            M_odd /= 2;
            k_pow2++;
        }

        init_odd_part(M_odd);

        std::mt19937_64 gen(seed);
        initial_count_2k = (k_pow2 > 0) ? (gen() % (1ULL << k_pow2)) : 0;
        count_2k = initial_count_2k;
    }

    void reset() {
        current_zeta = initial_zeta;
        count_2k = initial_count_2k;
    }

    u64 next() {
        u64 res;
        if (M_odd <= 1) {
            res = count_2k;
            count_2k = (count_2k + 1) % (1ULL << k_pow2);
        } else {
            GF2EPush scope(modulus);
            u64 r = get_rank(current_zeta);
            res = (r << k_pow2) | count_2k;

            count_2k++;
            if (k_pow2 < 64 && count_2k == (1ULL << k_pow2)) {
                count_2k = 0;
                current_zeta *= zeta_step;
            } else if (k_pow2 >= 64) {
                // This shouldn't happen with u64 N
            }
        }
        return res;
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
        GF2X P; BuildSparseIrred(P, w);
        GF2EPush scope(P);
        ZZ order = (power_ZZ(2, w) - 1);
        GF2X alpha_poly = string_to_gf2x(obs.prefix[1]);
        GF2E alpha = conv<GF2E>(alpha_poly);
        if (IsZero(alpha)) return false;

        // Primitivity check
        ZZ d = order;
        ZZ p = to_ZZ(2);
        while (p * p <= d) {
            if (d % p == 0) {
                GF2E t; power(t, alpha, order / p);
                if (IsOne(t)) return false;
                while (d % p == 0) d /= p;
            }
            p++;
        }
        if (d > 1) {
            GF2E t; power(t, alpha, order / d);
            if (IsOne(t)) return false;
        }

        GF2E state(1);
        for (const auto& expected : obs.prefix) {
            if (as_string(state) != expected) return false;
            state *= alpha;
        }
        return true;
    } catch (...) { return false; }
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
// Main Demo
// -----------------------------------------------------------------------------

void test_geometric_traversal(u64 N, std::optional<u64> seed = std::nullopt) {
    std::cout << "--- Geometric Traversal (N=" << N << ") ---\n";
    try {
        GeometricTraverser traverser(N, seed);
        std::vector<u64> sequence;
        for (u64 i = 0; i < N; ++i) sequence.push_back(traverser.next());

        std::cout << "First 15: ";
        for(size_t i=0; i<std::min<size_t>(sequence.size(), 15); ++i) std::cout << sequence[i] << " ";
        std::cout << "\n";

        std::set<u64> seen;
        bool all_in_range = true;
        for(u64 x : sequence) {
            if(x >= N) all_in_range = false;
            seen.insert(x);
        }
        std::cout << "Verification: Size=" << sequence.size() << ", Unique=" << seen.size()
                  << ", All in range? " << (all_in_range ? "YES" : "NO") << "\n";
        std::cout << "Traversal Complete? " << (seen.size() == N ? "YES" : "NO") << "\n\n";
    } catch (std::exception& e) {
        std::cout << "Error: " << e.what() << "\n\n";
    }
}

int main() {
    try {
        std::cout << "=== Improved LFSR Suite: Quotient Geometry Edition ===\n\n";

        // 1. Functional Inference Demo
        std::cout << "--- 1. Inference Demo (GF(2^8)) ---\n";
        {
            GF2X P_test; BuildSparseIrred(P_test, 8);
            GF2EPush scope(P_test);
            SetSeed(to_ZZ(12345));
            GF2E alpha;
            ZZ ord = to_ZZ(255);
            do {
                random(alpha);
                bool ok = true;
                if (IsZero(alpha)) ok = false;
                else {
                    for(long p : {3, 5, 17}) {
                        GF2E t; power(t, alpha, ord/p);
                        if(IsOne(t)) { ok = false; break; }
                    }
                }
                if (ok) break;
            } while(true);

            Observation obs; GF2E s(1);
            for(int i=0; i<5; ++i) { obs.prefix.push_back(as_string(s)); s *= alpha; }

            auto res = ran_extend({prefix_recognizer}, obs);
            std::cout << "Observed prefix: ";
            for(auto& p : obs.prefix) std::cout << p << " ";
            std::cout << "\nInferred field width(s): ";
            for(long w : res.candidates) std::cout << w << " ";
            std::cout << "\n\n";
        }

        // 2. Geometric Traversal (No rejection sampling)
        test_geometric_traversal(100);    // Even N
        test_geometric_traversal(65535);  // Large Mersenne-like N (n=16)
        test_geometric_traversal(67);     // Odd N where n=66 (> 64)
        test_geometric_traversal(1024);   // Power of 2

        // Reproducibility
        std::cout << "--- 3. Reproducibility Test ---\n";
        u64 seed = 42;
        GeometricTraverser t1(50, seed), t2(50, seed);
        bool match = true;
        for(int i=0; i<50; ++i) if(t1.next() != t2.next()) match = false;
        std::cout << "Seeded traversers match? " << (match ? "YES" : "NO") << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Global Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

// lfsr_improved.cpp - Enhanced LFSR Suite: Algebraic Geography Edition
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
#include <numeric>
#include <map>

using namespace NTL;
using u64 = std::uint64_t;

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

static std::vector<ZZ> unique_prime_factors_ZZ(ZZ n) {
    std::vector<ZZ> factors;
    ZZ d = n;
    if (d <= 1) return factors;
    for (long p : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}) {
        if (d % p == 0) {
            factors.push_back(to_ZZ(p));
            while (d % p == 0) d /= p;
        }
    }
    if (d > 1) factors.push_back(d);
    return factors;
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

static GF2X bits_to_gf2x(u64 bits) {
    GF2X f;
    for (int i = 0; i < 64; ++i) if ((bits >> i) & 1) SetCoeff(f, i);
    return f;
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

static GF2E find_minimal_primitive_element(const ZZ& order, const std::vector<ZZ>& factors) {
    for (long i = 2; i < 65536; ++i) {
        GF2X p = bits_to_gf2x(i);
        GF2E alpha = conv<GF2E>(p);
        if (IsZero(alpha)) continue;
        bool ok = true;
        for (const auto& f : factors) {
            GF2E t; power(t, alpha, order / f);
            if (IsOne(t)) { ok = false; break; }
        }
        if (ok) return alpha;
    }
    throw std::runtime_error("Minimal primitive element not found");
}

// -----------------------------------------------------------------------------
// Geometric Atlas & Morphisms
// -----------------------------------------------------------------------------

class GeometricAtlas {
    GF2X modulus;
    mat_GF2 trace_map_inv;
    bool has_trace_map = false;

public:
    struct Chart {
        std::string name;
        std::function<u64(const GF2E&)> projection;
    };

    void init(const GF2X& mod) {
        modulus = mod;
        long n = deg(mod);
        GF2EPush scope(mod);
        mat_GF2 A; A.SetDims(n, n);
        GF2E alpha = conv<GF2E>(bits_to_gf2x(2));
        for (long i = 0; i < n; ++i) {
            for (long j = 0; j < n; ++j) {
                GF2E term; power(term, alpha, to_ZZ(i + j));
                if (IsOne(trace(term))) A[i][j] = 1;
            }
        }
        inv(trace_map_inv, A);
        has_trace_map = true;
    }

    static Chart CompanionChart() {
        return {"Companion", [](const GF2E& s) {
            return IsOne(coeff(rep(s), 0)) ? 1ULL : 0ULL;
        }};
    }

    static Chart TraceChart() {
        return {"Trace", [](const GF2E& s) {
            return IsOne(trace(s)) ? 1ULL : 0ULL;
        }};
    }

    GF2E reconstruct_from_trace(const std::vector<int>& bits) {
        if (!has_trace_map) throw std::runtime_error("Atlas not initialized");
        long n = deg(modulus);
        vec_GF2 b; b.SetLength(n);
        for (long i = 0; i < n; ++i) if (bits[i]) b[i] = 1;
        vec_GF2 x = trace_map_inv * b;
        GF2X res;
        for (long j = 0; j < n; ++j) if (IsOne(x[j])) SetCoeff(res, j);
        GF2EPush scope(modulus);
        return conv<GF2E>(res);
    }
};

// -----------------------------------------------------------------------------
// Geometric Traverser (Optimized for Scalability)
// -----------------------------------------------------------------------------

class GeometricTraverser {
    u64 N_total; u64 k_pow2; u64 M_odd;
    long field_n; GF2X modulus; GF2E zeta_step; GF2E current_zeta; GF2E initial_zeta;

    struct RankPair {
        GF2X poly; u64 rank;
        bool operator<(const RankPair& other) const { return poly_less(poly, other.poly); }
    };
    std::vector<RankPair> rank_map;
    bool use_lazy_rank = false;
    u64 count_2k; u64 initial_count_2k; u64 total_steps; u64 seed;

    void init_odd_part(u64 M) {
        if (M <= 1) { field_n = 0; return; }
        field_n = -1;
        for (long n = 1; n <= 1024; ++n) {
            if (PowerMod(to_ZZ(2), to_ZZ(n), to_ZZ(M)) == 1) { field_n = n; break; }
        }
        if (field_n == -1) throw std::runtime_error("Field search failed for M=" + std::to_string(M));

        BuildSparseIrred(modulus, field_n);
        GF2EPush scope(modulus);
        ZZ order = (power_ZZ(2, field_n) - 1);
        GF2E alpha = find_minimal_primitive_element(order, unique_prime_factors_ZZ(order));
        GF2E zeta; power(zeta, alpha, order / to_ZZ(M));

        if (M < 200000) {
            std::vector<GF2X> roots; roots.reserve(M);
            GF2E cur_z(1);
            for (u64 i = 0; i < M; ++i) { roots.push_back(rep(cur_z)); cur_z *= zeta; }
            std::vector<GF2X> sorted_roots = roots;
            std::sort(sorted_roots.begin(), sorted_roots.end(), poly_less);
            rank_map.clear();
            for (u64 i = 0; i < M; ++i) {
                auto it = std::lower_bound(sorted_roots.begin(), sorted_roots.end(), roots[i], poly_less);
                rank_map.push_back({roots[i], (u64)std::distance(sorted_roots.begin(), it)});
            }
            std::sort(rank_map.begin(), rank_map.end());
        } else {
            use_lazy_rank = true;
        }

        zeta_step = zeta;
        std::mt19937_64 gen(seed);
        u64 start_pow = gen() % M;
        power(initial_zeta, zeta, to_ZZ(start_pow));
        current_zeta = initial_zeta;
    }

    u64 get_rank(const GF2E& s) {
        if (use_lazy_rank) return static_cast<u64>(std::hash<std::string>{}(as_string(s)));
        GF2X p = rep(s);
        auto it = std::lower_bound(rank_map.begin(), rank_map.end(), RankPair{p, 0});
        return it->rank;
    }

public:
    GeometricTraverser(u64 N, std::optional<u64> s = std::nullopt) : N_total(N), total_steps(0) {
        if (N == 0) throw std::runtime_error("N > 0");
        seed = s ? *s : (((u64)std::random_device{}() << 32) | std::random_device{}());
        M_odd = N; k_pow2 = 0;
        while (M_odd > 0 && (M_odd & 1) == 0) { M_odd >>= 1; k_pow2++; }
        init_odd_part(M_odd);
        initial_count_2k = (k_pow2 > 0) ? (std::mt19937_64(seed)() % (1ULL << k_pow2)) : 0;
        count_2k = initial_count_2k;
    }

    void seek(u64 pos) {
        if (pos >= N_total) pos %= N_total;
        total_steps = pos;
        u64 c_val = (initial_count_2k + pos);
        u64 cycles = c_val >> k_pow2;
        count_2k = c_val & ((k_pow2 >= 64) ? ~0ULL : ((1ULL << k_pow2) - 1));
        if (M_odd > 1) {
            GF2EPush scope(modulus); GF2E shift;
            power(shift, zeta_step, to_ZZ(cycles));
            current_zeta = initial_zeta * shift;
        }
    }

    u64 next() {
        u64 r = (M_odd <= 1) ? 0 : get_rank(current_zeta);
        u64 res = (r << k_pow2) | count_2k;
        total_steps = (total_steps + 1) % N_total;
        count_2k++;
        if (k_pow2 < 64 && count_2k == (1ULL << k_pow2)) {
            count_2k = 0;
            if (M_odd > 1) { GF2EPush scope(modulus); current_zeta *= zeta_step; }
        }
        return res;
    }

    const GF2X& get_modulus() const { return modulus; }
    const GF2E& get_algebraic_state() const { return current_zeta; }
};

// -----------------------------------------------------------------------------
// Kan Certificates & Primitive Locus
// -----------------------------------------------------------------------------

static long kan_certificate(long n) {
    // Number of bits needed to uniquely identify the primitive polynomial of degree n
    return 2 * n;
}

static std::vector<GF2X> find_all_primitive_polynomials(long n) {
    std::vector<GF2X> locus;
    if (n > 16) return locus;
    ZZ order = (power_ZZ(2, n) - 1);
    auto factors = unique_prime_factors_ZZ(order);
    for (long i = 0; i < (1L << n); ++i) {
        GF2X p; SetCoeff(p, n);
        for (long j = 0; j < n; ++j) if ((i >> j) & 1) SetCoeff(p, j);
        if (IsZero(coeff(p, 0))) continue;
        if (IterIrredTest(p)) {
            GF2EPush scope(p);
            GF2E x = conv<GF2E>(bits_to_gf2x(2));
            bool prim = true;
            for (const auto& f : factors) {
                GF2E t; power(t, x, order / f);
                if (IsOne(t)) { prim = false; break; }
            }
            if (prim) locus.push_back(p);
        }
    }
    return locus;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
    try {
        std::cout << "=== LFSR Suite: Advanced Algebraic Geography ===\n\n";

        std::cout << "--- 1. Optimized Trace Reconstruction ---\n";
        {
            GF2X mod; BuildSparseIrred(mod, 10);
            GeometricAtlas atlas; atlas.init(mod);
            GF2EPush scope(mod);
            GF2E original_state; random(original_state);
            std::vector<int> trace_bits;
            GF2E alpha = conv<GF2E>(bits_to_gf2x(2));
            for (int i = 0; i < 10; ++i) {
                GF2E s; power(s, alpha, to_ZZ(i));
                trace_bits.push_back(IsOne(trace(s * original_state)) ? 1 : 0);
            }
            GF2E reconstructed = atlas.reconstruct_from_trace(trace_bits);
            std::cout << "Original:      " << original_state << "\n";
            std::cout << "Reconstructed: " << reconstructed << " ";
            std::cout << (reconstructed == original_state ? "(SUCCESS)" : "(FAIL)") << "\n\n";
        }

        std::cout << "--- 2. Primitive Locus (n=4) ---\n";
        auto locus = find_all_primitive_polynomials(4);
        std::cout << "Found " << locus.size() << " primitive polynomials of degree 4.\n";
        for(auto& p : locus) std::cout << "  " << p << "\n";
        std::cout << "\n";

        std::cout << "--- 3. Kan Certificates ---\n";
        std::cout << "Bits required for unique identification of degree 8 LFSR: " << kan_certificate(8) << "\n\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n"; return 1;
    }
    return 0;
}

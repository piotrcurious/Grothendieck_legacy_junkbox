// lfsr_improved.cpp - Enhanced LFSR Suite: Canonical Kan Extensions
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
public:
    struct Chart {
        std::string name;
        std::function<u64(const GF2E&)> projection;
    };

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

    static GF2E reconstruct_from_trace(const std::vector<int>& bits, const GF2X& mod) {
        long n = deg(mod);
        if ((long)bits.size() < n) throw std::runtime_error("Insufficient bits");

        mat_GF2 A; A.SetDims(n, n);
        vec_GF2 b; b.SetLength(n);

        GF2EPush scope(mod);
        GF2E alpha = conv<GF2E>(bits_to_gf2x(2)); // x
        for (long i = 0; i < n; ++i) {
            for (long j = 0; j < n; ++j) {
                GF2E term; power(term, alpha, to_ZZ(i + j));
                if (IsOne(trace(term))) A[i][j] = 1;
            }
            if (bits[i]) b[i] = 1;
        }

        GF2 det; vec_GF2 x;
        solve(det, x, A, b);
        if (IsZero(det)) throw std::runtime_error("Singular system");

        GF2X res;
        for (long j = 0; j < n; ++j) if (IsOne(x[j])) SetCoeff(res, j);
        return conv<GF2E>(res);
    }
};

// -----------------------------------------------------------------------------
// Geometric Traverser (Canonical Construction)
// -----------------------------------------------------------------------------

class GeometricTraverser {
    u64 N_total; u64 k_pow2; u64 M_odd;
    long field_n; GF2X modulus; GF2E zeta_step; GF2E current_zeta; GF2E initial_zeta;

    struct RankPair {
        GF2X poly; u64 rank;
        bool operator<(const RankPair& other) const { return poly_less(poly, other.poly); }
    };
    std::vector<RankPair> rank_map;
    u64 count_2k; u64 initial_count_2k; u64 total_steps; u64 seed;

    void init_odd_part(u64 M) {
        if (M <= 1) { field_n = 0; return; }
        field_n = -1;
        for (long n = 1; n <= 1024; ++n) {
            if (PowerMod(to_ZZ(2), to_ZZ(n), to_ZZ(M)) == 1) { field_n = n; break; }
        }
        if (field_n == -1) throw std::runtime_error("Field search failed");

        BuildSparseIrred(modulus, field_n);
        GF2EPush scope(modulus);
        ZZ order = (power_ZZ(2, field_n) - 1);
        GF2E alpha = find_minimal_primitive_element(order, unique_prime_factors_ZZ(order));

        GF2E zeta; power(zeta, alpha, order / to_ZZ(M));
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
        zeta_step = zeta;

        std::mt19937_64 gen(seed);
        u64 start_pow = gen() % M;
        power(initial_zeta, zeta, to_ZZ(start_pow));
        current_zeta = initial_zeta;
    }

    u64 get_rank(const GF2E& s) {
        GF2X p = rep(s);
        auto it = std::lower_bound(rank_map.begin(), rank_map.end(), RankPair{p, 0});
        return it->rank;
    }

public:
    GeometricTraverser(u64 N, std::optional<u64> s = std::nullopt) : N_total(N), total_steps(0) {
        if (N == 0) throw std::runtime_error("N > 0");
        seed = s ? *s : (((u64)std::random_device{}() << 32) | std::random_device{}());
        M_odd = N; k_pow2 = 0;
        while (M_odd > 0 && M_odd % 2 == 0) { M_odd /= 2; k_pow2++; }
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
// Inference Engine (Ran Extension)
// -----------------------------------------------------------------------------

struct InferenceResult {
    std::vector<long> candidates;
};

static InferenceResult run_multi_chart_inference(const std::vector<int>& companion_bits,
                                               const std::vector<int>& trace_bits, long max_w) {
    InferenceResult res;
    for (long w = 2; w <= max_w; ++w) {
        GF2X P; BuildSparseIrred(P, w);
        GF2EPush scope(P);
        ZZ order = (power_ZZ(2, w) - 1);
        auto factors = unique_prime_factors_ZZ(order);

        for (long i = 1; i < 512; ++i) {
            GF2E alpha = conv<GF2E>(bits_to_gf2x(i));
            if (IsZero(alpha)) continue;
            bool ok = true;
            for (const auto& f : factors) {
                GF2E t; power(t, alpha, order / f);
                if (IsOne(t)) { ok = false; break; }
            }
            if (!ok) continue;

            GF2E s(1); bool match = true;
            for (size_t j=0; j<companion_bits.size(); ++j) {
                if ((IsOne(coeff(rep(s), 0)) ? 1 : 0) != companion_bits[j]) { match = false; break; }
                s *= alpha;
            }
            if (!match) continue;

            s = GF2E(1);
            for (size_t j=0; j<trace_bits.size(); ++j) {
                if (IsOne(trace(s)) != (trace_bits[j] != 0)) { match = false; break; }
                s *= alpha;
            }
            if (match) { res.candidates.push_back(w); break; }
        }
    }
    return res;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
    try {
        std::cout << "=== LFSR Suite: Canonical Kan Extensions ===\n\n";

        std::cout << "--- 1. Chart Morphism (Trace -> Full State) ---\n";
        {
            GF2X mod; BuildSparseIrred(mod, 8);
            GF2EPush scope(mod);
            GF2E original_state; random(original_state);
            std::cout << "Original state: " << original_state << "\n";
            std::vector<int> trace_bits;
            GF2E alpha = conv<GF2E>(bits_to_gf2x(2));
            for (int i = 0; i < 8; ++i) {
                GF2E s; power(s, alpha, to_ZZ(i));
                trace_bits.push_back(IsOne(trace(s * original_state)) ? 1 : 0);
            }
            GF2E reconstructed = GeometricAtlas::reconstruct_from_trace(trace_bits, mod);
            std::cout << "Reconstructed:  " << reconstructed << " "
                      << (reconstructed == original_state ? "(SUCCESS)" : "(FAIL)") << "\n\n";
        }

        std::cout << "--- 2. Canonical Construction (Determinism) ---\n";
        {
            u64 N = 127;
            GeometricTraverser t1(N, 42ULL), t2(N, 42ULL);
            bool match = true;
            for(int i=0; i<N; ++i) if(t1.next() != t2.next()) match = false;
            std::cout << "N=127: Seeded instances are bit-identical? " << (match ? "YES" : "NO") << "\n\n";
        }

        std::cout << "--- 3. Multi-Chart Consensus Inference ---\n";
        {
            // Observations from GF(2^3) with p(x) = x^3 + x + 1 (sparse irred for n=3)
            // alpha = x = [0 1]
            // Orbit: 1, x, x^2, x+1, x^2+x, x^2+x+1, x^2+1
            // Companion bit-0: 1, 0, 0, 1, 0, 1, 1
            // Trace (aes): Tr(1)=1, Tr(x)=0, Tr(x^2)=0, Tr(x^3=x+1)=Tr(x)+Tr(1)=1...
            // Let's actually generate them to be sure.
            GF2X p_gen; BuildSparseIrred(p_gen, 3);
            GF2EPush scope(p_gen);
            GF2E a = conv<GF2E>(bits_to_gf2x(2));
            GF2E s(1);
            std::vector<int> c_obs, t_obs;
            for(int i=0; i<7; i++) {
                c_obs.push_back(IsOne(coeff(rep(s), 0)) ? 1 : 0);
                t_obs.push_back(IsOne(trace(s)) ? 1 : 0);
                s *= a;
            }

            auto res = run_multi_chart_inference(c_obs, t_obs, 8);
            std::cout << "Inferred field width(s) from consensus: ";
            for(long w : res.candidates) std::cout << w << " ";
            std::cout << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n"; return 1;
    }
    return 0;
}

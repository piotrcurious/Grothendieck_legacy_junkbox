// lfsr_improved.cpp - Enhanced LFSR Suite with Quotient Geometry and Atlas views
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
    for (long p : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}) {
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
// Geometric Atlas & Charts
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

    static Chart MatrixChart(const GF2X& mod) {
        long n = deg(mod);
        return {"Matrix-Form", [n](const GF2E& s) {
            return IsOne(coeff(rep(s), n-1)) ? 1ULL : 0ULL;
        }};
    }
};

// -----------------------------------------------------------------------------
// Geometric Traverser (The Algebraic Redesign)
// -----------------------------------------------------------------------------

class GeometricTraverser {
    u64 N_total;
    u64 k_pow2;
    u64 M_odd;

    long field_n;
    GF2X modulus;
    GF2E zeta_step;
    GF2E current_zeta;
    GF2E initial_zeta;

    struct RankPair {
        GF2X poly;
        u64 rank;
        bool operator<(const RankPair& other) const { return poly_less(poly, other.poly); }
    };
    std::vector<RankPair> rank_map;

    u64 count_2k;
    u64 initial_count_2k;
    u64 total_steps;
    u64 seed;

    void init_odd_part(u64 M) {
        if (M <= 1) { field_n = 0; return; }
        field_n = -1;
        for (long n = 1; n <= 1024; ++n) {
            if (PowerMod(to_ZZ(2), to_ZZ(n), to_ZZ(M)) == 1) { field_n = n; break; }
        }
        if (field_n == -1) throw std::runtime_error("Field degree search failed");

        BuildSparseIrred(modulus, field_n);
        GF2EPush scope(modulus);

        SetSeed(to_ZZ(seed));
        ZZ order = (power_ZZ(2, field_n) - 1);
        auto factors = unique_prime_factors_ZZ(order);

        GF2E alpha;
        for (long i = 0; i < 5000; ++i) {
            random(alpha);
            if (IsZero(alpha)) continue;
            bool ok = true;
            for (const auto& p : factors) {
                GF2E t; power(t, alpha, order / p);
                if (IsOne(t)) { ok = false; break; }
            }
            if (ok) break;
        }

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

        std::mt19937_64 gen(seed);
        u64 j = 1;
        if (M > 1) { do { j = (gen() % (M - 1)) + 1; } while (std::gcd(j, M) != 1); }
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

public:
    GeometricTraverser(u64 N, std::optional<u64> s = std::nullopt) : N_total(N), total_steps(0) {
        if (N == 0) throw std::runtime_error("N > 0 required");
        seed = s ? *s : (((u64)std::random_device{}() << 32) | std::random_device{}());
        M_odd = N; k_pow2 = 0;
        while (M_odd > 0 && M_odd % 2 == 0) { M_odd /= 2; k_pow2++; }
        init_odd_part(M_odd);
        initial_count_2k = (k_pow2 > 0) ? (std::mt19937_64(seed)() % (1ULL << k_pow2)) : 0;
        count_2k = initial_count_2k;
    }

    void jump(u64 steps) {
        if (N_total == 0) return;
        u64 cur_pos = (total_steps + steps) % N_total;
        seek(cur_pos);
    }

    void seek(u64 pos) {
        if (pos >= N_total) pos %= N_total;
        total_steps = pos;

        u64 c_val = (initial_count_2k + pos);
        u64 cycles = c_val >> k_pow2;
        count_2k = c_val & ((1ULL << k_pow2) - 1);
        if (k_pow2 >= 64) count_2k = (initial_count_2k + pos);

        if (M_odd > 1) {
            GF2EPush scope(modulus);
            GF2E shift;
            power(shift, zeta_step, to_ZZ(cycles));
            current_zeta = initial_zeta * shift;
        }
    }

    u64 tell() const { return total_steps; }

    u64 next() {
        u64 r = (M_odd <= 1) ? 0 : get_rank(current_zeta);
        u64 res = (r << k_pow2) | count_2k;

        total_steps = (total_steps + 1) % N_total;
        count_2k++;
        if (k_pow2 < 64 && count_2k == (1ULL << k_pow2)) {
            count_2k = 0;
            if (M_odd > 1) {
                GF2EPush scope(modulus);
                current_zeta *= zeta_step;
            }
        }
        return res;
    }

    const GF2X& get_modulus() const { return modulus; }
    const GF2E& get_algebraic_state() const { return current_zeta; }
};

// -----------------------------------------------------------------------------
// Locus Exploration
// -----------------------------------------------------------------------------

static std::vector<GF2X> find_all_primitive_polynomials(long n) {
    std::vector<GF2X> locus;
    if (n < 1 || n > 16) {
        GF2X p; BuildSparseIrred(p, n);
        locus.push_back(p);
        return locus;
    }

    ZZ order = (power_ZZ(2, n) - 1);
    auto factors = unique_prime_factors_ZZ(order);

    for (long i = 0; i < (1L << n); ++i) {
        GF2X p;
        SetCoeff(p, n);
        for (long j = 0; j < n; ++j) if ((i >> j) & 1) SetCoeff(p, j);
        if (IsZero(coeff(p, 0))) continue;

        if (IterIrredTest(p)) {
            GF2EPush scope(p);
            GF2E x = conv<GF2E>(string_to_gf2x("[0 1]"));
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
        std::cout << "=== LFSR Suite: Advanced Algebraic Tools ===\n\n";

        std::cout << "--- 1. Primitive Locus (n=5) ---\n";
        auto locus = find_all_primitive_polynomials(5);
        std::cout << "Found " << locus.size() << " primitive polynomials:\n";
        for (const auto& p : locus) std::cout << "  " << p << "\n";
        std::cout << "\n";

        std::cout << "--- 2. Jump & Seek Test (N=1000) ---\n";
        GeometricTraverser gt(1000, 12345ULL);
        u64 p0 = gt.next();
        std::cout << "pos 0: " << p0 << "\n";
        gt.jump(499);
        u64 p500 = gt.next();
        std::cout << "pos 500 (after jump 499): " << p500 << "\n";
        gt.seek(0);
        if (gt.next() == p0) std::cout << "Seek(0) successful.\n";
        gt.seek(500);
        if (gt.next() == p500) std::cout << "Seek(500) successful.\n\n";

        std::cout << "--- 3. Geometric Atlas Views (N=65535) ---\n";
        GeometricTraverser atlas_gt(65535, 42ULL);
        auto companion = GeometricAtlas::CompanionChart();
        auto trace_chart = GeometricAtlas::TraceChart();

        std::cout << "First 20 bits through different charts:\n";
        std::cout << "  [Companion]: ";
        for(int i=0; i<20; ++i) {
            GF2EPush scope(atlas_gt.get_modulus());
            std::cout << companion.projection(atlas_gt.get_algebraic_state());
            atlas_gt.next();
        }
        std::cout << "\n";

        atlas_gt.seek(0);
        std::cout << "  [Trace]:     ";
        for(int i=0; i<20; ++i) {
            GF2EPush scope(atlas_gt.get_modulus());
            std::cout << trace_chart.projection(atlas_gt.get_algebraic_state());
            atlas_gt.next();
        }
        std::cout << "\n\n";

        std::cout << "--- 4. Large Range Verification (N=65536) ---\n";
        GeometricTraverser large_gt(65536, 999ULL);
        std::set<u64> seen;
        for(u64 i=0; i<65536; ++i) seen.insert(large_gt.next());
        std::cout << "N=65536: Unique elements seen = " << seen.size()
                  << (seen.size() == 65536 ? " (PASS)" : " (FAIL)") << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

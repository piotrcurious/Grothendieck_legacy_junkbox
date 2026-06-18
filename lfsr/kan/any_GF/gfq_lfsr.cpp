// gfq_lfsr.cpp  —  LFSR suite over GF(p^n), lifting lfsr_improved.cpp
//                  from GF(2) to any prime characteristic p.
//
// Compile (Linux / macOS, NTL linked against GMP):
//   g++ -O2 -std=c++17 gfq_lfsr.cpp -lntl -lgmp -pthread -o gfq_lfsr
//
// ═══════════════════════════════════════════════════════════════════════
// CATEGORICAL & GEOMETRIC ARCHITECTURE
// ═══════════════════════════════════════════════════════════════════════
//
// 1. Global Object: A point in the variety of monic irreducible polynomials
//    f(x) over F_p of degree n, such that x mod f(x) is primitive.
//
// 2. Left Kan Extension (Lan): The realization of the global object into
//    an atlas of local implementation fragments (Charts).
//     Lan(f) = Glue { Companion, Matrix, Trace, Decimation, Reciprocal }
//
// 3. Right Kan Extension (Ran): The inference of the global object from
//    local bit-stream observations.
//     Ran(obs) = { f | compatible(f, obs) }
//
// ═══════════════════════════════════════════════════════════════════════

#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/ZZ_pX.h>
#include <NTL/ZZ_pE.h>
#include <NTL/ZZ_pXFactoring.h>
#include <NTL/vec_ZZ_p.h>
#include <NTL/mat_ZZ_p.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace NTL;
using u64 = std::uint64_t;

// ────────────────────────────────────────────────────────────────────────────
// §1  Utilities
// ────────────────────────────────────────────────────────────────────────────

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

static u64 p_power(long p, long n) {
    if (p < 2 || n < 0) throw std::invalid_argument("p_power: p>=2, n>=0 required");
    u64 r = 1;
    for (long i = 0; i < n; ++i) {
        if (r > std::numeric_limits<u64>::max() / (u64)p)
            throw std::overflow_error("p^n overflows u64");
        r *= (u64)p;
    }
    return r;
}

static ZZ u64_to_ZZ(u64 v) {
    ZZ r = to_ZZ((long)((v >> 48) & 0xFFFFULL));
    r = LeftShift(r, 16); r += to_ZZ((long)((v >> 32) & 0xFFFFULL));
    r = LeftShift(r, 16); r += to_ZZ((long)((v >> 16) & 0xFFFFULL));
    r = LeftShift(r, 16); r += to_ZZ((long)(v          & 0xFFFFULL));
    return r;
}

// ────────────────────────────────────────────────────────────────────────────
// §2  Base-p Encoding
// ────────────────────────────────────────────────────────────────────────────

static u64 zzpe_to_uint(const ZZ_pE& a, long p) {
    if (IsZero(a)) return 0ULL;
    const ZZ_pX& f = rep(a);
    long d = deg(f);
    u64 result = 0, base = 1;
    for (long i = 0; i <= d; ++i) {
        long c = conv<long>(rep(coeff(f, i)));
        result += (u64)c * base;
        if (i < d) {
            if (base > std::numeric_limits<u64>::max() / (u64)p)
                throw std::overflow_error("zzpe_to_uint: base overflow");
            base *= (u64)p;
        }
    }
    return result;
}

static ZZ_pE uint_to_zzpe(u64 val, long p) {
    if (val == 0) return ZZ_pE();
    ZZ_pX f;
    for (long i = 0; val > 0; ++i, val /= (u64)p) {
        long c = (long)(val % (u64)p);
        if (c) SetCoeff(f, i, ZZ_p(c));
    }
    return conv<ZZ_pE>(f);
}

// ────────────────────────────────────────────────────────────────────────────
// §3  Galois Field Locus
// ────────────────────────────────────────────────────────────────────────────

static bool is_primitive_element(const ZZ_pE& a, u64 group_order) {
    if (IsZero(a)) return false;
    if (group_order == 1) return true;
    for (u64 q : unique_prime_factors(group_order)) {
        ZZ_pE t;
        power(t, a, u64_to_ZZ(group_order / q));
        if (IsOne(t)) return false;
    }
    return true;
}

static ZZ_pE find_primitive_element(u64 group_order, long tries = 10000) {
    ZZ_pE a;
    for (long i = 0; i < tries; ++i) {
        random(a);
        if (is_primitive_element(a, group_order)) return a;
    }
    throw std::runtime_error("find_primitive_element: exhausted tries");
}

class FieldContext {
public:
    const long p, n;
    const u64  order;
    const u64  group_order;

private:
    ZZ_pContext       ctx_p_;
    ZZ_pEContext      ctx_pe_;
    u64               prim_uint_;
    std::vector<long> prim_c_;
    std::vector<long> mod_c_;

public:
    FieldContext(long p_, long n_, u64 ntl_seed = 0)
        : p(p_), n(n_),
          order(p_power(p_, n_)),
          group_order(p_power(p_, n_) - 1)
    {
        if (p_ < 2) throw std::invalid_argument("FieldContext: p >= 2 required");
        if (n_ < 1) throw std::invalid_argument("FieldContext: n >= 1 required");

        ZZ_pPush guard;
        ZZ_p::init(to_ZZ(p_));
        u64 effective_seed = ntl_seed ? ntl_seed : (u64(p_) * 1000003ULL + u64(n_));
        SetSeed(u64_to_ZZ(effective_seed));
        ZZ_pX irred;
        BuildIrred(irred, n_);
        ZZ_pE::init(irred);
        mod_c_.assign((size_t)(n_ + 1), 0L);
        for (long i = 0; i <= deg(irred); ++i)
            mod_c_[(size_t)i] = conv<long>(rep(coeff(irred, i)));
        ZZ_pE alpha = find_primitive_element(group_order);
        prim_uint_  = zzpe_to_uint(alpha, p_);
        const ZZ_pX& ap = rep(alpha);
        prim_c_.assign((size_t)(deg(ap) + 1), 0L);
        for (long i = 0; i <= deg(ap); ++i)
            prim_c_[(size_t)i] = conv<long>(rep(coeff(ap, i)));
        ctx_p_.save();
        ctx_pe_.save();
    }

    FieldContext(long p_, long n_, const ZZ_pX& irred, u64 ntl_seed = 0)
        : p(p_), n(n_),
          order(p_power(p_, n_)),
          group_order(p_power(p_, n_) - 1)
    {
        if (p_ < 2) throw std::invalid_argument("FieldContext: p >= 2 required");
        if (n_ < 1) throw std::invalid_argument("FieldContext: n >= 1 required");
        if (deg(irred) < 1) throw std::invalid_argument("FieldContext: deg(irred) >= 1 required");

        ZZ_pPush guard;
        ZZ_p::init(to_ZZ(p_));
        ZZ_pE::init(irred);
        mod_c_.assign((size_t)(n_ + 1), 0L);
        for (long i = 0; i <= deg(irred); ++i)
            mod_c_[(size_t)i] = conv<long>(rep(coeff(irred, i)));
        u64 effective_seed = ntl_seed ? ntl_seed : (u64(p_) * 1000003ULL + u64(n_) + 7);
        SetSeed(u64_to_ZZ(effective_seed));
        ZZ_pE alpha = find_primitive_element(group_order);
        prim_uint_  = zzpe_to_uint(alpha, p_);
        const ZZ_pX& ap = rep(alpha);
        prim_c_.assign((size_t)(deg(ap) + 1), 0L);
        for (long i = 0; i <= deg(ap); ++i)
            prim_c_[(size_t)i] = conv<long>(rep(coeff(ap, i)));
        ctx_p_.save();
        ctx_pe_.save();
    }

    void activate() const {
        ctx_p_.restore();
        ctx_pe_.restore();
    }

    ZZ_pE primitive() const {
        ZZ_pX f;
        for (long i = 0; i < (long)prim_c_.size(); ++i)
            if (prim_c_[(size_t)i]) SetCoeff(f, i, ZZ_p(prim_c_[(size_t)i]));
        return conv<ZZ_pE>(f);
    }

    u64 prim_as_uint() const { return prim_uint_; }
    ZZ_pE from_uint(u64 v)          const { return uint_to_zzpe(v, p); }
    u64   to_uint  (const ZZ_pE& a) const { return zzpe_to_uint(a, p); }

    void print_info(std::ostream& os = std::cout) const {
        os << "  GF(" << p << "^" << n << ")  |F*| = " << group_order << "\n";
        os << "  irred f(x) =";
        for (long i = 0; i < (long)mod_c_.size(); ++i)
            if (mod_c_[(size_t)i])
                os << " + " << mod_c_[(size_t)i] << "x^" << i;
        os << "\n  primitive \u03b1 (base-" << p << " int) = " << prim_uint_ << "\n";
    }

    std::unique_ptr<FieldContext> reciprocal_field() const {
        ZZ_pPush guard;
        ctx_p_.restore();
        ZZ_pX irred_star;
        for (long i = 0; i <= n; ++i) {
            if (mod_c_[(size_t)i])
                SetCoeff(irred_star, n - i, ZZ_p(mod_c_[(size_t)i]));
        }
        if (deg(irred_star) < 1) return nullptr;
        return std::make_unique<FieldContext>(p, n, irred_star);
    }
};

// ────────────────────────────────────────────────────────────────────────────
// §4  Kan Extensions (Lan) — Realization Atlas
// ────────────────────────────────────────────────────────────────────────────

struct Fragment {
    std::string chart_name;
    std::string coordinate_desc;
    std::function<u64(u64)> transition;
    std::function<bool(u64)> projection;
    u64 state = 1;
    const FieldContext* fc;

    virtual ~Fragment() = default;

    std::string sample(std::size_t len) {
        std::string s;
        u64 cur = state;
        for (std::size_t i = 0; i < len; ++i) {
            fc->activate();
            s.push_back(projection(cur) ? '1' : '0');
            cur = transition(cur);
        }
        return s;
    }
};

using ChartFunctor = std::function<std::unique_ptr<Fragment>(const FieldContext*)>;

static std::vector<std::unique_ptr<Fragment>> lan_extend(const std::vector<ChartFunctor>& atlas, const FieldContext* fc) {
    std::vector<std::unique_ptr<Fragment>> fragments;
    for (auto& f_gen : atlas) {
        auto res = f_gen(fc);
        if (res) fragments.push_back(std::move(res));
    }
    return fragments;
}

static ChartFunctor companion_chart = [](const FieldContext* fc) -> std::unique_ptr<Fragment> {
    auto f = std::make_unique<Fragment>();
    f->chart_name = "Companion";
    f->coordinate_desc = "Polynomial basis bit 0";
    f->fc = fc;
    f->transition = [fc](u64 s) {
        fc->activate();
        ZZ_pE ss = fc->from_uint(s);
        ZZ_pX x_poly; SetCoeff(x_poly, 1, 1);
        ZZ_pE x_ev = conv<ZZ_pE>(x_poly);
        ss *= x_ev;
        return fc->to_uint(ss);
    };
    f->projection = [fc](u64 s) {
        fc->activate();
        // Polynomial representation a0 + a1*x + ...
        // state s = a0 + a1*p + ...
        // bit 0 is a0 mod 2.
        return (u64)(s % fc->p) % 2 != 0;
    };
    return f;
};

static ChartFunctor trace_chart = [](const FieldContext* fc) -> std::unique_ptr<Fragment> {
    auto f = std::make_unique<Fragment>();
    f->chart_name = "Trace";
    f->coordinate_desc = "Field trace to GF(p) mod 2";
    f->fc = fc;
    f->transition = [fc](u64 s) {
        fc->activate();
        ZZ_pE ss = fc->from_uint(s);
        ZZ_pX x_poly; SetCoeff(x_poly, 1, 1);
        ZZ_pE x_ev = conv<ZZ_pE>(x_poly);
        ss *= x_ev;
        return fc->to_uint(ss);
    };
    f->projection = [fc](u64 s) {
        fc->activate();
        ZZ_pE ss = fc->from_uint(s);
        ZZ tr_zz = rep(trace(ss));
        return (conv<long>(tr_zz) % 2) != 0;
    };
    return f;
};

class ReciprocalFragment : public Fragment {
public:
    std::shared_ptr<FieldContext> owned_fc;
    ReciprocalFragment(std::shared_ptr<FieldContext> f) : owned_fc(f) {
        chart_name = "Reciprocal";
        coordinate_desc = "Dual basis / reversed orbit";
        fc = owned_fc.get();
        transition = [this](u64 s) {
            owned_fc->activate();
            ZZ_pE ss = owned_fc->from_uint(s);
            ZZ_pX x_poly; SetCoeff(x_poly, 1, 1);
            ZZ_pE x_ev = conv<ZZ_pE>(x_poly);
            ss *= x_ev;
            return owned_fc->to_uint(ss);
        };
        projection = [this](u64 s) {
            owned_fc->activate();
            ZZ_pE ss = owned_fc->from_uint(s);
            return (conv<long>(rep(coeff(rep(ss), 0))) & 1) != 0;
        };
    }
};

static ChartFunctor reciprocal_chart = [](const FieldContext* fc) -> std::unique_ptr<Fragment> {
    auto rfc_ptr = fc->reciprocal_field();
    if (!rfc_ptr) return nullptr;
    auto rfc = std::shared_ptr<FieldContext>(rfc_ptr.release());
    return std::make_unique<ReciprocalFragment>(rfc);
};

static ChartFunctor matrix_chart = [](const FieldContext* fc) -> std::unique_ptr<Fragment> {
    fc->activate();
    long n = fc->n;
    mat_ZZ_p M; M.SetDims(n, n);
    ZZ_pX x_poly; SetCoeff(x_poly, 1, 1);
    ZZ_pE x_ev = conv<ZZ_pE>(x_poly);
    for (long j = 0; j < n; ++j) {
        ZZ_pX col_poly; SetCoeff(col_poly, j, 1);
        ZZ_pE col_ev = conv<ZZ_pE>(col_poly);
        col_ev *= x_ev;
        ZZ_pX res_poly = rep(col_ev);
        for (long i = 0; i < n; ++i) M[i][j] = coeff(res_poly, i);
    }
    auto f = std::make_unique<Fragment>();
    f->chart_name = "Matrix";
    f->coordinate_desc = "Linear map transition";
    f->fc = fc;
    f->transition = [fc, M](u64 s) {
        fc->activate();
        vec_ZZ_p v; v.SetLength(fc->n);
        u64 tmp = s;
        for (long i = 0; i < fc->n; ++i) { v[i] = ZZ_p(tmp % fc->p); tmp /= fc->p; }
        vec_ZZ_p res = M * v;
        u64 out = 0, base = 1;
        for (long i = 0; i < fc->n; ++i) { out += conv<long>(rep(res[i])) * base; base *= fc->p; }
        return out;
    };
    f->projection = [fc](u64 s) { return (u64)(s % fc->p) % 2 != 0; };
    return f;
};

static ChartFunctor decimation_chart = [](const FieldContext* fc) -> std::unique_ptr<Fragment> {
    fc->activate();
    long k = 3;
    ZZ_pX x_poly; SetCoeff(x_poly, 1, 1);
    ZZ_pE x_ev = conv<ZZ_pE>(x_poly);
    ZZ_pE xk; power(xk, x_ev, k);
    u64 xk_uint = fc->to_uint(xk);
    auto f = std::make_unique<Fragment>();
    f->chart_name = "Decimate-3";
    f->coordinate_desc = "Every 3rd orbit point";
    f->fc = fc;
    f->transition = [fc, xk_uint](u64 s) {
        fc->activate();
        ZZ_pE ss = fc->from_uint(s);
        ss *= fc->from_uint(xk_uint);
        return fc->to_uint(ss);
    };
    f->projection = [fc](u64 s) {
        fc->activate();
        ZZ_pE ss = fc->from_uint(s);
        return (conv<long>(rep(coeff(rep(ss), 0))) & 1) != 0;
    };
    return f;
};

// ────────────────────────────────────────────────────────────────────────────
// §5  Orbit & RangeTraverser
// ────────────────────────────────────────────────────────────────────────────

struct Orbit {
    long p, n;
    std::vector<u64> states;
};

static Orbit build_orbit(long p, long n) {
    FieldContext fc(p, n);
    fc.activate();
    ZZ_pE alpha = fc.primitive();
    u64 len = fc.group_order;
    Orbit out; out.p = p; out.n = n; out.states.reserve(len);
    ZZ_pE s; set(s);
    for (u64 i = 0; i < len; ++i) {
        out.states.push_back(fc.to_uint(s));
        s *= alpha;
    }
    return out;
}

class RangeTraverser {
    long p_, n_;
    u64 max_val_, seed_;
    bool visited_zero_;
    u64 cur_;
    std::unique_ptr<FieldContext> fc_;

    void step() {
        fc_->activate();
        ZZ_pE s = fc_->from_uint(cur_);
        ZZ_pE alpha = fc_->from_uint(fc_->prim_as_uint());
        s *= alpha;
        cur_ = fc_->to_uint(s);
    }
    void advance_to_range() { while (cur_ > max_val_) step(); }

public:
    RangeTraverser(long p, u64 max_v, std::optional<u64> seed = std::nullopt)
        : p_(p), max_val_(max_v), visited_zero_(false)
    {
        seed_ = seed.value_or([] {
            std::random_device rd;
            return ((u64)rd() << 32) | (u64)rd();
        }());
        if (max_val_ == 0) { cur_ = 0; return; }
        n_ = 1;
        for (;;) {
            u64 cap;
            try { cap = p_power(p_, n_); }
            catch (...) { throw std::runtime_error("RangeTraverser overflow"); }
            if (cap - 1 >= max_val_) break;
            ++n_;
        }
        fc_ = std::make_unique<FieldContext>(p_, n_, seed_);
        fc_->activate();
        std::mt19937_64 gen(seed_);
        std::uniform_int_distribution<u64> dis(1, fc_->group_order);
        cur_ = dis(gen);
        advance_to_range();
    }
    void reset() {
        visited_zero_ = false;
        if (max_val_ == 0) return;
        fc_->activate();
        std::mt19937_64 gen(seed_);
        std::uniform_int_distribution<u64> dis(1, fc_->group_order);
        cur_ = dis(gen);
        advance_to_range();
    }
    u64 next() {
        if (max_val_ == 0) return 0;
        if (!visited_zero_) { visited_zero_ = true; return 0; }
        u64 val = cur_;
        step();
        advance_to_range();
        return val;
    }
};

// ────────────────────────────────────────────────────────────────────────────
// §6  Right Kan Extension (Ran) — Inference Machine
// ────────────────────────────────────────────────────────────────────────────

struct Observation {
    long p;
    std::string chart_name;
    std::vector<u64> prefix;
};

static std::vector<long> ran_extend(const Observation& obs, long n_min = 1, long n_max = 10) {
    std::vector<long> candidates;
    for (long n = std::max(1L, n_min); n <= n_max; ++n) {
        try {
            // In a fully developed Right Kan Extension, we don't assume the irreducible.
            // But NTL's BuildIrred is deterministic. To search the "variety",
            // we'd need to iterate all irreducible polynomials.
            // For now, we improve inference by searching the PRIMITIVE LOCUS
            // of the chosen field representation.

            FieldContext fc(obs.p, n, 0);
            fc.activate();

            // 1. Efficient Algebraic Inference for Companion/Matrix (State-level)
            if (obs.chart_name == "Companion" || obs.chart_name == "Matrix") {
                if (obs.prefix.size() >= 2 && obs.prefix[0] != 0) {
                    ZZ_pE s0 = fc.from_uint(obs.prefix[0]);
                    ZZ_pE s1 = fc.from_uint(obs.prefix[1]);
                    ZZ_pE s0inv, alpha;
                    inv(s0inv, s0);
                    mul(alpha, s1, s0inv);
                    // Algebraic property: alpha = s1/s0 MUST be primitive
                    if (is_primitive_element(alpha, fc.group_order)) {
                        ZZ_pE s = s0;
                        bool ok = true;
                        for (u64 val : obs.prefix) {
                            fc.activate();
                            if (fc.to_uint(s) != val) { ok = false; break; }
                            s *= alpha;
                        }
                        if (ok) { candidates.push_back(n); continue; }
                    }
                }
            }

            // 2. Fragment-based Inference (Projection-level or multi-chart)

            bool found_for_n = false;

            // Special optimization for Trace inference using Linear Algebra (especially for p=2)
            if (obs.chart_name == "Trace" && obs.p == 2) {
                if (obs.prefix.size() >= (size_t)n) {
                    mat_ZZ_p A; A.SetDims(n, n);
                    vec_ZZ_p b; b.SetLength(n);
                    ZZ_pE alpha = fc.primitive();
                    for (int i = 0; i < n; ++i) {
                        ZZ_pE alpha_i; power(alpha_i, alpha, i);
                        for (int j = 0; j < n; ++j) {
                            ZZ_pX xj_poly; SetCoeff(xj_poly, j, 1);
                            ZZ_pE xj_ev = conv<ZZ_pE>(xj_poly);
                            A[i][j] = conv<ZZ_p>(rep(trace(xj_ev * alpha_i)));
                        }
                        b[i] = conv<ZZ_p>(to_ZZ((long)obs.prefix[i]));
                    }

                    ZZ_p det_val;
                    vec_ZZ_p sol;
                    solve(det_val, sol, A, b);
                    if (!IsZero(det_val)) {
                        u64 s_uint = 0, base = 1;
                        for (int j = 0; j < n; ++j) {
                            s_uint += conv<long>(rep(sol[j])) * base;
                            base *= 2;
                        }
                        // Verify solution against remaining bits
                        ZZ_pE s_ev = fc.from_uint(s_uint);
                        bool ok = true;
                        for (size_t i = 0; i < obs.prefix.size(); ++i) {
                            ZZ_pE cur_s = s_ev;
                            ZZ_pE alpha_i; power(alpha_i, alpha, i);
                            if ((u64)(conv<long>(rep(trace(cur_s * alpha_i))) % 2) != obs.prefix[i]) {
                                ok = false; break;
                            }
                        }
                        if (ok) { candidates.push_back(n); found_for_n = true; break; }
                    }
                }
                if (found_for_n) break;
            }

            std::vector<ChartFunctor> atlas = {
                companion_chart, matrix_chart, trace_chart, decimation_chart, reciprocal_chart
            };
            auto fragments = lan_extend(atlas, &fc);

            for (auto& f : fragments) {
                if (f && f->chart_name == obs.chart_name) {
                    // Search the PRIMITIVE LOCUS
                    // For the Trace/Decimation charts, the generator matters.
                    // We try multiple primitive elements if the standard one doesn't work.
                    std::vector<u64> primitives;
        if (fc.order < 10000) {
                        for (u64 a = 1; a < fc.order; ++a) {
                            if (is_primitive_element(fc.from_uint(a), fc.group_order))
                                primitives.push_back(a);
                if (primitives.size() > 64) break; // Increased limit for Decimation search
                        }
                    } else {
                        primitives.push_back(fc.prim_as_uint());
                    }

                    for (u64 prim : primitives) {
                        // Update fragment's transition to use this primitive
                        f->transition = [&fc, prim](u64 s) {
                            fc.activate();
                            ZZ_pE ss = fc.from_uint(s);
                            ss *= fc.from_uint(prim);
                            return fc.to_uint(ss);
                        };

                        u64 search_limit = std::min<u64>(fc.order, 1024);
                        for (u64 s_idx = 1; s_idx < search_limit; ++s_idx) {
                            f->state = s_idx;
                            bool match = true;
                            u64 cur = f->state;
                            for (u64 val : obs.prefix) {
                                fc.activate();
                                if ((u64)f->projection(cur) != val) { match = false; break; }
                                cur = f->transition(cur);
                            }
                            if (match) { candidates.push_back(n); found_for_n = true; break; }
                        }
                        if (found_for_n) break;
                    }
                }
                if (found_for_n) break;
            }
        } catch (...) {}
    }
    return candidates;
}

// ────────────────────────────────────────────────────────────────────────────
// §7  Tests
// ────────────────────────────────────────────────────────────────────────────

static void test_kan_realization(long p, long n) {
    std::cout << "[ Lan Realization  p=" << p << "  n=" << n << " ]\n";
    FieldContext fc(p, n);
    fc.print_info();
    std::vector<ChartFunctor> atlas = {
        companion_chart, matrix_chart, trace_chart, decimation_chart, reciprocal_chart
    };
    auto fragments = lan_extend(atlas, &fc);
    std::cout << "  Atlas Realizations:\n";
    for (auto& f : fragments) {
        if (!f) continue;
        std::cout << "    [" << std::left << std::setw(12) << f->chart_name << "] "
                  << std::setw(25) << f->coordinate_desc << " | Sample: " << f->sample(15) << "\n";
    }
    std::cout << "\n";
}

static void test_kan_inference(long p, long n, std::string chart) {
    std::cout << "[ Ran Inference  p=" << p << "  target_n=" << n << "  chart=" << chart << " ]\n";
    FieldContext fc(p, n, 0);
    std::vector<ChartFunctor> atlas;
    if (chart == "Trace") atlas.push_back(trace_chart);
    else if (chart == "Reciprocal") atlas.push_back(reciprocal_chart);
    else if (chart == "Matrix") atlas.push_back(matrix_chart);
    else if (chart == "Decimate-3") atlas.push_back(decimation_chart);
    else atlas.push_back(companion_chart);

    auto fragments = lan_extend(atlas, &fc);
    if (fragments.empty() || !fragments[0]) {
        std::cout << "  Failed to create chart " << chart << "\n\n";
        return;
    }
    Observation obs; obs.p = p; obs.chart_name = chart;
    std::string smp = fragments[0]->sample(20);
    for (char c : smp) obs.prefix.push_back(c - '0');

    std::cout << "  Observation: " << smp << "\n";
    auto candidates = ran_extend(obs, 2, 10);
    std::cout << "  Inferred n:";
    for (long c : candidates) std::cout << " " << c;
    bool ok = std::find(candidates.begin(), candidates.end(), n) != candidates.end();
    std::cout << "  [OK: " << (ok ? "YES" : "NO") << "]\n\n";
}

static void test_categorical_gluing(long p, long n) {
    std::cout << "[ Categorical Gluing Test  p=" << p << "  n=" << n << " ]\n";
    FieldContext fc(p, n, 0);
    std::vector<ChartFunctor> atlas = {
        companion_chart, matrix_chart, trace_chart, decimation_chart, reciprocal_chart
    };
    auto fragments = lan_extend(atlas, &fc);

    // In a category, different presentations of the same object must commute.
    // Here we check that the Companion and Matrix charts generate identical state-sequences
    // when projected identically.

    if (fragments.size() >= 2) {
        std::string s1 = fragments[0]->sample(50);
        std::string s2 = fragments[1]->sample(50);
        bool match = (s1 == s2);
        std::cout << "  Companion vs Matrix bit-perfect consistency: " << (match ? "PASS" : "FAIL") << "\n";
    }
    std::cout << "\n";
}

static void test_range_traversal(long p, u64 max_val, std::optional<u64> seed = std::nullopt) {
    std::cout << "[ RangeTraversal  p=" << p << "  max_val=" << max_val << " ]\n";
    RangeTraverser tr(p, max_val, seed);
    std::vector<u64> seq;
    for (u64 i = 0; i <= max_val; ++i) seq.push_back(tr.next());
    std::set<u64> seen(seq.begin(), seq.end());
    bool ok = (seen.size() == max_val + 1);
    std::cout << "  All [0," << max_val << "] visited exactly once: " << (ok ? "YES" : "NO") << "\n";
    std::cout << "  First 20:";
    for (u64 i = 0; i < std::min<u64>(20, seq.size()); ++i) std::cout << " " << seq[i];
    std::cout << "\n\n";
}

static void test_build_orbit(long p, long n) {
    std::cout << "[ Orbit  p=" << p << "  n=" << n << " ]\n";
    u64 expected; try { expected = p_power(p, n) - 1; } catch (...) { return; }
    if (expected > 2000000) return;
    Orbit orb = build_orbit(p, n);
    std::set<u64> seen(orb.states.begin(), orb.states.end());
    bool ok = (seen.size() == expected && seen.find(0) == seen.end());
    std::cout << "  unique+nonzero: " << (ok ? "YES" : "NO") << "\n\n";
}

int main() {
    std::cout << "=== GF(p^n) LFSR Suite: Kan Extensions & Algebraic Geometry ===\n\n";
    try {
        std::cout << "--- Stage 1: Left Kan Extension (Lan) ---" << std::endl;
        test_kan_realization(2, 8);
        test_kan_realization(3, 4);
        test_kan_realization(5, 2);
        test_kan_realization(97, 2);

        std::cout << "--- Stage 2: Right Kan Extension (Ran) ---" << std::endl;
        test_kan_inference(2, 4, "Companion");
        test_kan_inference(2, 4, "Trace");
        test_kan_inference(2, 4, "Matrix");
        test_kan_inference(3, 3, "Reciprocal");
        test_kan_inference(5, 2, "Trace");

        std::cout << "--- Stage 3: Categorical Consistency ---" << std::endl;
        test_categorical_gluing(2, 8);
        test_categorical_gluing(3, 4);

        std::cout << "--- Stage 4: Core Utilities ---" << std::endl;
        test_range_traversal(2, 20, 12345ULL);
        test_build_orbit(2, 8);
        test_build_orbit(3, 4);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

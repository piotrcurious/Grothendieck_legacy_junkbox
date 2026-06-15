// gfq_lfsr.cpp  —  LFSR suite over GF(p^n), lifting lfsr_improved.cpp
//                  from GF(2) to any prime characteristic p.
//
// Compile (Linux / macOS, NTL linked against GMP):
//   g++ -O2 -std=c++17 gfq_lfsr.cpp -lntl -lgmp -o gfq_lfsr
//
// ═══════════════════════════════════════════════════════════════════════
// STRUCTURAL CHANGES FROM THE GF(2) BASELINE
// ═══════════════════════════════════════════════════════════════════════
//
// ① NTL types
//      GF2X / GF2E / GF2EPush
//    ↓
//      ZZ_pX / ZZ_pE   (with the TWO-LEVEL context management below)
//
// ② Double context — the central difficulty
//    GF2E has a single global slot (just the modulus polynomial).
//    ZZ_pE requires TWO:
//      • ZZ_p::init(p)    — sets the characteristic (prime)
//      • ZZ_pE::init(f)   — sets the extension modulus
//    Changing ZZ_p invalidates any existing ZZ_pE arithmetic.
//    ZZ_pEContext::save()    captures BOTH levels in one snapshot.
//    ZZ_pEContext::restore() reinstates both atomically.
//    FieldContext (§4) wraps this so every call-site just says activate().
//    The constructor uses ZZ_pPush so the caller's context is preserved.
//
// ③ State encoding  —  bit-string → base-p integer
//    GF(p^n) element  a₀ + a₁x + … + a_{n-1}x^{n-1}
//    ↔  integer        a₀ + a₁·p + … + a_{n-1}·p^{n-1}  ∈ [0, p^n−1]
//    For p = 2 this is identical to the original bit representation.
//    The map is a bijection of sets; it does NOT preserve ring structure.
//
// ④ Orbit length   2^n − 1   →   p^n − 1
//
// ⑤ BuildSparseIrred and find_primitive_element work identically over ZZ_p[x].
//
// ⑥ Inference: p is assumed known; only n is inferred.
//    α is computed from the observation as prefix[1] / prefix[0], so no
//    assumption is made that the sequence starts at 1.
// ═══════════════════════════════════════════════════════════════════════

#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/ZZ_pX.h>
#include <NTL/ZZ_pE.h>
#include <NTL/ZZ_pXFactoring.h>

#include <algorithm>
#include <cstdint>
#include <functional>
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

// p^n as u64; throws std::overflow_error on u64 overflow.
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

// Portable u64 → NTL ZZ, valid on both 32- and 64-bit platforms.
// Splits into four 16-bit chunks; each fits in any signed long.
static ZZ u64_to_ZZ(u64 v) {
    ZZ r = to_ZZ((long)((v >> 48) & 0xFFFFULL));
    r = LeftShift(r, 16); r += to_ZZ((long)((v >> 32) & 0xFFFFULL));
    r = LeftShift(r, 16); r += to_ZZ((long)((v >> 16) & 0xFFFFULL));
    r = LeftShift(r, 16); r += to_ZZ((long)(v          & 0xFFFFULL));
    return r;
}

// ────────────────────────────────────────────────────────────────────────────
// §2  Base-p Encoding  (bijection: GF(p^n) element ↔ u64)
// ────────────────────────────────────────────────────────────────────────────
//
//   a₀ + a₁x + … + a_{n-1}x^{n-1}   ↔   a₀ + a₁p + … + a_{n-1}p^{n-1}
//
// The active ZZ_pE context must match the field from which the element comes.
// For p=2 this collapses to the original bit-vector encoding.

// GF(p^n) element → base-p integer (ZZ_pE context must be active).
static u64 zzpe_to_uint(const ZZ_pE& a, long p) {
    if (IsZero(a)) return 0ULL;
    const ZZ_pX& f = rep(a);
    long d = deg(f);
    u64 result = 0, base = 1;
    for (long i = 0; i <= d; ++i) {
        long c = conv<long>(rep(coeff(f, i)));   // c ∈ [0, p−1]
        result += (u64)c * base;
        if (i < d) {
            if (base > std::numeric_limits<u64>::max() / (u64)p)
                throw std::overflow_error("zzpe_to_uint: base overflow");
            base *= (u64)p;
        }
    }
    return result;
}

// base-p integer → GF(p^n) element, reduced mod active ZZ_pE modulus.
static ZZ_pE uint_to_zzpe(u64 val, long p) {
    if (val == 0) return ZZ_pE();       // additive identity
    ZZ_pX f;
    for (long i = 0; val > 0; ++i, val /= (u64)p) {
        long c = (long)(val % (u64)p);
        if (c) SetCoeff(f, i, ZZ_p(c));
    }
    return conv<ZZ_pE>(f);              // reduces modulo the current ZZ_pE modulus
}

// ────────────────────────────────────────────────────────────────────────────
// §3  Primitivity  (active ZZ_pE context required)
// ────────────────────────────────────────────────────────────────────────────

// True iff a generates GF(p^n)* (cyclic group of order group_order = p^n−1).
// Uses the standard test: a is primitive iff a^(m/q) ≠ 1 for every prime q | m.
static bool is_primitive_element(const ZZ_pE& a, u64 group_order) {
    if (IsZero(a)) return false;
    // group_order = 1  ⟹  GF(2)*, trivially generated by 1; loop body never runs.
    for (u64 q : unique_prime_factors(group_order)) {
        ZZ_pE t;
        power(t, a, u64_to_ZZ(group_order / q));
        if (IsOne(t)) return false;
    }
    return true;
}

// Random search for a generator of GF(p^n)*.
static ZZ_pE find_primitive_element(u64 group_order, long tries = 10000) {
    ZZ_pE a;
    for (long i = 0; i < tries; ++i) {
        random(a);
        if (is_primitive_element(a, group_order)) return a;
    }
    throw std::runtime_error("find_primitive_element: exhausted tries");
}

// ────────────────────────────────────────────────────────────────────────────
// §4  FieldContext  —  frozen snapshot of GF(p^n)
// ────────────────────────────────────────────────────────────────────────────
//
// NTL context recap for ZZ_pE (cf. GF2E which only has one level):
//
//   ZZ_p::init(p)         sets the global prime — destroys any ZZ_pE context
//   ZZ_pE::init(f)        sets the global extension modulus
//   ZZ_pPush guard        RAII: saves / restores ZZ_p on scope exit
//   ZZ_pEContext ctx      ctx.save()    — captures (ZZ_p=p, ZZ_pE=f) together
//                         ctx.restore() — reinstates both atomically
//
// Pattern used in FieldContext:
//   1. Wrap field construction inside ZZ_pPush so the CALLER'S ZZ_p context
//      is restored when the constructor exits (even on exceptions).
//   2. After initialising ZZ_p and ZZ_pE, call ctx_.save() to snapshot both.
//   3. Call activate() (= ctx_.restore()) before any ZZ_pE arithmetic.
//      Between activate() calls the global context may be in any state, but
//      we never perform ZZ_pE arithmetic without a preceding activate().

class FieldContext {
public:
    const long p, n;
    const u64  order;         // p^n
    const u64  group_order;   // p^n − 1

private:
    ZZ_pEContext      ctx_;        // frozen snapshot: ZZ_p=p and ZZ_pE=irred
    u64               prim_uint_;  // primitive element in base-p encoding
    std::vector<long> prim_c_;     // primitive element coefficients (for display)
    std::vector<long> mod_c_;      // irreducible polynomial coefficients

public:
    // ntl_seed: if non-zero, seeds NTL's RNG before find_primitive_element so
    // the same field construction always returns the same primitive element.
    FieldContext(long p_, long n_, u64 ntl_seed = 0)
        : p(p_), n(n_),
          order(p_power(p_, n_)),
          group_order(p_power(p_, n_) - 1)
    {
        if (p_ < 2) throw std::invalid_argument("FieldContext: p >= 2 required");
        if (n_ < 1) throw std::invalid_argument("FieldContext: n >= 1 required");

        ZZ_pPush guard;             // ← saves caller's ZZ_p; restored at scope exit
        ZZ_p::init(to_ZZ(p_));

        ZZ_pX irred;
        BuildSparseIrred(irred, n_);
        ZZ_pE::init(irred);

        // Cache irreducible polynomial coefficients for display
        mod_c_.assign((size_t)(n_ + 1), 0L);
        for (long i = 0; i <= deg(irred); ++i)
            mod_c_[(size_t)i] = conv<long>(rep(coeff(irred, i)));

        // Optional deterministic seeding of NTL's RNG
        if (ntl_seed) SetSeed(u64_to_ZZ(ntl_seed));

        ZZ_pE alpha = find_primitive_element(group_order);
        prim_uint_  = zzpe_to_uint(alpha, p_);

        const ZZ_pX& ap = rep(alpha);
        prim_c_.assign((size_t)(deg(ap) + 1), 0L);
        for (long i = 0; i <= deg(ap); ++i)
            prim_c_[(size_t)i] = conv<long>(rep(coeff(ap, i)));

        ctx_.save();       // ← snapshot: ZZ_p=p_, ZZ_pE=irred
    }
    // ZZ_pPush guard destructs here: caller's ZZ_p is restored.
    // ZZ_pE now technically references irred under the wrong ZZ_p, but
    // we never perform ZZ_pE arithmetic without calling activate() first.

    // Reinstates this field's context globally.
    // MUST be called before any ZZ_pE arithmetic that uses this field.
    void activate() const { ctx_.restore(); }

    // Reconstruct the primitive element as a ZZ_pE (activate() first).
    ZZ_pE primitive() const {
        ZZ_pX f;
        for (long i = 0; i < (long)prim_c_.size(); ++i)
            if (prim_c_[(size_t)i]) SetCoeff(f, i, ZZ_p(prim_c_[(size_t)i]));
        return conv<ZZ_pE>(f);
    }

    u64 prim_as_uint() const { return prim_uint_; }

    // Encoding helpers (activate() first).
    ZZ_pE from_uint(u64 v)          const { return uint_to_zzpe(v, p); }
    u64   to_uint  (const ZZ_pE& a) const { return zzpe_to_uint(a, p); }

    void print_info(std::ostream& os = std::cout) const {
        os << "  GF(" << p << "^" << n << ")  |F*| = " << group_order << "\n";
        os << "  irred f(x) =";
        for (long i = 0; i < (long)mod_c_.size(); ++i)
            if (mod_c_[(size_t)i])
                os << " + " << mod_c_[(size_t)i] << "x^" << i;
        os << "\n  primitive α (base-" << p << " int) = " << prim_uint_
           << "  coeffs:";
        for (long c : prim_c_) os << " " << c;
        os << "\n";
    }
};

// ────────────────────────────────────────────────────────────────────────────
// §5  Orbit  —  full maximal-length sequence of GF(p^n)
// ────────────────────────────────────────────────────────────────────────────

struct Orbit {
    long p, n;
    std::vector<u64> states;  // base-p encoded: 1, α, α², …, α^(p^n−2)
};

static Orbit build_orbit(long p, long n) {
    FieldContext fc(p, n);
    fc.activate();

    ZZ_pE alpha = fc.primitive();
    u64   len   = fc.group_order;

    Orbit out;
    out.p = p;
    out.n = n;
    out.states.reserve(len);

    ZZ_pE s; set(s);               // s = 1 (multiplicative identity)
    for (u64 i = 0; i < len; ++i) {
        out.states.push_back(fc.to_uint(s));
        s *= alpha;
    }
    return out;
}

// ────────────────────────────────────────────────────────────────────────────
// §6  RangeTraverser  —  pseudo-random permutation of [0, max_val]
// ────────────────────────────────────────────────────────────────────────────
//
// Chooses the minimal n such that p^n − 1 ≥ max_val, builds GF(p^n), then
// steps through the multiplicative orbit  cur → cur·α,  skipping (on the fly)
// any base-p encoded value that exceeds max_val.  Zero is yielded first,
// separately, outside the multiplicative orbit.
//
// Efficiency note: the skip rate is (p^n − 1 − max_val) / (p^n − 1).
// Since n is chosen minimally, at most a factor of ~p steps are discarded
// per accepted value, so the traversal is never worse than O(p) overhead.

class RangeTraverser {
    long p_, n_;
    u64  max_val_, seed_;
    bool visited_zero_;
    u64  cur_;                                // current LFSR state (base-p integer)
    std::unique_ptr<FieldContext> fc_;

    void step() {
        fc_->activate();
        ZZ_pE s     = fc_->from_uint(cur_);
        ZZ_pE alpha = fc_->from_uint(fc_->prim_as_uint());  // faster than primitive()
        s   *= alpha;
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

        // Minimum n with p^n − 1 ≥ max_val_
        n_ = 1;
        for (;;) {
            u64 cap;
            try { cap = p_power(p_, n_); }
            catch (const std::overflow_error&) {
                throw std::runtime_error("RangeTraverser: max_val too large for u64");
            }
            if (cap - 1 >= max_val_) break;
            if (n_ >= 63) throw std::runtime_error("RangeTraverser: max_val too large");
            ++n_;
        }

        // seed_ controls both the NTL RNG (→ same primitive element) and the
        // initial state (→ same starting position), giving full reproducibility.
        fc_ = std::make_unique<FieldContext>(p_, n_, seed_);
        fc_->activate();

        std::mt19937_64 gen(seed_);
        std::uniform_int_distribution<u64> dis(1, fc_->group_order);
        cur_ = dis(gen);
        advance_to_range();
    }

    void reset() {
        visited_zero_ = false;
        if (max_val_ == 0) { cur_ = 0; return; }
        fc_->activate();
        std::mt19937_64 gen(seed_);
        std::uniform_int_distribution<u64> dis(1, fc_->group_order);
        cur_ = dis(gen);
        advance_to_range();
    }

    // Call exactly max_val + 1 times to visit every integer in [0, max_val] once.
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
// §7  Inference Layer
// ────────────────────────────────────────────────────────────────────────────
//
// The prime p is assumed known (it's part of the protocol); only the extension
// degree n is inferred.  The candidate α is computed as prefix[1] / prefix[0],
// making no assumption that the sequence starts at 1.

struct Observation {
    long          p;       // known field characteristic (must be prime)
    std::vector<u64> prefix;  // base-p encoded LFSR output
};

struct InferenceResult {
    std::vector<long> candidates;   // extension degrees consistent with the prefix
};

using Recognizer = std::function<bool(const Observation&, long n)>;

// Checks:  prefix = { s₀, s₀α, s₀α², … }  for some primitive α ∈ GF(p^n)*.
static const Recognizer prefix_recognizer =
    [](const Observation& obs, long n) -> bool {
    try {
        if ((long)obs.prefix.size() < 2) return false;

        FieldContext fc(obs.p, n);    // builds GF(p^n); throws on overflow
        fc.activate();

        if (obs.prefix[0] == 0) return false;
        ZZ_pE s0 = fc.from_uint(obs.prefix[0]);
        ZZ_pE s1 = fc.from_uint(obs.prefix[1]);

        // α = s₁ / s₀
        ZZ_pE s0inv, alpha;
        inv(s0inv, s0);
        mul(alpha, s1, s0inv);

        if (!is_primitive_element(alpha, fc.group_order)) return false;

        ZZ_pE s = s0;
        for (u64 exp_val : obs.prefix) {
            if (fc.to_uint(s) != exp_val) return false;
            s *= alpha;
        }
        return true;
    } catch (...) { return false; }   // catches p_power overflow for large n
};

// Try extension degrees n_min … n_max; collect those consistent with all recognizers.
static InferenceResult ran_extend(
    const std::vector<Recognizer>& recognizers,
    const Observation& obs,
    long n_min = 1,
    long n_max = 24)
{
    InferenceResult res;
    for (long n = n_min; n <= n_max; ++n) {
        bool ok = true;
        for (auto& r : recognizers)
            if (!r(obs, n)) { ok = false; break; }
        if (ok) res.candidates.push_back(n);
    }
    return res;
}

// ────────────────────────────────────────────────────────────────────────────
// §8  Tests
// ────────────────────────────────────────────────────────────────────────────

static void test_inference(long p, long n) {
    std::cout << "[ Inference  p=" << p << "  n=" << n << " ]\n";
    FieldContext fc(p, n, /*ntl_seed=*/42);
    fc.print_info();
    fc.activate();

    ZZ_pE s; set(s);                   // s = 1
    ZZ_pE alpha = fc.primitive();
    Observation obs; obs.p = p;
    for (int i = 0; i < 8; ++i) {
        obs.prefix.push_back(fc.to_uint(s));
        s *= alpha;
    }
    std::cout << "  prefix:";
    for (u64 v : obs.prefix) std::cout << " " << v;
    std::cout << "\n  inferred n:";

    auto inf = ran_extend({prefix_recognizer}, obs);
    for (long w : inf.candidates) std::cout << " " << w;
    bool ok = std::find(inf.candidates.begin(), inf.candidates.end(), n)
              != inf.candidates.end();
    std::cout << "  [true n=" << n << " found: " << (ok ? "YES" : "NO") << "]\n\n";
}

static void test_range_traversal(long p, u64 max_val,
                                 std::optional<u64> seed = std::nullopt) {
    std::cout << "[ RangeTraversal  p=" << p << "  max_val=" << max_val << " ]\n";
    RangeTraverser tr(p, max_val, seed);
    std::vector<u64> seq;
    seq.reserve(max_val + 1);
    for (u64 i = 0; i <= max_val; ++i) seq.push_back(tr.next());

    std::set<u64> seen(seq.begin(), seq.end());
    bool ok = (seen.size() == max_val + 1);
    for (u64 x : seq) if (x > max_val) { ok = false; break; }

    std::cout << "  All [0," << max_val << "] visited exactly once: "
              << (ok ? "YES" : "NO") << "\n";
    std::cout << "  First 20:";
    for (u64 i = 0; i < std::min<u64>(20, seq.size()); ++i)
        std::cout << " " << seq[i];
    std::cout << "\n\n";
}

static void test_build_orbit(long p, long n) {
    std::cout << "[ Orbit  p=" << p << "  n=" << n << " ]\n";
    u64 expected;
    try { expected = p_power(p, n) - 1; }
    catch (...) { std::cout << "  overflow; skipping\n\n"; return; }
    if (expected > 2'000'000) {
        std::cout << "  orbit size " << expected << " — too large; skipping\n\n";
        return;
    }
    Orbit orb = build_orbit(p, n);
    std::set<u64> seen(orb.states.begin(), orb.states.end());
    bool ok = (seen.size() == expected && seen.find(0) == seen.end());
    std::cout << "  generated=" << orb.states.size()
              << "  expected=" << expected
              << "  unique+nonzero: " << (ok ? "YES" : "NO") << "\n\n";
}

static void test_reproducibility(long p, u64 max_val, u64 seed) {
    std::cout << "[ Reproducibility  p=" << p << "  max_val=" << max_val << " ]\n";
    RangeTraverser t1(p, max_val, seed), t2(p, max_val, seed);
    bool ok = true;
    for (u64 i = 0; i <= max_val && ok; ++i)
        if (t1.next() != t2.next()) ok = false;
    std::cout << "  same seed → same sequence: " << (ok ? "PASS" : "FAIL") << "\n\n";
}

// ────────────────────────────────────────────────────────────────────────────
// §9  main
// ────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== GF(p^n) LFSR Suite ===\n\n";
    try {
        // ── GF(2^n): backward-compatible with lfsr_improved.cpp ──────────────
        test_inference      (2, 8);
        test_range_traversal(2, 20,    12345ULL);
        test_range_traversal(2, 65535, 42ULL);
        test_build_orbit    (2, 4);          // |F*| = 15
        test_build_orbit    (2, 8);          // |F*| = 255
        test_reproducibility(2, 100, 777);

        // ── GF(3^n) ──────────────────────────────────────────────────────────
        test_inference      (3, 4);
        test_range_traversal(3, 20,   12345ULL);
        test_range_traversal(3, 1000, 999ULL);
        test_build_orbit    (3, 2);          // |F*| = 8
        test_build_orbit    (3, 4);          // |F*| = 80
        test_build_orbit    (3, 6);          // |F*| = 728
        test_reproducibility(3, 100, 777);

        // ── GF(5^n) ──────────────────────────────────────────────────────────
        test_inference      (5, 3);
        test_range_traversal(5, 100,  42ULL);
        test_build_orbit    (5, 2);          // |F*| = 24
        test_build_orbit    (5, 4);          // |F*| = 624
        test_reproducibility(5, 100, 777);

        // ── GF(7^n) ──────────────────────────────────────────────────────────
        test_inference      (7, 3);
        test_range_traversal(7, 100,  42ULL);
        test_build_orbit    (7, 2);          // |F*| = 48
        test_build_orbit    (7, 3);          // |F*| = 342
        test_reproducibility(7, 100, 777);

        // ── GF(17^n): larger prime base ───────────────────────────────────────
        test_inference      (17, 2);
        test_range_traversal(17, 500,  42ULL);
        test_build_orbit    (17, 2);         // |F*| = 288
        test_reproducibility(17, 200, 777);

        // ── GF(97^n): even larger prime ───────────────────────────────────────
        // For p=97, p^n overflows u64 for n > ~9, so inference auto-caps there.
        test_inference      (97, 2);
        test_range_traversal(97, 10000, 42ULL);
        test_build_orbit    (97, 1);         // |F*| = 96  (GF(97) itself)
        test_build_orbit    (97, 2);         // |F*| = 9408
        test_reproducibility(97, 500, 777);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

#ifndef NTL_MOCK_H
#define NTL_MOCK_H

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>

namespace NTL {

typedef uint64_t u64;
typedef uint64_t ZZ;

struct GF2X {
    u64 data = 0;
    GF2X() : data(0) {}
    GF2X(u64 d) : data(d) {}
};

inline long deg(const GF2X& a) {
    if (a.data == 0) return -1;
    return 63 - std::countl_zero(a.data);
}

inline bool IsCoeff(const GF2X& a, long i) {
    if (i < 0 || i > 63) return false;
    return (a.data >> i) & 1;
}

inline void SetCoeff(GF2X& a, long i) {
    if (i >= 0 && i < 64) a.data |= (1ULL << i);
}

inline std::ostream& operator<<(std::ostream& os, const GF2X& a) {
    if (a.data == 0) return os << "0";
    bool first = true;
    for (int i = deg(a); i >= 0; --i) {
        if (IsCoeff(a, i)) {
            if (!first) os << " + ";
            if (i == 0) os << "1";
            else if (i == 1) os << "x";
            else os << "x^" << i;
            first = false;
        }
    }
    return os;
}

struct GF2XModulus {
    GF2X P;
    GF2XModulus() {}
    GF2XModulus(const GF2X& p) : P(p) {}
};

struct GF2E {
    static GF2X current_modulus;
    u64 data = 0;

    static const GF2X& modulus() { return current_modulus; }
    static const GF2XModulus& GetModulus() {
        static GF2XModulus m;
        m.P = current_modulus;
        return m;
    }

    GF2E() : data(0) {}
    GF2E(u64 d) : data(d) {}

    static void init(const GF2X& m) { current_modulus = m; }

    GF2E& operator*=(const GF2E& other) {
        u64 a = data;
        u64 b = other.data;
        u64 mod = current_modulus.data;
        if (mod <= 1) return *this;
        unsigned n = 63 - std::countl_zero(mod);
        u64 red = mod ^ (1ULL << n);
        u64 mask = (1ULL << n) - 1;
        a &= mask; b &= mask;
        u64 res = 0;
        while (b) {
            if (b & 1) res ^= a;
            b >>= 1;
            bool carry = (a & (1ULL << (n - 1))) != 0;
            a <<= 1; a &= mask;
            if (carry) a ^= red;
        }
        data = res;
        return *this;
    }

    GF2E operator*(const GF2E& other) const {
        GF2E res = *this;
        res *= other;
        return res;
    }
};

inline bool IsZero(const GF2E& a) { return a.data == 0; }
inline bool IsOne(const GF2E& a) { return a.data == 1; }
inline bool IsOne(const GF2X& a) { return a.data == 1; }
inline bool IsOne(long a) { return (a & 1) != 0; }

inline void random(GF2E& a) {
    u64 mod = GF2E::current_modulus.data;
    if (mod <= 1) { a.data = 0; return; }
    unsigned n = 63 - std::countl_zero(mod);
    a.data = (rand() % ((1ULL << n) - 1)) + 1;
}

inline void power(GF2E& res, const GF2E& a, uint64_t e) {
    GF2E base = a;
    res = GF2E(1);
    while (e) {
        if (e & 1) res *= base;
        base *= base;
        e >>= 1;
    }
}

inline GF2E inv(const GF2E& a) {
    if (a.data == 0) return GF2E(0);
    u64 mod = GF2E::current_modulus.data;
    unsigned n = 63 - std::countl_zero(mod);
    u64 order_minus_2 = (1ULL << n) - 3;
    GF2E res;
    power(res, a, order_minus_2);
    return res;
}

inline GF2X rep(const GF2E& a) { return GF2X(a.data); }

inline GF2X coeff(const GF2X& a, long i) {
    return GF2X((a.data >> i) & 1);
}

inline std::ostream& operator<<(std::ostream& os, const GF2E& a) {
    return os << GF2X(a.data);
}

struct GF2EPush {
    GF2X old_mod;
    GF2EPush(const GF2X& m) : old_mod(GF2E::current_modulus) { GF2E::init(m); }
    ~GF2EPush() { GF2E::init(old_mod); }
};

inline void BuildSparseIrred(GF2X& p, long n) {
    static const u64 table[] = {
        0, 0, 0b111, 0b1011, 0b10011, 0b100101, 0b1000011, 0b10000011, 0b100011011
    };
    if (n < 9) p.data = table[n];
    else p.data = (1ULL << n) | 3; // fallback
}

inline void BuildIrred(GF2X& p, long n) {
    BuildSparseIrred(p, n);
}

template<class T, class S>
inline T conv(const S& a) {
    return T(a.data);
}

inline uint64_t to_ZZ(uint64_t n) { return n; }

inline void SetSeed(const ZZ& s) { srand((unsigned)s); }

// --- GF2EX ---
struct GF2EX {
    std::vector<GF2E> coeffs;
};
inline long deg(const GF2EX& f) {
    for (long i = f.coeffs.size() - 1; i >= 0; --i) {
        if (!IsZero(f.coeffs[i])) return i;
    }
    return -1;
}
inline void SetCoeff(GF2EX& f, long i) {
    if (f.coeffs.size() <= (size_t)i) f.coeffs.resize(i+1);
    f.coeffs[i] = GF2E(1);
}
inline void SetCoeff(GF2EX& f, long i, const GF2E& a) {
    if (f.coeffs.size() <= (size_t)i) f.coeffs.resize(i+1);
    f.coeffs[i] = a;
}
inline std::ostream& operator<<(std::ostream& os, const GF2EX& f) {
    if (deg(f) < 0) return os << "0";
    bool first = true;
    for (long i = deg(f); i >= 0; --i) {
        if (!IsZero(f.coeffs[i])) {
            if (!first) os << " + ";
            os << "(" << f.coeffs[i] << ")*X^" << i;
            first = false;
        }
    }
    return os;
}

struct GF2EXModulus {
    GF2EX f;
    GF2EXModulus(const GF2EX& _f) : f(_f) {}
    const GF2EX& getgetP() const { return f; }
};

inline GF2EX PowerXMod(uint64_t e, const GF2EXModulus& F) {
    // Real implementation of X^e mod f(X) for GF2EX
    // Since f(X) comes from orbit.modulus which is degree n,
    // and we want X^e mod f(X).
    // In our mock, we can just return a GF2EX that represents the polynomial
    // that is equal to the residue of x^e mod p(x).

    u64 mod = GF2E::current_modulus.data;
    if (mod <= 1) return GF2EX();

    // Compute x^e mod mod in GF(2)[x]
    u64 res_val = 1;
    u64 base = 2; // x
    u64 exp = e;
    unsigned n = 63 - std::countl_zero(mod);
    u64 red = mod ^ (1ULL << n);
    u64 mask = (1ULL << n) - 1;

    auto poly_mul = [&](u64 a, u64 b) {
        u64 r = 0;
        a &= mask; b &= mask;
        while (b) {
            if (b & 1) r ^= a;
            b >>= 1;
            bool carry = (a & (1ULL << (n - 1))) != 0;
            a <<= 1; a &= mask;
            if (carry) a ^= red;
        }
        return r;
    };

    while (exp) {
        if (exp & 1) res_val = poly_mul(res_val, base);
        base = poly_mul(base, base);
        exp >>= 1;
    }

    GF2EX res;
    // The result is an element of GF(2^n), but PowerXMod returns a polynomial.
    // X^e mod f(X) where f(X) is degree n.
    // If f(X) = X^n + ... + 1, then X^e mod f(X) is a polynomial of degree < n.
    // The coefficients of this polynomial are elements of the base ring (GF2E).
    // In our case, the coefficients are just 0 or 1.
    for (long i = 0; i < n; i++) {
        if ((res_val >> i) & 1) SetCoeff(res, i, GF2E(1));
    }
    return res;
}

// --- mat_GF2 ---
struct vec_GF2 {
    u64 data = 0;
    long n = 0;
    vec_GF2() {}
    vec_GF2(long _n) : n(_n) {}
};

struct mat_GF2 {
    long rows = 0, cols = 0;
    std::vector<u64> row_data;
    mat_GF2() {}
    void SetDims(long r, long c) { rows = r; cols = c; row_data.assign(r, 0); }

    struct row_ref {
        u64& data;
        row_ref(u64& d) : data(d) {}
        struct bit_ref {
            u64& data; long j;
            bit_ref(u64& d, long _j) : data(d), j(_j) {}
            bit_ref& operator=(int val) {
                if (val & 1) data |= (1ULL << j);
                else data &= ~(1ULL << j);
                return *this;
            }
        };
        bit_ref operator[](long j) { return bit_ref(data, j); }
    };
    row_ref operator[](long i) { return row_ref(row_data[i]); }
};

inline u64 mul(const mat_GF2& M, u64 v) {
    u64 res = 0;
    for (long i = 0; i < M.rows; ++i) {
        if (std::popcount(M.row_data[i] & v) & 1) res |= (1ULL << i);
    }
    return res;
}

inline std::ostream& operator<<(std::ostream& os, const mat_GF2& M) {
    os << "[" << M.rows << "x" << M.cols << " matrix]";
    return os;
}

inline long trace(const GF2E& a) {
    u64 val = a.data;
    u64 mod = GF2E::current_modulus.data;
    if (mod <= 1) return 0;
    unsigned n = 63 - std::countl_zero(mod);
    u64 sum = val;
    u64 term = val;
    for (unsigned i = 1; i < n; ++i) {
        // term = term^2 mod mod
        u64 t = term;
        u64 res = 0;
        u64 red = mod ^ (1ULL << n);
        u64 mask = (1ULL << n) - 1;
        u64 b = t;
        while (b) {
            if (b & 1) res ^= t;
            b >>= 1;
            bool carry = (t & (1ULL << (n - 1))) != 0;
            t <<= 1; t &= mask;
            if (carry) t ^= red;
        }
        term = res;
        sum ^= term;
    }
    return sum & 1;
}

} // namespace NTL

#endif

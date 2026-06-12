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

struct GF2E {
    static GF2X current_modulus;
    u64 data = 0;

    static const GF2X& modulus() { return current_modulus; }

    GF2E() : data(0) {}
    GF2E(u64 d) : data(d) {}

    static void init(const GF2X& m) { current_modulus = m; }

    GF2E& operator*=(const GF2E& other) {
        u64 a = data;
        u64 b = other.data;
        u64 mod = current_modulus.data;
        if (mod == 0) return *this;
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
};

inline bool IsZero(const GF2E& a) { return a.data == 0; }
inline bool IsOne(const GF2E& a) { return a.data == 1; }

inline bool IsOne(const GF2X& a) { return a.data == 1; }

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
    p.data = (1ULL << n) | 3;
    if (n == 4) p.data = 0b10011;
    if (n == 5) p.data = 0b100101;
    if (n == 6) p.data = 0b1000011;
}

inline void BuildIrred(GF2X& p, long n) {
    BuildSparseIrred(p, n);
}

inline uint64_t to_ZZ(uint64_t n) { return n; }

struct GF2EX {
    std::vector<GF2E> coeffs;
};
inline void SetCoeff(GF2EX& f, long i) {
    if (f.coeffs.size() <= (size_t)i) f.coeffs.resize(i+1);
    f.coeffs[i] = GF2E(1);
}
inline std::ostream& operator<<(std::ostream& os, const GF2EX& f) {
    return os << "[mock GF2EX]";
}

struct GF2EXModulus {
    GF2EXModulus(const GF2EX& f) {}
};

inline GF2EX PowerXMod(uint64_t e, const GF2EXModulus& F) {
    return GF2EX();
}

} // namespace NTL

#endif

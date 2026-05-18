// FieldExtension.h // Compact field extension library for ESP32 Arduino // Allows higher precision and numerical stability by truncated polynomial extensions over a transcendental constant

#ifndef FIELD_EXTENSION_H_V2
#define FIELD_EXTENSION_H_V2

#include <Arduino.h>

// Template for field extension over a transcendental constant C // Represents numbers in Q(C) truncated to degree N-1: a0 + a1C + a2C^2 + ... + a_{N-1}*C^{N-1}

template<typename T, T C, int N> struct FieldExt { T a[N];

// Constructors
FieldExt() { for (int i = 0; i < N; ++i) a[i] = T(0); }
FieldExt(T x) { a[0] = x; for (int i = 1; i < N; ++i) a[i] = T(0); }

// Access coefficient
T& operator[](int i) { return a[i]; }
const T& operator[](int i) const { return a[i]; }

// Addition
FieldExt operator+(const FieldExt& o) const {
    FieldExt r;
    for (int i = 0; i < N; ++i) r.a[i] = a[i] + o.a[i];
    return r;
}
FieldExt& operator+=(const FieldExt& o) {
    for (int i = 0; i < N; ++i) a[i] += o.a[i];
    return *this;
}
friend FieldExt operator+(T lhs, const FieldExt& rhs) { return FieldExt(lhs) + rhs; }
friend FieldExt operator+(const FieldExt& lhs, T rhs) { return lhs + FieldExt(rhs); }

FieldExt operator-(const FieldExt& o) const {
    FieldExt r;
    for (int i = 0; i < N; ++i) r.a[i] = a[i] - o.a[i];
    return r;
}
FieldExt& operator-=(const FieldExt& o) {
    for (int i = 0; i < N; ++i) a[i] -= o.a[i];
    return *this;
}
friend FieldExt operator-(T lhs, const FieldExt& rhs) { return FieldExt(lhs) - rhs; }
friend FieldExt operator-(const FieldExt& lhs, T rhs) { return lhs - FieldExt(rhs); }

// Multiplication (truncated)
FieldExt operator*(const FieldExt& o) const {
    FieldExt r;
    for (int i = 0; i < N; ++i) {
        if (a[i] == 0) continue;
        for (int j = 0; j < N - i; ++j) {
            r.a[i + j] += a[i] * o.a[j];
        }
    }
    return r;
}
FieldExt& operator*=(const FieldExt& o) { return *this = (*this) * o; }
friend FieldExt operator*(T lhs, const FieldExt& rhs) { return FieldExt(lhs) * rhs; }
friend FieldExt operator*(const FieldExt& lhs, T rhs) {
    FieldExt r;
    for (int i = 0; i < N; ++i) r.a[i] = lhs.a[i] * rhs;
    return r;
}

// Scalar division
FieldExt operator/(T s) const {
    FieldExt r;
    T inv = T(1) / s;
    for (int i = 0; i < N; ++i) r.a[i] = a[i] * inv;
    return r;
}

// Approximate reciprocal by Newton-Raphson on scalar part
FieldExt recip(int iter = 3) const {
    // Only valid if a[0] != 0
    FieldExt x; x.a[0] = 1.0 / a[0];
    // refine x = x*(2 - this*x)
    for (int k = 0; k < iter; ++k) {
        FieldExt prod = (*this) * x;
        // compute (2 - prod)
        FieldExt two_minus;
        two_minus.a[0] = 2 - prod.a[0];
        for (int i = 1; i < N; ++i) two_minus.a[i] = -prod.a[i];
        x = x * two_minus;
    }
    return x;
}

FieldExt operator/(const FieldExt& o) const {
    return (*this) * o.recip();
}
FieldExt& operator/=(const FieldExt& o) { return *this = (*this) / o; }

friend FieldExt sin(const FieldExt& x) {
    FieldExt res;
    res.a[0] = std::sin(x.eval());
    return res;
}
friend FieldExt cos(const FieldExt& x) {
    FieldExt res;
    res.a[0] = std::cos(x.eval());
    return res;
}
friend FieldExt exp(const FieldExt& x) {
    FieldExt res;
    res.a[0] = std::exp(x.eval());
    return res;
}
friend FieldExt log(const FieldExt& x) {
    FieldExt res;
    res.a[0] = std::log(x.eval());
    return res;
}
friend FieldExt sqrt(const FieldExt& x) {
    FieldExt res;
    res.a[0] = std::sqrt(x.eval());
    return res;
}

// Convert to T by evaluating polynomial: Horner's method
T eval() const {
    T res = a[N-1];
    for (int i = N-2; i >= 0; --i) {
        res = res * C + a[i];
    }
    return res;
}

};

#endif // FIELD_EXTENSION_H_V2

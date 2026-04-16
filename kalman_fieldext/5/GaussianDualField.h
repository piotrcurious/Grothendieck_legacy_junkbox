// GaussianDualField.h
#ifndef GAUSSIAN_DUAL_FIELD_H
#define GAUSSIAN_DUAL_FIELD_H

#include <cmath>
#include <iostream>
#include <algorithm>

/**
 * @brief GaussianDualField implements a hybrid number system: a + b*ε + c*δ
 * ε² = σ² (Gaussian/Hyperbolic component)
 * δ² = 0  (Dual component for sensitivity analysis)
 * εδ = 0  (Cross-term assumed negligible/zero)
 *
 * Includes Kahan compensation for addition/subtraction.
 */
template<typename T = double>
class GaussianDualField {
public:
    T sigma2 = 0.01;
    T nominal;   // a
    T noise;     // b (ε component)
    T delta;     // c (δ component)

    // Kahan compensation terms
    T nominal_c = 0;
    T noise_c   = 0;
    T delta_c   = 0;

    constexpr GaussianDualField(T a = 0, T b = 0, T c = 0, T s2 = 0.01)
      : sigma2(s2), nominal(a), noise(b), delta(c) {}

    // Kahan add: sum += value
    static void kahanAdd(T &sum, T &comp, T value) {
        T y = value - comp;
        T t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }

    // precise multiplication via FMA
    static void preciseMul(T a, T b, T &prod, T &err) {
        prod = a * b;
        err  = std::fma(a, b, -prod);
    }

    GaussianDualField operator+(const GaussianDualField &o) const {
        GaussianDualField r(nominal, noise, delta, sigma2);
        r.nominal_c = nominal_c;
        r.noise_c = noise_c;
        r.delta_c = delta_c;
        kahanAdd(r.nominal, r.nominal_c, o.nominal);
        kahanAdd(r.nominal, r.nominal_c, -o.nominal_c);
        kahanAdd(r.noise,   r.noise_c,   o.noise);
        kahanAdd(r.noise,   r.noise_c,   -o.noise_c);
        kahanAdd(r.delta,   r.delta_c,   o.delta);
        kahanAdd(r.delta,   r.delta_c,   -o.delta_c);
        return r;
    }

    GaussianDualField operator-(const GaussianDualField &o) const {
        GaussianDualField r(nominal, noise, delta, sigma2);
        r.nominal_c = nominal_c;
        r.noise_c = noise_c;
        r.delta_c = delta_c;
        kahanAdd(r.nominal, r.nominal_c, -o.nominal);
        kahanAdd(r.nominal, r.nominal_c, o.nominal_c);
        kahanAdd(r.noise,   r.noise_c,   -o.noise);
        kahanAdd(r.noise,   r.noise_c,   o.noise_c);
        kahanAdd(r.delta,   r.delta_c,   -o.delta);
        kahanAdd(r.delta,   r.delta_c,   o.delta_c);
        return r;
    }

    GaussianDualField operator*(const GaussianDualField &o) const {
        T p1, e1; preciseMul(nominal, o.nominal, p1, e1);
        T p2, e2; preciseMul(noise,   o.noise,   p2, e2);
        T p2s = p2 * sigma2;
        T e2s = std::fma(p2, sigma2, -p2s) + sigma2 * e2;

        GaussianDualField r(0, 0, 0, sigma2);
        kahanAdd(r.nominal, r.nominal_c, p1);
        kahanAdd(r.nominal, r.nominal_c, p2s);
        kahanAdd(r.nominal, r.nominal_c, e1 + e2s);

        T q1, f1; preciseMul(nominal, o.noise, q1, f1);
        T q2, f2; preciseMul(noise,   o.nominal, q2, f2);
        kahanAdd(r.noise, r.noise_c, q1);
        kahanAdd(r.noise, r.noise_c, q2);
        kahanAdd(r.noise, r.noise_c, f1 + f2);

        T d1, g1; preciseMul(nominal, o.delta, d1, g1);
        T d2, g2; preciseMul(delta,   o.nominal, d2, g2);
        kahanAdd(r.delta, r.delta_c, d1);
        kahanAdd(r.delta, r.delta_c, d2);
        kahanAdd(r.delta, r.delta_c, g1 + g2);

        return r;
    }

    GaussianDualField inv() const {
        T d1, e1; preciseMul(nominal, nominal, d1, e1);
        T d2, e2; preciseMul(noise,   noise,   d2, e2);
        T d2s = d2 * sigma2;
        T e2s = std::fma(d2, sigma2, -d2s) + sigma2 * e2;
        T denom = (d1 + e1) - (d2s + e2s);
        if (std::abs(denom) < 1e-20) return GaussianDualField(0, 0, 0, sigma2);
        T invDenom = 1.0 / denom;
        invDenom = invDenom * (2.0 - denom * invDenom);

        GaussianDualField r(0, 0, 0, sigma2);
        T a, ea; preciseMul(nominal, invDenom, a, ea);
        r.nominal = a;
        kahanAdd(r.nominal, r.nominal_c, ea);

        T b, eb; preciseMul(-noise, invDenom, b, eb);
        r.noise = b;
        kahanAdd(r.noise, r.noise_c, eb);

        if (std::abs(denom) > 1e-20) {
            r.delta = -delta / denom;
        }
        return r;
    }

    GaussianDualField operator/(const GaussianDualField &o) const {
        return (*this) * o.inv();
    }

    GaussianDualField operator+(T s) const { return (*this) + GaussianDualField(s, 0, 0, sigma2); }
    GaussianDualField operator-(T s) const { return (*this) - GaussianDualField(s, 0, 0, sigma2); }

    GaussianDualField operator*(T s) const {
        GaussianDualField r(0, 0, 0, sigma2);
        T p, e;
        preciseMul(nominal, s, p, e);
        r.nominal = p;
        kahanAdd(r.nominal, r.nominal_c, e + nominal_c * s);
        preciseMul(noise, s, p, e);
        r.noise = p;
        kahanAdd(r.noise, r.noise_c, e + noise_c * s);
        preciseMul(delta, s, p, e);
        r.delta = p;
        kahanAdd(r.delta, r.delta_c, e + delta_c * s);
        return r;
    }

    GaussianDualField operator/(T s) const {
        if (std::abs(s) < 1e-20) return GaussianDualField(0, 0, 0, sigma2);
        return (*this) * (1.0 / s);
    }

    // Non-member friend operators for T op Field
    friend GaussianDualField operator+(T s, const GaussianDualField& f) { return f + s; }
    friend GaussianDualField operator-(T s, const GaussianDualField& f) { return GaussianDualField(s, 0, 0, f.sigma2) - f; }
    friend GaussianDualField operator*(T s, const GaussianDualField& f) { return f * s; }
    friend GaussianDualField operator/(T s, const GaussianDualField& f) { return GaussianDualField(s, 0, 0, f.sigma2) / f; }

    static GaussianDualField exp(const GaussianDualField &x) {
        T ea = std::exp(x.nominal);
        T s = std::sqrt(x.sigma2);
        T bs = x.noise * s;
        T ch = std::cosh(bs);
        T sh = std::sinh(bs);
        T res_nom = ea * ch;
        T res_noise = (s > 1e-15) ? ea * sh / s : ea * x.noise;
        T res_delta = res_nom * x.delta;
        return GaussianDualField(res_nom, res_noise, res_delta, x.sigma2);
    }

    static GaussianDualField log(const GaussianDualField &x) {
        T s = std::sqrt(x.sigma2);
        T a = x.nominal;
        T b = x.noise;
        T denom = a * a - x.sigma2 * b * b;
        if (denom <= 0) return GaussianDualField(0, 0, 0, x.sigma2);
        T res_nom = 0.5 * std::log(denom);
        T res_noise = (s > 1e-15) ? (1.0 / s) * std::atanh(b * s / a) : b / a;
        T res_delta = (x.delta * a) / denom;
        return GaussianDualField(res_nom, res_noise, res_delta, x.sigma2);
    }

    static GaussianDualField sqrt(const GaussianDualField &x) {
        T s2 = x.sigma2;
        T a = x.nominal;
        T b = x.noise;
        T denom = a * a - s2 * b * b;
        T disc = std::sqrt(std::max(T(0), denom));
        T val_x = std::sqrt((a + disc) / 2.0);
        if (val_x <= 0) return GaussianDualField(0, 0, 0, s2);
        T res_nom = val_x;
        T res_noise = b / (2.0 * val_x);
        T res_delta = x.delta / (2.0 * val_x);
        return GaussianDualField(res_nom, res_noise, res_delta, s2);
    }

    static GaussianDualField sin(const GaussianDualField &x) {
        T a = x.nominal; T b = x.noise; T s = std::sqrt(x.sigma2);
        T bs = b * s;
        T ch = std::cosh(bs);
        T sh = std::sinh(bs);
        T res_nom = std::sin(a) * ch;
        T res_noise = (s > 1e-15) ? std::cos(a) * (sh / s) : std::cos(a) * b;
        T res_delta = std::cos(a) * ch * x.delta;
        return GaussianDualField(res_nom, res_noise, res_delta, x.sigma2);
    }

    static GaussianDualField cos(const GaussianDualField &x) {
        T a = x.nominal; T b = x.noise; T s = std::sqrt(x.sigma2);
        T bs = b * s;
        T ch = std::cosh(bs);
        T sh = std::sinh(bs);
        T res_nom = std::cos(a) * ch;
        T res_noise = (s > 1e-15) ? -std::sin(a) * (sh / s) : -std::sin(a) * b;
        T res_delta = -std::sin(a) * ch * x.delta;
        return GaussianDualField(res_nom, res_noise, res_delta, x.sigma2);
    }

    static GaussianDualField tan(const GaussianDualField &x) {
        return sin(x) / cos(x);
    }

    static GaussianDualField asin(const GaussianDualField &x) {
        T a = x.nominal;
        T limit = 1.0 - 1e-15;
        if (a > limit) a = limit;
        if (a < -limit) a = -limit;
        T der = 1.0 / std::sqrt(1.0 - a * a);
        return GaussianDualField(std::asin(a), x.noise * der, x.delta * der, x.sigma2);
    }

    static GaussianDualField acos(const GaussianDualField &x) {
        T a = x.nominal;
        T limit = 1.0 - 1e-15;
        if (a > limit) a = limit;
        if (a < -limit) a = -limit;
        T der = -1.0 / std::sqrt(1.0 - a * a);
        return GaussianDualField(std::acos(a), x.noise * der, x.delta * der, x.sigma2);
    }

    static GaussianDualField atan(const GaussianDualField &x) {
        T a = x.nominal;
        T der = 1.0 / (1.0 + a * a);
        return GaussianDualField(std::atan(a), x.noise * der, x.delta * der, x.sigma2);
    }

    static GaussianDualField atan2(const GaussianDualField &y, const GaussianDualField &x) {
        T a = x.nominal; T b = y.nominal;
        T d = a * a + b * b;
        if (d < 1e-20) return GaussianDualField(0, 0, 0, x.sigma2);
        T res_nom = std::atan2(b, a);
        T res_noise = (y.noise * a - x.noise * b) / d;
        T res_delta = (y.delta * a - x.delta * b) / d;
        return GaussianDualField(res_nom, res_noise, res_delta, x.sigma2);
    }

    static GaussianDualField sinh(const GaussianDualField &x) {
        T a = x.nominal; T b = x.noise; T s = std::sqrt(x.sigma2);
        T bs = b * s;
        T ch = std::cosh(bs);
        T sh = std::sinh(bs);
        T res_nom = std::sinh(a) * ch;
        T res_noise = (s > 1e-15) ? std::cosh(a) * (sh / s) : std::cosh(a) * b;
        T res_delta = std::cosh(a) * ch * x.delta;
        return GaussianDualField(res_nom, res_noise, res_delta, x.sigma2);
    }

    static GaussianDualField cosh(const GaussianDualField &x) {
        T a = x.nominal; T b = x.noise; T s = std::sqrt(x.sigma2);
        T bs = b * s;
        T ch = std::cosh(bs);
        T sh = std::sinh(bs);
        T res_nom = std::cosh(a) * ch;
        T res_noise = (s > 1e-15) ? std::sinh(a) * (sh / s) : std::sinh(a) * b;
        T res_delta = std::sinh(a) * ch * x.delta;
        return GaussianDualField(res_nom, res_noise, res_delta, x.sigma2);
    }

    static GaussianDualField tanh(const GaussianDualField &x) {
        return sinh(x) / cosh(x);
    }

    static GaussianDualField pow(const GaussianDualField &x, double y) {
        return exp(log(x) * y);
    }

    static GaussianDualField pow(const GaussianDualField &x, const GaussianDualField &y) {
        return exp(log(x) * y);
    }

    static T norm(const GaussianDualField &x) {
        return std::sqrt(std::abs(x.nominal * x.nominal - x.sigma2 * x.noise * x.noise));
    }

    static GaussianDualField abs(const GaussianDualField &x) {
        if (x.nominal >= 0) return x;
        return x * T(-1.0);
    }

    bool is_finite() const {
        return std::isfinite(nominal) && std::isfinite(noise) && std::isfinite(delta);
    }

    void clear_compensation() {
        nominal_c = 0; noise_c = 0; delta_c = 0;
    }

    operator T() const { return nominal; }
};

#endif // GAUSSIAN_DUAL_FIELD_H

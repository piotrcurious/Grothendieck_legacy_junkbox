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
        // (a1 + b1ε + c1δ)*(a2 + b2ε + c2δ) = a1a2 + σ²b1b2 + (a1b2 + b1a2)ε + (a1c2 + c1a2)δ
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

        // delta = -c / (a² - σ²b²)
        if (std::abs(denom) > 1e-20) {
            r.delta = -delta / denom;
        }

        return r;
    }

    GaussianDualField operator/(const GaussianDualField &o) const {
        T d1, e1; preciseMul(o.nominal, o.nominal, d1, e1);
        T d2, e2; preciseMul(o.noise,   o.noise,   d2, e2);
        T d2s = d2 * o.sigma2;
        T e2s = std::fma(d2, o.sigma2, -d2s) + o.sigma2 * e2;
        T denom = (d1 + e1) - (d2s + e2s);
        if (std::abs(denom) < 1e-20) return GaussianDualField(0, 0, 0, sigma2);
        T invDenom = 1.0 / denom;

        T n1, f1; preciseMul(nominal, o.nominal, n1, f1);
        T n2, f2; preciseMul(noise,   o.noise,   n2, f2);
        T n2s = n2 * sigma2;
        T f2s = std::fma(n2, sigma2, -n2s) + sigma2 * f2;
        T numNom = n1 - n2s;
        T errNom = (n1 - numNom) - n2s + f1 - f2s;

        GaussianDualField r(0, 0, 0, sigma2);
        r.nominal = numNom * invDenom;
        kahanAdd(r.nominal, r.nominal_c, errNom * invDenom);

        T m1, h1; preciseMul(noise,   o.nominal, m1, h1);
        T m2, h2; preciseMul(nominal, o.noise,   m2, h2);
        T numN = m1 - m2;
        T errN = (m1 - numN) - m2 + h1 - h2;
        r.noise = numN * invDenom;
        kahanAdd(r.noise, r.noise_c, errN * invDenom);

        // delta = (c1 * nominal(Z2) - nominal(Z1) * c2) / denom
        // Z = a + be. 1/(Z2 + c2d) = 1/Z2 - c2/Z2^2 d
        // (Z1 + c1d)/(Z2 + c2d) = Z1/Z2 + (c1*Z2 - Z1*c2)/Z2^2 d
        // nominal part of (c1*Z2 - Z1*c2) is c1*a2 - a1*c2
        // nominal part of Z2^2 is a2^2 + sigma2*b2^2
        // But wait, Z2^2 in hyperbolic is a2^2 + sigma2*b2^2 + 2*a2*b2*e.
        // The nominal part of 1/Z2^2 is (a2^2 + sigma2*b2^2) / (a2^2 - sigma2*b2^2)^2
        // This is getting complicated. Let's use the property that Z2*conj(Z2) = a2^2 - sigma2*b2^2 = denom.
        // Z1/Z2 = Z1*conj(Z2)/denom
        // Sensitivity c of (Z1/Z2) is the dual part.
        // d/dp (Z1/Z2) = ( (dZ1/dp)*Z2 - Z1*(dZ2/dp) ) / Z2^2
        // If we only care about the nominal part of the sensitivity:
        // nominal( (c1*Z2 - Z1*c2)/Z2^2 ) = nominal( (c1*Z2 - Z1*c2)*conj(Z2)^2 / denom^2 )
        // conj(Z2)^2 = (a2 - b2e)^2 = a2^2 + sigma2*b2^2 - 2*a2*b2*e
        // c1*Z2 - Z1*c2 = (c1*a2 - a1*c2) + (c1*b2 - b1*c2)e
        // nominal( ((c1*a2-a1*c2) + (c1*b2-b1*c2)e) * (a2^2+sigma2*b2^2 - 2*a2*b2*e) )
        // = (c1*a2-a1*c2)*(a2^2+sigma2*b2^2) - (c1*b2-b1*c2)*(2*a2*b2*sigma2)

        T a1 = nominal; T b1 = noise; T c1 = delta;
        T a2 = o.nominal; T b2 = o.noise; T c2 = o.delta;
        T s2 = sigma2;

        T term1 = (c1 * a2 - a1 * c2) * (a2 * a2 + s2 * b2 * b2);
        T term2 = (c1 * b2 - b1 * c2) * (2.0 * a2 * b2 * s2);
        r.delta = (term1 - term2) / (denom * denom);

        return r;
    }

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

    static GaussianDualField exp(const GaussianDualField &x) {
        T ea = std::exp(x.nominal);
        T s = std::sqrt(x.sigma2);
        T bs = x.noise * s;
        T ch = std::cosh(bs);
        T sh = std::sinh(bs);

        T res_nom = ea * ch;
        T res_noise = ea * sh / s;

        // d/dp exp(Z) = exp(Z) * dZ/dp
        // x.delta is the dual part of Z.
        // exp(Z + c*d) = exp(Z) * (1 + c*d) = exp(Z) + exp(Z)*c*d
        // nominal part of exp(Z)*c is nominal(exp(Z))*c = res_nom * x.delta
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
        T res_noise = (1.0 / s) * std::atanh(b * s / a);

        // d/dp log(Z) = (1/Z) * dZ/dp
        // nominal part of (1/Z * c) is nominal(1/Z) * c
        // nominal(1/Z) = a / (a^2 - s2*b^2) = a / denom
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

        // d/dp sqrt(Z) = 1/(2*sqrt(Z)) * dZ/dp
        // nominal(1/(2*sqrt(Z)) * c) = 1/(2*res_nom) * c
        T res_delta = x.delta / (2.0 * val_x);

        return GaussianDualField(res_nom, res_noise, res_delta, s2);
    }

    operator T() const { return nominal; }
};

#endif // GAUSSIAN_DUAL_FIELD_H

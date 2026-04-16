// GaussianDualField.h
#ifndef GAUSSIAN_DUAL_FIELD_H
#define GAUSSIAN_DUAL_FIELD_H

#include <cmath>

template<typename T = double>
class GaussianDualField {
public:
    T sigma2 = 0.01;
    T nominal;   // a
    T noise;     // b (ε component, ε² = σ²)
    T delta;     // c (δ component, δ² = 0)

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
        err  = fma(a, b, -prod);
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
        // Compute nominal = a1*a2 + σ²*(b1*b2), with FMA error
        T p1, e1; preciseMul(nominal, o.nominal, p1, e1);
        T p2, e2; preciseMul(noise,   o.noise,   p2, e2);
        T p2s = p2 * sigma2;
        T e2s = fma(p2, sigma2, -p2s) + sigma2 * e2;

        GaussianDualField r(0, 0, 0, sigma2);
        kahanAdd(r.nominal, r.nominal_c, p1);
        kahanAdd(r.nominal, r.nominal_c, p2s);
        kahanAdd(r.nominal, r.nominal_c, e1 + e2s);

        // noise   = a1*b2 + b1*a2
        T q1, f1; preciseMul(nominal, o.noise, q1, f1);
        T q2, f2; preciseMul(noise,   o.nominal, q2, f2);
        kahanAdd(r.noise, r.noise_c, q1);
        kahanAdd(r.noise, r.noise_c, q2);
        kahanAdd(r.noise, r.noise_c, f1 + f2);

        // delta   = a1*d2 + d1*a2
        T d1, g1; preciseMul(nominal, o.delta, d1, g1);
        T d2, g2; preciseMul(delta,   o.nominal, d2, g2);
        kahanAdd(r.delta, r.delta_c, d1);
        kahanAdd(r.delta, r.delta_c, d2);
        kahanAdd(r.delta, r.delta_c, g1 + g2);

        return r;
    }

    GaussianDualField operator/(const GaussianDualField &o) const {
        // denominator = o.nominal² - σ² * o.noise²
        T d1, e1; preciseMul(o.nominal, o.nominal, d1, e1);
        T d2, e2; preciseMul(o.noise,   o.noise,   d2, e2);
        T d2s = d2 * sigma2;
        T e2s = fma(d2, sigma2, -d2s) + sigma2 * e2;

        T denom = (d1 + e1) - (d2s + e2s);
        if (std::abs(denom) < 1e-20) return GaussianDualField(0, 0, 0, sigma2);

        T inv = 1.0 / denom;
        inv = inv * (2.0 - denom * inv);

        T n1, f1; preciseMul(nominal, o.nominal, n1, f1);
        T n2, f2; preciseMul(noise,   o.noise,   n2, f2);
        T n2s = n2 * sigma2;
        T f2s = fma(n2, sigma2, -n2s) + sigma2 * f2;

        T numNom = n1 - n2s;
        T errNom = (n1 - numNom) - n2s + f1 - f2s;

        GaussianDualField r(0, 0, 0, sigma2);
        T a, ea; preciseMul(numNom, inv, a, ea);
        r.nominal = a; r.nominal_c = 0;
        kahanAdd(r.nominal, r.nominal_c, errNom * inv + ea);

        T m1, h1; preciseMul(noise,   o.nominal, m1, h1);
        T m2, h2; preciseMul(nominal, o.noise,   m2, h2);
        T numN = m1 - m2;
        T errN = (m1 - numN) - m2 + h1 - h2;
        T b, eb; preciseMul(numN, inv, b, eb);
        r.noise = b; r.noise_c = 0;
        kahanAdd(r.noise, r.noise_c, errN * inv + eb);

        T u1, j1; preciseMul(delta,   o.nominal, u1, j1);
        T u2, j2; preciseMul(nominal, o.delta,   u2, j2);
        T numD = (u1 + j1) - (u2 + j2);
        T a2 = o.nominal * o.nominal;
        if (std::abs(a2) > 1e-20) {
            r.delta = numD / a2;
        }

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

    operator T() const { return nominal; }
};

#endif // GAUSSIAN_DUAL_FIELD_H

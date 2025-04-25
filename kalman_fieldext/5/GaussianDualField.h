// GaussianDualField.h
#ifndef GAUSSIAN_DUAL_FIELD_H
#define GAUSSIAN_DUAL_FIELD_H

#include <math.h>

template<typename T = double, T sigma2 = 0.01>
class GaussianDualField {
public:
    T nominal;   // a
    T noise;     // b (ε component, ε² = σ²)
    T delta;     // c (δ component, δ² = 0)

    // Kahan compensation terms
    T nominal_c = 0;
    T noise_c   = 0;
    T delta_c   = 0;

    constexpr GaussianDualField(T a = 0, T b = 0, T c = 0)
      : nominal(a), noise(b), delta(c) {}

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
        GaussianDualField r = *this;
        kahanAdd(r.nominal, r.nominal_c, o.nominal);
        kahanAdd(r.noise,   r.noise_c,   o.noise);
        kahanAdd(r.delta,   r.delta_c,   o.delta);
        return r;
    }

    GaussianDualField operator-(const GaussianDualField &o) const {
        GaussianDualField r = *this;
        kahanAdd(r.nominal, r.nominal_c, -o.nominal);
        kahanAdd(r.noise,   r.noise_c,   -o.noise);
        kahanAdd(r.delta,   r.delta_c,   -o.delta);
        return r;
    }

    GaussianDualField operator*(const GaussianDualField &o) const {
        // Compute nominal = a1*a2 + σ²*(b1*b2), with FMA error
        T p1, e1; preciseMul(nominal, o.nominal, p1, e1);
        T p2, e2; preciseMul(noise,   o.noise,   p2, e2);
        p2 *= sigma2;
        e2 = fma(noise * o.noise, sigma2, -p2) + sigma2 * e2;
        GaussianDualField r;
        r.nominal = 0; r.nominal_c = 0;
        kahanAdd(r.nominal, r.nominal_c, p1);
        kahanAdd(r.nominal, r.nominal_c, p2);
        kahanAdd(r.nominal, r.nominal_c, e1 + e2);

        // noise   = a1*b2 + b1*a2
        T q1, f1; preciseMul(nominal, o.noise, q1, f1);
        T q2, f2; preciseMul(noise,   o.nominal, q2, f2);
        r.noise = 0; r.noise_c = 0;
        kahanAdd(r.noise, r.noise_c, q1);
        kahanAdd(r.noise, r.noise_c, q2);
        kahanAdd(r.noise, r.noise_c, f1 + f2);

        // delta   = a1*d2 + d1*a2
        T d1, g1; preciseMul(nominal, o.delta, d1, g1);
        T d2, g2; preciseMul(delta,   o.nominal, d2, g2);
        r.delta = 0; r.delta_c = 0;
        kahanAdd(r.delta, r.delta_c, d1);
        kahanAdd(r.delta, r.delta_c, d2);
        kahanAdd(r.delta, r.delta_c, g1 + g2);

        return r;
    }

    GaussianDualField operator/(const GaussianDualField &o) const {
        // denominator = o.nominal² - σ² * o.noise²
        T d1, e1; preciseMul(o.nominal, o.nominal, d1, e1);
        T d2, e2; preciseMul(o.noise,   o.noise,   d2, e2);
        d2 *= sigma2;
        e2 = fma(o.noise * o.noise, sigma2, -d2) + sigma2 * e2;
        T denom = d1 - d2;
        // one Newton–Raphson iteration for 1/denom
        T inv  = 1.0 / denom;
        inv = inv * (2.0 - denom * inv);

        // numerator real = nominal*o.nominal - σ²*(noise*o.noise)
        T n1, f1; preciseMul(nominal, o.nominal, n1, f1);
        T n2, f2; preciseMul(noise,   o.noise,   n2, f2);
        n2 *= sigma2;
        f2 = fma(noise * o.noise, sigma2, -n2) + sigma2 * f2;
        T numReal = n1 - n2;
        T numErr  = f1 - f2;

        // build result
        GaussianDualField r;
        T a, ea; preciseMul(numReal, inv, a, ea);
        r.nominal = a; r.nominal_c = 0;
        kahanAdd(r.nominal, r.nominal_c, numErr * inv + ea);

        // noise   = (noise*o.nominal - nominal*o.noise) / denom
        T m1, h1; preciseMul(noise,   o.nominal, m1, h1);
        T m2, h2; preciseMul(nominal, o.noise,   m2, h2);
        T numN = m1 - m2;
        T errN = h1 - h2;
        T b, eb; preciseMul(numN, inv, b, eb);
        r.noise = b; r.noise_c = 0;
        kahanAdd(r.noise, r.noise_c, errN * inv + eb);

        // delta   = (delta*o.nominal - nominal*o.delta) / denom
        T u1, j1; preciseMul(delta,   o.nominal, u1, j1);
        T u2, j2; preciseMul(nominal, o.delta,   u2, j2);
        T numD = u1 - u2;
        T errD = j1 - j2;
        T c, ec; preciseMul(numD, inv, c, ec);
        r.delta = c; r.delta_c = 0;
        kahanAdd(r.delta, r.delta_c, errD * inv + ec);

        return r;
    }

    operator T() const { return nominal; }
};

#endif // GAUSSIAN_DUAL_FIELD_H

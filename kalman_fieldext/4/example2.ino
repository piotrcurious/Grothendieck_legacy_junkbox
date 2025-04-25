/* Enhanced Gaussian Dual Field Extension with Kahan Summation for ESP32 Arduino Kalman Filter */

#ifndef GAUSSIAN_DUAL_FIELD_H #define GAUSSIAN_DUAL_FIELD_H

#include <math.h>

template<typename T = double, T sigma2 = 0.01> class GaussianDualField { public: T nominal;    // a T noise;      // b (ε, where ε² = sigma2) T delta;      // c (δ, where δ² = 0)

// Kahan compensation terms for each component
T compNominal;
T compNoise;
T compDelta;

GaussianDualField(T a = 0, T b = 0, T c = 0)
    : nominal(a), noise(b), delta(c), compNominal(0), compNoise(0), compDelta(0) {}

// Kahan-style addition
GaussianDualField& operator+=(const GaussianDualField& other) {
    // nominal component
    T y0 = other.nominal - compNominal;
    T t0 = nominal + y0;
    compNominal = (t0 - nominal) - y0;
    nominal = t0;
    // noise component
    T y1 = other.noise - compNoise;
    T t1 = noise + y1;
    compNoise = (t1 - noise) - y1;
    noise = t1;
    // delta component
    T y2 = other.delta - compDelta;
    T t2 = delta + y2;
    compDelta = (t2 - delta) - y2;
    delta = t2;
    return *this;
}

GaussianDualField operator+(const GaussianDualField& other) const {
    GaussianDualField res = *this;
    res += other;
    return res;
}

GaussianDualField operator-(const GaussianDualField& other) const {
    return GaussianDualField(
        nominal - other.nominal,
        noise - other.noise,
        delta - other.delta
    );
}

GaussianDualField operator*(const GaussianDualField& other) const {
    // (a + bε + cδ)*(A + Bε + Cδ)
    T a = nominal * other.nominal + sigma2 * noise * other.noise;
    T b = nominal * other.noise + noise * other.nominal;
    T c = nominal * other.delta + delta * other.nominal;
    return GaussianDualField(a, b, c);
}

GaussianDualField operator/(const GaussianDualField& other) const {
    T denom = other.nominal * other.nominal - sigma2 * other.noise * other.noise;
    if (fabs(denom) < 1e-12) return GaussianDualField(0,0,0);
    T a = (nominal * other.nominal - sigma2 * noise * other.noise) / denom;
    T b = (noise * other.nominal - nominal * other.noise) / denom;
    T c = (delta * other.nominal - nominal * other.delta) / denom;
    return GaussianDualField(a, b, c);
}

operator T() const { return nominal; }

};

// Kalman filter over GaussianDualField template<typename Field> class KalmanFieldFilter { public: KalmanFieldFilter(Field q_, Field r_, Field p0_, Field x0_) : q(q_), r(r_), p(p0_), x(x0_) {}

double update(double zRaw) {
    Field z(zRaw, 0.0, 0.0);

    // Prediction
    p += q;

    // Measurement update
    k = p / (p + r);
    x += k * (z - x);
    p = (Field(1.0,0.0,0.0) - k) * p;

    return x.nominal;
}

private: Field q, r, p, x, k; };

#endif // GAUSSIAN_DUAL_FIELD_H

// Example Arduino sketch using the enhanced filter #include <Arduino.h> #include "GaussianDualField.h"

using Field = GaussianDualField<double, 0.0025>;  // sigma^2 = 0.0025

KalmanFieldFilter<Field> filter( Field(0.001,0.0,0.0),  // process noise Field(0.01,0.0,0.0),   // measurement noise Field(1.0,0.0,0.0),    // initial covariance Field(0.0,0.0,0.0)     // initial state );

void setup() { Serial.begin(115200); }

void loop() { double rawMeasurement = analogRead(34) * (3.3 / 4095.0); double estimate = filter.update(rawMeasurement); Serial.println(estimate, 6); delay(100); }


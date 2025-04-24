/* KalmanFilterDD.ino ESP32 Arduino Kalman filter with double-double precision using compensated arithmetic for improved numerical stability in addition, multiplication, and division. */

#include <math.h>

// Double-double precision type struct dd_real { double hi, lo; };

// Error-free transformation for addition\static inline dd_real two_sum(double a, double b) { double sum = a + b; double bb = sum - a; double err = (a - (sum - bb)) + (b - bb); return {sum, err}; }

// Error-free transformation for multiplication using FMA static inline dd_real two_prod(double a, double b) { double prod = a * b; double err = fma(a, b, -prod); return {prod, err}; }

// Double-double addition static inline dd_real dd_add(const dd_real &a, const dd_real &b) { dd_real s = two_sum(a.hi, b.hi); double t = a.lo + b.lo + s.lo; dd_real result = two_sum(s.hi, t); return result; }

// Double-double subtraction static inline dd_real dd_sub(const dd_real &a, const dd_real &b) { return dd_add(a, {-b.hi, -b.lo}); }

// Double-double multiplication static inline dd_real dd_mul(const dd_real &a, const dd_real &b) { dd_real p = two_prod(a.hi, b.hi); double err = p.lo + (a.hi * b.lo + a.lo * b.hi); dd_real result = two_sum(p.hi, err); return result; }

// Double-double division static inline dd_real dd_div(const dd_real &a, const dd_real &b) { double approx = a.hi / b.hi; dd_real approx_dd = {approx, 0.0}; dd_real prod = dd_mul(b, approx_dd); dd_real diff = dd_sub(a, prod); double corr = (diff.hi + diff.lo) / b.hi; return dd_add(approx_dd, {corr, 0.0}); }

// 1D Kalman filter using double-double arithmetic class KalmanFilterDD { public: KalmanFilterDD(double processNoise, double measurementNoise, double estimationError, double initialEstimate) : q{processNoise, 0}, r{measurementNoise, 0}, p{estimationError, 0}, x{initialEstimate, 0}, k{0, 0} {}

// Update filter with new measurement, returns filtered estimate (double precision)
double update(double measurement) {
    // Prediction update: p = p + q
    p = dd_add(p, q);

    // Compute Kalman gain: k = p / (p + r)
    dd_real denom = dd_add(p, r);
    k = dd_div(p, denom);

    // Measurement update: x = x + k * (measurement - x)
    dd_real z = {measurement, 0};
    dd_real innovation = dd_sub(z, x);
    x = dd_add(x, dd_mul(k, innovation));

    // Update estimation error: p = (1 - k) * p
    dd_real one = {1.0, 0.0};
    p = dd_mul(dd_sub(one, k), p);

    return x.hi;
}

private: dd_real q, r, p, x, k; };

// Instantiate Kalman filter: tune q, r, p0, x0 as needed KalmanFilterDD kf(0.01, 0.1, 1.0, 0.0);

void setup() { Serial.begin(115200); analogReadResolution(12); }

void loop() { int raw = analogRead(34); double voltage = (raw / 4095.0) * 3.3; double filtered = kf.update(voltage);

Serial.print("Raw: ");
Serial.print(voltage, 6);
Serial.print(" | Filtered: ");
Serial.println(filtered, 6);

delay(100);

}


// KalmanFilter.cpp
#include "KalmanFilter.h"
#include <math.h>

KalmanFilter::KalmanFilter(double processNoise, double measurementNoise, double estimationError, double initialEstimate)
    : q(processNoise), r(measurementNoise), p(estimationError), x(initialEstimate), compensation(0.0) {}

double KalmanFilter::update(double measurement) {
    // Prediction update
    p = kahanSum(q + p - compensation);  // Update error covariance

    // Measurement update
    k = safeDiv(p, p + r);  // Compute Kalman gain
    double innovation = measurement - x;
    x = kahanSum(preciseMul(k, innovation));  // Update estimate
    p = preciseMul((1.0 - k), p);  // Update error covariance

    return x;
}

double KalmanFilter::kahanSum(double increment) {
    double y = increment - compensation;
    double t = x + y;
    compensation = (t - x) - y;
    x = t;
    return x;
}

// Two-product style precise multiplication
double KalmanFilter::preciseMul(double a, double b) {
    double result = a * b;
    double error = fma(a, b, -result);  // fused multiply-add captures the rounding error
    return result + error;
}

// Safe division with fallback for near-zero denominators
double KalmanFilter::safeDiv(double numerator, double denominator) {
    if (fabs(denominator) < 1e-12) return 0.0;
    double result = numerator / denominator;
    // Optionally refine result here using Newton-Raphson or similar
    return result;
}

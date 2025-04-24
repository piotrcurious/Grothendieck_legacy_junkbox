// KalmanFilter.h
#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

class KalmanFilter {
public:
    KalmanFilter(double processNoise, double measurementNoise, double estimationError, double initialEstimate);

    double update(double measurement);

private:
    double q;  // Process noise covariance
    double r;  // Measurement noise covariance
    double p;  // Estimation error covariance
    double x;  // Value
    double k;  // Kalman gain

    // Kahan summation error compensation
    double compensation;

    // Precision-aware helpers
    double kahanSum(double increment);
    double preciseMul(double a, double b);
    double safeDiv(double numerator, double denominator);
};

#endif

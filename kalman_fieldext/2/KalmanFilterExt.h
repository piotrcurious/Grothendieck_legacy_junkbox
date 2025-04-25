// KalmanFilterExt.h
#ifndef KALMAN_FILTER_EXT_H
#define KALMAN_FILTER_EXT_H

#include "FieldExtension.h"

template<typename F>
class KalmanFilterExt {
public:
    KalmanFilterExt(F processNoise, F measurementNoise, F estimationError, F initialEstimate)
        : q(processNoise), r(measurementNoise), p(estimationError), x(initialEstimate) {}

    double update(double z) {
        F measurement(z);       // Lift into field extension
        p = p + q;
        k = p / (p + r);
        x = x + k * (measurement - x);
        p = (F(1.0) - k) * p;
        return x;
    }

private:
    F q, r, p, x, k;
};

#endif

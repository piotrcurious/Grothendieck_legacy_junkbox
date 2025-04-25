// KalmanField.h
#ifndef KALMAN_FIELD_H
#define KALMAN_FIELD_H

#include "FieldElement.h"

template<typename T = double>
class KalmanField {
public:
    using Field = FieldElement<T>;

    KalmanField(Field processNoise, Field measurementNoise, Field estimationError, Field initialEstimate)
        : q(processNoise), r(measurementNoise), p(estimationError), x(initialEstimate) {}

    T update(T measurementValue) {
        Field measurement(measurementValue);
        p = p + q;
        k = p / (p + r);
        x = x + k * (measurement - x);
        p = (Field(1.0) - k) * p;
        return x;
    }

private:
    Field q, r, p, x, k;
};

#endif

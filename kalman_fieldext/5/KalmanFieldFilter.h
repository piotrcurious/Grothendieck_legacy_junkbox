// KalmanFieldFilter.h
#ifndef KALMAN_FIELD_FILTER_H
#define KALMAN_FIELD_FILTER_H

#include "GaussianDualField.h"

template<typename Field>
class KalmanFieldFilter {
public:
    KalmanFieldFilter(Field q, Field r, Field p0, Field x0)
      : q(q), r(r), p(p0), x(x0) {}

    double update(double zRaw) {
        return update_field(zRaw).nominal;
    }

    /**
     * @brief Update the filter with a new measurement.
     * @param zRaw The raw measurement value.
     * @param qOverride Optional process noise override.
     * @param rOverride Optional measurement noise override.
     */
    Field update_field(double zRaw, const Field* qOverride = nullptr, const Field* rOverride = nullptr) {
        Field currentQ = qOverride ? *qOverride : q;
        Field currentR = rOverride ? *rOverride : r;
        Field z(zRaw, 0.0, 0.0, x.sigma2);

        // Predict
        p = p + currentQ;

        // Update
        Field k = p / (p + currentR);
        x = x + k * (z - x);

        // Joseph form for better stability: P = (I-k)P(I-k)^T + kRk^T
        Field one(1.0, 0.0, 0.0, x.sigma2);
        Field imk = one - k;
        p = imk * imk * p + k * k * currentR;

        return x;
    }

    Field getX() const { return x; }
    Field getP() const { return p; }

    void reset(Field x0, Field p0) {
        x = x0;
        p = p0;
    }

    void init(Field q_new, Field r_new, Field x0, Field p0) {
        q = q_new;
        r = r_new;
        x = x0;
        p = p0;
    }

private:
    Field q, r, p, x;
};

#endif // KALMAN_FIELD_FILTER_H

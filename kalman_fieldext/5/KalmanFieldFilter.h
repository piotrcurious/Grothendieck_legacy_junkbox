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
     * @brief Update the filter with a new measurement (Standard Linear).
     */
    Field update_field(double zRaw, const Field* qOverride = nullptr, const Field* rOverride = nullptr) {
        Field currentQ = qOverride ? *qOverride : q;
        Field currentR = rOverride ? *rOverride : r;
        Field z(zRaw, 0.0, 0.0, x.sigma2);

        p = p + currentQ;
        Field k = p / (p + currentR);
        x = x + k * (z - x);

        Field one(1.0, 0.0, 0.0, x.sigma2);
        Field imk = one - k;
        p = imk * imk * p + k * k * currentR;

        return x;
    }

    /**
     * @brief Extended Kalman Filter update using Dual Numbers for automatic Jacobian.
     * @tparam MeasurementFunc A function or functor `Field h(Field x)`
     */
    template<typename MeasurementFunc>
    Field update_ekf(double zRaw, MeasurementFunc h, const Field* qOverride = nullptr, const Field* rOverride = nullptr) {
        Field currentQ = qOverride ? *qOverride : q;
        Field currentR = rOverride ? *rOverride : r;

        // 1. Predict
        p = p + currentQ;

        // 2. Automatic Jacobian via Dual component
        Field x_for_h = x;
        x_for_h.delta = 1.0; // Seed for derivative
        Field h_x = h(x_for_h);

        double H_val = h_x.delta; // dh/dx

        // 3. Update using Field operators for consistency and numerical stability
        Field H_field(H_val, 0, 0, x.sigma2);
        Field K_field = (p * H_field) / (H_field * p * H_field + currentR);

        Field z_field(zRaw, 0, 0, x.sigma2);
        Field innovation = z_field - Field(h_x.nominal, h_x.noise, 0, x.sigma2);
        x = x + K_field * innovation;

        Field one(1.0, 0.0, 0.0, x.sigma2);
        Field imkH = one - K_field * H_field;
        p = imkH * imkH * p + K_field * K_field * currentR; // Joseph form

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

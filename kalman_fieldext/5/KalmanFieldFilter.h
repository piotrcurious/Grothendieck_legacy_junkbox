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
        predict();
        return update_linear(zRaw).nominal;
    }

    /**
     * @brief Linear Prediction step.
     */
    void predict(const Field* qOverride = nullptr) {
        Field currentQ = qOverride ? *qOverride : q;
        p = p + currentQ;
    }

    /**
     * @brief EKF Prediction step using automatic Jacobian for non-linear state transition.
     * @tparam TransitionFunc A function or functor `Field f(Field x)`
     */
    template<typename TransitionFunc>
    void predict_ekf(TransitionFunc f, const Field* qOverride = nullptr) {
        Field currentQ = qOverride ? *qOverride : q;

        // 1. Automatic Jacobian via Dual component
        Field x_for_f = x;
        x_for_f.delta = 1.0;
        Field f_x = f(x_for_f);

        double F_val = f_x.delta;

        // 2. Propagate state
        // Use nominal and noise from f(x)
        x = Field(f_x.nominal, f_x.noise, 0, x.sigma2);

        // 3. Propagate covariance (field-based)
        Field F_field(F_val, 0, 0, x.sigma2);
        p = F_field * p * F_field + currentQ;
    }

    /**
     * @brief Linear measurement update step.
     */
    Field update_linear(double zRaw, const Field* rOverride = nullptr) {
        Field currentR = rOverride ? *rOverride : r;
        Field z(zRaw, 0.0, 0.0, x.sigma2);

        Field k = p / (p + currentR);
        x = x + k * (z - x);

        Field one(1.0, 0.0, 0.0, x.sigma2);
        Field imk = one - k;
        p = imk * imk * p + k * k * currentR;

        return x;
    }

    /**
     * @brief EKF measurement update step using Dual Numbers for automatic Jacobian.
     */
    template<typename MeasurementFunc>
    Field update_ekf_step(double zRaw, MeasurementFunc h, const Field* rOverride = nullptr) {
        Field currentR = rOverride ? *rOverride : r;

        Field x_for_h = x;
        x_for_h.delta = 1.0;
        Field h_x = h(x_for_h);

        double H_val = h_x.delta;

        Field H_field(H_val, 0, 0, x.sigma2);
        Field K_field = (p * H_field) / (H_field * p * H_field + currentR);

        Field z_field(zRaw, 0, 0, x.sigma2);
        Field innovation = z_field - Field(h_x.nominal, h_x.noise, 0, x.sigma2);
        x = x + K_field * innovation;

        Field one(1.0, 0.0, 0.0, x.sigma2);
        Field imkH = one - K_field * H_field;
        p = imkH * imkH * p + K_field * K_field * currentR;

        return x;
    }

    /**
     * @brief Combined Predict + EKF Update for convenience.
     */
    template<typename MeasurementFunc>
    Field update_ekf(double zRaw, MeasurementFunc h, const Field* qOverride = nullptr, const Field* rOverride = nullptr) {
        predict(qOverride);
        return update_ekf_step(zRaw, h, rOverride);
    }

    /**
     * @brief Standard combined Predict + Linear Update.
     */
    Field update_field(double zRaw, const Field* qOverride = nullptr, const Field* rOverride = nullptr) {
        predict(qOverride);
        return update_linear(zRaw, rOverride);
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

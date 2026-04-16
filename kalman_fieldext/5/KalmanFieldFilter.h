// KalmanFieldFilter.h
#ifndef KALMAN_FIELD_FILTER_H
#define KALMAN_FIELD_FILTER_H

template<typename Field>
class KalmanFieldFilter {
public:
    KalmanFieldFilter(Field q, Field r, Field p0, Field x0)
      : q(q), r(r), p(p0), x(x0) {}

    double update(double zRaw) {
        return update_field(zRaw).nominal;
    }

    Field update_field(double zRaw) {
        Field z(zRaw, 0.0, 0.0, x.sigma2);

        p = p + q;
        Field k = p / (p + r);
        x = x + k * (z - x);
        // Joseph form for better stability: P = (I-k)P(I-k)^T + kRk^T
        // For 1D scalar, P = (1-k)^2 * P + k^2 * R
        Field one(1.0, 0.0, 0.0, x.sigma2);
        Field imk = one - k;
        p = imk * imk * p + k * k * r;

        return x;
    }

private:
    Field q, r, p, x;
};

#endif // KALMAN_FIELD_FILTER_H

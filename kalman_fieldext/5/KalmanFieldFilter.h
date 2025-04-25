// KalmanFieldFilter.h
#ifndef KALMAN_FIELD_FILTER_H
#define KALMAN_FIELD_FILTER_H

template<typename Field>
class KalmanFieldFilter {
public:
    KalmanFieldFilter(Field q, Field r, Field p0, Field x0)
      : q(q), r(r), p(p0), x(x0) {}

    double update(double zRaw) {
        Field z(zRaw, 0.0, 0.0);

        p = p + q;
        Field k = p / (p + r);
        x = x + k * (z - x);
        p = (Field(1.0,0.0,0.0) - k) * p;

        return x.nominal;
    }

private:
    Field q, r, p, x;
};

#endif // KALMAN_FIELD_FILTER_H

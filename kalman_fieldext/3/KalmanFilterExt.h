// KalmanFilterExt.h
#ifndef KALMAN_FILTER_EXT_H
#define KALMAN_FILTER_EXT_H

#include "FieldExtension.h"

template<typename T, T R, T Q>
class KalmanFilterExt {
public:
    using F = FieldExtension<T, R, Q>;

    // q = process variance Q, r = measurement variance R
    KalmanFilterExt()
      : q(F(0,0,1)),        // Q enters via εp²=Q, so bp=1
        r(F(0,1,0)),        // R enters via εm²=R, so bm=1
        p(F(1,0,0)),        // initial covariance =1
        x(F(0,0,0))         // initial state =0
    {}

    // optionally set initial state & cov:
    void init(T x0, T p0) {
      x = F(x0,0,0);
      p = F(p0,0,0);
    }

    T update(T z_raw) {
        F z = F::lift(z_raw);

        // time update
        p = p + q;

        // measurement update
        F k = p / (p + r);
        x = x + k * (z - x);
        p = (F(1,0,0) - k) * p;

        return x.project();
    }

private:
    F q, r, p, x;
};

#endif

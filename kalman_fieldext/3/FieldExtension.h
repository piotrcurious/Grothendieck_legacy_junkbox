// FieldExtension.h
#ifndef FIELD_EXTENSION_H
#define FIELD_EXTENSION_H

#include <cmath>

template<typename T, T R, T Q>
struct FieldExtension {
    T a;   // nominal value
    T bm;  // measurement-noise coefficient (εm)
    T bp;  // process-noise coefficient     (εp)

    constexpr FieldExtension(T a_=0, T bm_=0, T bp_=0)
      : a(a_), bm(bm_), bp(bp_) {}

    // addition
    FieldExtension operator+(FieldExtension o) const {
        return { a+o.a, bm+o.bm, bp+o.bp };
    }
    // subtraction
    FieldExtension operator-(FieldExtension o) const {
        return { a-o.a, bm-o.bm, bp-o.bp };
    }
    // multiplication: (a + bm εm + bp εp)*(c + dm εm + dp εp)
    FieldExtension operator*(FieldExtension o) const {
        T real = a*o.a + R*bm*o.bm + Q*bp*o.bp;
        T mcoef = a*o.bm + bm*o.a;
        T pcoef = a*o.bp + bp*o.a;
        return { real, mcoef, pcoef };
    }
    // division: multiply by inverse of denominator
    FieldExtension operator/(FieldExtension o) const {
        // inverse of (c + dm εm + dp εp):
        T denom = o.a*o.a - R*o.bm*o.bm - Q*o.bp*o.bp;
        if (std::fabs(denom) < 1e-15) return {0,0,0};
        T A =  o.a/denom;
        T B = -o.bm/denom;
        T C = -o.bp/denom;
        // (a+bmεm+bpεp)*(A + Bεm + Cεp)
        T real =  a*A + R*bm*B + Q*bp*C;
        T mcoef = a*B + bm*A;
        T pcoef = a*C + bp*A;
        return { real, mcoef, pcoef };
    }

    // lift a raw measurement z∈Q₂ → a+0·εm+0·εp
    static FieldExtension lift(T z) { return {z, 0, 0}; }

    // project back to nominal
    T project() const { return a; }
};

#endif

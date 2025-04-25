// FieldElement.h
#ifndef FIELD_ELEMENT_H
#define FIELD_ELEMENT_H

template<typename T = double>
class FieldElement {
public:
    T value;
    T error;

    FieldElement(T v = 0.0, T e = 0.0) : value(v), error(e) {}

    // Accurate addition using Kahan-style compensation
    FieldElement operator+(const FieldElement& other) const {
        T sum = value + other.value;
        T delta = ((value - sum) + other.value) + error + other.error;
        return FieldElement(sum, delta);
    }

    // Accurate subtraction
    FieldElement operator-(const FieldElement& other) const {
        T diff = value - other.value;
        T delta = ((value - diff) - other.value) + error - other.error;
        return FieldElement(diff, delta);
    }

    // Two-product style multiplication with error correction
    FieldElement operator*(const FieldElement& other) const {
        T prod = value * other.value;
        T err = fma(value, other.value, -prod) + value * other.error + other.value * error;
        return FieldElement(prod, err);
    }

    // Division with compensation
    FieldElement operator/(const FieldElement& other) const {
        if (fabs(other.value) < 1e-12) return FieldElement(0.0);
        T div = value / other.value;
        T residual = (value - div * other.value + error - div * other.error) / other.value;
        return FieldElement(div, residual);
    }

    FieldElement& operator+=(const FieldElement& other) { *this = *this + other; return *this; }
    FieldElement& operator*=(const FieldElement& other) { *this = *this * other; return *this; }

    operator T() const { return value + error; }
};

#endif

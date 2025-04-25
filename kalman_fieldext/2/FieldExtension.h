// FieldExtension.h
#ifndef FIELD_EXTENSION_H
#define FIELD_EXTENSION_H

template<typename T = double, T D = 2>
class FieldExtension {
public:
    T a;  // Real part
    T b;  // Coefficient of √D

    constexpr FieldExtension(T a_ = 0.0, T b_ = 0.0) : a(a_), b(b_) {}

    FieldExtension operator+(const FieldExtension& other) const {
        return FieldExtension(a + other.a, b + other.b);
    }

    FieldExtension operator-(const FieldExtension& other) const {
        return FieldExtension(a - other.a, b - other.b);
    }

    FieldExtension operator*(const FieldExtension& other) const {
        // (a + b√d)(c + d√d) = (ac + bdD) + (ad + bc)√d
        T real = a * other.a + D * b * other.b;
        T ext = a * other.b + b * other.a;
        return FieldExtension(real, ext);
    }

    FieldExtension operator/(const FieldExtension& other) const {
        // Multiply numerator and denominator by conjugate
        // (a + b√d) / (c + d√d) = [(a + b√d)(c - d√d)] / (c² - d²D)
        T denom = other.a * other.a - D * other.b * other.b;
        if (fabs(denom) < 1e-12) return FieldExtension(0.0, 0.0);

        T numReal = a * other.a - D * b * other.b;
        T numExt = b * other.a - a * other.b;
        return FieldExtension(numReal / denom, numExt / denom);
    }

    FieldExtension operator*(T scalar) const {
        return FieldExtension(a * scalar, b * scalar);
    }

    FieldExtension operator+(T scalar) const {
        return FieldExtension(a + scalar, b);
    }

    operator T() const {
        return a + b * sqrt(D);  // Approximate real value
    }
};

#endif

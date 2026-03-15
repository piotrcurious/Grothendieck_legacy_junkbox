#ifndef ARDUINO_POLYFIT_HPP
#define ARDUINO_POLYFIT_HPP

#include <stdint.h>
#include <stddef.h>

namespace polyfit {

/**
 * @brief Represents a number as a polynomial over F2 (binary representation).
 */
class BitField {
public:
    uint64_t value;
    uint8_t width;

    BitField(uint64_t val = 0, uint8_t w = 32) : value(val), width(w) {
        if (width < 64) {
            value &= (1ULL << width) - 1;
        }
    }

    bool get_coeff(uint8_t i) const {
        if (i >= width) return false;
        return (value >> i) & 1;
    }

    BitField operator+(const BitField& other) const {
        uint8_t w = width > other.width ? width : other.width;
        return BitField(value ^ other.value, w);
    }

    BitField operator*(const BitField& other) const {
        uint64_t res = 0;
        for (uint8_t i = 0; i < width; ++i) {
            if ((value >> i) & 1) {
                res ^= (other.value << i);
            }
        }
        uint8_t w = width + other.width;
        if (w > 64) w = 64;
        return BitField(res, w);
    }

    BitField frobenius() const {
        uint64_t res = 0;
        for (uint8_t i = 0; i < 32; ++i) {
            if ((value >> i) & 1) {
                res |= (1ULL << (2 * i));
            }
        }
        uint8_t w = width * 2;
        if (w > 64) w = 64;
        return BitField(res, w);
    }
};

/**
 * @brief Represents different machine number formats.
 */
enum class NumType {
    INT32,
    FLOAT32
};

/**
 * @brief Base class for machine numbers with scheme structure.
 */
struct MachineNumber {
    NumType type;
    union {
        int32_t i32;
        float f32;
    } val;

    MachineNumber(int32_t v) : type(NumType::INT32) { val.i32 = v; }
    MachineNumber(float v) : type(NumType::FLOAT32) { val.f32 = v; }
    MachineNumber() : type(NumType::INT32) { val.i32 = 0; }

    BitField get_bitfield() const;
};

/**
 * @brief Implements arithmetic as scheme morphisms.
 */
class FieldMorphism {
public:
    static MachineNumber add(MachineNumber x, MachineNumber y);
    static MachineNumber multiply(MachineNumber x, MachineNumber y);
    static float to_float(MachineNumber x);
};

/**
 * @brief Feature extractor preserving field structure.
 */
class PolynomialFeatureExtractor {
public:
    uint8_t max_degree;

    PolynomialFeatureExtractor(uint8_t degree) : max_degree(degree) {}

    void extract(float x, float* features) const;
};

/**
 * @brief Simple polynomial fitter using normal equations.
 */
class PolynomialFitter {
public:
    uint8_t degree;
    float* weights;

    PolynomialFitter(uint8_t d);
    ~PolynomialFitter();

    bool fit(const float* x, const float* y, size_t n, float lambda = 0.0f);
    float predict(float x) const;
};

/**
 * @brief Purely algebraic feature extractor based on scheme theory.
 */
class AlgebraicFeatureExtractor {
public:
    uint8_t max_degree;
    bool use_frobenius;

    AlgebraicFeatureExtractor(uint8_t degree, bool frob = false) : max_degree(degree), use_frobenius(frob) {}

    // Extracts features by treating float input as a polynomial over F2
    // and generating higher order terms in F2[x]
    void extract(float x, float* features) const;
};

} // namespace polyfit

#endif // ARDUINO_POLYFIT_HPP

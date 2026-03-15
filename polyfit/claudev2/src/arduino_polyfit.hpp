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

    BitField(uint64_t val = 0, uint8_t w = 32) : value(val & (0xFFFFFFFFFFFFFFFF >> (64 - w))), width(w) {}

    bool get_coeff(uint8_t i) const {
        if (i >= width) return false;
        return (value >> i) & 1;
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

} // namespace polyfit

#endif // ARDUINO_POLYFIT_HPP

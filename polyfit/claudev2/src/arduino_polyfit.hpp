#ifndef ARDUINO_POLYFIT_HPP
#define ARDUINO_POLYFIT_HPP

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

namespace polyfit {

/**
 * @brief F2Polynomial - Element of the ring F2[x].
 * Foundation of computer arithmetic as Grothendieck's Spec(F2).
 */
class F2Polynomial {
public:
    uint64_t data;
    uint8_t width;

    F2Polynomial(uint64_t val = 0, uint8_t w = 64) : data(val), width(w) {
        if (width < 64) data &= (1ULL << width) - 1;
    }

    F2Polynomial operator+(const F2Polynomial& other) const {
        return F2Polynomial(data ^ other.data, width > other.width ? width : other.width);
    }

    F2Polynomial operator*(const F2Polynomial& other) const {
        uint64_t res = 0;
        for (uint8_t i = 0; i < width; ++i) {
            if ((data >> i) & 1) res ^= (other.data << i);
        }
        return F2Polynomial(res, 64);
    }

    F2Polynomial frobenius() const {
        uint64_t res = 0;
        for (uint8_t i = 0; i < 32; ++i) {
            if ((data >> i) & 1) res |= (1ULL << (2 * i));
        }
        return F2Polynomial(res, 64);
    }
};

/**
 * @brief MachineScheme - Categorical representation of numeric types.
 * Each instance is a point in the scheme Spec(Z/2Z[x]/(constraints)).
 */
struct MachineScheme {
    enum Type { FLOAT32, INT32 } type;

    // Components of the structure sheaf
    F2Polynomial sign;
    F2Polynomial exponent;
    F2Polynomial mantissa;

    MachineScheme(float f);
    MachineScheme(int32_t i);

    float to_float() const;
    int32_t to_int32() const;
    F2Polynomial to_poly() const;
};

/**
 * @brief SchemeMorphism - Implements arithmetic as transitions between schemes.
 */
class SchemeMorphism {
public:
    static MachineScheme add(MachineScheme a, MachineScheme b);
    static MachineScheme multiply(MachineScheme a, MachineScheme b);
};

/**
 * @brief QuantizedField - Defines the boundaries and quantization of the machine field.
 */
struct QuantizedField {
    float epsilon;
    float step;

    static QuantizedField float32() {
        return { 1.1920929e-7f, 1.0f / (float)(1 << 23) };
    }

    float quantize(float x) const {
        if (step == 0) return x;
        return roundf(x / step) * step;
    }
};

/**
 * @brief CategoricalFeatureExtractor - Functorial extraction of features.
 * Maps the machine scheme into a higher-dimensional feature space.
 */
class CategoricalFeatureExtractor {
public:
    uint8_t max_degree;

    CategoricalFeatureExtractor(uint8_t degree) : max_degree(degree) {}

    // Extracts features by applying natural transformations (powers, frobenius)
    void extract(float x, float* features) const;

    // Cyclotomic features capturing discrete symmetries
    void extract_cyclotomic(float x, float* features) const;
};

/**
 * @brief PolynomialFitter - Robust fitter respecting categorical boundaries.
 */
class PolynomialFitter {
public:
    uint8_t degree;
    float* weights;
    QuantizedField qfield;

    PolynomialFitter(uint8_t d, QuantizedField q = QuantizedField::float32());
    ~PolynomialFitter();

    // Disable copying due to raw pointer ownership
    PolynomialFitter(const PolynomialFitter&) = delete;
    PolynomialFitter& operator=(const PolynomialFitter&) = delete;

    bool fit(const float* x, const float* y, size_t n, float lambda = 0.0f);

    /**
     * @brief Lebesgue-based fitting using projection onto Legendre basis.
     */
    bool fit_lebesgue(const float* x, const float* y, size_t n);

    float predict(float x) const;
    float predict_lebesgue(float x, float x_min, float x_max) const;
};

/**
 * @brief LegendreBasis - Orthogonal basis for L2[-1, 1].
 */
class LegendreBasis {
public:
    static float eval(uint8_t n, float x);
};

} // namespace polyfit

#endif // ARDUINO_POLYFIT_HPP

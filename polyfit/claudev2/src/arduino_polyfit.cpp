#include "arduino_polyfit.hpp"
#include <string.h>
#include <math.h>

namespace polyfit {

BitField MachineNumber::get_bitfield() const {
    if (type == NumType::INT32) {
        return BitField((uint64_t)val.i32, 32);
    } else {
        uint32_t bits;
        memcpy(&bits, &val.f32, sizeof(float));
        return BitField((uint64_t)bits, 32);
    }
}

BitField MachineNumber::get_sign() const {
    if (type == NumType::INT32) return BitField(val.i32 < 0 ? 1 : 0, 1);
    uint32_t bits;
    memcpy(&bits, &val.f32, sizeof(float));
    return BitField(bits >> 31, 1);
}

BitField MachineNumber::get_exponent() const {
    if (type == NumType::INT32) return BitField(0, 8);
    uint32_t bits;
    memcpy(&bits, &val.f32, sizeof(float));
    return BitField((bits >> 23) & 0xFF, 8);
}

BitField MachineNumber::get_mantissa() const {
    if (type == NumType::INT32) return BitField(val.i32, 23);
    uint32_t bits;
    memcpy(&bits, &val.f32, sizeof(float));
    return BitField(bits & 0x7FFFFF, 23);
}

MachineNumber MachineNumber::from_scheme(BitField sign, BitField exp, BitField mant) {
    uint32_t bits = ((sign.value & 1) << 31) |
                    ((exp.value & 0xFF) << 23) |
                    (mant.value & 0x7FFFFF);
    float f;
    memcpy(&f, &bits, sizeof(float));
    return MachineNumber(f);
}

MachineNumber FieldMorphism::add(MachineNumber x, MachineNumber y) {
    if (x.type != y.type) return MachineNumber(0);
    if (x.type == NumType::INT32) {
        return MachineNumber(x.val.i32 + y.val.i32);
    } else {
        return MachineNumber(x.val.f32 + y.val.f32);
    }
}

MachineNumber FieldMorphism::multiply(MachineNumber x, MachineNumber y) {
    if (x.type != y.type) return MachineNumber(0);
    if (x.type == NumType::INT32) {
        return MachineNumber(x.val.i32 * y.val.i32);
    } else {
        return MachineNumber(x.val.f32 * y.val.f32);
    }
}

float FieldMorphism::to_float(MachineNumber x) {
    if (x.type == NumType::INT32) return (float)x.val.i32;
    return x.val.f32;
}

void PolynomialFeatureExtractor::extract(float x, float* features) const {
    features[0] = 1.0f;
    for (uint8_t d = 1; d <= max_degree; ++d) {
        features[d] = features[d-1] * x;
    }
}

PolynomialFitter::PolynomialFitter(uint8_t d) : degree(d) {
    weights = new float[degree + 1];
    memset(weights, 0, (degree + 1) * sizeof(float));
}

PolynomialFitter::~PolynomialFitter() {
    delete[] weights;
}

float PolynomialFitter::predict(float x) const {
    float features[degree + 1];
    float result = 0;
    features[0] = 1.0f;
    result += weights[0];
    for (uint8_t d = 1; d <= degree; ++d) {
        features[d] = features[d-1] * x;
        result += weights[d] * features[d];
    }
    return result;
}

// Simple solver for small systems (Normal Equations)
// (X^T * X + lambda * I) * w = X^T * y
bool PolynomialFitter::fit(const float* x, const float* y, size_t n, float lambda, float epsilon) {
    uint8_t dim = degree + 1;
    float XtX[dim * dim];
    float Xty[dim];
    memset(XtX, 0, sizeof(XtX));
    memset(Xty, 0, sizeof(Xty));

    for (size_t i = 0; i < n; ++i) {
        float features[dim];
        features[0] = 1.0f;
        for (uint8_t d = 1; d <= degree; ++d) {
            features[d] = features[d-1] * x[i];
        }

        for (uint8_t r = 0; r < dim; ++r) {
            for (uint8_t c = 0; c < dim; ++c) {
                XtX[r * dim + c] += features[r] * features[c];
            }
            Xty[r] += features[r] * y[i];
        }
    }

    // Add regularization (Ridge) - do not regularize the bias term
    for (uint8_t i = 1; i < dim; ++i) {
        XtX[i * dim + i] += lambda;
    }

    // Gaussian elimination with pivoting
    for (uint8_t i = 0; i < dim; ++i) {
        uint8_t pivot = i;
        for (uint8_t j = i + 1; j < dim; ++j) {
            if (fabs(XtX[j * dim + i]) > fabs(XtX[pivot * dim + i])) pivot = j;
        }

        for (uint8_t j = 0; j < dim; ++j) {
            float tmp = XtX[i * dim + j];
            XtX[i * dim + j] = XtX[pivot * dim + j];
            XtX[pivot * dim + j] = tmp;
        }
        float tmp = Xty[i];
        Xty[i] = Xty[pivot];
        Xty[pivot] = tmp;

        if (fabs(XtX[i * dim + i]) < epsilon) return false;

        for (uint8_t j = i + 1; j < dim; ++j) {
            float factor = XtX[j * dim + i] / XtX[i * dim + i];
            Xty[j] -= factor * Xty[i];
            for (uint8_t k = i; k < dim; ++k) {
                XtX[j * dim + k] -= factor * XtX[i * dim + k];
            }
        }
    }

    for (int8_t i = dim - 1; i >= 0; --i) {
        float sum = 0;
        for (uint8_t j = i + 1; j < dim; ++j) {
            sum += XtX[i * dim + j] * weights[j];
        }
        weights[i] = (Xty[i] - sum) / XtX[i * dim + i];
    }

    return true;
}

float LegendreBasis::eval(uint8_t n, float x) {
    if (n == 0) return 1.0f;
    if (n == 1) return x;
    float p0 = 1.0f, p1 = x, p2 = 0;
    for (uint8_t i = 2; i <= n; ++i) {
        p2 = ((2.0f * i - 1.0f) * x * p1 - (i - 1.0f) * p0) / i;
        p0 = p1; p1 = p2;
    }
    return p1;
}

bool PolynomialFitter::fit_lebesgue(const float* x, const float* y, size_t n) {
    if (n == 0) return false;

    // Normalize domain to [-1, 1] for Legendre orthogonality
    float x_min = x[0], x_max = x[0];
    for(size_t i = 1; i < n; ++i) {
        if (x[i] < x_min) x_min = x[i];
        if (x[i] > x_max) x_max = x[i];
    }
    float x_range = x_max - x_min;
    if (x_range < 1e-9) x_range = 1.0f;

    for (uint8_t d = 0; d <= degree; ++d) {
        float coeff = 0;
        float norm_factor = 0;

        for (size_t i = 0; i < n; ++i) {
            float x_norm = 2.0f * (x[i] - x_min) / x_range - 1.0f;
            float p = LegendreBasis::eval(d, x_norm);
            coeff += y[i] * p;
            norm_factor += p * p;
        }

        // Projection: c_n = <f, p_n> / <p_n, p_n>
        if (norm_factor > 1e-9) {
            weights[d] = coeff / norm_factor;
        } else {
            weights[d] = 0;
        }
    }

    // Note: To use weights[d] with predict(), we'd need to evaluate
    // Legendre polynomials instead of simple powers.
    // For simplicity, we'll keep weights as Legendre coefficients
    // and we'd need a modified predict().
    // Let's modify predict() or add predict_lebesgue().

    return true;
}

float PolynomialFitter::predict_lebesgue(float x, float x_min, float x_max) const {
    float x_range = x_max - x_min;
    if (x_range < 1e-9) x_range = 1.0f;
    float x_norm = 2.0f * (x - x_min) / x_range - 1.0f;

    float result = 0;
    for (uint8_t d = 0; d <= degree; ++d) {
        result += weights[d] * LegendreBasis::eval(d, x_norm);
    }
    return result;
}

void AlgebraicFeatureExtractor::extract(float x, float* features) const {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(float));
    BitField base_poly(bits, 32);

    features[0] = 1.0f;
    BitField current = base_poly;

    for (uint8_t d = 1; d <= max_degree; ++d) {
        if (use_frobenius) {
            current = current.frobenius();
        }

        uint32_t res_bits = (uint32_t)current.value;
        float res_float;
        memcpy(&res_float, &res_bits, sizeof(float));

        // Handle potential NaN/Inf from bit manipulation
        if (!isfinite(res_float)) res_float = 0.0f;

        features[d] = res_float;

        if (!use_frobenius) {
            current = current * base_poly;
        }
    }
}

void GaloisActionExtractor::extract_frobenius_orbit(float x, float* features) const {
    MachineNumber num(x);
    BitField current_poly = num.get_bitfield();

    for (uint8_t d = 0; d < max_degree; ++d) {
        current_poly = current_poly.frobenius();
        uint32_t bits = (uint32_t)current_poly.value;
        float f;
        memcpy(&f, &bits, sizeof(float));
        features[d] = isfinite(f) ? f : 0.0f;
    }
}

void GaloisActionExtractor::extract_cyclotomic(float x, float* features) const {
    for (uint8_t d = 1; d <= max_degree; ++d) {
        features[(d-1)*2] = sinf(2.0f * M_PI * x / (float)d);
        features[(d-1)*2 + 1] = cosf(2.0f * M_PI * x / (float)d);
    }
}

} // namespace polyfit

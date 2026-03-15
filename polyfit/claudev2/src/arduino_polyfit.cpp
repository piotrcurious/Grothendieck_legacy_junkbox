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
bool PolynomialFitter::fit(const float* x, const float* y, size_t n, float lambda) {
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

        if (fabs(XtX[i * dim + i]) < 1e-9) return false;

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

} // namespace polyfit

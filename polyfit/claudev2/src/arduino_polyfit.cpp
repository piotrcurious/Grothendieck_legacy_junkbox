#include "arduino_polyfit.hpp"

namespace polyfit {

MachineScheme::MachineScheme(float f) : type(FLOAT32) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    sign = F2Polynomial(bits >> 31, 1);
    exponent = F2Polynomial((bits >> 23) & 0xFF, 8);
    mantissa = F2Polynomial(bits & 0x7FFFFF, 23);
}

MachineScheme::MachineScheme(int32_t i) : type(INT32) {
    sign = F2Polynomial(i < 0 ? 1 : 0, 1);
    exponent = F2Polynomial(0, 8);
    mantissa = F2Polynomial((uint64_t)i, 32);
}

float MachineScheme::to_float() const {
    uint32_t bits = ((sign.data & 1) << 31) |
                    ((exponent.data & 0xFF) << 23) |
                    (mantissa.data & 0x7FFFFF);
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

int32_t MachineScheme::to_int32() const {
    return (int32_t)mantissa.data;
}

F2Polynomial F2Polynomial::operator/(const F2Polynomial& other) const {
    if (other.data == 0) return F2Polynomial(0);
    uint8_t d_this = degree();
    uint8_t d_other = other.degree();
    if (d_this < d_other) return F2Polynomial(0);

    uint64_t q = 0;
    uint64_t r = data;
    for (int8_t i = (int8_t)d_this - (int8_t)d_other; i >= 0; --i) {
        if ((r >> (i + d_other)) & 1) {
            q |= (1ULL << i);
            r ^= (other.data << i);
        }
    }
    return F2Polynomial(q, 64);
}

F2Polynomial F2Polynomial::operator%(const F2Polynomial& other) const {
    if (other.data == 0) return F2Polynomial(0);
    uint8_t d_this = degree();
    uint8_t d_other = other.degree();
    if (d_this < d_other) return F2Polynomial(data, 64);

    uint64_t r = data;
    for (int8_t i = (int8_t)d_this - (int8_t)d_other; i >= 0; --i) {
        if ((r >> (i + d_other)) & 1) {
            r ^= (other.data << i);
        }
    }
    return F2Polynomial(r, 64);
}

MachineScheme MachineScheme::relative(const MachineScheme& X, const MachineScheme& S) {
    if (X.type == FLOAT32 && S.type == FLOAT32) {
        // Morphism as difference in the affine space of floats
        return MachineScheme(X.to_float() - S.to_float());
    } else {
        // Morphism in the discrete space
        return MachineScheme(X.to_int32() - S.to_int32());
    }
}

F2Polynomial MachineScheme::to_poly() const {
    if (type == FLOAT32) {
        uint32_t bits = ((sign.data & 1) << 31) |
                        ((exponent.data & 0xFF) << 23) |
                        (mantissa.data & 0x7FFFFF);
        return F2Polynomial(bits, 32);
    } else {
        return F2Polynomial(mantissa.data, 32);
    }
}

MachineScheme SchemeMorphism::add(MachineScheme a, MachineScheme b) {
    if (a.type == MachineScheme::FLOAT32) {
        return MachineScheme(a.to_float() + b.to_float());
    } else {
        return MachineScheme(a.to_int32() + b.to_int32());
    }
}

MachineScheme SchemeMorphism::multiply(MachineScheme a, MachineScheme b) {
    if (a.type == MachineScheme::FLOAT32) {
        return MachineScheme(a.to_float() * b.to_float());
    } else {
        return MachineScheme(a.to_int32() * b.to_int32());
    }
}

void CategoricalFeatureExtractor::extract(float x, float* features) const {
    MachineScheme base(x);
    features[0] = 1.0f;

    F2Polynomial current = base.to_poly();
    F2Polynomial base_poly = current;

    for (uint8_t d = 1; d <= max_degree; ++d) {
        uint32_t bits = (uint32_t)current.data;
        float f;
        memcpy(&f, &bits, sizeof(float));
        features[d] = isfinite(f) ? f : 0.0f;
        current = current * base_poly;
    }
}

void CategoricalFeatureExtractor::extract_cyclotomic(float x, float* features) const {
    for (uint8_t d = 1; d <= max_degree; ++d) {
        features[(d-1)*2] = sinf(2.0f * (float)M_PI * x / (float)d);
        features[(d-1)*2 + 1] = cosf(2.0f * (float)M_PI * x / (float)d);
    }
}

PolynomialFitter::PolynomialFitter(uint8_t d, QuantizedField q) : degree(d), qfield(q) {
    weights = new float[degree + 1];
    memset(weights, 0, (degree + 1) * sizeof(float));
}

PolynomialFitter::~PolynomialFitter() {
    delete[] weights;
}

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
            features[d] = qfield.quantize(powf(x[i], (float)d));
        }

        for (uint8_t r = 0; r < dim; ++r) {
            for (uint8_t c = 0; c < dim; ++c) {
                XtX[r * dim + c] += features[r] * features[c];
            }
            Xty[r] += features[r] * y[i];
        }
    }

    for (uint8_t i = 1; i < dim; ++i) XtX[i * dim + i] += lambda;

    // Gaussian elimination
    for (uint8_t i = 0; i < dim; ++i) {
        uint8_t pivot = i;
        for (uint8_t j = i + 1; j < dim; ++j) {
            if (fabsf(XtX[j * dim + i]) > fabsf(XtX[pivot * dim + i])) pivot = j;
        }
        for (uint8_t j = 0; j < dim; ++j) {
            float tmp = XtX[i * dim + j];
            XtX[i * dim + j] = XtX[pivot * dim + j];
            XtX[pivot * dim + j] = tmp;
        }
        float tmp = Xty[i];
        Xty[i] = Xty[pivot];
        Xty[pivot] = tmp;

        if (fabsf(XtX[i * dim + i]) < qfield.epsilon) return false;

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
        for (uint8_t j = i + 1; j < dim; ++j) sum += XtX[i * dim + j] * weights[j];
        weights[i] = (Xty[i] - sum) / XtX[i * dim + i];
    }

    return true;
}

bool PolynomialFitter::fit_lebesgue(const float* x, const float* y, size_t n) {
    if (n == 0) return false;
    float x_min = x[0], x_max = x[0];
    for(size_t i = 1; i < n; ++i) {
        if (x[i] < x_min) x_min = x[i];
        if (x[i] > x_max) x_max = x[i];
    }
    float x_range = x_max - x_min;
    if (x_range < qfield.epsilon) x_range = 1.0f;

    for (uint8_t d = 0; d <= degree; ++d) {
        float coeff = 0, norm_factor = 0;
        for (size_t i = 0; i < n; ++i) {
            float x_norm = 2.0f * (x[i] - x_min) / x_range - 1.0f;
            float p = LegendreBasis::eval(d, x_norm);
            coeff += y[i] * p;
            norm_factor += p * p;
        }
        weights[d] = (norm_factor > qfield.epsilon) ? coeff / norm_factor : 0;
    }
    return true;
}

float PolynomialFitter::predict(float x) const {
    float res = weights[0];
    float current_x = 1.0f;
    for (uint8_t d = 1; d <= degree; ++d) {
        current_x *= x;
        res += weights[d] * current_x;
    }
    return res;
}

ResidualFitter::ResidualFitter(uint8_t d_base, uint8_t d_res)
    : base_fitter(d_base), residual_fitter(d_res) {}

bool ResidualFitter::fit(const float* x, const float* y, size_t n, float lambda) {
    if (!base_fitter.fit(x, y, n, lambda)) return false;

    float* residuals = new float[n];
    for (size_t i = 0; i < n; ++i) {
        residuals[i] = y[i] - base_fitter.predict(x[i]);
    }

    bool res = residual_fitter.fit(x, residuals, n, lambda);
    delete[] residuals;
    return res;
}

float ResidualFitter::predict(float x) const {
    return base_fitter.predict(x) + residual_fitter.predict(x);
}

float PolynomialFitter::predict_lebesgue(float x, float x_min, float x_max) const {
    float x_range = x_max - x_min;
    if (x_range < qfield.epsilon) x_range = 1.0f;
    float x_norm = 2.0f * (x - x_min) / x_range - 1.0f;
    float res = 0;
    for (uint8_t d = 0; d <= degree; ++d) res += weights[d] * LegendreBasis::eval(d, x_norm);
    return res;
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

} // namespace polyfit

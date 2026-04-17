#ifndef WEYL_FILTER_UTILS_H
#define WEYL_FILTER_UTILS_H

#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ***************************************************************************
// Fixed‑Point Arithmetic Setup (Q16 Format)
// ***************************************************************************

#define Q_SHIFT 16
#define Q_ONE   (1 << Q_SHIFT)

static inline int32_t double_to_fixed(double d) { return (int32_t)(d * Q_ONE); }
static inline double fixed_to_double(int32_t q) { return ((double)q) / Q_ONE; }

static inline int32_t q_mul(int32_t a, int32_t b) {
  int64_t temp = (int64_t)a * (int64_t)b;
  int64_t result = temp >> Q_SHIFT;
  if (result > 2147483647LL) result = 2147483647LL;
  if (result < -2147483648LL) result = -2147483648LL;
  return (int32_t)result;
}

static inline int32_t q_div(int32_t a, int32_t b) {
  if (b == 0) return (a >= 0 ? 2147483647 : -2147483648);
  int64_t temp = ((int64_t)a << Q_SHIFT);
  int64_t result = temp / b;
  if (result > 2147483647LL) result = 2147483647LL;
  if (result < -2147483648LL) result = -2147483648LL;
  return (int32_t)result;
}

// ***************************************************************************
// Fixed‑Point Complex Number Structure and Operations
// ***************************************************************************

struct fixed_complex {
  int32_t real;
  int32_t imag;
};

static inline fixed_complex fc_add(fixed_complex a, fixed_complex b) {
  return {a.real + b.real, a.imag + b.imag};
}

static inline fixed_complex fc_sub(fixed_complex a, fixed_complex b) {
  return {a.real - b.real, a.imag - b.imag};
}

static inline fixed_complex fc_mul(fixed_complex a, fixed_complex b) {
  return {q_mul(a.real, b.real) - q_mul(a.imag, b.imag),
          q_mul(a.real, b.imag) + q_mul(a.imag, b.real)};
}

static inline fixed_complex fc_mul_scalar(fixed_complex a, int32_t scalar) {
  return {q_mul(a.real, scalar), q_mul(a.imag, scalar)};
}

static inline fixed_complex fc_from_double(double r, double i) {
  return {double_to_fixed(r), double_to_fixed(i)};
}

static inline void fc_to_double(fixed_complex fc, double &r, double &i) {
  r = fixed_to_double(fc.real);
  i = fixed_to_double(fc.imag);
}

// ***************************************************************************
// Field Filter Configuration
// ***************************************************************************

struct FieldConfig {
  int32_t lambda;
  int32_t eps;
  int32_t grad_weight;
  int32_t laplacian_weight;
};

// ***************************************************************************
// FFT and Sampling Setup
// ***************************************************************************

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 128
#endif

extern double realBuffer[BUFFER_SIZE];
extern double imagBuffer[BUFFER_SIZE];
extern fixed_complex fixedF[BUFFER_SIZE];

/**
 * processFrequencyDomain()
 *
 * Implements a regularized field operator in the frequency domain.
 */
inline void processFrequencyDomain(const FieldConfig& config) {
  // 1. Convert FFT results to fixed‑point
  for (int i = 0; i < BUFFER_SIZE; i++) {
    fixedF[i] = fc_from_double(realBuffer[i], imagBuffer[i]);
  }

  // 2. Constants for field equations
  int32_t deltaNorm = double_to_fixed(1.0 / BUFFER_SIZE);

  // 3. Compute the field gradient D and Laplacian L
  fixed_complex D[BUFFER_SIZE];
  fixed_complex L[BUFFER_SIZE];

  for (int i = 0; i < BUFFER_SIZE; i++) {
    // 4-point central difference for first derivative if possible
    if (i >= 2 && i < BUFFER_SIZE - 2) {
      fixed_complex term1 = fc_mul_scalar(fc_sub(fixedF[i+1], fixedF[i-1]), double_to_fixed(8.0));
      fixed_complex term2 = fc_sub(fixedF[i+2], fixedF[i-2]);
      D[i] = fc_mul_scalar(fc_sub(term1, term2), q_div(Q_ONE, q_mul(deltaNorm, double_to_fixed(12.0))));
    } else if (i > 0 && i < BUFFER_SIZE - 1) {
      D[i] = fc_mul_scalar(fc_sub(fixedF[i+1], fixedF[i-1]), q_div(Q_ONE, q_mul(deltaNorm, double_to_fixed(2.0))));
    } else if (i == 0) {
      D[0] = fc_mul_scalar(fc_sub(fixedF[1], fixedF[0]), q_div(Q_ONE, deltaNorm));
    } else {
      D[i] = fc_mul_scalar(fc_sub(fixedF[i], fixedF[i-1]), q_div(Q_ONE, deltaNorm));
    }

    // Laplacian: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
    if (i > 0 && i < BUFFER_SIZE - 1) {
      fixed_complex term1 = fc_add(fixedF[i+1], fixedF[i-1]);
      fixed_complex term2 = fc_mul_scalar(fixedF[i], double_to_fixed(2.0));
      L[i] = fc_mul_scalar(fc_sub(term1, term2), q_div(Q_ONE, q_mul(deltaNorm, deltaNorm)));
    } else {
      L[i] = {0, 0};
    }
  }

  // 4. Apply Regularized Field Operator
  for (int i = 0; i < BUFFER_SIZE; i++) {
    int32_t x;
    if (i <= BUFFER_SIZE / 2) {
      x = double_to_fixed((double)i / (BUFFER_SIZE / 2.0));
    } else {
      x = double_to_fixed((double)(BUFFER_SIZE - i) / (BUFFER_SIZE / 2.0));
    }

    int32_t x_sq = q_mul(x, x);
    int32_t V = x_sq - x + config.lambda;
    int32_t grad_V = q_mul(double_to_fixed(2.0), x) - Q_ONE;

    // Regularized field operator: P = F - D + F·(∇V) + L * weight
    fixed_complex grad_term = fc_mul_scalar(fixedF[i], q_mul(grad_V, config.grad_weight));
    fixed_complex lap_term = fc_mul_scalar(L[i], config.laplacian_weight);

    fixed_complex P = fc_sub(fixedF[i], D[i]);
    P = fc_add(P, grad_term);
    P = fc_add(P, lap_term);

    // H(x) = 1/(1 + V² + ε)
    int32_t V_sq = q_mul(V, V);
    int32_t denom = Q_ONE + V_sq + config.eps;
    int32_t attenuation = q_div(Q_ONE, denom);

    fixedF[i] = fc_mul_scalar(P, attenuation);
  }

  // 5. Convert back to double precision
  for (int i = 0; i < BUFFER_SIZE; i++) {
    fc_to_double(fixedF[i], realBuffer[i], imagBuffer[i]);
  }
}

inline void processFrequencyDomain() {
  FieldConfig defaultConfig = {
    double_to_fixed(0.1),  // lambda
    double_to_fixed(0.01), // eps
    double_to_fixed(0.1),  // grad_weight
    double_to_fixed(0.01)  // laplacian_weight
  };
  processFrequencyDomain(defaultConfig);
}

#endif // WEYL_FILTER_UTILS_H

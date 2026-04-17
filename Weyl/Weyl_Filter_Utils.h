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

#define Q_SHIFT 16              // 16 fractional bits
#define Q_ONE   (1 << Q_SHIFT)  // Representation of 1.0 in fixed‑point

// Convert a double to fixed‑point.
static inline int32_t double_to_fixed(double d) {
  return (int32_t)(d * Q_ONE);
}

// Convert fixed‑point to double.
static inline double fixed_to_double(int32_t q) {
  return ((double)q) / Q_ONE;
}

/**
 * Fixed-point multiplication with basic saturation.
 */
static inline int32_t q_mul(int32_t a, int32_t b) {
  int64_t temp = (int64_t)a * (int64_t)b;
  int64_t result = temp >> Q_SHIFT;
  if (result > 2147483647LL) result = 2147483647LL;
  if (result < -2147483648LL) result = -2147483648LL;
  return (int32_t)result;
}

/**
 * Fixed-point division with rudimentary zero‑check.
 */
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
  fixed_complex result;
  result.real = a.real + b.real;
  result.imag = a.imag + b.imag;
  return result;
}

static inline fixed_complex fc_sub(fixed_complex a, fixed_complex b) {
  fixed_complex result;
  result.real = a.real - b.real;
  result.imag = a.imag - b.imag;
  return result;
}

static inline fixed_complex fc_mul(fixed_complex a, fixed_complex b) {
  fixed_complex result;
  result.real = q_mul(a.real, b.real) - q_mul(a.imag, b.imag);
  result.imag = q_mul(a.real, b.imag) + q_mul(a.imag, b.real);
  return result;
}

static inline fixed_complex fc_mul_scalar(fixed_complex a, int32_t scalar) {
  fixed_complex result;
  result.real = q_mul(a.real, scalar);
  result.imag = q_mul(a.imag, scalar);
  return result;
}

static inline fixed_complex fc_from_double(double r, double i) {
  fixed_complex fc;
  fc.real = double_to_fixed(r);
  fc.imag = double_to_fixed(i);
  return fc;
}

static inline void fc_to_double(fixed_complex fc, double &r, double &i) {
  r = fixed_to_double(fc.real);
  i = fixed_to_double(fc.imag);
}

// ***************************************************************************
// Field Filter Configuration
// ***************************************************************************

struct FieldConfig {
  int32_t lambda;     // Field strength parameter
  int32_t eps;        // Regularization constant
  int32_t grad_weight; // Weight of the gradient term
};

// ***************************************************************************
// FFT and Sampling Setup
// ***************************************************************************

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 128
#endif

// Shared buffers
extern double realBuffer[BUFFER_SIZE];
extern double imagBuffer[BUFFER_SIZE];
extern fixed_complex fixedF[BUFFER_SIZE];

/**
 * processFrequencyDomain()
 *
 * Implements a regularized field operator in the frequency domain.
 * This version uses second-order finite differences for interior points.
 */
inline void processFrequencyDomain(const FieldConfig& config) {
  // 1. Convert FFT results to fixed‑point
  for (int i = 0; i < BUFFER_SIZE; i++) {
    fixedF[i] = fc_from_double(realBuffer[i], imagBuffer[i]);
  }

  // 2. Constants for field equations
  int32_t deltaNorm = double_to_fixed(1.0 / BUFFER_SIZE);

  // 3. Compute the field gradient (finite difference) D for each frequency bin
  fixed_complex D[BUFFER_SIZE];

  // Boundary: Bin 0 (DC) - First-order forward difference
  if (BUFFER_SIZE > 1) {
    fixed_complex diff = fc_sub(fixedF[1], fixedF[0]);
    int32_t scale = q_div(Q_ONE, deltaNorm);
    D[0] = fc_mul_scalar(diff, scale);
  } else {
    D[0] = {0, 0};
  }

  // Interior points: Second-order central difference for first derivative:
  // f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / 12h
  // For simplicity and to avoid too many boundary issues, we use 4-point central difference where possible
  for (int i = 1; i < BUFFER_SIZE - 1; i++) {
    if (i >= 2 && i < BUFFER_SIZE - 2) {
      // 4-point central difference
      fixed_complex term1 = fc_mul_scalar(fc_sub(fixedF[i+1], fixedF[i-1]), double_to_fixed(8.0));
      fixed_complex term2 = fc_sub(fixedF[i+2], fixedF[i-2]);
      fixed_complex diff = fc_sub(term1, term2);
      int32_t scale = q_div(Q_ONE, q_mul(deltaNorm, double_to_fixed(12.0)));
      D[i] = fc_mul_scalar(diff, scale);
    } else {
      // Standard 2-point central difference
      fixed_complex diff = fc_sub(fixedF[i+1], fixedF[i-1]);
      int32_t scale = q_div(Q_ONE, q_mul(deltaNorm, double_to_fixed(2.0)));
      D[i] = fc_mul_scalar(diff, scale);
    }
  }

  // Boundary: Nyquist frequency - First-order backward difference
  if (BUFFER_SIZE > 1) {
    fixed_complex diff = fc_sub(fixedF[BUFFER_SIZE-1], fixedF[BUFFER_SIZE-2]);
    int32_t scale = q_div(Q_ONE, deltaNorm);
    D[BUFFER_SIZE-1] = fc_mul_scalar(diff, scale);
  } else {
    D[BUFFER_SIZE-1] = {0, 0};
  }

  // 4. Apply Regularized Field Operator
  for (int i = 0; i < BUFFER_SIZE; i++) {
    // Normalized frequency x in [0, 1]
    int32_t x;
    if (i <= BUFFER_SIZE / 2) {
      x = double_to_fixed((double)i / (BUFFER_SIZE / 2.0));
    } else {
      x = double_to_fixed((double)(BUFFER_SIZE - i) / (BUFFER_SIZE / 2.0));
    }

    // Field potential: V(x) = x² - x + λ
    int32_t x_sq = q_mul(x, x);
    int32_t V = x_sq - x + config.lambda;

    // Field gradient: ∇V = 2x - 1
    int32_t grad_V = q_mul(double_to_fixed(2.0), x) - Q_ONE;

    // Regularized field operator: P = F - D + F·(∇V) * weight
    fixed_complex grad_term = fc_mul_scalar(fixedF[i], q_mul(grad_V, config.grad_weight));
    fixed_complex P = fc_sub(fixedF[i], D[i]);
    P = fc_add(P, grad_term);

    // Attenuation function: H(x) = 1/(1 + V² + ε)
    int32_t V_sq = q_mul(V, V);
    int32_t denom = Q_ONE + V_sq + config.eps;
    int32_t attenuation = q_div(Q_ONE, denom);

    // Final coefficient
    fixedF[i] = fc_mul_scalar(P, attenuation);
  }

  // 5. Convert back to double precision
  for (int i = 0; i < BUFFER_SIZE; i++) {
    double r, im;
    fc_to_double(fixedF[i], r, im);
    realBuffer[i] = r;
    imagBuffer[i] = im;
  }
}

/**
 * Overload for backward compatibility or simple use.
 */
inline void processFrequencyDomain() {
  FieldConfig defaultConfig = {
    double_to_fixed(0.3),  // lambda
    double_to_fixed(0.01), // eps
    Q_ONE                  // grad_weight = 1.0
  };
  processFrequencyDomain(defaultConfig);
}

#endif // WEYL_FILTER_UTILS_H

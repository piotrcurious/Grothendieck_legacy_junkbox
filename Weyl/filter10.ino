#include <Arduino.h>
#include <arduinoFFT.h>
#include <stdint.h>
#include <limits.h>

// ***************************************************************************
// Fixed‑Point Arithmetic Setup (Q16 Format)
// ***************************************************************************

#define Q_SHIFT 16              // 16 fractional bits
#define Q_ONE   (1 << Q_SHIFT)  // Fixed‑point representation of 1.0

// Convert a double to fixed‑point.
static inline int32_t double_to_fixed(double d) {
  return (int32_t)(d * Q_ONE);
}

// Convert fixed‑point to double.
static inline double fixed_to_double(int32_t q) {
  return ((double)q) / Q_ONE;
}

// Fixed‑point multiplication with basic saturation.
static inline int32_t q_mul(int32_t a, int32_t b) {
  int64_t temp = (int64_t)a * (int64_t)b;
  int64_t result = temp >> Q_SHIFT;
  // Basic saturation check:
  if(result > INT32_MAX) result = INT32_MAX;
  if(result < INT32_MIN) result = INT32_MIN;
  return (int32_t)result;
}

// Fixed‑point division with rudimentary zero‑check.
static inline int32_t q_div(int32_t a, int32_t b) {
  if (b == 0) return (a >= 0 ? INT32_MAX : INT32_MIN);
  int64_t temp = ((int64_t)a << Q_SHIFT);
  int64_t result = temp / b;
  if(result > INT32_MAX) result = INT32_MAX;
  if(result < INT32_MIN) result = INT32_MIN;
  return (int32_t)result;
}

// ***************************************************************************
// Fixed‑Point Complex Number Structure and Operations
// ***************************************************************************

struct fixed_complex {
  int32_t real;
  int32_t imag;
};

static inline fixed_complex fc_add(const fixed_complex &a, const fixed_complex &b) {
  fixed_complex result;
  result.real = a.real + b.real;
  result.imag = a.imag + b.imag;
  return result;
}

static inline fixed_complex fc_sub(const fixed_complex &a, const fixed_complex &b) {
  fixed_complex result;
  result.real = a.real - b.real;
  result.imag = a.imag - b.imag;
  return result;
}

static inline fixed_complex fc_mul(const fixed_complex &a, const fixed_complex &b) {
  fixed_complex result;
  // Complex multiplication: (a.real + i a.imag) * (b.real + i b.imag)
  result.real = q_mul(a.real, b.real) - q_mul(a.imag, b.imag);
  result.imag = q_mul(a.real, b.imag) + q_mul(a.imag, b.real);
  return result;
}

static inline fixed_complex fc_mul_scalar(const fixed_complex &a, int32_t scalar) {
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

static inline void fc_to_double(const fixed_complex &fc, double &r, double &i) {
  r = fixed_to_double(fc.real);
  i = fixed_to_double(fc.imag);
}

// ***************************************************************************
// FFT and Sampling Setup
// ***************************************************************************

#define SAMPLE_RATE 5000   // Hz
#define BUFFER_SIZE 128    // Must be a power of 2
#define ADC_PIN     34     // ESP32 ADC Input Pin

// Buffers for FFT (time domain, in doubles)
double realBuffer[BUFFER_SIZE];
double imagBuffer[BUFFER_SIZE];

// Create an FFT instance using the arduinoFFT library.
arduinoFFT FFT = arduinoFFT(realBuffer, imagBuffer, BUFFER_SIZE, SAMPLE_RATE);

// Global fixed‑point frequency domain array.
fixed_complex fixedF[BUFFER_SIZE];

// ***************************************************************************
// Frequency‑Domain Processing Using Fixed‑Point Algebra
// ***************************************************************************

// This function converts the FFT’s double complex coefficients into fixed‑point,
// computes a finite‑difference derivative, applies a simple Weyl operator, and
// then multiplies by an algebraic “constraint” polynomial.
void processFrequencyDomain() {
  // Convert FFT results (in doubles) to fixed‑point complex numbers.
  for (int i = 0; i < BUFFER_SIZE; i++) {
    fixedF[i] = fc_from_double(realBuffer[i], imagBuffer[i]);
  }

  // Compute a finite difference derivative for each frequency bin.
  // Here, for bin i, we approximate the derivative D[i] as:
  //    D[i] ≈ (F[i+1] - F[i]) / Δf, with Δf ≈ 1/BUFFER_SIZE.
  // Multiplying by BUFFER_SIZE (in fixed‑point) accounts for the spacing.
  fixed_complex D[BUFFER_SIZE];
  for (int i = 0; i < BUFFER_SIZE - 1; i++) {
    fixed_complex diff = fc_sub(fixedF[i+1], fixedF[i]);
    D[i] = fc_mul_scalar(diff, double_to_fixed((double)BUFFER_SIZE));
  }
  // For the last bin, set the derivative to zero.
  D[BUFFER_SIZE - 1].real = 0;
  D[BUFFER_SIZE - 1].imag = 0;

  // For each frequency bin, apply the Weyl operator:
  //    P(F) = F - D + F·D.
  // Then, apply an algebraic filter defined by g(x) = x² - x + λ,
  // where x is the normalized frequency (x = i/BUFFER_SIZE) and λ is a tuning constant.
  for (int i = 0; i < BUFFER_SIZE; i++) {
    fixed_complex op = fc_add( fc_sub(fixedF[i], D[i]), fc_mul(fixedF[i], D[i]) );

    // Compute normalized frequency x in fixed‑point.
    int32_t x = double_to_fixed((double)i / (double)BUFFER_SIZE);
    // Compute x² in fixed‑point.
    int32_t x_sq = q_mul(x, x);
    int32_t lambda = double_to_fixed(0.3); // Tuning parameter
    // The constraint polynomial: g(x) = x² - x + λ.
    int32_t constraint = x_sq - x + lambda;

    // Multiply the operator result by the constraint.
    fixed_complex filtered = fc_mul_scalar(op, constraint);

    // Store the filtered coefficient.
    fixedF[i] = filtered;
  }

  // Convert the fixed‑point frequency domain data back to doubles for inverse FFT.
  for (int i = 0; i < BUFFER_SIZE; i++) {
    double r, im;
    fc_to_double(fixedF[i], r, im);
    realBuffer[i] = r;
    imagBuffer[i] = im;
  }
}

// ***************************************************************************
// Sampling Function
// ***************************************************************************

// Samples the ADC, normalizes the value, and fills the time‑domain buffers.
// (For a robust design, consider using timer interrupts rather than delayMicroseconds.)
void sampleSignal() {
  for (int i = 0; i < BUFFER_SIZE; i++) {
    // Read the ADC value and normalize to [0, 1].
    double sample = analogRead(ADC_PIN) / 4095.0;
    realBuffer[i] = sample;
    imagBuffer[i] = 0.0;
    delayMicroseconds(1000000 / SAMPLE_RATE);
  }
}

// ***************************************************************************
// Arduino Setup and Loop
// ***************************************************************************

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
}

void loop() {
  // Sample the signal from the ADC.
  sampleSignal();

  // Compute the FFT of the sampled time-domain signal.
  FFT.Windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.Compute(FFT_FORWARD);
  // Note: We keep the full complex FFT data (do not discard phase).

  // Process the frequency domain using our fixed‑point Weyl operator and constraint.
  processFrequencyDomain();

  // Compute the inverse FFT to reconstruct the filtered time‑domain signal.
  FFT.Compute(FFT_INVERSE);

  // Output the reconstructed signal for debugging.
  for (int i = 0; i < BUFFER_SIZE; i++) {
    Serial.println(realBuffer[i]);
  }
  
  delay(500);
}

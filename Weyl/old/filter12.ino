#include <Arduino.h>
#include <arduinoFFT.h>
#include <stdint.h>
#include <limits.h>

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

// Fixed‑point multiplication with basic saturation.
static inline int32_t q_mul(int32_t a, int32_t b) {
  int64_t temp = (int64_t)a * (int64_t)b;
  int64_t result = temp >> Q_SHIFT;
  if (result > INT32_MAX) result = INT32_MAX;
  if (result < INT32_MIN) result = INT32_MIN;
  return (int32_t)result;
}

// Fixed‑point division with rudimentary zero‑check.
static inline int32_t q_div(int32_t a, int32_t b) {
  if (b == 0) return (a >= 0 ? INT32_MAX : INT32_MIN);
  int64_t temp = ((int64_t)a << Q_SHIFT);
  int64_t result = temp / b;
  if (result > INT32_MAX) result = INT32_MAX;
  if (result < INT32_MIN) result = INT32_MIN;
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
  // (a.real + i*a.imag) * (b.real + i*b.imag) = (a.real*b.real - a.imag*b.imag) + i(a.real*b.imag + a.imag*b.real)
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

// Time‑domain buffers (using doubles for the FFT library).
double realBuffer[BUFFER_SIZE];
double imagBuffer[BUFFER_SIZE];

// Create an FFT instance using the arduinoFFT library.
arduinoFFT FFT = arduinoFFT(realBuffer, imagBuffer, BUFFER_SIZE, SAMPLE_RATE);

// Global fixed‑point frequency-domain array.
fixed_complex fixedF[BUFFER_SIZE];

// ***************************************************************************
// Accurate Sampling Using micros()
// ***************************************************************************
//
// This routine schedules sampling times using micros() so that the sampling
// rate remains as constant as possible.
void sampleSignal() {
  const unsigned long sampleInterval = 1000000UL / SAMPLE_RATE;
  unsigned long nextSampleTime = micros();
  
  for (int i = 0; i < BUFFER_SIZE; i++) {
    while (micros() < nextSampleTime) {
      // Busy-wait; in a more robust implementation you might use interrupts.
    }
    int adcValue = analogRead(ADC_PIN);
    if (adcValue < 0) adcValue = 0;
    if (adcValue > 4095) adcValue = 4095;
    double sample = adcValue / 4095.0;
    
    realBuffer[i] = sample;
    imagBuffer[i] = 0.0;
    
    nextSampleTime += sampleInterval;
  }
}

// ***************************************************************************
// Frequency‑Domain Processing Using Fixed‑Point Algebra
// ***************************************************************************
//
// For each frequency bin we do the following:
//   1. Convert the FFT output (double) to a fixed‑point complex number.
//   2. Compute a finite-difference derivative D using forward, central,
//      or backward differences.
//   3. Form the Weyl operator: P(F) = F - D + F·D.
//   4. Compute a normalized frequency x = i / BUFFER_SIZE, and then a
//      constraint value g(x) = x² - x + λ.
//   5. Define an attenuation (filter) function H(x) = 1 / (1 + [g(x)]²).
//   6. Multiply the Weyl operator result by H(x) to obtain the final filtered
//      coefficient.
void processFrequencyDomain() {
  // Convert FFT results to fixed‑point.
  for (int i = 0; i < BUFFER_SIZE; i++) {
    fixedF[i] = fc_from_double(realBuffer[i], imagBuffer[i]);
  }
  
  // Normalized frequency spacing Δx ≈ 1/BUFFER_SIZE (in fixed‑point).
  int32_t deltaNorm = double_to_fixed(1.0 / BUFFER_SIZE);
  
  // Compute the finite-difference derivative D for each frequency bin.
  fixed_complex D[BUFFER_SIZE];
  
  // Bin 0: forward difference.
  if (BUFFER_SIZE > 1) {
    fixed_complex diff = fc_sub(fixedF[1], fixedF[0]);
    D[0].real = q_div(diff.real, deltaNorm);
    D[0].imag = q_div(diff.imag, deltaNorm);
  } else {
    D[0].real = 0;
    D[0].imag = 0;
  }
  
  // Bins 1 to BUFFER_SIZE-2: central difference.
  for (int i = 1; i < BUFFER_SIZE - 1; i++) {
    fixed_complex diff = fc_sub(fixedF[i+1], fixedF[i-1]);
    // Denominator is 2*deltaNorm.
    int32_t twoDelta = q_mul(deltaNorm, double_to_fixed(2.0));
    D[i].real = q_div(diff.real, twoDelta);
    D[i].imag = q_div(diff.imag, twoDelta);
  }
  
  // Last bin: backward difference.
  if (BUFFER_SIZE > 1) {
    fixed_complex diff = fc_sub(fixedF[BUFFER_SIZE-1], fixedF[BUFFER_SIZE-2]);
    D[BUFFER_SIZE-1].real = q_div(diff.real, deltaNorm);
    D[BUFFER_SIZE-1].imag = q_div(diff.imag, deltaNorm);
  } else {
    D[BUFFER_SIZE-1].real = 0;
    D[BUFFER_SIZE-1].imag = 0;
  }
  
  // Process each frequency bin.
  for (int i = 0; i < BUFFER_SIZE; i++) {
    // Apply the Weyl operator: P(F) = F - D + F·D.
    fixed_complex term1 = fc_sub(fixedF[i], D[i]);
    fixed_complex term2 = fc_mul(fixedF[i], D[i]);
    fixed_complex P = fc_add(term1, term2);
    
    // Compute normalized frequency x = i / BUFFER_SIZE in fixed‑point.
    int32_t x = double_to_fixed((double)i / (double)BUFFER_SIZE);
    int32_t x_sq = q_mul(x, x);
    int32_t lambda = double_to_fixed(0.3); // Tuning parameter.
    
    // Compute the constraint value: g(x) = x² - x + λ.
    int32_t g_val = x_sq - x + lambda;
    
    // Define a filter (attenuation) function:
    //    H(x) = 1 / (1 + (g(x))²)
    int32_t g_sq = q_mul(g_val, g_val);
    int32_t onePlus_g_sq = Q_ONE + g_sq;
    int32_t attenuation = q_div(Q_ONE, onePlus_g_sq);
    
    // Final filtered coefficient: Q(F) = P(F) * H(x).
    fixed_complex filtered = fc_mul_scalar(P, attenuation);
    
    // Store the filtered coefficient.
    fixedF[i] = filtered;
  }
  
  // Convert the fixed‑point frequency-domain data back to doubles for the inverse FFT.
  for (int i = 0; i < BUFFER_SIZE; i++) {
    double r, im;
    fc_to_double(fixedF[i], r, im);
    realBuffer[i] = r;
    imagBuffer[i] = im;
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
  // 1. Sample the ADC signal using accurate timing.
  sampleSignal();
  
  // 2. Compute the FFT of the time‑domain signal.
  FFT.Windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.Compute(FFT_FORWARD);
  // We retain both magnitude and phase.
  
  // 3. Process the frequency-domain data using our fixed‑point Weyl operator
  //    and the completed attenuation logic.
  processFrequencyDomain();
  
  // 4. Compute the inverse FFT to reconstruct the filtered time‑domain signal.
  FFT.Compute(FFT_INVERSE);
  
  // 5. Output the reconstructed signal for debugging.
  for (int i = 0; i < BUFFER_SIZE; i++) {
    Serial.println(realBuffer[i]);
  }
  
  // Short delay before processing the next block.
  delay(500);
}

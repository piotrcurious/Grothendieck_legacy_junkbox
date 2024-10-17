#include <Arduino.h>

const int MAX_DEGREE = 5;
const int MAX_SAMPLES = 30;

// GF(8) operations
byte gf8_add(byte a, byte b) { return a ^ b; }

byte gf8_mul(byte a, byte b) {
  byte p = 0;
  for (int i = 0; i < 3; i++) {
    if (b & 1) p ^= a;
    byte high_bit = a & 4;
    a <<= 1;
    if (high_bit) a ^= 0x0B; // x^3 + x + 1
    b >>= 1;
  }
  return p;
}

byte gf8_inv(byte a) {
  for (byte i = 1; i < 8; i++)
    if (gf8_mul(a, i) == 1) return i;
  return 0; // This should never happen for non-zero elements
}

// Berlekamp-Massey algorithm
void berlekamp_massey(byte* sequence, int n, byte* coeffs, int& length) {
  byte c[MAX_SAMPLES] = {1}, b[MAX_SAMPLES] = {1};
  int l = 0, m = 1, b_len = 1;

  for (int i = 0; i < n; i++) {
    byte d = sequence[i];
    for (int j = 1; j <= l; j++)
      d = gf8_add(d, gf8_mul(c[j], sequence[i - j]));
    if (d == 0) {
      m++;
    } else if (2 * l <= i) {
      byte temp[MAX_SAMPLES];
      memcpy(temp, c, MAX_SAMPLES);
      for (int j = 0; j < b_len; j++)
        c[j + i - m] = gf8_add(c[j + i - m], gf8_mul(d, b[j]));
      l = i + 1 - l;
      memcpy(b, temp, MAX_SAMPLES);
      b_len = MAX_SAMPLES;
      m = 1;
    } else {
      for (int j = 0; j < b_len; j++)
        c[j + i - m] = gf8_add(c[j + i - m], gf8_mul(d, b[j]));
      m++;
    }
  }
  
  memcpy(coeffs, c, l + 1);
  length = l + 1;
}

// Polynomial evaluation over GF(8)
byte evaluate_poly(byte* coeffs, int degree, byte x) {
  byte result = 0;
  byte power = 1;
  for (int i = 0; i <= degree; i++) {
    result = gf8_add(result, gf8_mul(coeffs[i], power));
    power = gf8_mul(power, x);
  }
  return result;
}

// LFSR run function
void run_lfsr(byte* coeffs, int length, byte* state, int steps, byte* output) {
  for (int i = 0; i < steps; i++) {
    output[i] = state[0];
    byte feedback = 0;
    for (int j = 0; j < length; j++)
      feedback = gf8_add(feedback, gf8_mul(coeffs[j], state[j]));
    for (int j = 0; j < length - 1; j++)
      state[j] = state[j + 1];
    state[length - 1] = feedback;
  }
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Generate a random polynomial
  byte true_poly[MAX_DEGREE + 1];
  for (int i = 0; i <= MAX_DEGREE; i++)
    true_poly[i] = random(8);

  Serial.println("True polynomial coefficients:");
  for (int i = 0; i <= MAX_DEGREE; i++) {
    Serial.print(true_poly[i]);
    Serial.print(" ");
  }
  Serial.println();

  // Generate noisy samples
  byte samples[MAX_SAMPLES];
  for (int i = 0; i < MAX_SAMPLES; i++) {
    byte x = (i % 7) + 1; // Non-zero elements of GF(8)
    samples[i] = evaluate_poly(true_poly, MAX_DEGREE, x);
    // Add noise with 20% probability
    if (random(100) < 20)
      samples[i] = gf8_add(samples[i], random(1, 8));
  }

  // Fit LFSR using Berlekamp-Massey
  byte lfsr_coeffs[MAX_SAMPLES];
  int lfsr_length;
  unsigned long start_time = micros();
  berlekamp_massey(samples, MAX_SAMPLES, lfsr_coeffs, lfsr_length);
  unsigned long end_time = micros();

  Serial.println("LFSR coefficients:");
  for (int i = 0; i < lfsr_length; i++) {
    Serial.print(lfsr_coeffs[i]);
    Serial.print(" ");
  }
  Serial.println();

  Serial.print("Fitting time (microseconds): ");
  Serial.println(end_time - start_time);

  // Run LFSR
  byte state[MAX_SAMPLES];
  memcpy(state, samples, lfsr_length - 1);
  byte lfsr_output[MAX_SAMPLES];
  run_lfsr(lfsr_coeffs, lfsr_length, state, MAX_SAMPLES, lfsr_output);

  // Compare LFSR output with original samples
  Serial.println("Comparison (Original, LFSR output):");
  int mismatches = 0;
  for (int i = 0; i < MAX_SAMPLES; i++) {
    Serial.print(samples[i]);
    Serial.print(", ");
    Serial.println(lfsr_output[i]);
    if (samples[i] != lfsr_output[i]) mismatches++;
  }

  Serial.print("Total mismatches: ");
  Serial.print(mismatches);
  Serial.print(" out of ");
  Serial.println(MAX_SAMPLES);
}

void loop() {
  // Nothing to do in the loop
}

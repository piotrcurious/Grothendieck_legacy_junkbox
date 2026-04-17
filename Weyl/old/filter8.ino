#include <Arduino.h>
#include <arduinoFFT.h>
#include <math.h>

// --- Configuration parameters ---
#define SAMPLE_RATE   5000        // Sampling rate in Hz
#define BUFFER_SIZE   128         // Must be a power of 2 for FFT
#define ADC_PIN       34          // ADC input pin
const int Q = 251;                // A small prime for finite-field (quantized) arithmetic

// --- Global arrays for FFT (the signal will be interpreted as a function over a quantized field) ---
double vReal[BUFFER_SIZE];
double vImag[BUFFER_SIZE];
arduinoFFT FFT = arduinoFFT(vReal, vImag, BUFFER_SIZE, SAMPLE_RATE);

// --- Step 1. Sampling the Signal ---
// Each sample is already quantized by the ADC resolution and sampling frequency.
void sampleSignal() {
  for (int i = 0; i < BUFFER_SIZE; i++) {
    // Read and normalize the ADC value (0 to 1)
    double sample = analogRead(ADC_PIN) / 4095.0;
    vReal[i] = sample;
    vImag[i] = 0;   // The imaginary part is zero in time domain.
    delayMicroseconds(1000000 / SAMPLE_RATE);
  }
}

// --- Step 2. Transformation to the Frequency Domain ---
// The FFT gives us a quantized (discrete) frequency model. Each bin is an “atomic” element.
  
// --- Helper: Quantize a floating–point magnitude into a finite field element ---
// (Conceptually, we map the magnitude into the finite field F_Q.)
int quantize(double value, int Q) {
  // Assume the signal is normalized (0 <= value <= 1).
  // Multiply by Q and round; then take modulo Q.
  int q_val = (int)(value * Q);
  q_val %= Q;
  if (q_val < 0) q_val += Q;
  return q_val;
}

// --- Step 3. Apply a Weyl–algebra operator in the quantized frequency domain ---
// We define a toy Weyl operator acting on the quantized magnitudes.
// Here the operator is defined as:  
//      W(u)[i] = ( (i mod Q) * u[i]  -  (u[i+1] - u[i]) )  mod Q
// The first term is like a “position” (multiplication by frequency index)
// and the finite–difference approximates the “momentum” (derivative).
void applyWeylOperator(const int *quantMag, int *weylOut, int N, int Q) {
  for (int i = 0; i < N; i++) {
    // Finite difference (cyclic difference for the quantized field)
    int d = (quantMag[(i + 1) % N] - quantMag[i]) % Q;
    if (d < 0) d += Q;
    int term = ((i % Q) * quantMag[i]) % Q;
    int W = (term - d) % Q;
    if (W < 0) W += Q;
    weylOut[i] = W;
  }
}

// --- Step 4. Process the Signal in the Frequency Domain ---
// We convert the time–domain signal to its frequency representation,
// “lift” the magnitudes into the finite field, apply the Weyl operator,
// then map back to a modified frequency profile that enhances (for example)
// the lower–frequency “subvariety.”
void processSignal() {
  // Apply windowing (Hamming) to reduce leakage.
  FFT.Windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  // Compute FFT: the result is stored in vReal (real parts) and vImag (imaginary parts)
  FFT.Compute(FFT_FORWARD);
  
  int halfSize = BUFFER_SIZE / 2; // For a real–valued time signal, only half the bins are independent.
  int quantMag[halfSize];  // Array holding quantized magnitudes (finite-field elements)
  int weylMag[halfSize];   // Array holding the result after applying the Weyl operator
  
  // For each frequency bin in the lower half, compute the magnitude and quantize it.
  for (int i = 0; i < halfSize; i++) {
    double mag = sqrt(vReal[i] * vReal[i] + vImag[i] * vImag[i]);
    // (Assume that the ADC and FFT have normalized the signal appropriately.)
    quantMag[i] = quantize(mag, Q);
  }
  
  // Apply the Weyl operator to the quantized magnitudes.
  applyWeylOperator(quantMag, weylMag, halfSize, Q);
  
  // Now map the quantized result back to a floating–point “magnitude.”
  // Also, we preserve the original phase of each FFT bin.
  for (int i = 0; i < halfSize; i++) {
    double phase = atan2(vImag[i], vReal[i]);
    // Map the quantized value back to a magnitude in [0,1].
    double newMag = ((double)weylMag[i]) / ((double)Q);
    
    // --- Band Splitting ---  
    // For demonstration, we only modify bins below a threshold (i.e. lower frequencies).
    int lowBandThreshold = halfSize / 4;
    if (i > lowBandThreshold) {
      // For higher frequencies, we leave the magnitude unchanged.
      newMag = sqrt(vReal[i] * vReal[i] + vImag[i] * vImag[i]);
    }
    
    // Update the frequency bin with the new magnitude and original phase.
    vReal[i] = newMag * cos(phase);
    vImag[i] = newMag * sin(phase);
    // For the inverse FFT of a real signal, enforce conjugate symmetry.
    if (i != 0) {
      vReal[BUFFER_SIZE - i] = vReal[i];
      vImag[BUFFER_SIZE - i] = -vImag[i];
    }
  }
  
  // --- Step 5. Inverse Transform ---
  // Compute the inverse FFT to reconstruct the time–domain signal.
  FFT.Compute(FFT_INVERSE);
  FFT.ComplexToMagnitude();  // Now vReal holds the (filtered) time–domain signal.
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);   // ESP32 ADC: 12-bit resolution.
  Serial.println("Starting Quantized Frequency–Domain Processing...");
}

void loop() {
  // 1. Sample the input signal.
  sampleSignal();
  
  // 2–5. Process it in the quantized frequency domain via a Weyl–algebra operator.
  processSignal();
  
  // 6. Output the processed (time–domain) signal.
  // (For demonstration, we print the first BUFFER_SIZE samples.)
  for (int i = 0; i < BUFFER_SIZE; i++) {
    Serial.println(vReal[i]);
  }
  delay(500);
}

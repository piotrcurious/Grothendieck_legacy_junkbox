/*
  Example: Band-Splitting Filter with Weyl-Algebra-Inspired Coefficient Modulation

  This sketch demonstrates a block-based filter on an ESP32 that extracts the lower frequency
  band from an incoming analog signal. Although the signal is already quantized at the ADC
  (Nyquist limit), we “leverage” its field (by computing an average) to simulate an atomic quantization
  effect. The filter coefficients are computed from a sinc (low-pass) kernel multiplied by a Hamming
  window, then modulated by a phase factor derived from the input’s “field” – a concept inspired by
  Weyl algebra and constructive algebraic geometry. (This is a conceptual demonstration rather than a
  rigorous implementation.)

  Hardware setup:
   - An analog signal is fed to ADC pin 34.
   - The ESP32 ADC is set to 12‐bit resolution.
*/

#include <Arduino.h>
#include <math.h>

// Define PI if not defined
#ifndef PI
#define PI 3.14159265358979323846
#endif

// Sampling and filter parameters
#define BUFFER_SIZE      256        // Number of samples in one processing block
#define SAMPLE_RATE      10000.0    // Sample rate in Hz (adjust as needed)
#define CUTOFF_FREQUENCY 1000.0     // Low-pass cutoff (lower frequency band)
#define FILTER_ORDER     51         // Filter order (should be odd for symmetric FIR)

// Global sample buffer and index
float sampleBuffer[BUFFER_SIZE];
volatile int sampleIndex = 0;

/*
  Class: WeylAlgebraFilter

  This class implements a FIR filter whose coefficients are computed from a sinc function
  (to form a low-pass filter) and then modulated by a cosine factor. That modulation uses
  a phase derived from the “average field” of the input signal (i.e. the block of samples).
  The idea is to mimic, in a very abstract way, the procedure of using Weyl algebra to
  “quantize” the field and thereby select atomic frequency components.
*/
class WeylAlgebraFilter {
  public:
    int order;          // Number of filter taps
    float cutoff;       // Cutoff frequency (Hz)
    float sampleRate;   // Sampling rate (Hz)
    float *coeffs;      // Pointer to filter coefficients (length = order)

    // Constructor: allocate coefficient array.
    WeylAlgebraFilter(int order, float cutoff, float sampleRate) {
      this->order = order;
      this->cutoff = cutoff;
      this->sampleRate = sampleRate;
      coeffs = new float[order];
      // Initialize coefficients (will be updated later)
      for (int i = 0; i < order; i++) {
        coeffs[i] = 0.0;
      }
    }
    ~WeylAlgebraFilter() {
      delete[] coeffs;
    }

    // updateCoeffs: (Re)compute the filter coefficients based on the current input field.
    // The “field” here is the current block of samples.
    void updateCoeffs(const float *field, int fieldSize) {
      // Compute the average of the input field. This average is our “atomic” parameter.
      float avgField = 0.0;
      for (int i = 0; i < fieldSize; i++) {
        avgField += field[i];
      }
      avgField /= fieldSize;
      // Compute a phase factor from the average (simulate atomic quantization)
      float phase = fmod(avgField, 2 * PI);

      int M = order - 1;  // M = filter delay in taps
      // Compute the base (sinc) low-pass filter coefficients with a Hamming window.
      // Then modulate each tap with a cosine factor whose argument is scaled by the phase.
      for (int n = 0; n < order; n++) {
        float x = n - (M / 2.0);
        float baseCoeff;
        if (fabs(x) < 1e-6) {
          baseCoeff = cutoff / sampleRate;
        } else {
          baseCoeff = sin(PI * cutoff * x / sampleRate) / (PI * x);
        }
        // Apply a Hamming window
        baseCoeff *= (0.54 - 0.46 * cos(2 * PI * n / M));
        // Modulate by a cosine factor derived from the phase (Weyl algebra inspiration)
        coeffs[n] = baseCoeff * cos(phase * x);
      }
    }

    // apply: Convolve the filter with the input signal (taken from the provided buffer)
    float apply(const float *signal, int index, int signalSize) {
      float result = 0.0;
      int half = order / 2;
      // Convolution sum: for each tap, use the signal sample (with appropriate boundary handling)
      for (int n = 0; n < order; n++) {
        int idx = index - half + n;
        if (idx < 0) idx = 0;
        if (idx >= signalSize) idx = signalSize - 1;
        result += coeffs[n] * signal[idx];
      }
      return result;
    }
};

// Instantiate the filter object (global instance)
WeylAlgebraFilter lowPassFilter(FILTER_ORDER, CUTOFF_FREQUENCY, SAMPLE_RATE);

void setup() {
  Serial.begin(115200);
  // Set ADC resolution to 12 bits on the ESP32
  analogReadResolution(12);
  // (Optional) ADC setup: depending on your board/configuration
}

void loop() {
  // Read one analog sample from pin 34
  int rawValue = analogRead(34);
  // Convert the raw ADC value (0..4095) to a voltage (assuming 3.3V reference)
  float voltage = (rawValue / 4095.0) * 3.3;
  
  // Store the sample in the global circular buffer
  sampleBuffer[sampleIndex] = voltage;
  sampleIndex++;

  // When the buffer is full, process the block.
  if (sampleIndex >= BUFFER_SIZE) {
    // Update the filter coefficients using the current block.
    // (This is where we “leverage the field” of the input signal.)
    lowPassFilter.updateCoeffs(sampleBuffer, BUFFER_SIZE);

    // For demonstration, apply the filter to each sample in the block and print the results.
    Serial.println("Filtered Block:");
    for (int i = 0; i < BUFFER_SIZE; i++) {
      float filteredValue = lowPassFilter.apply(sampleBuffer, i, BUFFER_SIZE);
      Serial.println(filteredValue);
    }
    Serial.println("----- End Block -----\n");

    // Reset the buffer index to start a new block.
    sampleIndex = 0;
  }

  // Wait for the next sample period (in microseconds).
  delayMicroseconds((unsigned long)(1000000.0 / SAMPLE_RATE));
}

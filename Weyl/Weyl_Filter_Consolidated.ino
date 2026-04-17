#include <Arduino.h>
#include <arduinoFFT.h>
#include "Weyl_Filter_Utils.h"

/**
 * Consolidated Weyl Filter for ESP32
 *
 * This sketch demonstrates the application of Weyl algebra concepts to
 * frequency-domain signal processing. It uses fixed-point arithmetic
 * for the core field operator calculations.
 */

#define SAMPLE_RATE 5000   // Hz
#define ADC_PIN     34     // ESP32 ADC Input Pin

// Global buffers defined in Weyl_Filter_Utils.h
double realBuffer[BUFFER_SIZE];
double imagBuffer[BUFFER_SIZE];
fixed_complex fixedF[BUFFER_SIZE];

// FFT instance
arduinoFFT FFT = arduinoFFT(realBuffer, imagBuffer, BUFFER_SIZE, SAMPLE_RATE);

/**
 * Accurately sample the ADC signal using micros() to maintain constant sample rate.
 */
void sampleSignal() {
  const unsigned long sampleInterval = 1000000UL / SAMPLE_RATE;
  unsigned long nextSampleTime = micros();

  for (int i = 0; i < BUFFER_SIZE; i++) {
    while (micros() < nextSampleTime) {
      // Busy-wait
    }
    int adcValue = analogRead(ADC_PIN);
    // ADC resolution is typically 12-bit on ESP32 (0-4095)
    realBuffer[i] = (double)adcValue / 4095.0;
    imagBuffer[i] = 0.0;

    nextSampleTime += sampleInterval;
  }
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  Serial.println("Weyl Filter Consolidated Initialized");
}

void loop() {
  // 1. Sample ADC
  sampleSignal();

  // 2. Forward FFT
  FFT.Windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.Compute(FFT_FORWARD);

  // 3. Process in Frequency Domain (Fixed-Point Field Operator)
  processFrequencyDomain();

  // 4. Inverse FFT
  FFT.Compute(FFT_INVERSE);

  // 5. Output reconstructed signal
  for (int i = 0; i < BUFFER_SIZE; i++) {
    Serial.println(realBuffer[i]);
  }

  delay(500);
}

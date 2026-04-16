#include <math.h>
#include "fitting_utils.h"

const int sensorPin = A0; // Thermistor connected to analog pin A0
const int bufferSize = FITTING_BUFFER_SIZE; // Number of readings in the buffer
float temperatureBuffer[bufferSize]; // Buffer to store temperature readings
unsigned long timeBuffer[bufferSize]; // Buffer to store corresponding timestamps
int bufferIndex = 0; // Index to keep track of the buffer position
const unsigned long sampleInterval = 1000; // Sampling interval in milliseconds (1 second)

float smoothedTemp = 0;
bool firstReading = true;

// Function to read temperature from the sensor
float readTemperature() {
  int rawValue = analogRead(sensorPin);
  float voltage = rawValue * (5.0 / 1023.0);
  float temperatureC = (voltage - 0.5) * 100.0; // Convert voltage to temperature (assuming TMP36)

  if (firstReading) {
      smoothedTemp = temperatureC;
      firstReading = false;
  } else {
      smoothedTemp = exponentialMovingAverage(temperatureC, smoothedTemp, 0.3);
  }

  return smoothedTemp;
}

void setup() {
  Serial.begin(9600);
  // Initialize buffers with zero values
  for (int i = 0; i < bufferSize; i++) {
    temperatureBuffer[i] = 0;
    timeBuffer[i] = 0;
  }
}

void loop() {
  static unsigned long lastSampleTime = 0;
  if (millis() - lastSampleTime >= sampleInterval) {
    lastSampleTime = millis();
    float temperature = readTemperature();
    unsigned long currentTime = millis();

    // Store the reading in the buffers
    temperatureBuffer[bufferIndex] = temperature;
    timeBuffer[bufferIndex] = currentTime;
    bufferIndex = (bufferIndex + 1) % bufferSize;

    // Only analyze data if the buffer is full
    if (bufferIndex == 0) {
      // Robust outlier detection using Hampel Filter
      float filteredBuffer[bufferSize];
      for (int i = 0; i < bufferSize; i++) {
          filteredBuffer[i] = hampelFilter(temperatureBuffer, bufferSize, i, 5);
      }

      // Convert time to seconds for analysis
      float timeSeconds[bufferSize];
      for (int i = 0; i < bufferSize; i++) {
        timeSeconds[i] = (timeBuffer[i] - timeBuffer[0]) / 1000.0;
      }

      // First pass: Hierarchical fit to detect growth type
      ResidualFitter fitter;
      fitter.fit(timeSeconds, filteredBuffer, bufferSize);

      float confidence = fitter.growthConfidence();

      if (confidence > 0.2 && fitter.rmse_residual < 2.0) {
        Serial.println("Exponential growth detected on first pass!");

        // Second pass: Fit polynomial for detailed rate analysis
        const int degree = 3;
        float coeffs[degree + 1];
        Normalizer polyNorm;
        polyNorm.compute(timeSeconds, filteredBuffer, bufferSize);
        if (polynomialFitRidge(timeSeconds, filteredBuffer, bufferSize, degree, coeffs)) {
            float latestTime = timeSeconds[bufferSize - 1];
            float rateOfGrowth = polynomialDerivativeNormalized(coeffs, degree, latestTime, polyNorm);

            Serial.print("Rate of exponential growth: ");
            Serial.println(rateOfGrowth);
            Serial.print("Growth Confidence: ");
            Serial.println(confidence);

            // Determine magnitude
            float magnitude = exponentialModel(latestTime, fitter.a, fitter.exp_b);
            Serial.print("Magnitude of growth: ");
            Serial.println(magnitude);

            if (rateOfGrowth > 0.5 && magnitude > 25.0) {
              Serial.println("Confirmed exponential growth detected with significant rate and magnitude.");
            } else {
              Serial.println("Exponential growth detected, but not significant.");
            }
        }
      } else if (fitter.rmse_linear < 2.0) {
        Serial.println("Linear growth detected.");
      } else {
        Serial.println("No significant growth detected or short-term noise.");
      }

      // Debug output
      Serial.print("RMSE Linear: ");
      Serial.println(fitter.rmse_linear);
      Serial.print("RMSE Exponential: ");
      Serial.println(fitter.rmse_residual);
      Serial.println();
    }
  }
}

#include <math.h>
#include "fitting_utils.h"

const int sensorPin = A0; // Thermistor connected to analog pin A0
const int bufferSize = FITTING_BUFFER_SIZE; // Number of readings in the buffer
float temperatureBuffer[bufferSize]; // Buffer to store temperature readings
unsigned long timeBuffer[bufferSize]; // Buffer to store corresponding timestamps
int bufferIndex = 0; // Index to keep track of the buffer position
const unsigned long sampleInterval = 1000; // Sampling interval in milliseconds (1 second)

// Function to read temperature from the sensor
float readTemperature() {
  int rawValue = analogRead(sensorPin);
  float voltage = rawValue * (5.0 / 1023.0);
  float temperatureC = (voltage - 0.5) * 100.0; // Convert voltage to temperature (assuming TMP36)
  return temperatureC;
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
      // Convert time to seconds for analysis
      float timeSeconds[bufferSize];
      for (int i = 0; i < bufferSize; i++) {
        timeSeconds[i] = (timeBuffer[i] - timeBuffer[0]) / 1000.0;
      }

      // Perform linear fit
      float m, b;
      linearFit(timeSeconds, temperatureBuffer, bufferSize, m, b);

      // Perform exponential fit using Fredholm kernel approach
      float a, expB;
      exponentialFitFredholm(timeSeconds, temperatureBuffer, bufferSize, a, expB);

      // Calculate goodness of fit for both models
      float r2Linear = goodnessOfFit(timeSeconds, temperatureBuffer, bufferSize, linearModel, m, b);
      float r2Exponential = goodnessOfFit(timeSeconds, temperatureBuffer, bufferSize, exponentialModel, a, expB);

      // Determine the best model based on the first pass
      if (r2Exponential > r2Linear && r2Exponential > 0.9) { // Threshold for good exponential fit
        Serial.println("Exponential growth detected on first pass!");

        // Second pass: Fit polynomial and calculate derivative
        const int degree = 3; // Degree of the polynomial
        float coeffs[degree + 1];
        polynomialFit(timeSeconds, temperatureBuffer, bufferSize, degree, coeffs);

        // Calculate the first derivative at the latest time point
        float latestTime = timeSeconds[bufferSize - 1];
        float rateOfGrowth = polynomialDerivative(coeffs, degree, latestTime);

        Serial.print("Rate of exponential growth: ");
        Serial.println(rateOfGrowth);

        // Output polynomial coefficients for debugging
        Serial.println("Polynomial coefficients:");
        for (int i = 0; i <= degree; i++) {
          Serial.print("a");
          Serial.print(i);
          Serial.print(" = ");
          Serial.println(coeffs[i]);
        }

        // Determine the magnitude of exponential growth
        float magnitude = exponentialModel(latestTime, a, expB);
        Serial.print("Magnitude of exponential growth: ");
        Serial.println(magnitude);

        // Additional logic to handle detection based on magnitude and rate
        if (rateOfGrowth > 0.5 && magnitude > 25.0) {
          Serial.println("Confirmed exponential growth detected with significant rate and magnitude.");
        } else {
          Serial.println("Exponential growth detected, but not significant.");
        }

      } else if (r2Linear > 0.9) { // Threshold for good linear fit
        Serial.println("Linear growth detected.");
      } else {
        Serial.println("No significant growth detected or short-term noise.");
      }

      // Debug output
      Serial.print("R^2 Linear: ");
      Serial.println(r2Linear);
      Serial.print("R^2 Exponential: ");
      Serial.println(r2Exponential);
      Serial.print("Temperature Readings: ");
      for (int i = 0; i < bufferSize; i++) {
        Serial.print(temperatureBuffer[i]);
        Serial.print(" ");
      }
      Serial.println();
    }
  }
}

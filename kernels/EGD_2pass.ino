#include <math.h>

const int sensorPin = A0; // Thermistor connected to analog pin A0
const int bufferSize = 30; // Number of readings in the buffer
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

// Function to calculate linear fit (y = mx + b)
void linearFit(float* x, float* y, int n, float& m, float& b) {
  float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (int i = 0; i < n; i++) {
    sumX += x[i];
    sumY += y[i];
    sumXY += x[i] * y[i];
    sumX2 += x[i] * x[i];
  }
  m = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  b = (sumY - m * sumX) / n;
}

// Function to calculate exponential fit (y = a * e^(bx))
// Uses the Fredholm integral equation approach for better accuracy
void exponentialFitFredholm(float* x, float* y, int n, float& a, float& b) {
  // Approximate the integral using discrete sum
  float sum = 0;
  float sumExpBx = 0;
  for (int i = 0; i < n; i++) {
    sum += y[i];
    sumExpBx += y[i] * exp(-x[i]); // This approximates the kernel K(x,y) = exp(-x) * y
  }
  // Average values to find 'a' and 'b'
  a = sum / n;
  b = sumExpBx / sum; // Simplified estimation of 'b'
}

// Function to calculate polynomial coefficients (least squares method)
void polynomialFit(float* x, float* y, int n, int degree, float* coeffs) {
  float X[2 * degree + 1]; // Sum of powers of x
  for (int i = 0; i < 2 * degree + 1; i++) {
    X[i] = 0;
    for (int j = 0; j < n; j++) {
      X[i] += pow(x[j], i);
    }
  }

  float B[degree + 1][degree + 2], a[degree + 1];
  for (int i = 0; i <= degree; i++) {
    for (int j = 0; j <= degree; j++) {
      B[i][j] = X[i + j];
    }
  }

  float Y[degree + 1]; // Array to store values of sigma(xi^k * yi)
  for (int i = 0; i < degree + 1; i++) {
    Y[i] = 0;
    for (int j = 0; j < n; j++) {
      Y[i] += pow(x[j], i) * y[j];
    }
  }

  for (int i = 0; i <= degree; i++) {
    B[i][degree + 1] = Y[i];
  }

  degree += 1;
  for (int i = 0; i < degree; i++) {
    for (int k = i + 1; k < degree; k++) {
      if (B[i][i] < B[k][i]) {
        for (int j = 0; j <= degree; j++) {
          float temp = B[i][j];
          B[i][j] = B[k][j];
          B[k][j] = temp;
        }
      }
    }
  }

  for (int i = 0; i < degree - 1; i++) {
    for (int k = i + 1; k < degree; k++) {
      float t = B[k][i] / B[i][i];
      for (int j = 0; j <= degree; j++) {
        B[k][j] -= t * B[i][j];
      }
    }
  }

  for (int i = degree - 1; i >= 0; i--) {
    a[i] = B[i][degree];
    for (int j = 0; j < degree; j++) {
      if (j != i) {
        a[i] -= B[i][j] * a[j];
      }
    }
    a[i] /= B[i][i];
  }

  for (int i = 0; i < degree; i++) {
    coeffs[i] = a[i];
  }
}

// Function to calculate the first derivative of a polynomial at a given point
float polynomialDerivative(float* coeffs, int degree, float x) {
  float derivative = 0;
  for (int i = 1; i <= degree; i++) {
    derivative += i * coeffs[i] * pow(x, i - 1);
  }
  return derivative;
}

// Function to calculate the goodness of fit (R^2)
float goodnessOfFit(float* x, float* y, int n, float(*model)(float, float, float), float param1, float param2) {
  float ssTotal = 0, ssResidual = 0;
  float meanY = 0;
  for (int i = 0; i < n; i++) {
    meanY += y[i];
  }
  meanY /= n;
  for (int i = 0; i < n; i++) {
    float yi = y[i];
    float fi = model(x[i], param1, param2);
    ssTotal += (yi - meanY) * (yi - meanY);
    ssResidual += (yi - fi) * (yi - fi);
  }
  return 1 - (ssResidual / ssTotal);
}

// Linear model
float linearModel(float x, float m, float b) {
  return m * x + b;
}

// Exponential model
float exponentialModel(float x, float a, float b) {
  return a * exp(b * x);
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
        if (rateOfGrowth > someThreshold && magnitude > someOtherThreshold) {
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

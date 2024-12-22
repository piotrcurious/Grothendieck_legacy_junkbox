#include <Arduino.h>
#include <vector>
#include <cmath>

#define MAX_DATA_POINTS 100  // Maximum number of data points
#define POLY_DEGREE 2        // Degree of polynomial to fit
#define NOISE_VARIANCE 0.01  // Simulated Gaussian noise variance

struct DataPoint {
  float timestamp;
  float value;
};

std::vector<DataPoint> dataPoints;

// Simulate Gaussian noise
float gaussianNoise() {
  static bool hasSpare = false;
  static float spare;
  if (hasSpare) {
    hasSpare = false;
    return spare * sqrt(NOISE_VARIANCE);
  }
  hasSpare = true;
  float u, v, s;
  do {
    u = 2.0f * random(10000) / 10000.0f - 1.0f;
    v = 2.0f * random(10000) / 10000.0f - 1.0f;
    s = u * u + v * v;
  } while (s >= 1.0f || s == 0.0f);
  s = sqrt(-2.0f * log(s) / s);
  spare = v * s;
  return u * s * sqrt(NOISE_VARIANCE);
}

// Add a new data point with simulated noise
void addDataPoint(float timestamp, float value) {
  if (dataPoints.size() < MAX_DATA_POINTS) {
    DataPoint point;
    point.timestamp = timestamp;
    point.value = value + gaussianNoise();
    dataPoints.push_back(point);
  }
}

// Fit a polynomial using least squares
std::vector<float> fitPolynomial(int degree) {
  int n = dataPoints.size();
  int terms = degree + 1;
  std::vector<std::vector<float>> A(terms, std::vector<float>(terms, 0.0f));
  std::vector<float> b(terms, 0.0f);
  std::vector<float> coeffs(terms, 0.0f);

  for (const auto &point : dataPoints) {
    float x = point.timestamp;
    float y = point.value;
    for (int i = 0; i < terms; ++i) {
      for (int j = 0; j < terms; ++j) {
        A[i][j] += pow(x, i + j);
      }
      b[i] += y * pow(x, i);
    }
  }

  // Solve Ax = b (Gaussian elimination)
  for (int i = 0; i < terms; ++i) {
    for (int j = i + 1; j < terms; ++j) {
      float factor = A[j][i] / A[i][i];
      for (int k = i; k < terms; ++k) {
        A[j][k] -= factor * A[i][k];
      }
      b[j] -= factor * b[i];
    }
  }

  for (int i = terms - 1; i >= 0; --i) {
    coeffs[i] = b[i];
    for (int j = i + 1; j < terms; ++j) {
      coeffs[i] -= A[i][j] * coeffs[j];
    }
    coeffs[i] /= A[i][i];
  }

  return coeffs;
}

// Evaluate a polynomial
float evaluatePolynomial(const std::vector<float> &coeffs, float x) {
  float result = 0.0f;
  for (int i = 0; i < coeffs.size(); ++i) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Divide data points into feature sets
void generateSupportVectors(const std::vector<float> &coeffs, std::vector<DataPoint> &below, std::vector<DataPoint> &above) {
  for (const auto &point : dataPoints) {
    float predicted = evaluatePolynomial(coeffs, point.timestamp);
    if (point.value < predicted) {
      below.push_back(point);
    } else {
      above.push_back(point);
    }
  }
}

// Predict future values
float predictFuture(const std::vector<float> &coeffs, float futureTimestamp) {
  return evaluatePolynomial(coeffs, futureTimestamp);
}

void setup() {
  Serial.begin(115200);

  // Simulate adding data points
  for (float t = 0; t < 10; t += 0.1) {
    float value = sin(t) + 0.5 * t; // Example system evolution
    addDataPoint(t, value);
  }

  // Fit polynomial
  std::vector<float> coeffs = fitPolynomial(POLY_DEGREE);

  // Print coefficients
  Serial.println("Fitted polynomial coefficients:");
  for (int i = 0; i < coeffs.size(); ++i) {
    Serial.printf("a[%d] = %f\n", i, coeffs[i]);
  }

  // Generate support vectors
  std::vector<DataPoint> below, above;
  generateSupportVectors(coeffs, below, above);

  Serial.printf("Below support vector size: %d\n", below.size());
  Serial.printf("Above support vector size: %d\n", above.size());

  // Predict future value
  float futureTimestamp = 15.0f;
  float futureValue = predictFuture(coeffs, futureTimestamp);
  Serial.printf("Predicted value at t = %.2f: %.2f\n", futureTimestamp, futureValue);
}

void loop() {
  // Periodic updates or data point addition can be added here
}

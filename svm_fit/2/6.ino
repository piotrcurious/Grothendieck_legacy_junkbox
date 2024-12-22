#include <Arduino.h>
#include <vector>
#include <cmath>

#define MAX_DATA_POINTS 100  // Maximum number of data points
#define POLY_DEGREE 2        // Degree of polynomial to fit
#define NOISE_VARIANCE 0.01  // Simulated Gaussian noise variance
#define MAX_ITERATIONS 10    // Maximum number of iterations for refinement
#define ERROR_THRESHOLD 1.5  // Multiplier for standard deviation threshold

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
std::vector<float> fitPolynomial(const std::vector<DataPoint> &subset, int degree) {
  int n = subset.size();
  int terms = degree + 1;
  std::vector<std::vector<float>> A(terms, std::vector<float>(terms, 0.0f));
  std::vector<float> b(terms, 0.0f);
  std::vector<float> coeffs(terms, 0.0f);

  for (const auto &point : subset) {
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

// Compute residuals and standard deviation
float computeStandardDeviation(const std::vector<DataPoint> &subset, const std::vector<float> &coeffs) {
  float sum = 0.0f, sumSq = 0.0f;
  for (const auto &point : subset) {
    float predicted = evaluatePolynomial(coeffs, point.timestamp);
    float residual = point.value - predicted;
    sum += residual;
    sumSq += residual * residual;
  }
  int n = subset.size();
  return sqrt((sumSq - (sum * sum) / n) / (n - 1));
}

// Refine support vectors iteratively
void refineSupportVectors(std::vector<DataPoint> &below, std::vector<DataPoint> &above) {
  std::vector<DataPoint> refinedDataset = dataPoints;
  std::vector<float> coeffs;

  for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
    coeffs = fitPolynomial(refinedDataset, POLY_DEGREE);
    float stdDev = computeStandardDeviation(refinedDataset, coeffs);

    std::vector<DataPoint> newBelow, newAbove;
    for (const auto &point : refinedDataset) {
      float predicted = evaluatePolynomial(coeffs, point.timestamp);
      float residual = point.value - predicted;

      // Classify points based on residuals and threshold
      if (residual < -ERROR_THRESHOLD * stdDev) {
        newBelow.push_back(point);
      } else if (residual > ERROR_THRESHOLD * stdDev) {
        newAbove.push_back(point);
      }
    }

    // Update refined dataset
    refinedDataset = newBelow;
    refinedDataset.insert(refinedDataset.end(), newAbove.begin(), newAbove.end());

    // Check convergence
    if (newBelow.size() == below.size() && newAbove.size() == above.size()) {
      break;
    }
    below = newBelow;
    above = newAbove;
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

  // Initial polynomial fit
  std::vector<float> coeffs = fitPolynomial(dataPoints, POLY_DEGREE);

  // Refine support vectors
  std::vector<DataPoint> below, above;
  refineSupportVectors(below, above);

  // Output results
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

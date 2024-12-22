#include <Arduino.h>
#include <Eigen.h> // Include Eigen for matrix operations
using namespace Eigen;

// Constants
const int MAX_DATA_POINTS = 100;       // Sliding window size
const int MAX_CANDIDATES = 5;          // Number of candidate polynomials to test
const int POLY_ORDER = 3;              // Polynomial order
const float GAUSSIAN_SIGMA = 1.0;      // Gaussian kernel parameter

// Data buffers
float timestamps[MAX_DATA_POINTS];
float values[MAX_DATA_POINTS];
int dataCount = 0;

// Candidate polynomial coefficients and scores
MatrixXf candidateCoeffs(POLY_ORDER + 1, MAX_CANDIDATES);
VectorXf candidateScores(MAX_CANDIDATES);

// Function prototypes
void addData(float timestamp, float value);
void fitCandidateFunctions();
VectorXf fitWeightedPolynomial(const float* timestamps, const float* values, const VectorXf& weights, int count, int order);
void scoreCandidates();
void findSupportVectors(const VectorXf& coeffs);
float gaussianErrorWeight(float residual);
float calculateResidual(const VectorXf& coeffs, float timestamp, float value);
float predictEvolution(float futureTimestamp);

// Add data point to the sliding window
void addData(float timestamp, float value) {
  // Add new data point and maintain sliding window
  if (dataCount < MAX_DATA_POINTS) {
    timestamps[dataCount] = timestamp;
    values[dataCount] = value;
    dataCount++;
  } else {
    // Shift data to make space for new point
    memmove(timestamps, timestamps + 1, (MAX_DATA_POINTS - 1) * sizeof(float));
    memmove(values, values + 1, (MAX_DATA_POINTS - 1) * sizeof(float));
    timestamps[MAX_DATA_POINTS - 1] = timestamp;
    values[MAX_DATA_POINTS - 1] = value;
  }

  // Fit candidate functions and refine support vectors dynamically
  fitCandidateFunctions();
}

// Fit multiple candidate polynomials dynamically
void fitCandidateFunctions() {
  if (dataCount < POLY_ORDER + 1) {
    Serial.println("Not enough data to fit polynomials.");
    return;
  }

  Serial.println("Fitting candidate polynomials...");
  VectorXf weights = VectorXf::Ones(dataCount);

  for (int i = 0; i < MAX_CANDIDATES; i++) {
    // Simulate noise in weights for different candidate fits
    for (int j = 0; j < dataCount; j++) {
      weights(j) += ((float)random(-100, 100)) / 10000.0;
    }

    // Fit polynomial with weighted data
    candidateCoeffs.col(i) = fitWeightedPolynomial(timestamps, values, weights, dataCount, POLY_ORDER);
  }

  // Score candidates based on their fit quality
  scoreCandidates();
}

// Fit a polynomial using weighted least squares
VectorXf fitWeightedPolynomial(const float* x, const float* y, const VectorXf& weights, int count, int order) {
  MatrixXf vandermonde(count, order + 1);
  VectorXf values(count);
  VectorXf W = weights.asDiagonal(); // Weight matrix

  for (int i = 0; i < count; i++) {
    values(i) = y[i];
    for (int j = 0; j <= order; j++) {
      vandermonde(i, j) = pow(x[i], j);
    }
  }

  // Solve weighted least squares
  return (vandermonde.transpose() * W * vandermonde)
             .ldlt()
             .solve(vandermonde.transpose() * W * values);
}

// Score candidate polynomials dynamically
void scoreCandidates() {
  Serial.println("Scoring candidates...");
  for (int i = 0; i < MAX_CANDIDATES; i++) {
    float score = 0.0;

    for (int j = 0; j < dataCount; j++) {
      float residual = calculateResidual(candidateCoeffs.col(i), timestamps[j], values[j]);
      score += gaussianErrorWeight(residual);
    }

    candidateScores(i) = score;
    Serial.printf("Candidate %d Score: %.4f\n", i + 1, score);
  }
}

// Calculate residual for a data point
float calculateResidual(const VectorXf& coeffs, float timestamp, float value) {
  float fittedValue = 0.0;
  for (int i = 0; i <= POLY_ORDER; i++) {
    fittedValue += coeffs(i) * pow(timestamp, i);
  }
  return value - fittedValue;
}

// Gaussian error weight
float gaussianErrorWeight(float residual) {
  return exp(-pow(residual, 2) / (2 * pow(GAUSSIAN_SIGMA, 2)));
}

// Predict future value using the best-fit candidate
float predictEvolution(float futureTimestamp) {
  // Select the best candidate based on scores
  int bestIndex = 0;
  float bestScore = candidateScores(0);
  for (int i = 1; i < MAX_CANDIDATES; i++) {
    if (candidateScores(i) > bestScore) {
      bestIndex = i;
      bestScore = candidateScores(i);
    }
  }

  // Use the best candidate for prediction
  const VectorXf& bestFit = candidateCoeffs.col(bestIndex);

  float prediction = 0.0;
  for (int i = 0; i <= POLY_ORDER; i++) {
    prediction += bestFit(i) * pow(futureTimestamp, i);
  }

  Serial.printf("Predicted value at timestamp %.2f: %.4f\n", futureTimestamp, prediction);
  return prediction;
}

// Find support vectors for the best candidate dynamically
void findSupportVectors(const VectorXf& coeffs) {
  Serial.println("Support Vectors:");
  for (int i = 0; i < dataCount; i++) {
    float fittedValue = 0.0;
    for (int j = 0; j <= POLY_ORDER; j++) {
      fittedValue += coeffs(j) * pow(timestamps[i], j);
    }

    float residual = values[i] - fittedValue;
    float weight = gaussianErrorWeight(residual);

    if (residual > 0) {
      Serial.printf("Above Fit: Timestamp: %.2f, Value: %.2f, Residual: %.4f, Weight: %.4f\n",
                    timestamps[i], values[i], residual, weight);
    } else {
      Serial.printf("Below Fit: Timestamp: %.2f, Value: %.2f, Residual: %.4f, Weight: %.4f\n",
                    timestamps[i], values[i], residual, weight);
    }
  }
}

void setup() {
  Serial.begin(115200);

  // Initial data for testing
  addData(1.0, 10.0);
  addData(2.0, 20.5);
  addData(3.0, 29.7);
  addData(4.0, 39.2);

  // Predict future values dynamically
  predictEvolution(5.0);
}

void loop() {
  // Simulate periodic data updates
  static float timestamp = 5.0;
  static float value = 50.0;

  addData(timestamp, value);
  predictEvolution(timestamp + 1.0);

  timestamp += 1.0;
  value += random(-5, 5) / 10.0; // Simulate noisy data
  delay(1000); // Update every second
}

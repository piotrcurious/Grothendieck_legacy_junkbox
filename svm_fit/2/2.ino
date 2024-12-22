#include <Arduino.h>
#include <Eigen.h> // Include Eigen for matrix operations
using namespace Eigen;

// Define constants
const int MAX_DATA_POINTS = 100;       // Maximum number of data points to store
const int MAX_CANDIDATES = 5;          // Number of candidate polynomials to test
const int POLY_ORDER = 3;              // Polynomial order
const float GAUSSIAN_SIGMA = 1.0;      // Gaussian kernel parameter
const float QUANTIZATION_ERROR = 0.01; // Assumed quantization error
const float ADC_ERROR = 0.01;          // ADC quantization error
const float SAMPLING_NOISE = 0.02;     // Sampling noise error

// Data buffers
float timestamps[MAX_DATA_POINTS];
float values[MAX_DATA_POINTS];
int dataCount = 0;

// Candidate polynomial coefficients
MatrixXf candidateCoeffs(POLY_ORDER + 1, MAX_CANDIDATES);

// Function prototypes
void addData(float timestamp, float value);
void fitCandidateFunctions();
VectorXf fitPolynomial(const float* timestamps, const float* values, int count, int order);
void findSupportVectors(const VectorXf& coeffs);
float gaussianErrorWeight(float residual);
void predictEvolution(float futureTimestamp, const VectorXf& coeffs);

// Incoming data callback
void addData(float timestamp, float value) {
  if (dataCount < MAX_DATA_POINTS) {
    timestamps[dataCount] = timestamp;
    values[dataCount] = value;
    dataCount++;
  } else {
    Serial.println("Data buffer full! Cannot add more data.");
  }
}

// Fit multiple candidate polynomials
void fitCandidateFunctions() {
  if (dataCount < POLY_ORDER + 1) {
    Serial.println("Not enough data to fit polynomials.");
    return;
  }

  // Fit multiple candidates by slightly varying the data weights
  for (int i = 0; i < MAX_CANDIDATES; i++) {
    float noiseFactor = ((float)random(-100, 100)) / 10000.0; // Simulate noise
    VectorXf coeffs = fitPolynomial(timestamps, values, dataCount, POLY_ORDER);
    coeffs += VectorXf::Constant(POLY_ORDER + 1, noiseFactor); // Add noise
    candidateCoeffs.col(i) = coeffs;
  }

  // Output candidates
  Serial.println("Candidate Polynomials:");
  for (int i = 0; i < MAX_CANDIDATES; i++) {
    Serial.printf("Candidate %d Coefficients: ", i + 1);
    for (int j = 0; j <= POLY_ORDER; j++) {
      Serial.printf("%.4f ", candidateCoeffs(j, i));
    }
    Serial.println();
  }
}

// Fit a polynomial using least squares
VectorXf fitPolynomial(const float* x, const float* y, int count, int order) {
  MatrixXf vandermonde(count, order + 1);
  VectorXf values(count);

  for (int i = 0; i < count; i++) {
    values(i) = y[i];
    for (int j = 0; j <= order; j++) {
      vandermonde(i, j) = pow(x[i], j);
    }
  }

  // Solve for polynomial coefficients
  return vandermonde.colPivHouseholderQr().solve(values);
}

// Find support vectors
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

// Gaussian error weight
float gaussianErrorWeight(float residual) {
  return exp(-pow(residual, 2) / (2 * pow(GAUSSIAN_SIGMA, 2)));
}

// Predict evolution
void predictEvolution(float futureTimestamp, const VectorXf& coeffs) {
  float prediction = 0.0;
  for (int i = 0; i <= POLY_ORDER; i++) {
    prediction += coeffs(i) * pow(futureTimestamp, i);
  }

  Serial.printf("Predicted value at timestamp %.2f: %.4f\n", futureTimestamp, prediction);
}

void setup() {
  Serial.begin(115200);

  // Example data
  addData(1.0, 10.0);
  addData(2.0, 20.5);
  addData(3.0, 29.7);
  addData(4.0, 39.2);

  // Fit candidate polynomials
  fitCandidateFunctions();

  // Use the first candidate for support vector calculation
  VectorXf bestFit = candidateCoeffs.col(0);
  findSupportVectors(bestFit);

  // Predict future value
  predictEvolution(5.0, bestFit);
}

void loop() {
  // Add more data and refine predictions dynamically
  delay(1000); // Simulate periodic updates
}

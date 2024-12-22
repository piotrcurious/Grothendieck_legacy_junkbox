#include <Arduino.h>
#include <Eigen.h> // Include Eigen for matrix operations
using namespace Eigen;

// Define constants
const int MAX_DATA_POINTS = 100;       // Maximum number of data points to store
const int POLY_ORDER = 3;              // Polynomial order
const float GAUSSIAN_SIGMA = 1.0;      // Gaussian kernel parameter
const float QUANTIZATION_ERROR = 0.01; // Assumed quantization error
const float ADC_ERROR = 0.01;          // ADC quantization error
const float SAMPLING_NOISE = 0.02;     // Sampling noise error

// Data buffers
float timestamps[MAX_DATA_POINTS];
float values[MAX_DATA_POINTS];
int dataCount = 0;

// Polynomial coefficients
VectorXf polyCoeffs(POLY_ORDER + 1);

// Function prototypes
void fitPolynomial();
float gaussianKernel(float x1, float x2);
void predictEvolution(float futureTimestamp);

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

void fitPolynomial() {
  if (dataCount < POLY_ORDER + 1) {
    Serial.println("Not enough data to fit polynomial.");
    return;
  }

  // Build the Vandermonde matrix
  MatrixXf vandermonde(dataCount, POLY_ORDER + 1);
  VectorXf y(dataCount);

  for (int i = 0; i < dataCount; i++) {
    y(i) = values[i];
    for (int j = 0; j <= POLY_ORDER; j++) {
      vandermonde(i, j) = pow(timestamps[i], j);
    }
  }

  // Solve for polynomial coefficients using least squares
  polyCoeffs = vandermonde.colPivHouseholderQr().solve(y);
  Serial.println("Fitted Polynomial Coefficients:");
  for (int i = 0; i <= POLY_ORDER; i++) {
    Serial.printf("Coeff[%d]: %.4f\n", i, polyCoeffs(i));
  }
}

float gaussianKernel(float x1, float x2) {
  float diff = x1 - x2;
  return exp(-pow(diff, 2) / (2 * pow(GAUSSIAN_SIGMA, 2)));
}

void predictEvolution(float futureTimestamp) {
  if (dataCount == 0) {
    Serial.println("No data to make a prediction.");
    return;
  }

  // Compute Gaussian kernel weights
  VectorXf weights(dataCount);
  for (int i = 0; i < dataCount; i++) {
    weights(i) = gaussianKernel(futureTimestamp, timestamps[i]);
  }

  // Normalize weights
  float weightSum = weights.sum();
  weights /= weightSum;

  // Weighted prediction using polynomial model
  float prediction = 0.0;
  for (int i = 0; i <= POLY_ORDER; i++) {
    prediction += polyCoeffs(i) * pow(futureTimestamp, i);
  }

  // Apply kernel weights to refine prediction
  float refinedPrediction = 0.0;
  for (int i = 0; i < dataCount; i++) {
    refinedPrediction += weights(i) * prediction;
  }

  Serial.printf("Predicted value at timestamp %.2f: %.4f\n", futureTimestamp, refinedPrediction);
}

void setup() {
  Serial.begin(115200);

  // Example data
  addData(1.0, 10.0);
  addData(2.0, 20.5);
  addData(3.0, 29.7);
  addData(4.0, 39.2);

  // Fit polynomial
  fitPolynomial();

  // Predict future value
  predictEvolution(5.0); // Predict at timestamp = 5.0
}

void loop() {
  // Add more data and refine prediction dynamically
  delay(1000); // Simulate periodic updates
}

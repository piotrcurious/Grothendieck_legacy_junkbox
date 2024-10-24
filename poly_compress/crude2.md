To improve the code so it supports variable-length windows and variable-degree polynomials, we need to generalize the fitting process and allow more flexibility in both the number of data points (window size) and the degree of the polynomial. This will create a more sophisticated compression system that can adapt based on memory and accuracy requirements.

Step 1: Generalizing Polynomial Fitting for Variable Degree

The least squares polynomial fitting function needs to handle polynomials of arbitrary degrees. The matrix equations to solve for the polynomial coefficients can be extended for higher-degree polynomials.

Step 2: Generalizing the Arduino Code for Variable Window and Polynomial Degree

We will modify the polynomialFit function to support variable polynomial degrees and allow the buffer to dynamically adapt based on the window size and polynomial degree.

Here’s the improved code:

1. Polynomial Fitting for Variable Degrees

This function computes the least squares polynomial fit for any degree and window size.

#include <Arduino.h>

// Function to calculate least squares polynomial fit of variable degree
void polynomialFit(int* x, int* y, int n, int degree, float* coeffs) {
  // Create matrices to hold the sums
  float X[degree+1][degree+1];
  float Y[degree+1];

  // Initialize the matrices to zero
  for (int i = 0; i <= degree; i++) {
    Y[i] = 0;
    for (int j = 0; j <= degree; j++) {
      X[i][j] = 0;
    }
  }

  // Compute the sums for the normal equation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= degree; j++) {
      Y[j] += y[i] * pow(x[i], j);
      for (int k = 0; k <= degree; k++) {
        X[j][k] += pow(x[i], j + k);
      }
    }
  }

  // Solve the system using Gaussian elimination
  for (int i = 0; i <= degree; i++) {
    for (int k = i + 1; k <= degree; k++) {
      if (X[i][i] < X[k][i]) {
        // Swap rows
        for (int j = 0; j <= degree; j++) {
          float temp = X[i][j];
          X[i][j] = X[k][j];
          X[k][j] = temp;
        }
        float tempY = Y[i];
        Y[i] = Y[k];
        Y[k] = tempY;
      }
    }

    // Perform Gaussian elimination
    for (int k = i + 1; k <= degree; k++) {
      float t = X[k][i] / X[i][i];
      for (int j = 0; j <= degree; j++) {
        X[k][j] -= t * X[i][j];
      }
      Y[k] -= t * Y[i];
    }
  }

  // Back substitution to find the coefficients
  for (int i = degree; i >= 0; i--) {
    coeffs[i] = Y[i];
    for (int j = i + 1; j <= degree; j++) {
      coeffs[i] -= X[i][j] * coeffs[j];
    }
    coeffs[i] /= X[i][i];
  }
}

2. Generalizing the Rolling Buffer for Variable Degree and Window Size

Now, we modify the RollingBuffer structure to store different window sizes and degrees for polynomial compression.

// Rolling buffer structure to hold polynomial coefficients
struct RollingBuffer {
  float* buffer;       // Buffer to store polynomial coefficients
  int bufferSize;      // Number of windows to store
  int* degrees;        // Array to store the polynomial degree for each window
  int* windowSizes;    // Array to store the number of data points for each window
  int currentIndex;    // Current buffer index
};

// Initialize the rolling buffer for variable window sizes and polynomial degrees
void initBuffer(RollingBuffer* buf, int bufferSize) {
  buf->bufferSize = bufferSize;
  buf->buffer = new float[bufferSize * 10];  // Pre-allocate space for up to 10 coefficients per window
  buf->degrees = new int[bufferSize];
  buf->windowSizes = new int[bufferSize];
  buf->currentIndex = 0;
}

// Add polynomial coefficients to the rolling buffer with associated window size and degree
void addToBuffer(RollingBuffer* buf, float* coeffs, int degree, int windowSize) {
  // Add the coefficients to the buffer
  for (int i = 0; i <= degree; i++) {
    buf->buffer[buf->currentIndex * 10 + i] = coeffs[i];  // Store coefficients in chunks of 10
  }

  // Store the window size and degree for this window
  buf->degrees[buf->currentIndex] = degree;
  buf->windowSizes[buf->currentIndex] = windowSize;

  // Update the buffer index
  buf->currentIndex = (buf->currentIndex + 1) % buf->bufferSize;
}

3. Arduino Loop to Gather Data and Compress Using Variable Window and Degree

In this loop, we can dynamically choose the window size and polynomial degree for each data batch. This allows for more sophisticated data compression based on memory constraints or required accuracy.

#define BUFFER_SIZE 10   // Number of windows to store

RollingBuffer buffer;
int x[10];              // Time indices for the data points
int y[10];              // Data points (e.g., sensor readings)
float coeffs[10];       // Polynomial coefficients

void setup() {
  Serial.begin(9600);
  initBuffer(&buffer, BUFFER_SIZE);
}

void loop() {
  int windowSize = random(5, 10);  // Dynamically choose window size (5 to 10 points)
  int degree = random(1, 4);       // Dynamically choose polynomial degree (1 to 3)

  // Collect data in batches based on the chosen window size
  for (int i = 0; i < windowSize; i++) {
    x[i] = i;  // Simulating time or index
    y[i] = analogRead(A0);  // Example sensor data
    delay(100);  // Simulate sampling delay
  }

  // Apply polynomial fitting to compress data using the chosen degree
  polynomialFit(x, y, windowSize, degree, coeffs);

  // Add the compressed polynomial coefficients to the rolling buffer
  addToBuffer(&buffer, coeffs, degree, windowSize);

  // Debug output to show the coefficients stored
  Serial.print("Stored Coefficients (Degree ");
  Serial.print(degree);
  Serial.print("): ");
  for (int i = 0; i <= degree; i++) {
    Serial.print(coeffs[i]);
    Serial.print(" ");
  }
  Serial.println();

  delay(1000);  // Pause before next data collection
}

Step 3: Reconstructing Data from the Polynomial with Variable Degree

To reconstruct data for any window in the buffer, you need to evaluate the polynomial at the desired time index (x) based on the stored coefficients and degree:

// Reconstruct the data point for a given x using the polynomial coefficients and degree
float reconstructData(float* coeffs, int degree, int x) {
  float result = 0;
  for (int i = 0; i <= degree; i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

Step 4: Dynamic Compression Optimization with JuMP

Using JuMP, you can further optimize how you choose the window size and degree. You can create a model to balance accuracy and memory usage by setting constraints on these factors. Here’s how you could define it in JuMP:

using JuMP, GLPK

# Data points (as an example)
data_points = [10, 12, 15, 14, 13, 11, 12, 16, 17, 18]

# Set up the JuMP model to optimize window size and polynomial degree
model = Model(GLPK.Optimizer)

# Decision variable: the degree of the polynomial (could be 1, 2, or 3)
@variable(model, degree >= 1, Int)

# Decision variable: window size (number of points to fit the polynomial)
@variable(model, window_size >= 5, Int)

# Objective: minimize memory usage (related to window_size and degree)
@objective(model, Min, window_size * (degree + 1))

# Add a constraint based on data fitting accuracy (e.g., mean squared error)
# Use real data fitting results to derive this
@constraint(model, some_error_metric <= tolerance)  # Define a real fitting error constraint

# Solve the model
optimize!(model)

# Get the optimal window size and degree
optimal_window_size = value(window_size)
optimal_degree = value(degree)

println("Optimal window size: $optimal_window_size, Optimal polynomial degree: $optimal_degree")

Conclusion

This improved system allows for variable-length windows and variable-degree polynomials in the rolling buffer, making it more flexible and adaptive for data compression. By dynamically adjusting the polynomial degree and window size based on accuracy and memory constraints, you can achieve efficient data compression on Arduino. JuMP can further refine this process by optimizing these parameters based on defined objectives and constraints.


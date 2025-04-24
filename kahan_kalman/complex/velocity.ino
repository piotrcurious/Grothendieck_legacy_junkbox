#include <Arduino.h>
#include <cmath> // For sqrt(), fabs()
#include <utility> // For std::swap

// Define the analog input pin (replace with the actual pin on your ESP32 board)
const int ANALOG_PIN = 34; // Example: GPIO 34

// Define tuning parameters for process noise (adjust these based on expected signal variability)
// These influence the filter's responsiveness to changes. Lower values mean more smoothing (lower pass).
const double PROCESS_NOISE_POSITION_RATE = 0.1; // Tune this
const double PROCESS_NOISE_VELOCITY_RATE = 0.5; // Tune this

// Define the standard deviation of the measurement noise (adjust based on sensor noise)
// Higher values mean the filter trusts the measurements less, leading to more smoothing (lower pass).
const double MEASUREMENT_NOISE_STDDEV = 2.0; // Example: standard deviation of analog readings when input is stable

// Define the minimum and maximum random interval between readings (in milliseconds)
const unsigned long MIN_READ_INTERVAL_MS = 100;
const unsigned long MAX_READ_INTERVAL_MS = 500;

// --- Kahan Summation for double precision ---
double kahanSum(const double* input, size_t size) {
    double sum = 0.0;
    double c = 0.0; // Compensation
    for (size_t i = 0; i < size; ++i) {
        double y = input[i] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

// --- Basic Matrix Class with Dynamic Allocation ---
// (Same as in the previous example, included here for completeness)
class Matrix {
public:
    double* data;
    int rows;
    int cols;

    // Constructor
    Matrix(int r, int c) : rows(r), cols(c) {
        if (rows <= 0 || cols <= 0) {
            rows = 0;
            cols = 0;
            data = nullptr;
            Serial.println("Error: Matrix dimensions must be positive.");
            return;
        }
        data = new double[rows * cols];
        if (!data) {
            Serial.println("Error: Failed to allocate memory for matrix.");
            rows = 0;
            cols = 0;
        } else {
            // Initialize with zeros
            for (int i = 0; i < rows * cols; ++i) {
                data[i] = 0.0;
            }
        }
    }

    // Destructor
    ~Matrix() {
        if (data) {
            delete[] data;
            data = nullptr;
        }
    }

    // Copy Constructor
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
        data = new double[rows * cols];
        if (!data) {
             Serial.println("Error: Failed to allocate memory for matrix copy.");
             rows = 0;
             cols = 0;
        } else {
            memcpy(data, other.data, rows * cols * sizeof(double));
        }
    }

    // Assignment Operator
    Matrix& operator=(const Matrix& other) {
        if (this == &other) {
            return *this;
        }
        if (data) {
            delete[] data;
        }
        rows = other.rows;
        cols = other.cols;
        data = new double[rows * cols];
         if (!data) {
             Serial.println("Error: Failed to allocate memory for matrix assignment.");
             rows = 0;
             cols = 0;
        } else {
            memcpy(data, other.data, rows * cols * sizeof(double));
        }
        return *this;
    }

    // Element Access
    double& operator()(int r, int c) {
        if (r < 0 || r >= rows || c < 0 || c >= cols) {
            Serial.printf("Error: Matrix index out of bounds (%d, %d) for matrix size (%d, %d)\n", r, c, rows, cols);
            static double dummy = NAN; // Use NAN to indicate an invalid access
            return dummy;
        }
        return data[r * cols + c];
    }

    const double& operator()(int r, int c) const {
         if (r < 0 || r >= rows || c < 0 || c >= cols) {
            Serial.printf("Error: Const Matrix index out of bounds (%d, %d) for matrix size (%d, %d)\n", r, c, rows, cols);
            static const double dummy = NAN; // Use NAN to indicate an invalid access
            return dummy;
        }
        return data[r * cols + c];
    }

    // --- Matrix Operations using Kahan Summation ---

    // Matrix Addition (this + other)
    Matrix add(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            Serial.println("Error: Matrix dimensions do not match for addition.");
            return Matrix(0, 0); // Return an empty matrix
        }
        Matrix result(rows, cols);
        if (!result.data) return Matrix(0,0);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double terms[] = {(*this)(i, j), other(i, j)};
                result(i, j) = kahanSum(terms, 2);
            }
        }
        return result;
    }

    // Matrix Subtraction (this - other)
     Matrix subtract(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            Serial.println("Error: Matrix dimensions do not match for subtraction.");
            return Matrix(0, 0); // Return an empty matrix
        }
        Matrix result(rows, cols);
        if (!result.data) return Matrix(0,0);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double terms[] = {(*this)(i, j), -other(i, j)}; // Subtract by adding negative
                result(i, j) = kahanSum(terms, 2);
            }
        }
        return result;
    }


    // Matrix Multiplication (this * other)
    Matrix multiply(const Matrix& other) const {
        if (cols != other.rows) {
            Serial.println("Error: Matrix dimensions do not match for multiplication.");
            return Matrix(0, 0); // Return an empty matrix
        }
        Matrix result(rows, other.cols);
         if (!result.data) return Matrix(0,0);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                // Accumulate the sum of products using Kahan summation
                double sum = 0.0;
                double c = 0.0; // Compensation for Kahan summation
                for (int k = 0; k < cols; ++k) {
                    double term = (*this)(i, k) * other(k, j);
                    double y = term - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    // Scalar Multiplication (this * scalar)
    Matrix multiply_scalar(double scalar) const {
        Matrix result(rows, cols);
         if (!result.data) return Matrix(0,0);
        for (int i = 0; i < rows * cols; ++i) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    // Matrix Transpose
    Matrix transpose() const {
        Matrix result(cols, rows);
         if (!result.data) return Matrix(0,0);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

     // --- Utility Function to Print Matrix ---
    void print() const {
        if (!data) {
            Serial.println("Empty Matrix");
            return;
        }
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                Serial.print((*this)(i, j), 6); // Print with 6 decimal places
                Serial.print("\t");
            }
            Serial.println();
        }
    }
};


// --- Linear System Solver using Gaussian Elimination with Partial Pivoting ---
// Solves Ax = b for x
// NOTE: For symmetric positive-definite matrices (like S_k in a Kalman filter),
// a solver based on Cholesky or LDLT decomposition is generally more numerically stable.
// This Gaussian elimination solver is provided as a functional example.
Matrix solveLinear(const Matrix& A, const Matrix& b) {
    if (A.rows != A.cols || A.rows != b.rows || b.cols != 1) {
        Serial.println("Error: Invalid dimensions for solveLinear.");
        return Matrix(0, 0);
    }

    int n = A.rows;
    // Create an augmented matrix [A | b]
    Matrix augmentedMatrix(n, n + 1);
    if (!augmentedMatrix.data) return Matrix(0,0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmentedMatrix(i, j) = A(i, j);
        }
        augmentedMatrix(i, n) = b(i, 0);
    }

    // Gaussian Elimination with Partial Pivoting
    for (int i = 0; i < n; ++i) {
        // Find pivot row
        int pivotRow = i;
        for (int k = i + 1; k < n; ++k) {
            if (fabs(augmentedMatrix(k, i)) > fabs(augmentedMatrix(pivotRow, i))) {
                pivotRow = k;
            }
        }

        // Swap pivot row
        if (pivotRow != i) {
            for (int j = i; j <= n; ++j) {
                std::swap(augmentedMatrix(i, j), augmentedMatrix(pivotRow, j));
            }
        }

        // Check for singular matrix
        if (fabs(augmentedMatrix(i, i)) < 1e-12) { // Use a small tolerance for zero check
            Serial.println("Error: Singular matrix detected in solveLinear.");
            return Matrix(0, 0);
        }

        // Eliminate
        for (int k = i + 1; k < n; ++k) {
            double factor = augmentedMatrix(k, i) / augmentedMatrix(i, i);
            // Apply elimination to the rest of the row using Kahan summation
            for (int j = i; j <= n; ++j) {
                 double term = factor * augmentedMatrix(i, j);
                 double original_val = augmentedMatrix(k, j);
                 double terms[] = {original_val, -term};
                 augmentedMatrix(k, j) = kahanSum(terms, 2);
            }
        }
    }

    // Back Substitution
    Matrix x(n, 1);
     if (!x.data) return Matrix(0,0);

    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        double c = 0.0; // Compensation for Kahan summation
        double rhs = augmentedMatrix(i, n);
        for (int j = i + 1; j < n; ++j) {
            double term = augmentedMatrix(i, j) * x(j, 0);
             double y = term - c;
             double t = sum + y;
             c = (t - sum) - y;
             sum = t;
        }
        x(i, 0) = (rhs - sum) / augmentedMatrix(i, i);
    }

    return x;
}


// --- Kalman Filter for Filtering (Constant Velocity Model) ---
// State vector: [position, velocity]' (2D state)
// This filter is designed to estimate the underlying state (position and velocity)
// and naturally provides a smoothed position output (low-pass).
// High-pass related information can be extracted from the estimated velocity or residuals.
const int n_states = 2; // Number of states
const int n_measurements = 1; // Number of measurements (measuring position)

// State Transition Matrix (F) - 2x2, will be updated based on dt
Matrix F(n_states, n_states);
// Control Matrix (B) - 2x1, assuming no control input
Matrix B(n_states, 1);
// Observation Matrix (H) - 1x2, measuring position
Matrix H(n_measurements, n_states);
// Process Noise Covariance (Q) - 2x2, will be updated based on dt and tuning
Matrix Q(n_states, n_states);
// Measurement Noise Covariance (R) - 1x1, based on sensor noise characteristics
Matrix R(n_measurements, n_measurements);

// State Estimate (x_hat) - 2x1, [filtered_position, filtered_velocity]'
Matrix x_hat(n_states, 1);
// Error Covariance (P) - 2x2
Matrix P(n_states, n_states);

// Identity Matrix (for update step) - 2x2
Matrix Identity(n_states, n_states);

// Variables for timing
unsigned long last_read_time = 0;

// --- Setup ---
void setup() {
    Serial.begin(115200);
    while (!Serial); // Wait for serial port to connect

    Serial.println("Kalman Filter for Filtering (Constant Velocity Model) on ESP32");

    // Initialize Analog Input
    analogReadResolution(12); // Set ADC resolution (ESP32 typically 12-bit, range 0-4095)
    analogSetAttenuation(ADC_0db); // Set attenuation if needed

    // Initialize Constant Kalman Filter Matrices
    // B = [0] (No control input)
    //     [0]
    B(0, 0) = 0.0;
    B(1, 0) = 0.0;

    // H = [1 0] (Measuring position)
    H(0, 0) = 1.0;
    H(0, 1) = 0.0;

    // R - Measurement Noise Covariance (based on estimated sensor noise variance)
    R(0, 0) = MEASUREMENT_NOISE_STDDEV * MEASUREMENT_NOISE_STDDEV;

    // Initial State Estimate - Assume initial velocity is zero
    // Start with a plausible initial position.
    x_hat(0, 0) = analogRead(ANALOG_PIN); // Initial position from first reading
    x_hat(1, 0) = 0.0; // Initial velocity

    // Initial Error Covariance - High uncertainty initially
    // Position uncertainty can be set relatively high, velocity uncertainty even higher.
    P(0, 0) = R(0,0); // Initial position uncertainty similar to measurement noise
    P(1, 1) = 1.0; // Initial velocity uncertainty (tune this)
    P(0, 1) = 0.0;
    P(1, 0) = 0.0;

    // Identity Matrix (2x2)
    Identity(0, 0) = 1.0;
    Identity(1, 1) = 1.0;

    // Seed random number generator for random intervals
    randomSeed(analogRead(ANALOG_PIN)); // Use analog reading for a more random seed

    last_read_time = millis(); // Initialize the last read time
}

// --- Loop ---
void loop() {
    unsigned long current_time = millis();

    // Only run the filter if a new reading is taken after a random delay
    static unsigned long next_read_time = 0;
    if (current_time >= next_read_time) {
        double dt_seconds = (double)(current_time - last_read_time) / 1000.0;
        last_read_time = current_time; // Update last read time
        unsigned long random_delay = random(MIN_READ_INTERVAL_MS, MAX_READ_INTERVAL_MS + 1);
        next_read_time = current_time + random_delay; // Schedule next reading

        // Read the analog sensor
        double measured_value = analogRead(ANALOG_PIN);
        Matrix z_k(n_measurements, 1);
        z_k(0, 0) = measured_value;

        // --- Update F and Q based on dt ---
        // F = [1 dt]
        //     [0 1 ]
        F(0, 0) = 1.0;
        F(0, 1) = dt_seconds;
        F(1, 0) = 0.0;
        F(1, 1) = 1.0;

        // Q - Process Noise Covariance (tuned based on expected changes in position and velocity)
        // Using a simple diagonal model scaled by dt
        // These values represent the variance of unmodeled acceleration integrated over dt.
        // Q_pos = 0.25 * a_noise^2 * dt^4, Q_vel = a_noise^2 * dt^2, Q_pos_vel = 0.5 * a_noise^2 * dt^3
        // A simpler diagonal Q scaled by dt is often used for tuning.
        Q(0, 0) = PROCESS_NOISE_POSITION_RATE * dt_seconds; // Uncertainty in position grows with time
        Q(1, 1) = PROCESS_NOISE_VELOCITY_RATE * dt_seconds; // Uncertainty in velocity grows with time
        Q(0, 1) = 0.0;
        Q(1, 0) = 0.0;

        // --- Prediction Step ---
        // Predicted state estimate: x_hat_k_k_minus_1 = F * x_hat_k_minus_1_k_minus_1 + B * u_k
        // Assuming u_k is a zero vector
        Matrix u_k(n_states, 1); // Zero control input
        Matrix predicted_x_hat = F.multiply(x_hat); // B*u_k is zero matrix

        // Predicted error covariance: P_k_k_minus_1 = F * P_k_minus_1_k_minus_1 * F_transpose + Q
        Matrix F_transpose = F.transpose();
        Matrix F_P = F.multiply(P);
        Matrix F_P_F_transpose = F_P.multiply(F_transpose);
        Matrix predicted_P = F_P_F_transpose.add(Q);


        // --- Update Step ---
        // Innovation: y_tilde_k = z_k - H * predicted_x_hat
        Matrix H_predicted_x_hat = H.multiply(predicted_x_hat); // H is 1x2, predicted_x_hat is 2x1 -> result is 1x1 (predicted measurement)
        Matrix y_tilde = z_k.subtract(H_predicted_x_hat); // z_k is 1x1, H_predicted_x_hat is 1x1 -> result is 1x1

        // Innovation covariance: S_k = H * predicted_P * H_transpose + R
        Matrix H_transpose = H.transpose(); // H is 1x2, transpose is 2x1
        Matrix H_predicted_P = H.multiply(predicted_P); // H is 1x2, predicted_P is 2x2 -> result is 1x2
        Matrix H_predicted_P_H_transpose = H_predicted_P.multiply(H_transpose); // H_predicted_P is 1x2, H_transpose is 2x1 -> result is 1x1 (innovation covariance)
        Matrix S_k = H_predicted_P_H_transpose.add(R); // S_k is 1x1, R is 1x1 -> result is 1x1

        // Kalman gain: K_k = predicted_P * H_transpose * S_k_inverse
        // S_k is 1x1, so S_k_inverse is 1.0 / S_k(0,0)
        Matrix K_k(n_states, n_measurements); // K_k is 2x1
        if (fabs(S_k(0, 0)) > 1e-12) { // Check for near-zero determinant
            double S_k_inverse_scalar = 1.0 / S_k(0, 0);
            Matrix predicted_P_H_transpose = predicted_P.multiply(H_transpose); // predicted_P is 2x2, H_transpose is 2x1 -> result is 2x1
            K_k = predicted_P_H_transpose.multiply_scalar(S_k_inverse_scalar); // K_k is 2x1
        } else {
            Serial.println("Error: S_k is close to singular. Skipping update.");
            // In a real application, implement more robust handling.
            return;
        }

        // Updated state estimate: x_hat_k_k = predicted_x_hat + K_k * y_tilde
        Matrix K_k_y_tilde = K_k.multiply(y_tilde); // K_k is 2x1, y_tilde is 1x1 -> result is 2x1
        x_hat = predicted_x_hat.add(K_k_y_tilde); // predicted_x_hat is 2x1

        // Updated error covariance: P_k_k = (I - K_k * H) * predicted_P
        Matrix K_k_H = K_k.multiply(H); // K_k is 2x1, H is 1x2 -> result is 2x2
        Matrix I_minus_K_k_H = Identity.subtract(K_k_H); // Identity is 2x2
        P = I_minus_K_k_H.multiply(predicted_P); // Predicted_P is 2x2

        // Optional: Use the Joseph form for better symmetry preservation
        // This form is generally recommended for numerical stability of the covariance matrix.
        // Matrix I_minus_K_k_H_transpose = I_minus_K_k_H.transpose();
        // Matrix KR = K_k.multiply(R); // K_k is 2x1, R is 1x1 -> result is 2x1
        // Matrix KRK_transpose = KR.multiply(K_k.transpose()); // KR is 2x1, K_k_transpose is 1x2 -> result is 2x2
        // Matrix IPH = I_minus_K_k_H.multiply(predicted_P); // I_minus_K_k_H is 2x2, predicted_P is 2x2 -> result is 2x2
        // P = IPH.multiply(I_minus_K_k_H_transpose).add(KRK_transpose);


        // --- Filtering Outputs ---
        // Low-Pass Filtered Position: The estimated position provides a smoothed version of the raw signal.
        double low_pass_output_pos = x_hat(0, 0);

        // Estimated Velocity: Represents the rate of change, related to high-frequency content.
        double estimated_velocity = x_hat(1, 0);

        // High-Pass Residual: The unpredicted part of the measurement, representing higher frequencies and noise.
        double high_pass_residual = measured_value - low_pass_output_pos;


        Serial.printf("dt: %.4f s, Raw: %.2f, Low-Pass (Pos): %.2f, Est. Vel (High-Pass Related): %.4f, High-Pass (Residual): %.2f\n",
                      dt_seconds, measured_value, low_pass_output_pos, estimated_velocity, high_pass_residual);
    }

    // Short delay to avoid continuous checking of millis()
    delay(5);
}

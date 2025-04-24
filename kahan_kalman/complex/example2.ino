#include <Arduino.h>
#include <cmath> // For fabs()
#include <utility> // For std::swap

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
            // Return a reference to a dummy static variable or handle error more robustly
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

// --- Linear System Solver using Gaussian Elimination with Partial Pivoting and Kahan Summation ---
// Solves Ax = b for x
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
        if (augmentedMatrix(i, i) == 0.0) {
            Serial.println("Error: Singular matrix in solveLinear.");
            return Matrix(0, 0);
        }

        // Eliminate
        for (int k = i + 1; k < n; ++k) {
            double factor = augmentedMatrix(k, i) / augmentedMatrix(i, i);
            // Apply elimination to the rest of the row using Kahan summation
            // augmentedMatrix(k, j) = augmentedMatrix(k, j) - factor * augmentedMatrix(i, j);
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
        // sum = augmentedMatrix(i, n) - sum(augmentedMatrix(i, j) * x(j, 0)) for j = i+1 to n-1
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


// --- Kalman Filter Parameters (Example: 1D Constant Velocity) ---
// State vector: [position, velocity]'
const int n_states = 2; // Number of states
const int n_measurements = 1; // Number of measurements

// State Transition Matrix (F)
Matrix F(n_states, n_states);
// Control Matrix (B) - Assuming no control input for simplicity
Matrix B(n_states, 1);
// Observation Matrix (H) - Measuring position
Matrix H(n_measurements, n_states);
// Process Noise Covariance (Q)
Matrix Q(n_states, n_states);
// Measurement Noise Covariance (R)
Matrix R(n_measurements, n_measurements);

// Initial State Estimate (x_hat)
Matrix x_hat(n_states, 1);
// Initial Error Covariance (P)
Matrix P(n_states, n_states);

// Identity Matrix (for update step)
Matrix Identity(n_states, n_states);


// Time step (seconds)
double dt = 0.1;

// --- Setup ---
void setup() {
    Serial.begin(115200);
    while (!Serial); // Wait for serial port to connect

    Serial.println("Kalman Filter with Kahan Summation on ESP32");

    // Initialize Kalman Filter Matrices
    // F = [1 dt]
    //     [0 1 ]
    F(0, 0) = 1.0;
    F(0, 1) = dt;
    F(1, 0) = 0.0;
    F(1, 1) = 1.0;

    // B = [0] (no control input)
    //     [0]
    B(0, 0) = 0.0;
    B(1, 0) = 0.0;

    // H = [1 0] (measure position)
    H(0, 0) = 1.0;
    H(0, 1) = 0.0;

    // Q - Process Noise Covariance (tune these values based on system dynamics uncertainty)
    // Q = [q_p 0  ]
    //     [0   q_v]
    Q(0, 0) = 0.01; // Noise in position
    Q(1, 1) = 0.1;  // Noise in velocity
    Q(0, 1) = 0.0;
    Q(1, 0) = 0.0;


    // R - Measurement Noise Covariance (tune this value based on sensor noise)
    R(0, 0) = 0.5; // Noise in measurement (position)

    // Initial State Estimate - Assume initial position 0, velocity 0
    x_hat(0, 0) = 0.0;
    x_hat(1, 0) = 0.0;

    // Initial Error Covariance - High uncertainty initially
    P(0, 0) = 1.0;
    P(1, 1) = 1.0;
    P(0, 1) = 0.0;
    P(1, 0) = 0.0;

    // Identity Matrix
    Identity(0, 0) = 1.0;
    Identity(1, 1) = 1.0;

    // Seed random number generator for simulating noise
    randomSeed(analogRead(0)); // Use an unconnected analog pin for better randomness
}

// --- Loop ---
void loop() {
    // In a real application, read sensor data here.
    // For this example, we'll simulate a simple movement and add noise.
    static double actual_position = 0.0;
    static double actual_velocity = 0.5; // Constant velocity
    static double time = 0.0;

    time += dt;
    actual_position += actual_velocity * dt; // Simulate actual system state

    // Simulate noisy measurement
    double measurement_noise = ((double)random(-1000, 1001) / 1000.0) * sqrt(R(0,0)); // Random noise scaled by sqrt(R)
    double measured_position = actual_position + measurement_noise;
    Matrix z_k(n_measurements, 1);
    z_k(0, 0) = measured_position;

    Serial.printf("Time: %.2f, Actual Pos: %.2f, Measured Pos: %.2f\n", time, actual_position, measured_position);

    // --- Prediction Step ---
    // Predicted state estimate: x_hat_k_k_minus_1 = F * x_hat_k_minus_1_k_minus_1 + B * u_k
    // Assuming u_k is a zero vector for this example
    Matrix u_k(n_states, 1); // Zero control input
    Matrix predicted_x_hat = F.multiply(x_hat).add(B.multiply(u_k));

    // Predicted error covariance: P_k_k_minus_1 = F * P_k_minus_1_k_minus_1 * F_transpose + Q
    Matrix F_transpose = F.transpose();
    Matrix F_P = F.multiply(P);
    Matrix F_P_F_transpose = F_P.multiply(F_transpose);
    Matrix predicted_P = F_P_F_transpose.add(Q);


    // --- Update Step ---
    // Innovation: y_tilde_k = z_k - H * predicted_x_hat
    Matrix H_predicted_x_hat = H.multiply(predicted_x_hat);
    Matrix y_tilde = z_k.subtract(H_predicted_x_hat);

    // Innovation covariance: S_k = H * predicted_P * H_transpose + R
    Matrix H_transpose = H.transpose();
    Matrix H_predicted_P = H.multiply(predicted_P);
    Matrix H_predicted_P_H_transpose = H_predicted_P.multiply(H_transpose);
    Matrix S_k = H_predicted_P_H_transpose.add(R);

    // Kalman gain: K_k = predicted_P * H_transpose * S_k_inverse
    // Instead of explicit inverse, solve the linear system: S_k * K_k_transpose = (predicted_P * H_transpose)_transpose
    // Since S_k is symmetric (S_k = S_k_transpose), we solve S_k * K_k_transpose = H * predicted_P_transpose
    // Or more directly, solve S_k * K_k_transpose = (predicted_P * H_transpose) and then transpose the result
    // Let's form the right side (RHS) for the linear system: RHS = predicted_P * H_transpose
    Matrix RHS_for_K_transpose = predicted_P.multiply(H_transpose);

    // Solve S_k * K_k_transpose = RHS_for_K_transpose for K_k_transpose
    // Note: solveLinear is designed for a single right-hand side vector (b in Ax=b).
    // For a matrix right-hand side (AX=B), you would typically solve multiple systems,
    // one for each column of B. Since H_transpose is 2x1 and predicted_P is 2x2,
    // predicted_P * H_transpose is 2x1. S_k is 1x1.
    // The equation is actually K_k = predicted_P * H_transpose * S_k_inverse
    // For 1x1 S_k, S_k_inverse is simply 1/S_k(0,0).
    // The solveLinear function is more general for n_measurements > 1.
    // Let's use solveLinear in a way that works for n_measurements > 1,
    // where S_k would be n_measurements x n_measurements and K_k is n_states x n_measurements.
    // We solve S_k * X = (predicted_P * H_transpose) where X is the part of K_k.
    // In the 1D case, S_k is 1x1, H_transpose is 2x1. predicted_P is 2x2.
    // K_k is 2x1.
    // K_k = [ k0 ]
    //       [ k1 ]
    // S_k = [ s00 ]
    // H_transpose = [ h0 ]
    //                 [ h1 ]
    // predicted_P = [ p00 p01 ]
    //               [ p10 p11 ]
    // K_k = predicted_P * H_transpose * (1/s00)
    // K_k = [ p00 p01 ] [ h0 ] * (1/s00) = [ p00*h0 + p01*h1 ] * (1/s00)
    //       [ p10 p11 ] [ h1 ]           [ p10*h0 + p11*h1 ]
    // K_k = [ (p00*h0 + p01*h1)/s00 ]
    //       [ (p10*h0 + p11*h1)/s00 ]

    // Let's use the solveLinear to get K_k_transpose (1 x n_states)
    // S_k * K_k_transpose = (predicted_P * H_transpose)_transpose
    // S_k (1x1) * K_k_transpose (1x2) = (predicted_P (2x2) * H_transpose (2x1))_transpose (1x2)
    // This dimension doesn't match.

    // Let's go back to K_k = predicted_P * H_transpose * S_k_inverse
    // Where S_k_inverse is obtained by solving S_k * X = I for X.
    // For 1x1 matrix S_k, S_k_inverse is simply 1/S_k(0,0).
    // For higher dimensions, we would solve S_k * X = I col by col for X.

    // Using the solveLinear to find the inverse column by column:
    Matrix S_k_inverse(n_measurements, n_measurements);
    bool singular = false;
    if (S_k.rows > 0 && S_k.cols > 0) {
        Matrix Identity_measurements(n_measurements, n_measurements);
        for(int i=0; i<n_measurements; ++i) Identity_measurements(i,i) = 1.0;

        for (int i = 0; i < n_measurements; ++i) {
            Matrix identity_col(n_measurements, 1);
            identity_col(i, 0) = 1.0;
            Matrix inverse_col = solveLinear(S_k, identity_col);
            if (inverse_col.rows == 0) { // solveLinear failed
                singular = true;
                break;
            }
            for (int j = 0; j < n_measurements; ++j) {
                S_k_inverse(j, i) = inverse_col(j, 0);
            }
        }
    } else {
        singular = true;
    }


    if (singular) {
        Serial.println("Error: Could not compute S_k_inverse (singular or invalid S_k). Skipping update.");
        delay(100); // Prevent tight loop on error
        return;
    }


    Matrix predicted_P_H_transpose = predicted_P.multiply(H_transpose);
    Matrix K_k = predicted_P_H_transpose.multiply(S_k_inverse);

    // Updated state estimate: x_hat_k_k = predicted_x_hat + K_k * y_tilde
    Matrix K_k_y_tilde = K_k.multiply(y_tilde);
    x_hat = predicted_x_hat.add(K_k_y_tilde);

    // Updated error covariance: P_k_k = (I - K_k * H) * predicted_P
    Matrix K_k_H = K_k.multiply(H);
    Matrix I_minus_K_k_H = Identity.subtract(K_k_H);
    P = I_minus_K_k_H.multiply(predicted_P);

    // Optional: Use the Joseph form for better symmetry preservation
    // Matrix I_minus_K_k_H_transpose = I_minus_K_k_H.transpose();
    // Matrix KRK_transpose = K_k.multiply(R).multiply(K_k.transpose());
    // P = I_minus_K_k_H.multiply(predicted_P).multiply(I_minus_K_k_H_transpose).add(KRK_transpose);


    Serial.print("Filtered Pos: ");
    Serial.println(x_hat(0, 0), 6); // Print filtered position

    // Delay for the next iteration
    delay(dt * 1000); // Convert dt to milliseconds
}

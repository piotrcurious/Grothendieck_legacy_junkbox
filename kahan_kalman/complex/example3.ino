#include <Arduino.h>
#include <cmath>   // For fabs(), sqrt()
#include <utility> // For std::swap

// Define a small tolerance for floating-point comparisons
#define EPSILON 1e-9

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


// --- LDLT Decomposition (for symmetric matrices) ---
// Computes A = L * D * L^T where L is lower triangular with unit diagonal
// and D is a diagonal matrix.
// Returns true on success, false on failure (not symmetric positive-definite).
bool ldltDecomposition(const Matrix& A, Matrix& L, Matrix& D) {
    if (A.rows != A.cols) {
        Serial.println("Error: LDLT decomposition requires a square matrix.");
        return false;
    }
    int n = A.rows;
    L = Matrix(n, n); // L will store the lower triangular matrix (unit diagonal implicitly 1)
    D = Matrix(n, n); // D will store the diagonal matrix

    if (!L.data || !D.data) return false;

    // Assume A is symmetric, only use lower triangle (including diagonal)
    for (int i = 0; i < n; ++i) {
        // Compute D(i, i)
        double sum_D = 0.0;
        double c_D = 0.0; // Kahan compensation for D(i,i)
        for (int k = 0; k < i; ++k) {
            double term = L(i, k) * D(k, k) * L(i, k); // L(i,k)^2 * D(k,k)
             double y = term - c_D;
             double t = sum_D + y;
             c_D = (t - sum_D) - y;
             sum_D = t;
        }
        D(i, i) = A(i, i) - sum_D;

        // Check for positive definiteness (D elements should be > 0)
        if (D(i, i) < EPSILON) {
             Serial.printf("Error: Matrix is not symmetric positive-definite (D(%d,%d) <= 0).\n", i, i);
            return false;
        }

        L(i, i) = 1.0; // Unit diagonal

        // Compute L(j, i) for j > i
        for (int j = i + 1; j < n; ++j) {
            double sum_L = 0.0;
            double c_L = 0.0; // Kahan compensation for L(j,i)
            for (int k = 0; k < i; ++k) {
                double term = L(j, k) * D(k, k) * L(i, k);
                 double y = term - c_L;
                 double t = sum_L + y;
                 c_L = (t - sum_L) - y;
                 sum_L = t;
            }
            // Use A(j, i) because A is symmetric
            L(j, i) = (A(j, i) - sum_L) / D(i, i);
        }
    }
    return true;
}

// --- Forward Substitution (solves Ly = b for y, L is lower triangular with unit diagonal) ---
Matrix forwardSubstitution(const Matrix& L, const Matrix& b) {
    if (L.rows != L.cols || L.rows != b.rows || b.cols != 1) {
        Serial.println("Error: Invalid dimensions for forwardSubstitution.");
        return Matrix(0, 0);
    }
     int n = L.rows;
     Matrix y(n, 1);
      if (!y.data) return Matrix(0,0);

     for (int i = 0; i < n; ++i) {
         double sum = 0.0;
         double c = 0.0; // Kahan compensation
         for (int j = 0; j < i; ++j) {
             double term = L(i, j) * y(j, 0);
             double y_kahan = term - c;
             double t = sum + y_kahan;
             c = (t - sum) - y_kahan;
             sum = t;
         }
         y(i, 0) = b(i, 0) - sum; // Divide by L(i,i) which is 1.0
     }
     return y;
}

// --- Backward Substitution (solves U x = y for x, U is upper triangular) ---
// Solves L^T x = y for x, where L^T is upper triangular.
Matrix backwardSubstitution(const Matrix& L_transpose, const Matrix& y) {
     if (L_transpose.rows != L_transpose.cols || L_transpose.rows != y.rows || y.cols != 1) {
        Serial.println("Error: Invalid dimensions for backwardSubstitution.");
        return Matrix(0, 0);
    }
     int n = L_transpose.rows;
     Matrix x(n, 1);
      if (!x.data) return Matrix(0,0);

     for (int i = n - 1; i >= 0; --i) {
         double sum = 0.0;
         double c = 0.0; // Kahan compensation
         for (int j = i + 1; j < n; ++j) {
             double term = L_transpose(i, j) * x(j, 0);
             double y_kahan = term - c;
             double t = sum + y_kahan;
             c = (t - sum) - y_kahan;
             sum = t;
         }
         // L_transpose(i,i) is L(i,i) which is 1.0 for unit diagonal L
         x(i, 0) = (y(i, 0) - sum) / L_transpose(i, i);
     }
     return x;
}

// --- Solve System using LDLT (solves A x = b for x where A is symmetric positive-definite) ---
// Solves A x = b using LDLT decomposition: A = LDL^T.
// Solves LDL^T x = b in three steps: Ly = b, Dz = y, L^T x = z.
Matrix solveLDLT(const Matrix& A, const Matrix& b) {
    Matrix L(A.rows, A.cols), D(A.rows, A.cols);
    if (!ldltDecomposition(A, L, D)) {
        Serial.println("Error: LDLT decomposition failed in solveLDLT.");
        return Matrix(0, 0);
    }

    // Solve Ly = b for y
    Matrix y = forwardSubstitution(L, b);
     if (!y.data) return Matrix(0,0);

    // Solve Dz = y for z (Diagonal solve)
    Matrix z(A.rows, 1);
     if (!z.data) return Matrix(0,0);

    for (int i = 0; i < A.rows; ++i) {
        if (fabs(D(i, i)) < EPSILON) {
            Serial.println("Error: Division by zero in diagonal solve (LDLT).");
            return Matrix(0, 0);
        }
        z(i, 0) = y(i, 0) / D(i, i);
    }

    // Solve L^T x = z for x
    Matrix L_transpose = L.transpose();
    Matrix x = backwardSubstitution(L_transpose, z);
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

    Serial.println("Kalman Filter with LDLT Decomposition and Kahan Summation on ESP32");

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
    // We solve S_k * X = (predicted_P * H_transpose) for X, and X will be K_k.
    // Using LDLT decomposition to solve the system.
    // S_k * K_k = predicted_P * H_transpose

    Matrix K_k = solveLDLT(S_k, predicted_P.multiply(H_transpose));

    if (K_k.rows == 0) { // solveLDLT failed
        Serial.println("Error: Could not compute Kalman Gain (solveLDLT failed). Skipping update.");
        delay(100); // Prevent tight loop on error
        return;
    }


    // Updated state estimate: x_hat_k_k = predicted_x_hat + K_k * y_tilde
    Matrix K_k_y_tilde = K_k.multiply(y_tilde);
    x_hat = predicted_x_hat.add(K_k_y_tilde);

    // Updated error covariance: P_k_k = (I - K_k * H) * predicted_P
    Matrix K_k_H = K_k.multiply(H);
    Matrix I_minus_K_k_H = Identity.subtract(K_k_H);
    P = I_minus_K_k_H.multiply(predicted_P);

    // Optional: Use the Joseph form for better symmetry preservation
    // This form is generally recommended for numerical stability of the covariance matrix.
    // Matrix I_minus_K_k_H_transpose = I_minus_K_k_H.transpose();
    // Matrix KR = K_k.multiply(R);
    // Matrix KRK_transpose = KR.multiply(K_k.transpose());
    // Matrix IPH = I_minus_K_k_H.multiply(predicted_P);
    // P = IPH.multiply(I_minus_K_k_H_transpose).add(KRK_transpose);


    Serial.print("Filtered Pos: ");
    Serial.println(x_hat(0, 0), 6); // Print filtered position

    // Delay for the next iteration
    delay(dt * 1000); // Convert dt to milliseconds
}

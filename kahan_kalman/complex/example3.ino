// Numerically stable Kalman Filter utilizing LDLT Decomposition.
#include "KahanMatrix.h"

#include <cmath>   // For fabs(), sqrt()
#include <utility> // For std::swap

// Define a small tolerance for floating-point comparisons

// --- Kahan Summation for double precision ---

// --- Basic Matrix Class with Dynamic Allocation ---

// --- LDLT Decomposition (for symmetric matrices) ---
// Computes A = L * D * L^T where L is lower triangular with unit diagonal
// and D is a diagonal matrix.
// Returns true on success, false on failure (not symmetric positive-definite).

// --- Forward Substitution (solves Ly = b for y, L is lower triangular with unit diagonal) ---

// --- Backward Substitution (solves U x = y for x, U is upper triangular) ---
// Solves L^T x = y for x, where L^T is upper triangular.

// --- Solve System using LDLT (solves A x = b for x where A is symmetric positive-definite) ---
// Solves A x = b using LDLT decomposition: A = LDL^T.
// Solves LDL^T x = b in three steps: Ly = b, Dz = y, L^T x = z.

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
    // while (!Serial); // Wait for serial port to connect

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
    Identity = Matrix::Identity(n_states);

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
    Matrix predicted_x_hat = F.multiply(x_hat);

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

    Matrix K_k_trans = solveLDLT(S_k, H.multiply(predicted_P.transpose()));
    Matrix K_k = K_k_trans.transpose();

    if (K_k.rows == 0) { // solveLDLT failed
        Serial.println("Error: Could not compute Kalman Gain (solveLDLT failed). Skipping update.");
        delay(100); // Prevent tight loop on error
        return;
    }

    // Updated state estimate: x_hat_k_k = predicted_x_hat + K_k * y_tilde
    Matrix K_k_y_tilde = K_k.multiply(y_tilde);
    x_hat = predicted_x_hat.add(K_k_y_tilde);

    // Updated error covariance: P_k_k = (I - K_k * H) * predicted_P
    // Updated error covariance using Joseph form
        {
            Matrix K_H = K_k.multiply(H);
            Matrix I_KH = Identity.subtract(K_H);
            Matrix KRK_T = K_k.multiply(R).multiply(K_k.transpose());
            P = I_KH.multiply(predicted_P).multiply(I_KH.transpose()).add(KRK_T);
        }

    // Optional: Use the Joseph form for better symmetry preservation
    // This form is generally recommended for numerical stability of the covariance matrix.
    //
    //
    //
    //
    //

    Serial.print("Filtered Pos: ");
    Serial.println(x_hat(0, 0), 6); // Print filtered position

    // Delay for the next iteration
    delay(dt * 1000); // Convert dt to milliseconds
}

#include "KahanMatrix.h"

#include <cmath> // For sqrt(), fabs(), pow()
#include <utility> // For std::swap

// Define the analog input pin (replace with the actual pin on your ESP32 board)
const int ANALOG_PIN = 34; // Example: GPIO 34

// Define the spectral density of continuous-time process noise acceleration.
// This is the primary tuning parameter for the filter's bandwidth/smoothing.
// Smaller values result in more smoothing (lower cutoff frequency).
const double PROCESS_NOISE_SPECTRAL_DENSITY_ACCEL = 0.01; // Tune this parameter

// Define the standard deviation of the measurement noise (adjust based on sensor noise)
// Higher values mean the filter trusts the measurements less, leading to more smoothing.
const double MEASUREMENT_NOISE_STDDEV = 5.0; // Example: standard deviation of analog readings when input is stable

// Define the minimum and maximum random interval between readings (in milliseconds)
const unsigned long MIN_READ_INTERVAL_MS = 100;
const unsigned long MAX_READ_INTERVAL_MS = 500;

// --- Kahan Summation for double precision ---

// --- Basic Matrix Class with Dynamic Allocation ---
// (Same as in the previous example, included here for completeness)

// --- Linear System Solver using Gaussian Elimination with Partial Pivoting ---
// Solves Ax = b for x
// NOTE: For symmetric positive-definite matrices (like S_k in a Kalman filter),
// a solver based on Cholesky or LDLT decomposition is generally more numerically stable.
// This Gaussian elimination solver is provided as a functional example.

// --- Kalman Filter for Filtering (Constant Velocity Model) ---
// State vector: [position, velocity]' (2D state)
// This filter estimates the underlying state (position and velocity).
// Low-pass filtering is achieved through tuning Q and R.
// High-pass related information comes from estimated velocity and residuals.
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
    // while (!Serial); // Wait for serial port to connect

    Serial.println("Kalman Filter with Standard Q Calculation and Kahan Summation on ESP32");

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
    Identity = Matrix::Identity(n_states);

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

        // Q - Process Noise Covariance (standard discretization for constant velocity driven by
        // continuous-time white noise acceleration with spectral density sigma_a^2)
        double dt2 = dt_seconds * dt_seconds;
        double dt3 = dt2 * dt_seconds;
        Q(0, 0) = PROCESS_NOISE_SPECTRAL_DENSITY_ACCEL * (dt3 / 3.0);
        Q(0, 1) = PROCESS_NOISE_SPECTRAL_DENSITY_ACCEL * (dt2 / 2.0);
        Q(1, 0) = Q(0, 1); // Q is symmetric
        Q(1, 1) = PROCESS_NOISE_SPECTRAL_DENSITY_ACCEL * dt_seconds;

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
        // Updated error covariance using Joseph form
        {
            Matrix K_H = K_k.multiply(H);
            Matrix I_KH = Identity.subtract(K_H);
            Matrix KRK_T = K_k.multiply(R).multiply(K_k.transpose());
            P = I_KH.multiply(predicted_P).multiply(I_KH.transpose()).add(KRK_T);
        } // Predicted_P is 2x2

        // Optional: Use the Joseph form for better symmetry preservation
        // This form is generally recommended for numerical stability of the covariance matrix.
        //
        //  // K_k is 2x1, R is 1x1 -> result is 2x1
        //  // KR is 2x1, K_k_transpose is 1x2 -> result is 2x2
        //  // I_minus_K_k_H is 2x2, predicted_P is 2x2 -> result is 2x2
        //

        // --- Filtering Outputs ---
        // Low-Pass Filtered Position: The estimated position provides a smoothed version of the raw signal.
        double low_pass_output_pos = x_hat(0, 0);

        // Estimated Velocity: Represents the rate of change, related to high-frequency content.
        // This can be considered a high-pass related output emphasizing dynamics.
        double estimated_velocity = x_hat(1, 0);

        // High-Pass Residual: The unpredicted part of the measurement, representing higher frequencies and noise.
        // This is a common way to obtain a high-pass filtered signal by removing the low-frequency estimate.
        double high_pass_residual = measured_value - low_pass_output_pos;

        Serial.printf("dt: %.4f s, Raw: %.2f, Low-Pass (Pos): %.2f, Est. Vel: %.4f, High-Pass (Residual): %.2f\n",
                      dt_seconds, measured_value, low_pass_output_pos, estimated_velocity, high_pass_residual);
    }

    // Short delay to avoid continuous checking of millis()
    delay(5);
}

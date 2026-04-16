// Kalman Filter optimized for position and velocity estimation.
#include "KahanMatrix.h"
#include <cmath>
#include <utility>

const int ANALOG_PIN = 34;
const double PROCESS_NOISE_POSITION_RATE = 1.0;
const double PROCESS_NOISE_VELOCITY_RATE = 0.1;
const double MEASUREMENT_NOISE_STDDEV = 5.0;

const unsigned long MIN_READ_INTERVAL_MS = 100;
const unsigned long MAX_READ_INTERVAL_MS = 500;

const int n_states = 2;
const int n_measurements = 1;

MatrixStatic<n_states, n_states> F;
MatrixStatic<n_measurements, n_states> H;
MatrixStatic<n_states, n_states> Q;
MatrixStatic<n_measurements, n_measurements> R;
MatrixStatic<n_states, 1> x_hat;
MatrixStatic<n_states, n_states> P;
MatrixStatic<n_states, n_states> Identity;

unsigned long last_read_time = 0;
unsigned long g_next_read_time = 0;

void setup() {
    Serial.begin(115200);
    Serial.println("Kalman Filter (Constant Velocity) on ESP32 (Static Memory)");
    analogReadResolution(12);
    analogSetAttenuation(ADC_0db);

    H(0, 0) = 1.0; H(0, 1) = 0.0;
    R(0, 0) = MEASUREMENT_NOISE_STDDEV * MEASUREMENT_NOISE_STDDEV;
    x_hat(0, 0) = (double)analogRead(ANALOG_PIN);
    x_hat(1, 0) = 0.0;
    P(0, 0) = R(0,0); P(1, 1) = 1.0;
    Identity = MatrixStatic<n_states, n_states>::Identity();
    last_read_time = millis();
    g_next_read_time = 0;
}

void loop() {
    unsigned long current_time = millis();
    if (current_time >= g_next_read_time) {
        double dt = (double)(current_time - last_read_time) / 1000.0;
        if (dt <= 0) dt = 0.001;
        last_read_time = current_time;
        g_next_read_time = current_time + (unsigned long)random(MIN_READ_INTERVAL_MS, MAX_READ_INTERVAL_MS + 1);

        double measured = (double)analogRead(ANALOG_PIN);
        MatrixStatic<n_measurements, 1> z; z(0, 0) = measured;

        F(0, 0) = 1.0; F(0, 1) = dt; F(1, 0) = 0.0; F(1, 1) = 1.0;
        Q(0, 0) = PROCESS_NOISE_POSITION_RATE * dt; Q(1, 1) = PROCESS_NOISE_VELOCITY_RATE * dt;

        auto px = F.multiply(x_hat);
        auto pP = F.multiply(P).multiply(F.transpose()).add(Q);
        auto y = z.subtract(H.multiply(px));
        auto S = H.multiply(pP).multiply(H.transpose()).add(R);

        Matrix Kt_dyn = solveLinear(S.toDynamic(), (H.multiply(pP)).toDynamic());
        if (Kt_dyn.rows > 0) {
            Matrix K_dyn = Kt_dyn.transpose();
            MatrixStatic<n_states, n_measurements> K;
            for(int i=0; i<n_states; ++i) K(i,0) = K_dyn(i,0);

            x_hat = px.add(K.multiply(y));
            auto KH = K.multiply(H);
            auto I_KH = Identity.subtract(KH);
            P = I_KH.multiply(pP).multiply(I_KH.transpose()).add(K.multiply(R).multiply(K.transpose()));

            Serial.printf("dt: %.4f s, Raw: %.2f, Filtered: %.2f\n", dt, measured, x_hat(0, 0));
        }
    }
}

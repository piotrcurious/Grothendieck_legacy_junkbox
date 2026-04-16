// Advanced Kalman Filter with dynamic time constant adjustment.
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

MatrixStatic<2, 2> F;
MatrixStatic<1, 2> H;
MatrixStatic<2, 2> Q;
MatrixStatic<1, 1> R;
MatrixStatic<2, 1> x_hat;
MatrixStatic<2, 2> P;
MatrixStatic<2, 2> Identity;

unsigned long last_read_time = 0;
unsigned long g_next_read_time = 0;

void setup() {
    Serial.begin(115200);
    H(0, 0) = 1.0; H(0, 1) = 0.0;
    R(0, 0) = MEASUREMENT_NOISE_STDDEV * MEASUREMENT_NOISE_STDDEV;
    x_hat(0, 0) = (double)analogRead(ANALOG_PIN);
    x_hat(1, 0) = 0.0;
    P(0, 0) = R(0,0); P(1, 1) = 1.0;
    Identity = MatrixStatic<2, 2>::Identity();
    last_read_time = millis();
    g_next_read_time = 0;
}

void loop() {
    unsigned long current_time = millis();
    if (current_time >= g_next_read_time) {
        double dt = (double)(current_time - last_read_time) / 1000.0;
        if (dt <= 0) dt = 0.001;
        last_read_time = current_time;

        unsigned long interval = MIN_READ_INTERVAL_MS;
        if (fabs(x_hat(1, 0)) < 1.0) interval = MAX_READ_INTERVAL_MS;
        g_next_read_time = current_time + interval;

        double measured = (double)analogRead(ANALOG_PIN);
        MatrixStatic<1, 1> z; z(0, 0) = measured;

        F(0, 0) = 1.0; F(0, 1) = dt; F(1, 0) = 0.0; F(1, 1) = 1.0;
        Q(0, 0) = PROCESS_NOISE_POSITION_RATE * dt; Q(1, 1) = PROCESS_NOISE_VELOCITY_RATE * dt;

        auto px = F.multiply(x_hat);
        auto pP = F.multiply(P).multiply(F.transpose()).add(Q);
        auto y = z.subtract(H.multiply(px));
        auto S = H.multiply(pP).multiply(H.transpose()).add(R);

        Matrix Kt = solveLinear(S.toDynamic(), (H.multiply(pP)).toDynamic());
        if (Kt.rows > 0) {
            Matrix K_dyn = Kt.transpose();
            MatrixStatic<2, 1> K; K(0,0) = K_dyn(0,0); K(1,0) = K_dyn(1,0);
            x_hat = px.add(K.multiply(y));
            auto I_KH = Identity.subtract(K.multiply(H));
            P = I_KH.multiply(pP).multiply(I_KH.transpose()).add(K.multiply(R).multiply(K.transpose()));
            Serial.printf("dt: %.4f Filtered: %.2f Vel: %.2f\n", dt, x_hat(0, 0), x_hat(1, 0));
        }
    }
}

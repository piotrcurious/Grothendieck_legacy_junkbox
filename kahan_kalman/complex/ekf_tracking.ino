// Extended Kalman Filter (EKF) for tracking a falling object with air resistance.
// State vector: [position, velocity, drag_coefficient]'
#include "KahanMatrix.h"
#include <cmath>
#include <utility>

const int n_states = 3;
const int n_measurements = 1;

MatrixStatic<n_states, 1> x_hat;
MatrixStatic<n_states, n_states> P;
MatrixStatic<n_states, n_states> Q;
MatrixStatic<n_measurements, n_measurements> R;
MatrixStatic<n_states, n_states> Identity;

double dt = 0.1;
const double g = 9.81;

void setup() {
    Serial.begin(115200);
    Serial.println("EKF Tracking Initialized (Static)");

    // Initial state: 1000m high, 0 velocity, 0.1 drag
    x_hat(0, 0) = 1000.0; x_hat(1, 0) = 0.0; x_hat(2, 0) = 0.1;

    P = MatrixStatic<n_states, n_states>::Identity();
    P(0,0) = 10.0; P(1,1) = 10.0; P(2,2) = 0.01;

    Q = MatrixStatic<n_states, n_states>::Identity().multiply_scalar(0.01);
    R(0, 0) = 5.0; // 5m measurement noise
    Identity = MatrixStatic<n_states, n_states>::Identity();
}

void loop() {
    // 1. Prediction (Non-linear state transition)
    double p = x_hat(0, 0);
    double v = x_hat(1, 0);
    double d = x_hat(2, 0);

    // Non-linear f(x)
    // p_new = p + v*dt
    // v_new = v + (g - d*v*v)*dt
    // d_new = d
    double p_pred = p + v * dt;
    double v_pred = v + (g - d * v * v) * dt;
    double d_pred = d;

    MatrixStatic<n_states, 1> px;
    px(0,0) = p_pred; px(1,0) = v_pred; px(2,0) = d_pred;

    // Jacobian F = df/dx
    // [ 1   dt         0 ]
    // [ 0  1-2*d*v*dt  -v*v*dt ]
    // [ 0   0          1 ]
    MatrixStatic<n_states, n_states> F_jac;
    F_jac(0,0) = 1.0; F_jac(0,1) = dt;
    F_jac(1,1) = 1.0 - 2.0 * d * v * dt; F_jac(1,2) = -v * v * dt;
    F_jac(2,2) = 1.0;

    auto pP = F_jac.multiply(P).multiply(F_jac.transpose()).add(Q);

    // 2. Update (Linear measurement model h(x) = p)
    double measured = p_pred + (random(-100, 101) / 100.0) * sqrt(R(0,0));
    MatrixStatic<n_measurements, 1> z; z(0,0) = measured;

    MatrixStatic<n_measurements, n_states> H_jac;
    H_jac(0, 0) = 1.0; // h(x) = p, so dh/dp = 1

    auto y = z.subtract(H_jac.multiply(px));
    auto S = H_jac.multiply(pP).multiply(H_jac.transpose()).add(R);

    Matrix Kt = solveLinear(S.toDynamic(), (H_jac.multiply(pP)).toDynamic());
    if (Kt.rows > 0) {
        Matrix K_dyn = Kt.transpose();
        MatrixStatic<n_states, n_measurements> K;
        for(int i=0; i<n_states; ++i) K(i,0) = K_dyn(i,0);

        x_hat = px.add(K.multiply(y));
        auto I_KH = Identity.subtract(K.multiply(H_jac));
        P = I_KH.multiply(pP).multiply(I_KH.transpose()).add(K.multiply(R).multiply(K.transpose()));

        Serial.printf("Alt: %.1f Vel: %.1f Drag: %.3f\n", x_hat(0,0), x_hat(1,0), x_hat(2,0));
    }
    delay(dt * 1000);
}

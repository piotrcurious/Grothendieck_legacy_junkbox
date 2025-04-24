/* ESP32 Arduino Kalman Filter with Kahan Summation

This sketch implements a 1D Kalman filter using Kahan summation for improved numerical stability in accumulator operations.

Author: ChatGPT Date: 2025-04-24

KalmanFilter class handles prediction & update steps.

KahanSum struct provides compensated summation.

Example reads a noisy analog signal on pin 34.


Connections:

Signal source -> GPIO34 (ADC1_CH6)


Configure ADC attenuation as needed. */

#include <Arduino.h>

// Kahan summation for improved precision struct KahanSum { double sum;      // Running sum double c;        // Compensation for lost low-order bits

KahanSum() : sum(0.0), c(0.0) {}

void add(double value) { double y = value - c; double t = sum + y; c = (t - sum) - y; sum = t; }

double get() const { return sum; }

void reset(double value = 0.0) { sum = value; c = 0.0; } };

// 1D Kalman filter class class KalmanFilter { public: KalmanFilter(double q, double r, double initial_x = 0, double initial_p = 1) { Q = q;    // process noise covariance R = r;    // measurement noise covariance x = initial_x;  // state estimate P = initial_p;  // estimation error covariance

// Initialize Kahan accumulators
sumX.reset(x);
sumP.reset(P);

}

// Predict step: x = x; P = P + Q; void predict() { // P = P + Q using Kahan sum sumP.add(Q); P = sumP.get(); }

// Update step with measurement z void update(double z) { // Kalman gain: K = P / (P + R) double K_gain = P / (P + R);

// x = x + K * (z - x)
double innovation = z - x;
sumX.add(K_gain * innovation);
x = sumX.get();

// P = (1 - K) * P
P = (1.0 - K_gain) * P;
// Reset KahanSum for P to avoid drift
sumP.reset(P);

}

double getState() const { return x; } double getCovariance() const { return P; }

private: double Q, R;    // noise covariances double x, P;    // state & covariance

KahanSum sumX, sumP; };

// ADC pin const int analogPin = 34;

// Filter parameters // Tune Q and R to your system KalmanFilter kf(0.001, 0.1, 0.0, 1.0);

void setup() { Serial.begin(115200); analogReadResolution(12);      // 12-bit ADC resolution analogSetAttenuation(ADC_11db);// Full-scale voltage delay(1000); Serial.println("ESP32 Kalman Filter with Kahan Summation"); }

void loop() { // Read raw sensor (noisy) int raw = analogRead(analogPin); // Convert to voltage double measurement = (raw / 4095.0) * 3.3;

// Kalman filter kf.predict(); kf.update(measurement); double filtered = kf.getState();

// Print results Serial.print("Raw: "); Serial.print(measurement, 4); Serial.print(" V, Filtered: "); Serial.print(filtered, 4); Serial.print(" V, P: "); Serial.println(kf.getCovariance(), 6);

delay(10); }


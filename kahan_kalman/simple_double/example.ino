#include "KalmanFilter.h"

KalmanFilter kf(0.0001, 0.01, 1.0, 0.0);  // q, r, p, x (double precision)

void setup() {
    Serial.begin(115200);
}

void loop() {
    int raw = analogRead(34);
    double voltage = (static_cast<double>(raw) / 4095.0) * 3.3;
    double filtered = kf.update(voltage);

    Serial.print("Raw: ");
    Serial.print(voltage, 10);
    Serial.print(" | Filtered: ");
    Serial.println(filtered, 10);

    delay(100);
}

// ESP32_Example.ino
#include <Arduino.h>
#include "GaussianDualField.h"
#include "KalmanFieldFilter.h"

// Choose σ² = 0.0025
using Field = GaussianDualField<double, 0.0025>;

KalmanFieldFilter<Field> filter(
    Field(0.001,0,0),   // process noise
    Field(0.01,0,0),    // measurement noise
    Field(1.0,0,0),     // initial P
    Field(0.0,0,0)      // initial X
);

void setup() {
    Serial.begin(115200);
    analogReadResolution(12);
}

void loop() {
    double raw = analogRead(34) * (3.3 / 4095.0);
    double est = filter.update(raw);
    Serial.println(est, 6);
    delay(100);
}

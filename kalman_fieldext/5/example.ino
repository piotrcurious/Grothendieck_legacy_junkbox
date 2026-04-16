// example.ino
#include <Arduino.h>
#include "GaussianDualField.h"
#include "KalmanFieldFilter.h"

// Initialize Field with sigma^2 = 0.0025
using Field = GaussianDualField<double>;
const double s2 = 0.0025;

KalmanFieldFilter<Field> filter(
    Field(0.001, 0, 0, s2),   // process noise Q
    Field(0.01,  0, 0, s2),   // measurement noise R
    Field(1.0,   0, 0, s2),   // initial P
    Field(0.0,   0, 0, s2)    // initial X
);

void setup() {
    Serial.begin(115200);
    analogReadResolution(12);
}

void loop() {
    // Read analog value and normalize
    double raw = analogRead(34) * (3.3 / 4095.0);

    // Update filter
    double est = filter.update(raw);

    // Output estimate and sensitivity to measurement noise
    Field state = filter.getX();
    Serial.print("Est: ");
    Serial.print(state.nominal, 6);
    Serial.print(" | dEst/dR: ");
    Serial.println(state.delta, 6);

    delay(100);
}

#include "KalmanFilterExt.h"
#include "FieldExtension.h"

using F = FieldExtension<double, 2>;  // Q(√2) field

KalmanFilterExt<F> kf(
    F(0.0001), 
    F(0.01), 
    F(1.0), 
    F(0.0)
);

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

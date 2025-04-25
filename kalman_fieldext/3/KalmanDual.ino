// KalmanDual.ino
#include <Arduino.h>
#include "KalmanFilterExt.h"

// choose variances R and Q to match your sensor/noise
static constexpr double R = 0.01;    // measurement variance
static constexpr double Q = 0.0001;  // process variance

// instantiate 1-D Kalman in field extension driven by R,Q
KalmanFilterExt<double, (double)R, (double)Q> kf;

void setup() {
  Serial.begin(115200);
  kf.init(0.0, 1.0);         // optional: initial state=0, cov=1
  analogReadResolution(12);  // ESP32 12-bit ADC
}

void loop() {
  int raw = analogRead(34);
  double voltage = raw * (3.3 / 4095.0);

  double filtered = kf.update(voltage);

  Serial.print("Raw: ");
  Serial.print(voltage, 6);
  Serial.print("  |  Filtered: ");
  Serial.println(filtered, 6);

  delay(100);
}

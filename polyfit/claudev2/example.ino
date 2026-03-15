#include "src/arduino_polyfit.hpp"
#ifdef ARDUINO
#include <Arduino.h>
#else
#include "src/mock_arduino.hpp"
#endif

using namespace polyfit;

PolynomialFitter fitter(3); // Degree 3 polynomial

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("Galois-Discrete Hybrid Polyfit Example");

    // Sample data: y = 0.5x^3 - 2x^2 + x + 5
    float x_data[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    float y_data[] = {5.0, 4.5, 3.0, 3.5, 9.0, 22.5};
    size_t n = 6;

    Serial.println("Fitting data...");
    if (fitter.fit(x_data, y_data, n)) {
        Serial.println("Fit successful!");
        Serial.print("Weights: ");
        for (int i = 0; i <= fitter.degree; ++i) {
            Serial.print(fitter.weights[i]);
            Serial.print(" ");
        }
        Serial.println("");
    } else {
        Serial.println("Fit failed!");
    }
}

void loop() {
    float test_x[] = {0.5, 1.5, 2.5, 3.5, 4.5};
    Serial.println("Predictions:");
    for (int i = 0; i < 5; ++i) {
        float y = fitter.predict(test_x[i]);
        Serial.print("x = ");
        Serial.print(test_x[i]);
        Serial.print(" -> y = ");
        Serial.println(y);
    }
    delay(5000);
}

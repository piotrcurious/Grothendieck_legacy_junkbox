#include "src/arduino_polyfit.hpp"
#ifdef ARDUINO
#include <Arduino.h>
#else
#include "src/mock_arduino.hpp"
#endif

using namespace polyfit;

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("--- Galois-Discrete Hybrid Polyfit Example ---");

    // 1. Standard Polynomial Fitting with Ridge Regularization
    Serial.println("\n[1] Standard Polynomial Fitting (Degree 3)");
    PolynomialFitter fitter(3);
    float x_data[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    float y_data[] = {5.0, 4.5, 3.0, 3.5, 9.0, 22.5}; // y = 0.5x^3 - 2x^2 + x + 5
    size_t n = 6;

    if (fitter.fit(x_data, y_data, n, 0.1f)) {
        Serial.print("Weights: ");
        for (int i = 0; i <= fitter.degree; ++i) {
            Serial.print(fitter.weights[i]);
            Serial.print(" ");
        }
        Serial.println("");
        Serial.print("Prediction at x=2.5: ");
        Serial.println(fitter.predict(2.5f));
    }

    // 2. Lebesgue-based Orthogonal Projection (Legendre)
    Serial.println("\n[2] Lebesgue-based Fitting (Legendre deg 4)");
    PolynomialFitter lebesgue_fitter(4);
    if (lebesgue_fitter.fit_lebesgue(x_data, y_data, n)) {
        Serial.print("Legendre Coefficients: ");
        for (int i = 0; i <= lebesgue_fitter.degree; ++i) {
            Serial.print(lebesgue_fitter.weights[i]);
            Serial.print(" ");
        }
        Serial.println("");
        Serial.print("Lebesgue Prediction at x=2.5: ");
        Serial.println(lebesgue_fitter.predict_lebesgue(2.5f, 0.0f, 5.0f));
    }

    // 3. Algebraic Feature Extraction (F2[x] morphisms)
    Serial.println("\n[3] Algebraic Feature Extraction (Scheme Theory)");
    AlgebraicFeatureExtractor extractor(4, true); // degree 4, use Frobenius
    float features[5];
    float sample_x = 1.234f;
    extractor.extract(sample_x, features);
    Serial.print("Features for x=1.234: ");
    for(int i=0; i<5; ++i) {
        Serial.print(features[i]);
        Serial.print(" ");
    }
    Serial.println("");
}

void loop() {
    delay(10000);
}

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
    Serial.println("=== Categorical Galois-Discrete Hybrid Polyfit ===");

    // 1. Scheme Representation and Morphisms
    Serial.println("\n[1] Scheme Representation (Spec F2)");
    MachineScheme x(1.234f);
    MachineScheme y(2.345f);
    MachineScheme z = SchemeMorphism::multiply(x, y);
    Serial.print("x * y as Scheme Morphism: "); Serial.println(z.to_float());

    // 2. F2 Polynomial Ring Operations
    Serial.println("\n[2] F2Polynomial Ring Actions");
    F2Polynomial p1 = x.to_poly();
    F2Polynomial p2 = y.to_poly();
    F2Polynomial p3 = p1 * p2;
    Serial.print("F2 Polynomial Multiplication (Bit-space): "); Serial.println((int)p3.data);

    // 3. Functorial Feature Extraction
    Serial.println("\n[3] Functorial Feature Extraction");
    CategoricalFeatureExtractor extractor(4);
    float features[5];
    extractor.extract(1.234f, features);
    Serial.print("Monomial Orbits: ");
    for(int i=0; i<5; ++i) { Serial.print(features[i]); Serial.print(" "); }
    Serial.println("");

    // 4. Quantized Robust Fitting
    Serial.println("\n[4] Quantized Robust Fitting (Ridge)");
    PolynomialFitter fitter(3, QuantizedField::float32());
    float x_data[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    float y_data[] = {5.0, 4.5, 3.0, 3.5, 9.0, 22.5};
    if (fitter.fit(x_data, y_data, 6, 0.1f)) {
        Serial.print("Weights: ");
        for(int i=0; i<=3; ++i) { Serial.print(fitter.weights[i]); Serial.print(" "); }
        Serial.println("");
    }
}

void loop() {
    delay(10000);
}

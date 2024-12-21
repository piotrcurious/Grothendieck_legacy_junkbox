#include <AlgebraicCompute.h>

TimeSeries sensorData;
BanachSpace stabilizer;

void setup() {
    sensorData.data = {{1, 20.5}, {2, 21.1}, {3, 22.3}};
    Polynomial fitted = sensorData.fitPolynomial(2);
    Polynomial reduced = fitted.reduceUsingGroebnerBasis();
    Polynomial stable = stabilizer.regularize(reduced, 0.01);
    Serial.println(sensorData.predictValue(4));
}

void loop() {
    // Periodic updates
}

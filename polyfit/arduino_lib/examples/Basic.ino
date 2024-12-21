#include <AlgebraicCompute.h>

void setup() {
    Serial.begin(115200);

    // Example Polynomial
    Polynomial poly1({1, 2, 3}, 7); // 3x^2 + 2x + 1 mod 7
    Polynomial poly2({0, 1, 4}, 7); // 4x^2 + x mod 7

    Polynomial sum = poly1 + poly2;
    Polynomial product = poly1 * poly2;

    // Display Results
    Serial.println("Sum:");
    for (int coeff : sum.coefficients) {
        Serial.print(coeff);
        Serial.print(" ");
    }
    Serial.println();

    Serial.println("Product:");
    for (int coeff : product.coefficients) {
        Serial.print(coeff);
        Serial.print(" ");
    }
    Serial.println();
}

void loop() {
    // Nothing to do here
}

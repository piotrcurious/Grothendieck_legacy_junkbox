#include <Arduino.h>
#include "FieldExtension.h"

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  
  Serial.println("ESP32 Field Extension Library Example");
  Serial.println("====================================");
  
  // Basic operations with field elements
  demoBasicOperations();
  
  // Demonstration of improved precision
  demoPrecisionComparison();
  
  // Numerical stability demonstration
  demoNumericalStability();
  
  // Transcendental function calculations
  demoTranscendentalFunctions();
}

void loop() {
  // Nothing to do here
  delay(10000);
}

void demoBasicOperations() {
  Serial.println("\n=== Basic Operations ===");
  
  // Create field elements
  FieldElement4 a(3.5);
  FieldElement4 b = FieldElement4::pi();
  FieldElement4 c = FieldElement4::e();
  
  // Addition
  FieldElement4 sum = a + b;
  Serial.print("3.5 + π = ");
  Serial.println(sum.toFloat());
  
  // Multiplication
  FieldElement4 product = b * c;
  Serial.print("π × e = ");
  Serial.println(product.toFloat());
  
  // More complex calculation
  FieldElement4 result = a * b + c / a;
  Serial.print("3.5 × π + e ÷ 3.5 = ");
  Serial.println(result.toFloat());
  
  // Display the coefficients
  Serial.println("Coefficients of result:");
  Serial.print("  Constant term: ");
  Serial.println(result.getCoefficient(0));
  Serial.print("  π term: ");
  Serial.println(result.getCoefficient(1));
  Serial.print("  e term: ");
  Serial.println(result.getCoefficient(2));
  Serial.print("  √2 term: ");
  Serial.println(result.getCoefficient(3));
}

void demoPrecisionComparison() {
  Serial.println("\n=== Precision Comparison ===");
  
  // Test case: calculate (π + e)^2 - π^2 - e^2 - 2πe
  // The analytical result should be exactly 0
  
  // Standard float calculation
  float pi_f = 3.14159265359;
  float e_f = 2.71828182846;
  float std_result = pow(pi_f + e_f, 2) - pow(pi_f, 2) - pow(e_f, 2) - 2 * pi_f * e_f;
  
  Serial.print("Standard float result (should be 0): ");
  Serial.println(std_result, 10);
  
  // Field extension calculation
  FieldElement4 pi = FieldElement4::pi();
  FieldElement4 e = FieldElement4::e();
  FieldElement4 sum = pi + e;
  FieldElement4 squared = sum * sum;
  FieldElement4 pi_squared = pi * pi;
  FieldElement4 e_squared = e * e;
  FieldElement4 two_pi_e = pi * e * 2.0;
  FieldElement4 field_result = squared - pi_squared - e_squared - two_pi_e;
  
  Serial.print("Field extension result (should be 0): ");
  Serial.println(field_result.toFloat(), 10);
  
  // Show the coefficients of the result
  Serial.println("Field result coefficients:");
  for (int i = 0; i < 4; i++) {
    Serial.print("  Term ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(field_result.getCoefficient(i), 10);
  }
}

void demoNumericalStability() {
  Serial.println("\n=== Numerical Stability ===");
  
  // Test case: calculate (1000000 + π) - 1000000, should be exactly π
  
  // Standard float calculation
  float large_num = 1000000.0;
  float pi_f = 3.14159265359;
  float std_result = (large_num + pi_f) - large_num;
  
  Serial.print("Standard float: (1000000 + π) - 1000000 = ");
  Serial.println(std_result, 10);
  Serial.print("Error from π = ");
  Serial.println(std_result - pi_f, 10);
  
  // Field extension calculation
  FieldElement4 large_fe(large_num);
  FieldElement4 pi = FieldElement4::pi();
  FieldElement4 sum = large_fe + pi;
  FieldElement4 field_result = sum - large_fe;
  
  Serial.print("Field extension: (1000000 + π) - 1000000 = ");
  Serial.println(field_result.toFloat(), 10);
  Serial.println("Field result coefficients:");
  for (int i = 0; i < 4; i++) {
    Serial.print("  Term ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(field_result.getCoefficient(i), 10);
  }
}

void demoTranscendentalFunctions() {
  Serial.println("\n=== Transcendental Functions ===");
  
  // Create field elements
  FieldElement4 x = FieldElement4::pi() * 0.5; // π/2
  
  // Sin and cos
  FieldElement4 sin_x = sin(x);
  FieldElement4 cos_x = cos(x);
  
  Serial.print("sin(π/2) = ");
  Serial.println(sin_x.toFloat());
  
  Serial.print("cos(π/2) = ");
  Serial.println(cos_x.toFloat());
  
  // Verify sin²(x) + cos²(x) = 1
  FieldElement4 sin_squared = sin_x * sin_x;
  FieldElement4 cos_squared = cos_x * cos_x;
  FieldElement4 identity = sin_squared + cos_squared;
  
  Serial.print("sin²(π/2) + cos²(π/2) = ");
  Serial.println(identity.toFloat());
  
  // Exponential and logarithm
  FieldElement4 y = FieldElement4::e();
  FieldElement4 exp_y = exp(y);
  FieldElement4 log_exp_y = log(exp_y);
  
  Serial.print("e = ");
  Serial.println(y.toFloat());
  Serial.print("exp(e) = ");
  Serial.println(exp_y.toFloat());
  Serial.print("log(exp(e)) = ");
  Serial.println(log_exp_y.toFloat());
}

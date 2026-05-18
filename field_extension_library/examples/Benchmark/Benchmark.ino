#include <Arduino.h>
#include "FieldExtension.h"

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  Serial.println("FieldExtension Performance Benchmark");
  Serial.println("====================================");

  benchmarkMultiplication();
  benchmarkTranscendental();
}

void loop() {
  delay(10000);
}

void benchmarkMultiplication() {
  Serial.println("\n--- Multiplication Benchmark (10,000 ops) ---");

  FieldElement<32> a;
  for(int i=0; i<32; i++) a.setCoefficient(i, (float)i/100.0f);
  FieldElement<32> b = a;

  uint32_t start = micros();
  for(int i=0; i<10000; i++) {
    a = a * b;
    if (a.norm() > 10.0f) a *= 0.1f;
  }
  uint32_t end = micros();

  Serial.printf("Time: %u us\n", end - start);
  Serial.printf("Avg: %.3f us per op\n", (float)(end - start) / 10000.0f);
}

void benchmarkTranscendental() {
  Serial.println("\n--- Transcendental Benchmark (1,000 ops) ---");

  FieldElement16 x(0.5f);
  FieldElement16 res;

  uint32_t start = micros();
  for(int i=0; i<1000; i++) {
    res = sin(x);
    x = res * 0.5f + 0.1f;
  }
  uint32_t end = micros();

  Serial.printf("Sin Time: %u us\n", end - start);
}

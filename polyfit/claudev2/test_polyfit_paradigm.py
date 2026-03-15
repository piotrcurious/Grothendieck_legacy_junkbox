import subprocess
import os
import numpy as np

def compile_cpp():
    cmd = [
        "g++",
        "-I", ".",
        "polyfit/claudev2/src/arduino_polyfit.cpp",
        "polyfit/claudev2/src/mock_arduino.cpp",
        "polyfit/claudev2/src/main.cpp",
        "polyfit/claudev2/test_case.cpp",
        "-o", "polyfit/claudev2/test_bin"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation failed!")
        print(result.stderr)
        return False
    return True

def test_categorical_paradigm():
    cpp_code = """
#include "polyfit/claudev2/src/arduino_polyfit.hpp"
#include "polyfit/claudev2/src/mock_arduino.hpp"

using namespace polyfit;

void setup() {
    Serial.println("--- Categorical Paradigm Validation ---");

    // Test 1: Scheme Morphism Preservation (Addition)
    MachineScheme a(1.5f);
    MachineScheme b(2.5f);
    MachineScheme c = SchemeMorphism::add(a, b);
    Serial.print("Morphism Add (1.5 + 2.5): "); Serial.println(c.to_float());

    // Test 2: Frobenius Natural Transformation
    F2Polynomial p(0b1101, 4); // 1 + x + x^3
    F2Polynomial pf = p.frobenius(); // 1 + x^2 + x^6
    Serial.print("F2 Poly: "); Serial.println((int)p.data);
    Serial.print("Frobenius (x->x^2): "); Serial.println((int)pf.data);

    // Test 3: Quantized Field Boundaries
    QuantizedField qf = QuantizedField::float32();
    float val = 1.0f + 1e-8f;
    float qval = qf.quantize(val);
    Serial.print("Original: "); Serial.println(val, 10);
    Serial.print("Quantized: "); Serial.println(qval, 10);
}

void loop() {}
"""
    with open("polyfit/claudev2/test_case.cpp", 'w') as f:
        f.write(cpp_code)

    if not compile_cpp(): return

    result = subprocess.run(["./polyfit/claudev2/test_bin"], capture_output=True, text=True)
    print(result.stdout)

if __name__ == "__main__":
    test_categorical_paradigm()

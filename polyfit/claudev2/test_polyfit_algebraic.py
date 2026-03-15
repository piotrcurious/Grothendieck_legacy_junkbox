import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import struct

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

def run_categorical_extraction(x_val, degree):
    cpp_code = f"""
#include "polyfit/claudev2/src/arduino_polyfit.hpp"
#include "polyfit/claudev2/src/mock_arduino.hpp"

using namespace polyfit;

void setup() {{
    CategoricalFeatureExtractor extractor({degree});
    float features[{degree} + 1];
    extractor.extract({x_val}f, features);

    Serial.print("Features: ");
    for (int i = 0; i <= {degree}; ++i) {{
        Serial.print(features[i]);
        Serial.print(" ");
    }}
    Serial.println("");
}}

void loop() {{}}
"""
    with open("polyfit/claudev2/test_case.cpp", 'w') as f:
        f.write(cpp_code)

    if not compile_cpp():
        return None

    result = subprocess.run(["./polyfit/claudev2/test_bin"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "Features:" in line:
            return [float(f) for f in line.split("Features:")[1].strip().split()]
    return None

def float_to_bits(f):
    return struct.unpack('>I', struct.pack('>f', f))[0]

def bits_to_float(i):
    return struct.unpack('>f', struct.pack('>I', i & 0xFFFFFFFF))[0]

def f2_poly_mul(a_bits, b_bits):
    res = 0
    for i in range(32):
        if (a_bits >> i) & 1:
            res ^= (b_bits << i)
    return res & 0xFFFFFFFF

def test_algebraic_rigor():
    print("Testing Algebraic Feature Extraction Rigor...")

    test_x = 1.234
    degree = 3

    cpp_features = run_categorical_extraction(test_x, degree)
    print(f"Input float: {test_x}, Bits: {bin(float_to_bits(test_x))}")
    print(f"C++ Categorical Features (F2 mul): {cpp_features}")

    # Python verification of F2 mul
    bits = float_to_bits(test_x)
    py_features = [1.0]
    curr = bits
    for _ in range(degree):
        py_features.append(bits_to_float(curr))
        curr = f2_poly_mul(curr, bits)
    print(f"Python verified: {py_features}")

if __name__ == "__main__":
    test_algebraic_rigor()

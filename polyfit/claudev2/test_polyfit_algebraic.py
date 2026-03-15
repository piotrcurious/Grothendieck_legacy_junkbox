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

def run_algebraic_extraction(x_val, degree, use_frobenius=False):
    cpp_code = f"""
#include "polyfit/claudev2/src/arduino_polyfit.hpp"
#include "polyfit/claudev2/src/mock_arduino.hpp"

using namespace polyfit;

void setup() {{
    AlgebraicFeatureExtractor extractor({degree}, {str(use_frobenius).lower()});
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

    # Test F2 Multiplication
    cpp_features = run_algebraic_extraction(test_x, degree, use_frobenius=False)
    print(f"Input float: {test_x}, Bits: {bin(float_to_bits(test_x))}")
    print(f"C++ Algebraic Features (F2 mul): {cpp_features}")

    # Python verification of F2 mul
    bits = float_to_bits(test_x)
    py_features = [1.0]
    current = bits
    for _ in range(degree):
        py_features.append(bits_to_float(current))
        current = f2_poly_mul(current, bits)
    # The C++ implementation stores current BEFORE multiplication in the loop effectively,
    # but let's re-check the C++ logic.
    # C++: current = base; for (1 to degree) { features[d] = current; current = current * base; }
    # So features[1] = base, features[2] = base*base, etc.

    print(f"Python expected (F2 mul): {[1.0] + [bits_to_float(f2_poly_mul(bits, bits) if i > 1 else bits) for i in range(1, degree+1)]}")
    # Wait, my Python expected logic is a bit flawed, let's just use the same loop.
    py_features = [1.0]
    curr = bits
    for _ in range(degree):
        py_features.append(bits_to_float(curr))
        curr = f2_poly_mul(curr, bits)
    print(f"Python verified: {py_features}")

    # Test Frobenius
    cpp_frob = run_algebraic_extraction(test_x, degree, use_frobenius=True)
    print(f"C++ Algebraic Features (Frobenius): {cpp_frob}")

    # Morphism check: (a+b)^2 = a^2 + b^2 in F2
    x1, x2 = 1.0, 2.0
    # bits(x1) ^ bits(x2) is not bits(x1+x2) usually,
    # but the MORPHISM is in the bit-space.

    print("\nVerifying Morphism Property in Bit-Space:")
    b1 = float_to_bits(x1)
    b2 = float_to_bits(x2)
    b_sum = b1 ^ b2

    # Frobenius in Python
    def frob(b):
        res = 0
        for i in range(32):
            if (b >> i) & 1:
                res |= (1 << (2 * i))
        return res # We only care about bits here

    f1 = frob(b1)
    f2 = frob(b2)
    f_sum = frob(b_sum)

    print(f"frob(b1) ^ frob(b2) == frob(b1 ^ b2): { (f1 ^ f2) == f_sum }")

if __name__ == "__main__":
    test_algebraic_rigor()

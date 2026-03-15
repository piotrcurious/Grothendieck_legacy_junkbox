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

def test_scheme_decomposition():
    cpp_code = """
#include "polyfit/claudev2/src/arduino_polyfit.hpp"
#include "polyfit/claudev2/src/mock_arduino.hpp"

using namespace polyfit;

void setup() {
    float val = 1.234f;
    MachineScheme num(val);

    F2Polynomial sign = num.sign;
    F2Polynomial exp = num.exponent;
    F2Polynomial mant = num.mantissa;

    Serial.print("Val: "); Serial.println(val);
    Serial.print("Sign: "); Serial.println((int)sign.data);
    Serial.print("Exp: "); Serial.println((int)exp.data);
    Serial.print("Mant: "); Serial.println((int)mant.data);

    Serial.print("Restored: "); Serial.println(num.to_float());
}

void loop() {}
"""
    with open("polyfit/claudev2/test_case.cpp", 'w') as f:
        f.write(cpp_code)

    if not compile_cpp(): return

    result = subprocess.run(["./polyfit/claudev2/test_bin"], capture_output=True, text=True)
    print("Scheme Decomposition Test Output:")
    print(result.stdout)

if __name__ == "__main__":
    test_scheme_decomposition()

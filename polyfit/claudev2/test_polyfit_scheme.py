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
    MachineNumber num(val);

    BitField sign = num.get_sign();
    BitField exp = num.get_exponent();
    BitField mant = num.get_mantissa();

    Serial.print("Val: "); Serial.println(val);
    Serial.print("Sign: "); Serial.println((int)sign.value);
    Serial.print("Exp: "); Serial.println((int)exp.value);
    Serial.print("Mant: "); Serial.println((int)mant.value);

    MachineNumber restored = MachineNumber::from_scheme(sign, exp, mant);
    Serial.print("Restored: "); Serial.println(restored.val.f32);
}

void loop() {}
"""
    with open("polyfit/claudev2/test_case.cpp", 'w') as f:
        f.write(cpp_code)

    if not compile_cpp(): return

    result = subprocess.run(["./polyfit/claudev2/test_bin"], capture_output=True, text=True)
    print("Scheme Decomposition Test Output:")
    print(result.stdout)

def test_galois_action():
    cpp_code = """
#include "polyfit/claudev2/src/arduino_polyfit.hpp"
#include "polyfit/claudev2/src/mock_arduino.hpp"

using namespace polyfit;

void setup() {
    float val = 0.5f;
    GaloisActionExtractor extractor(2);
    float frob_features[2];
    float cyc_features[4];

    extractor.extract_frobenius_orbit(val, frob_features);
    extractor.extract_cyclotomic(val, cyc_features);

    Serial.print("Frob: ");
    for(int i=0; i<2; ++i) { Serial.print(frob_features[i]); Serial.print(" "); }
    Serial.println("");

    Serial.print("Cyc: ");
    for(int i=0; i<4; ++i) { Serial.print(cyc_features[i]); Serial.print(" "); }
    Serial.println("");
}

void loop() {}
"""
    with open("polyfit/claudev2/test_case.cpp", 'w') as f:
        f.write(cpp_code)

    if not compile_cpp(): return

    result = subprocess.run(["./polyfit/claudev2/test_bin"], capture_output=True, text=True)
    print("Galois Action Test Output:")
    print(result.stdout)

if __name__ == "__main__":
    test_scheme_decomposition()
    test_galois_action()

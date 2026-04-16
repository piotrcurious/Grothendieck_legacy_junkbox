import subprocess
import os

def compile_and_run(cpp_file, output_bin):
    print(f"Compiling {cpp_file}...")
    compile_cmd = [
        "g++", "-I", ".",
        "mock_arduino.cpp", cpp_file,
        "-o", output_bin
    ]
    result = subprocess.run(compile_cmd, cwd="kernels/tests", capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation failed for {cpp_file}:")
        print(result.stderr)
        return None

    print(f"Running {output_bin}...")
    run_result = subprocess.run([f"./{output_bin}"], cwd="kernels/tests", capture_output=True, text=True)
    return run_result.stdout

def validate_egd_output(output):
    print("Validating exponent_growth_detector output...")
    lines = output.split('\n')

    linear_detected = False
    exponential_detected = False

    # Simple check: we expect "Linear growth detected" in the first half and
    # "Exponential growth detected" in the second half of the output
    for line in lines:
        if "Linear growth detected" in line:
            linear_detected = True
        if "Exponential growth detected" in line:
            exponential_detected = True

    if linear_detected:
        print("[PASS] Linear growth detected successfully.")
    else:
        print("[FAIL] Linear growth NOT detected.")

    if exponential_detected:
        print("[PASS] Exponential growth detected successfully.")
    else:
        print("[FAIL] Exponential growth NOT detected.")

    return linear_detected and exponential_detected

def validate_egd_2pass_output(output):
    print("Validating EGD_2pass output...")
    lines = output.split('\n')

    linear_detected = False
    exponential_detected = False
    rate_reported = False

    for line in lines:
        if "Linear growth detected" in line:
            linear_detected = True
        if "Exponential growth detected on first pass!" in line:
            exponential_detected = True
        if "Rate of exponential growth:" in line:
            rate_reported = True

    if linear_detected:
        print("[PASS] Linear growth detected successfully.")
    else:
        print("[FAIL] Linear growth NOT detected.")

    if exponential_detected:
        print("[PASS] Exponential growth detected successfully on first pass.")
    else:
        print("[FAIL] Exponential growth NOT detected on first pass.")

    if rate_reported:
        print("[PASS] Rate of exponential growth reported.")
    else:
        print("[FAIL] Rate of exponential growth NOT reported.")

    return linear_detected and exponential_detected and rate_reported

def main():
    # Test 1
    output_egd = compile_and_run("test_egd.cpp", "test_egd")
    if output_egd:
        egd_ok = validate_egd_output(output_egd)
    else:
        egd_ok = False

    print("\n" + "="*40 + "\n")

    # Test 2
    output_egd_2pass = compile_and_run("test_egd_2pass.cpp", "test_egd_2pass")
    if output_egd_2pass:
        egd_2pass_ok = validate_egd_2pass_output(output_egd_2pass)
    else:
        egd_2pass_ok = False

    if egd_ok and egd_2pass_ok:
        print("\nALL TESTS PASSED!")
        exit(0)
    else:
        print("\nSOME TESTS FAILED!")
        exit(1)

if __name__ == "__main__":
    main()

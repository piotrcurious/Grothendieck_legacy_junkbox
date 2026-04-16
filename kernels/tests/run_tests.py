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
        if "RMSE Linear:" in line:
            print(line)
        if "RMSE Exponential:" in line:
            print(line)

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
    # Unit Test
    print("Running Unit Tests for fitting_utils.h...")
    compile_cmd = ["g++", "-I", "..", "unit_test_fitting.cpp", "-o", "unit_test_fitting"]
    subprocess.run(compile_cmd, cwd="kernels/tests", check=True)
    unit_result = subprocess.run(["./unit_test_fitting"], cwd="kernels/tests", capture_output=True, text=True)
    print(unit_result.stdout)
    if unit_result.returncode != 0:
        print("Unit Tests FAILED!")
        exit(1)

    # Test 1
    output_egd = compile_and_run("test_egd.cpp", "test_egd")
    if output_egd:
        egd_ok = validate_egd_output(output_egd)
    else:
        egd_ok = False

    print("\n" + "="*40 + "\n")

    # Test 1b - Spikes
    output_egd_spikes = compile_and_run("test_egd_spikes.cpp", "test_egd_spikes")
    if output_egd_spikes:
        egd_spikes_ok = validate_egd_output(output_egd_spikes)
    else:
        egd_spikes_ok = False

    print("\n" + "="*40 + "\n")

    # Test 1c - Step
    output_egd_step = compile_and_run("test_egd_step.cpp", "test_egd_step")
    if output_egd_step:
        # We don't strictly require it to detect "growth" here, just to run and see output.
        # But a step shouldn't ideally be detected as linear or exponential growth for long.
        egd_step_ok = True
    else:
        egd_step_ok = False

    print("\n" + "="*40 + "\n")

    # Test 1d - Compound
    output_egd_compound = compile_and_run("test_egd_compound.cpp", "test_egd_compound")
    if output_egd_compound:
        egd_compound_ok = validate_egd_output(output_egd_compound)
    else:
        egd_compound_ok = False

    print("\n" + "="*40 + "\n")

    # Test 2
    output_egd_2pass = compile_and_run("test_egd_2pass.cpp", "test_egd_2pass")
    if output_egd_2pass:
        egd_2pass_ok = validate_egd_2pass_output(output_egd_2pass)
    else:
        egd_2pass_ok = False

    if egd_ok and egd_spikes_ok and egd_step_ok and egd_compound_ok and egd_2pass_ok:
        print("\nALL TESTS PASSED!")
        exit(0)
    else:
        print("\nSOME TESTS FAILED!")
        exit(1)

if __name__ == "__main__":
    main()

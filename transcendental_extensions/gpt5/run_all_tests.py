import subprocess
import os
import sys

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result

def main():
    repo_root = os.getcwd()
    target_dir = os.path.join(repo_root, "transcendental_extensions/gpt5")
    os.chdir(target_dir)

    cpp_files = ["1.cpp", "2.cpp", "3.cpp", "4.cpp", "5.cpp", "6.cpp"]
    all_passed = True

    print("=== Testing C++ Symbolic Engines ===")
    for f in cpp_files:
        executable = f.replace('.cpp', '.exe')
        print(f"\n--- {f} ---")

        # Compile
        compile_res = run_cmd(["g++", "-std=c++17", f, "-o", executable, "-lm"])
        if compile_res.returncode != 0:
            all_passed = False
            continue

        # Run
        run_res = run_cmd([f"./{executable}"])
        if run_res.returncode == 0:
            print("Output:")
            print(run_res.stdout)
        else:
            all_passed = False
            print(f"Execution failed for {f}")

        if os.path.exists(executable):
            os.remove(executable)

    print("\n=== Verilog SFPU Verification ===")
    sim_compile = run_cmd(["iverilog", "-g2012", "8.verilog", "tb_sfpu.verilog", "-o", "sfpu.vvp"])
    if sim_compile.returncode == 0:
        sim_run = run_cmd(["vvp", "sfpu.vvp"])
        if sim_run.returncode == 0:
            print("Simulation Output:")
            print(sim_run.stdout)
        else:
            all_passed = False
            print("Verilog simulation failed.")
        if os.path.exists("sfpu.vvp"):
            os.remove("sfpu.vvp")
    else:
        all_passed = False
        print("Verilog compilation failed.")

    if all_passed:
        print("\nSUCCESS: All C++ tests passed.")
        sys.exit(0)
    else:
        print("\nFAILURE: Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()

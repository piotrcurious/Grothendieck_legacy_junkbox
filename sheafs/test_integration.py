import subprocess
import time
import os

def test_integration():
    print("Starting integration test (Headless Visualizer)...")

    # Run the visualizer in headless mock mode, limit to 32 samples to trigger a completion or near completion
    cmd = ['python3', 'sheafs/sheafs3_visualizer.py', '--port', 'mock', '--headless', '--max-samples', '32']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    collected_points = 0
    cycle_complete_msg = False
    max_samples_msg = False

    start_time = time.time()
    timeout = 15 # Wait up to 15 seconds

    try:
        while time.time() - start_time < timeout:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            print(f"Visualizer output: {line}")

            if "Sample" in line:
                collected_points += 1
            if "Cycle complete" in line:
                cycle_complete_msg = True
            if "Reached max samples" in line:
                max_samples_msg = True
                break
    finally:
        proc.terminate()

    print(f"\nSummary:")
    print(f"- Samples processed: {collected_points}")
    print(f"- Max samples message seen: {max_samples_msg}")

    if max_samples_msg and collected_points >= 20:
        print("\nIntegration test PASSED.")
    else:
        print("\nIntegration test FAILED.")
        exit(1)

if __name__ == "__main__":
    test_integration()

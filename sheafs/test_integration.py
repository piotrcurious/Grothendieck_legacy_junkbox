import subprocess
import time
import os
import signal

def test_integration():
    print("Starting integration test...")

    # Start mock arduino
    mock_proc = subprocess.Popen(['python3', 'sheafs/mock_arduino.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    collected_points = 0
    cycle_complete = False
    best_poly_found = False

    start_time = time.time()
    timeout = 15 # Wait up to 15 seconds

    try:
        while time.time() - start_time < timeout:
            line = mock_proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            print(f"Mock output: {line}")

            if "Collected Data Point" in line:
                collected_points += 1
            if "Data Collection Complete" in line:
                cycle_complete = True
            if "Best Feedback Polynomial" in line:
                best_poly_found = True

            if cycle_complete and best_poly_found and collected_points >= 32:
                break
    finally:
        mock_proc.terminate()

    print(f"\nSummary:")
    print(f"- Points collected: {collected_points}")
    print(f"- Cycle complete message seen: {cycle_complete}")
    print(f"- Best polynomial message seen: {best_poly_found}")

    if cycle_complete and best_poly_found and collected_points >= 32:
        print("\nIntegration test PASSED.")
    else:
        print("\nIntegration test FAILED.")
        exit(1)

if __name__ == "__main__":
    test_integration()

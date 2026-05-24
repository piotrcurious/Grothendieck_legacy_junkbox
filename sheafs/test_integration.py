import subprocess
import time

def test_integration():
    print("Starting integration test (Algebraic Recovery)...")

    # Run the visualizer in headless mock mode
    cmd = ['python3', '-u', 'sheafs/sheafs3_visualizer.py', '--port', 'mock', '--headless']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    samples_processed = 0
    cycle_complete = False

    start_time = time.time()
    timeout = 10

    try:
        while time.time() - start_time < timeout:
            line = proc.stdout.readline()
            if not line: break
            line = line.strip()
            # print(f"Visualizer: {line}")

            if "Sample" in line:
                samples_processed += 1
            if "Cycle complete" in line:
                cycle_complete = True
                break
    finally:
        proc.terminate()

    print(f"\nSummary:")
    print(f"- Samples processed: {samples_processed}")
    print(f"- Cycle complete: {cycle_complete}")

    if cycle_complete and samples_processed >= 10:
        print("\nIntegration test PASSED.")
    else:
        print("\nIntegration test FAILED.")
        exit(1)

if __name__ == "__main__":
    test_integration()

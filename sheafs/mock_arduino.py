import time
import random
import sys
from sheaf_utils import lfsr_gen

def mock_arduino_output():
    data_points = 32
    # Standard polynomial: x^8 + x^4 + x^3 + x^2 + 1
    poly = 0x1D
    seed = random.randint(1, 255)
    gen = lfsr_gen(seed, poly)
    noise_level = 0.05

    while True:
        current_time = 0

        for i in range(data_points):
            interval = random.randint(50, 200)
            current_time += interval
            value = next(gen)
            if random.random() < noise_level:
                value ^= 1

            print(f"Collected Data Point: Time: {current_time} ms, Value: {value}")
            sys.stdout.flush()
            time.sleep(0.01)

        print("Data Collection Complete.")
        sys.stdout.flush()
        print("Constructing Sheaf (Local Sections Check):")
        sys.stdout.flush()
        print(f"Sheaf Consensus Best Polynomial: {bin(poly)[2:]} with score: 100")
        sys.stdout.flush()
        print(f"Final Minimal Error Polynomial: {bin(poly)[2:]} (Error: 0)")
        sys.stdout.flush()

        print("Resetting cycle...")
        sys.stdout.flush()
        time.sleep(1)

if __name__ == "__main__":
    try:
        mock_arduino_output()
    except KeyboardInterrupt:
        pass

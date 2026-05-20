import time
import random

def lfsr(seed, poly, bits=8):
    state = seed
    while True:
        yield state & 1
        feedback = 0
        for i in range(bits):
            if (poly >> i) & 1:
                feedback ^= (state >> i) & 1
        state = (state >> 1) | (feedback << (bits - 1))

def mock_arduino_output():
    data_points = 32
    # Example: 8-bit LFSR with polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x1D or 0x8D etc)
    # Let's pick a simple one: 0b10110001
    poly = 0b10110001
    seed = random.randint(1, 255)
    gen = lfsr(seed, poly)

    while True:
        current_time = 0
        data_seq = []

        # Data collection phase
        for i in range(data_points):
            interval = random.randint(50, 500)
            current_time += interval
            value = next(gen)
            data_seq.append(value)

            print(f"Collected Data Point: Time: {current_time} ms, Value: {value}")
            time.sleep(0.01) # Speed up

        print("Data Collection Complete.")
        print("Constructing Sheaf with Improved Polynomial Matching:")

        # In a real match, it would find the poly. Let's mock a successful match sometimes.
        best_poly = poly if random.random() > 0.3 else random.randint(0, 255)
        print(f"Selected Best Polynomial: {bin(best_poly)[2:]} with error: 0, starting at index: 0")

        print(f"Best Feedback Polynomial (Monte Carlo Search): {bin(best_poly)[2:]}")

        print("Resetting for next data collection cycle...")
        time.sleep(1)

if __name__ == "__main__":
    try:
        mock_arduino_output()
    except KeyboardInterrupt:
        pass

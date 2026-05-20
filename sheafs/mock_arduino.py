import time
import random

def mock_arduino_output():
    data_points = 32
    while True:
        timestamps = []
        data = []
        current_time = 0

        # Data collection phase
        for i in range(data_points):
            interval = random.randint(50, 500)
            current_time += interval
            value = random.randint(0, 1)
            timestamps.append(current_time)
            data.append(value)

            print(f"Collected Data Point: Time: {current_time} ms, Value: {value}")
            time.sleep(0.05) # Speed up for testing

        print("Data Collection Complete.")
        print("Constructing Sheaf with Improved Polynomial Matching:")
        # Mocking the matching output
        best_poly = random.randint(0, 31)
        best_error = random.randint(0, 10)
        best_start = random.randint(0, 10)
        print(f"Selected Best Polynomial: {bin(best_poly)[2:]} with error: {best_error}, starting at index: {best_start}")

        feedback_poly = random.randint(0, 255)
        print(f"Best Feedback Polynomial (Monte Carlo Search): {bin(feedback_poly)[2:]}")

        print("Resetting for next data collection cycle...")
        time.sleep(1)

if __name__ == "__main__":
    try:
        mock_arduino_output()
    except KeyboardInterrupt:
        pass

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

def lfsr(seed, taps, steps):
    """ Linear Feedback Shift Register (LFSR) generator """
    reg = seed.copy()
    output = []
    for _ in range(steps):
        output.append(reg[-1])
        feedback = sum([reg[tap] for tap in taps]) % 2
        reg = [feedback] + reg[:-1]
    return output

def verify_primitive(polynomial, order):
    """ Verify if a polynomial is primitive by checking its period """
    seed = [1] + [0] * (len(polynomial) - 1)
    period = lfsr(seed, polynomial, order * 2)  # Try twice the expected period
    unique_states = len(set(tuple(period[i:i+len(seed)]) for i in range(order)))
    return unique_states == order

def visualize_lfsr(x, y, seed, polynomial, delay):
    steps = x * y
    lfsr_output = lfsr(seed, polynomial, steps)

    output_matrix = np.array(lfsr_output).reshape((y, x))

    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(output_matrix, cmap='Greys', vmin=0, vmax=1)

    for i in range(y):
        for j in range(x):
            time.sleep(delay)
            output_matrix[i, j] = lfsr_output[i * x + j]
            im.set_data(output_matrix)
            plt.draw()
            plt.pause(0.001)
    
    plt.ioff()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize LFSR-generated area coverage.')
    parser.add_argument('--x', type=int, required=True, help='Width of the area to cover.')
    parser.add_argument('--y', type=int, required=True, help='Height of the area to cover.')
    parser.add_argument('--seed', type=str, default='1000', help='Initial seed for the LFSR as a binary string.')
    parser.add_argument('--poly', type=str, required=True, help='Feedback polynomial as a binary string (e.g., "11001").')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between each step in seconds.')
    
    args = parser.parse_args()

    x = args.x
    y = args.y
    seed = [int(bit) for bit in args.seed]
    polynomial = [i for i, bit in enumerate(map(int, args.poly)) if bit == 1]
    delay = args.delay

    order = 2 ** len(polynomial) - 1
    if verify_primitive(polynomial, order):
        print(f"Polynomial {args.poly} is primitive. Proceeding with visualization.")
        visualize_lfsr(x, y, seed, polynomial, delay)
    else:
        print(f"Polynomial {args.poly} is not primitive. Please provide a primitive polynomial.")

if __name__ == '__main__':
    main()

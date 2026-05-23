import serial
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import argparse
import sys
from sheaf_utils import MockSerial, parse_data_line, berlekamp_massey

def main():
    parser = argparse.ArgumentParser(description='Advanced Sheaf Visualizer')
    parser.add_argument('--port', type=str, default='mock', help='Serial port or "mock"')
    parser.add_argument('--headless', action='store_true', help='Run without visualization')
    parser.add_argument('--max-samples', type=int, default=0, help='Max samples to process')
    args = parser.parse_args()

    if args.port == 'mock':
        ser = MockSerial(use_lfsr=True, noise_level=0.05)
    else:
        ser = serial.Serial(args.port, 9600)
        ser.flushInput()

    timestamps, values = [], []

    if not args.headless:
        plt.ion()
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    print(f"Starting {__file__} on port {args.port}")
    sys.stdout.flush()
    try:
        while True:
            if ser.in_waiting > 0:
                raw = ser.readline()
                lines = raw.decode('utf-8', errors='replace').split('\n')
                for line in lines:
                    line = line.strip()
                    if not line: continue

                    # print(f"DEBUG: Processing line: {line}")
                    t, v = parse_data_line(line)
                    if t is not None and v is not None:
                        timestamps.append(t)
                        values.append(v)

                        if len(values) >= 16:
                            recent_seq = values[-16:]
                            bm_poly = berlekamp_massey(recent_seq)

                            if not args.headless:
                                axs[0].clear()
                                axs[0].step(range(len(values)), values, where='post', color='red', label='Sequence')
                                axs[0].set_title(f"Sequence (BM Recovery: {bin(bm_poly)})")
                                axs[0].legend()
                                plt.draw()
                                plt.pause(0.01)
                            else:
                                print(f"Sample {len(values)}: BM Poly {bin(bm_poly)}")
                                sys.stdout.flush()

                    if "Data Collection Complete" in line:
                        print("Cycle complete.")
                        sys.stdout.flush()
                        if args.headless and len(values) >= 32:
                            ser.close()
                            return
                        timestamps, values = [], []

                    if args.max_samples > 0 and len(values) >= args.max_samples:
                        ser.close()
                        return

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        ser.close()

if __name__ == "__main__":
    main()

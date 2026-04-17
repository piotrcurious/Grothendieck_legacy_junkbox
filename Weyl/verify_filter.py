import numpy as np
import subprocess
import os

def run_filter(signal):
    N = 128
    if len(signal) != N:
        raise ValueError(f"Signal must be length {N}")

    in_file = "input_signal.bin"
    out_file = "output_signal.bin"

    with open(in_file, "wb") as f:
        f.write(signal.astype(np.float64).tobytes())

    # Compile
    subprocess.run(["g++", "-I", "Weyl/", "Weyl/test_weyl_filter.cpp", "-o", "Weyl/test_weyl_filter"], check=True)
    # Run
    subprocess.run(["./Weyl/test_weyl_filter", in_file, out_file], check=True)

    with open(out_file, "rb") as f:
        output = np.frombuffer(f.read(), dtype=np.float64)

    # Cleanup
    if os.path.exists(in_file): os.remove(in_file)
    if os.path.exists(out_file): os.remove(out_file)

    return output

if __name__ == "__main__":
    N = 128
    t = np.linspace(0, 1, N)
    clean = np.sin(2 * np.pi * 10 * t)
    noisy = clean + 0.5 * np.random.randn(N)

    try:
        filtered = run_filter(noisy)
        snr_noisy = 10 * np.log10(np.sum(clean**2) / np.sum((noisy - clean)**2))
        snr_filtered = 10 * np.log10(np.sum(clean**2) / np.sum((filtered - clean)**2))

        print(f"SNR Noisy: {snr_noisy:.2f} dB")
        print(f"SNR Filtered: {snr_filtered:.2f} dB")

        if snr_filtered > snr_noisy:
            print("SUCCESS: Filter improved SNR.")
        else:
            print("Note: Filter did not improve SNR on this random noise sample (adjust parameters if needed).")

    except Exception as e:
        print(f"Error: {e}")

# RNN Absolute Image Compressor (Fractal & Multi-scale Adaptive)

This project implements an online adaptive image compression system using Gated Recurrent Neural Networks (RNN) with multi-scale spatial context.

## Core Features

- **GatedRNN Architecture**: A robust GRU-like cell with refined single-step BPTT (Backpropagation Through Time) for high-fidelity online learning.
- **Context Mixing**: Dynamically combines neural prediction with classical predictors (**Median Edge Detector**, **Gradient**, and **Average**) and positional encodings.
- **Fractal Multi-scale Context**: Leverages both immediate spatial neighbors and distal pixels at multiple scales (2, 4, 8) to exploit long-range self-similarities.
- **Lossless Reconstruction**: Stores residuals as finite field elements (mod 10007), ensuring bit-perfect, lossless recovery.
- **Spatial Curve Utilities**: Includes optimized primitives for **Hilbert** and **Morton** space-filling curve traversals.

## Performance

The system achieves a compression ratio of **~1.38:1** (entropy reduction) on standard anime-style test images, outperforming baseline adaptive predictors.

## How to Run

1. Build the explorer:
   ```bash
   g++ -O3 main_explorer.cpp -o rnn_explorer
   ```
2. Run benchmarks:
   ```bash
   ./rnn_explorer
   ```

Check `compression_report.md` for detailed results.

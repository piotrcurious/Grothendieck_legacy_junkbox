# RNN Absolute Image Compressor (Morton & Raster Adaptive)

This project implements an online adaptive image compression system using Gated Recurrent Neural Networks (RNN) and Morton-order (Z-curve) traversal.

## Core Features

- **GatedRNN Architecture**: A GRU-like cell with full single-step BPTT (Backpropagation Through Time) for real-time online learning.
- **Hybrid Prediction**: Combines neural prediction with the **Median Edge Detector (MED)** and positional encodings.
- **Morton Traversal**: Supports both standard Raster scan and Morton Order Z-curve traversal for better 2D spatial context.
- **Finite Field Residues**: Stores residuals as finite field elements (mod 10007) to ensure bit-perfect, lossless reconstruction.

## Performance

The system achieves a compression ratio of **~1.38:1** (entropy reduction) on standard anime-style test images using the Raster mode, and **~1.34:1** using the Morton Order traversal.

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

# RNN Absolute Image Compressor (Galois + Fractal + LZMA)

This project implements an advanced image compression system that merges Recurrent Neural Networks (RNN) with **Absolute Galois Group** theory and LZMA entropy coding.

## Core Features

- **Hybrid Prediction**: Combines a **GatedRNN** (GRU-like) online predictor with classical spatial priors (**MED**, **Gradient**, and **Average**).
- **Absolute Galois Group Residuals**: Prediction residuals are interpreted as elements of **GF(2^8)**. We utilize the **Frobenius Automorphism** to map each residual to its canonical orbit representative.
- **Fractal Multi-scale Context**: Features are extracted at multi-scale spatial offsets (2, 4, 8) to capture long-range self-similarities.
- **LZMA Integration**: Final compressed streams (orbits and conjugacy indices) are further reduced using the LZMA algorithm.
- **Bit-Perfect Lossless**: Ensures bit-perfect reconstruction through rigorous finite field mapping.

## Performance

The system achieves a peak compression ratio of **~1.39:1** on standard anime-style test images, significantly outperforming baseline neural and classical predictors.

## How to Build

1. Build the explorer:
   ```bash
   ./build.sh
   ```
2. Run benchmarks:
   ```bash
   ./rnn_explorer
   ```

Check `compression_report.md` for detailed results.

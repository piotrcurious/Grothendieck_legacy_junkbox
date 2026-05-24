# RNN Absolute Galois Group Compressor

This project implements an online adaptive image compression system using Recurrent Neural Networks (RNN) and Finite Field arithmetic.

## Key Components

- **rnn_absolute_core.h**: Core C++ implementation of the `OptimizedRNN` with Adam optimizer and `FiniteFieldElement` for modular arithmetic (mod 10007).
- **main_explorer.cpp**: Main entry point for compression/decompression verification on PGM images.
- **compressor1.py**: Python prototype for testing algorithmic variations.
- **documentation.md**: Technical details on the RNN architecture and finite field mapping.

## Algorithms

The system uses a 12-pixel spatial context for each pixel to predict its value. The prediction error (residual) is mapped into a finite field and then back, ensuring bit-perfect reconstruction. The RNN weights are updated online after each pixel using the Adam optimizer to adapt to local image statistics.

## How to Build

Run the provided build script:
```bash
./build.sh
```

This will compile the `rnn_explorer` utility.

## Performance

See `compression_report.md` for current entropy metrics and compression ratios.

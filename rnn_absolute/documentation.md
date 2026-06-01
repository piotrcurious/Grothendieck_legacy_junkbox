# RNN Absolute Image Compressor

An experimental image compressor combining Gated Recurrent Neural Networks (RNNs) with Absolute Galois Group theory and LZMA entropy coding.

## Tools

### 1. `rnn_tool` (C++)
The core compression and decompression utility.

**Usage:**
```bash
# Compression
./rnn_tool compress <input.pgm> <output.rnn> [quality]

# Decompression
./rnn_tool decompress <input.rnn> <output.pgm>
```
*   **Quality**: 0 (Lossless, default) to 100 (Max quality).
*   **Input**: P5 PGM format.

### 2. `rnn_benchmark.py` (Python)
Automated benchmarking suite.

**Usage:**
```bash
python3 rnn_benchmark.py
```
This tool compresses a set of images at various quality levels, verifies decompression, and converts the results to PNG for visual inspection.

## Architecture

*   **Neural Predictor**: A `DualPathGaloisGRU` architecture with Gated Fusion. It independently processes Spatial Manifold features (gradients, MED) and Fractal Manifold features (multi-scale self-similarities).
*   **Algebraic Signal Representation**:
    *   **Frobenius Orbits**: residuals are mapped to orbits in GF(2^8).
    *   **Normal Basis**: Coordinates in a normal basis are used to capture Frobenius-cyclic structures.
    *   **Holomorphy**: Multi-scale trace differentials capture the flow of algebraic invariants.
*   **Lossy Mode**: Implemented via Rate-Distortion Optimization (RDO), selecting pixels that minimize `-log2(p) + lambda * MSE`.
*   **Entropy Coding**: Uses LZMA for final bitstream reduction of the unified rank stream.

## Performance

The codec typically achieves a 1.5x to 2.5x compression ratio in lossless mode. In lossy mode, it is designed to be competitive with JPEG while maintaining structural integrity through its Galois-constrained prediction.

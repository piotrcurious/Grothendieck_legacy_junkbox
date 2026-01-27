# GDHC v11.1 - VLC Compact Spectral Hybrid Image Compressor

## Overview

**GDHC (Galois-Discrete Hybrid Compressor) v11.1** is a sophisticated, multi-layer image compression engine that combines several advanced techniques from signal processing and information theory. It utilizes a **Spectral Hybrid** approach, leveraging Quadtree Decomposition, Dictionary-based Vector Quantization (VQ), Discrete Cosine Transform (DCT) for atom storage, and Variable Length Coding (VLC) to achieve high compression ratios with significant fidelity.

The "v11.1" release introduces a triple-layer architecture: a base reconstruction layer followed by two successive residual refinement layers, each with its own optimized dictionary and parameters.

---

## Core Mechanisms

### 1. Multi-Layer Residual Architecture
The compressor operates in three distinct stages for each color channel:
- **Base Layer**: Captures the primary structural and luminance information using a Rate-Distortion Optimized (RDO) Quadtree.
- **Residual Layer 1**: Models the error (difference) between the original image and the base reconstruction.
- **Residual Layer 2**: Further refines the reconstruction by modeling the error left by the first residual layer.

This hierarchical approach allows the engine to capture coarse features efficiently while progressively adding fine-grained detail.

### 2. Quadtree Decomposition & RDO
For the base layer, GDHC uses a Quadtree structure to adaptively partition the image.
- **Decision Logic**: The engine evaluates whether to split a block into four sub-blocks based on a **Rate-Distortion Optimization (RDO)** cost function:
  `Cost = SSD + λ * Bits`
  where `SSD` is the Sum of Squared Differences (distortion), `λ` (lambda) is the Lagrangian multiplier controlling the trade-off, and `Bits` is the estimated bitstream size.
- **Leaf Nodes**: Represent regions where a single "atom" from the dictionary sufficiently describes the block.

### 3. Dictionary-based Vector Quantization (Atoms)
Instead of traditional transform coding (like standard JPEG), GDHC uses a learned dictionary of "atoms".
- **Training**: During compression, the engine scans the image to identify unique, high-variance patterns. These are normalized and filtered for redundancy to form a compact dictionary.
- **Geometric Symmetries (D4 Group)**: To maximize dictionary efficiency, atoms are stored in a canonical form. During matching, the engine considers all 8 symmetries of the square (rotations and flips) plus signal inversion, effectively providing **16 variations per atom** at no extra storage cost for the dictionary itself.
- **Cross-Layer Fallback**: Residual layers can "borrow" atoms from the previous layer's dictionary if they provide a better match, further improving coding efficiency.

### 4. Spectral Atom Storage (DCT)
While the dictionary is used in the spatial domain for matching, it is stored in the **spectral domain** using the **Discrete Cosine Transform (DCT)**.
- **Quantization**: DCT coefficients are quantized using a frequency-dependent mask: `1.0 + (i + j) * 0.5`. This prioritizes low-frequency components which are more perceptually significant.
- **Compactness**: This allows the dictionary itself to be highly compressed within the file header.

### 5. Variable Length Coding (VLC)
The `Entry` structures (which store atom ID, transformation, gain, and offset) are encoded using a customized VLC scheme:
- **Frequency Analysis**: The engine identifies the most frequently used entries in the current image.
- **VLC Dictionary**: Up to 254 of these "hot" entries are stored in a small VLC dictionary.
- **Single-Byte Coding**: Matching entries are encoded as a single byte (0-253). Rare entries are prefixed with `0xFF` and stored in full (6 bytes).

---

## Data Structures

### `Entry` (6 bytes, packed)
The fundamental unit of reconstruction for a block.
- `id` (uint16_t): The index of the atom in the dictionary. The MSB (0x8000) indicates if the atom is from the current layer's dictionary or the fallback dictionary.
- `m` (uint16_t): Transformation index (0-15), representing D4 symmetries and signal inversion.
- `off` (uint8_t): The quantized DC offset (mean) of the block.
- `gn` (uint8_t): The quantized gain (scaling factor) of the atom.

### `Dict`
Manages a collection of 8x8 `MatrixXf` atoms. Handles training via variance-based selection and redundancy pruning, as well as DCT-based serialization.

### `Config`
Holds the hyperparameters for the compression run:
- `bs`, `rs`, `rs2`: Block sizes for Base, Resid1, and Resid2 layers.
- `be`, `re`, `re2`: Maximum entry counts for dictionaries.
- `lb`, `lr`, `lr2`: Lambda values for RDO.
- `bvt`, `rvt`, `rvt2`: Variance thresholds for dictionary training.

---

## File Format

The `.gdhc` bitstream is structured as follows:

1. **Global Header**:
   - Magic Number (`0x47444843`)
   - Image Dimensions (Width, Height)
   - Layer Block Sizes
2. **Channel Data** (repeated for R, G, B):
   - **Base Layer**:
     - DCT-encoded Dictionary
     - Quadtree Flags
     - VLC-encoded Entries
   - **Residual Layer 1**:
     - DCT-encoded Dictionary
     - VLC-encoded Entries
   - **Residual Layer 2**:
     - DCT-encoded Dictionary
     - VLC-encoded Entries
3. **Compression**: The entire bitstream is wrapped in a **Zlib** envelope for final entropy reduction.

---

## Compilation

The engine requires **Eigen 3**, **Boost Iostreams**, and **zlib**.

```bash
g++ -O3 -march=native -I /usr/include/eigen3 shared_comp.cpp -o shared_comp -lboost_iostreams -lz -fopenmp
```

---

## Usage

### Compression
```bash
./shared_comp c input.ppm output.gdhc [options]
```

### Decompression
```bash
./shared_comp d output.gdhc decoded.ppm
```

### Key Options
- `--bs <int>`: Base block size (default: 16)
- `--lb <float>`: Base layer lambda/quality (default: 250)
- `--be <int>`: Max base dictionary entries (default: 256)
- `--rs`, `--rs2`: Residual block sizes.
- `--lr`, `--lr2`: Residual layer lambdas.

---

## Dependencies
- **Eigen**: High-performance linear algebra.
- **Boost.Iostreams**: Stream filtering and Zlib integration.
- **Zlib**: General purpose compression.
- **OpenMP**: Multi-threading support (used in `Dict` operations).

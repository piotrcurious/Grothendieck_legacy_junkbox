# BanachJunk

A collection of experimental Arduino (ESP32) sketches and C++ components exploring the intersection of functional analysis, algebraic geometry, and signal processing.

## Core Concepts

- **Banach Spaces**: Utilizing normed vector spaces for multi-dimensional signal analysis and feature detection.
- **Galois Fields**: Applying finite field arithmetic (modular primes) to polynomial fitting and spectral analysis.
- **Algebraic Feature Detection**: Identifying signal characteristics (spikes, steps, oscillations) using algebraic invariants and variety dimensions.
- **Numerical Stability**: Implementations focus on robust algorithms like Runge-Kutta 4th Order (RK4) integration and Least Squares fitting with Gaussian elimination.

## Directory Structure

- `2.ino`: Banach Space Analyzer incorporating Galois Field concepts for polynomial fitting and spectral characteristics.
- `3_space.ino`: Abstract Banach Space representation with multi-dimensional numerical projections and Lp-norm metrics.
- `exponent_stats.ino`: Statistical analysis in Banach spaces, featuring exponential growth detection, Shannon entropy, and covariance matrices.
- `not_strict.ino`: Robust polynomial fitting using Gaussian elimination with pivoting.
- `feature_detect/`:
  - `feature_detect.ino`: Signal feature detection using norm ratios.
  - `poly_features.ino`: Advanced feature detection using Lagrange interpolation over Galois Fields.
  - `poly_feat_opt1.ino`: Memory-optimized and fixed-point version of the algebraic feature detector.
  - `poly_feature2.ino`: Feature-rich algebraic detector with autocorrelation and root-finding.
- `test/`: Host-side C++17 testing suite with a mock Arduino environment and a Makefile.
- `tesselator_demo.cpp`: Qt-based visualizer for Banach space trajectories using RK4 integration.

## Building and Testing

To run the host-side tests:
```bash
cd banachjunk/test
make test
```

## Theoretical Background

The "junk" in the name refers to the experimental nature of these implementations—repurposing abstract mathematical "junk" into practical, robust algorithms for embedded real-time systems.

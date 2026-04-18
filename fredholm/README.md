# Fredholm Education Suite Master

This directory contains a highly comprehensive, research-grade visualization and interactive education suite for Fredholm and Volterra Integral Equations. It demonstrates the depth of integral operator theory through high-fidelity numerical methods.

## Components

- **FredholmEngine.h**: A professional-grade C++ header library featuring:
  - **Nystrom Solver**: High-precision solving with up to 16-point Gauss-Legendre quadrature.
  - **Volterra Solver**: Time-evolution solver for causal systems using the trapezoidal rule.
  - **Galerkin Solver**: Basis expansion method using Legendre polynomials for global approximation.
  - **Neumann Series**: Iterative solver functionality for visualizing successive approximations.
  - **Spectral Engine**: QR algorithm for eigenvalue/eigenfunction decomposition.
  - **Stability Analysis**: Condition number and spectral radius estimation via Power Iteration on $A^T A$.
- **fredholm_suite.cpp**: A sophisticated 9-mode interactive SDL2 application:
  - **Theory**: Basic integral equations with multiple kernel types (Gaussian, Lorentzian).
  - **Comp**: Real-time signal jitter and quantization noise reduction.
  - **BVP**: Solving Boundary Value Problems (beam deflection, etc.) via Green's functions.
  - **Deblur**: Signal restoration from ill-posed blurring using Tikhonov Regularization.
  - **Spectral**: Visualization of the natural modes (eigenfunctions) of interaction kernels.
  - **Volterra**: Modeling causal evolution where history determines future states.
  - **Alt**: Demonstrating the Fredholm Alternative and resonance near eigenvalues.
  - **Neumann**: Visualizing the "learning" process of iterative solving.
  - **Galerkin**: High-order polynomial expansion approach to solving equations.

## Building and Running

### Prerequisites

- C++17 compiler (e.g., g++)
- SDL2 and SDL2_ttf development libraries

On Ubuntu/Debian:
```bash
sudo apt-get install libsdl2-dev libsdl2-ttf-dev
```

### Build

```bash
g++ fredholm_suite.cpp -o fredholm_suite -lSDL2 -lSDL2_ttf -I .
```

### Run

```bash
./fredholm_suite
```

## Educational Insights

The suite provides a "Stability Index" which estimates the spectral radius or condition proxy of the system matrix. This helps students understand why certain kernels or parameters lead to divergent or unreliable results (ill-posedness), especially in the Deblur and Alternative modes.

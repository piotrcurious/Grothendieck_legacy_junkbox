# Fredholm Education Suite Master (Optimized)

This directory contains a research-grade visualization and interactive education suite for Fredholm and Volterra Integral Equations. It demonstrates the depth of integral operator theory through high-fidelity numerical methods and optimized matrix computations.

## Components

- **FredholmEngine.h**: A performance-optimized C++ header library featuring:
  - **Nystrom Solver**: High-precision solving with up to 16-point Gauss-Legendre quadrature.
  - **Volterra Solver**: Time-evolution solver for causal systems.
  - **Optimized Galerkin Solver**: High-speed basis expansion method using pre-calculated Legendre polynomial values.
  - **Neumann Series**: Iterative step functionality for visualizing convergence paths.
  - **Spectral Engine**: QR algorithm for eigenvalue/eigenfunction decomposition.
  - **Stability Analysis**: Real-time Condition Number proxy estimation via Power Iteration.
- **fredholm_suite.cpp**: A sophisticated 10-mode interactive SDL2 application:
  - **Theory**: Basic integral equations with Gaussian/Lorentzian kernel selection.
  - **Comp**: Real-time signal jitter and quantization noise reduction.
  - **BVP**: Solving Boundary Value Problems via Green's functions.
  - **Deblur**: Signal restoration from ill-posed blurring using Tikhonov Regularization.
  - **Spectral**: Visualization of kernel natural modes.
  - **Volterra**: Modeling causal history-dependent evolution.
  - **Alt**: Demonstrating resonance near characteristic values.
  - **Neumann**: Visualizing iterative successive approximations.
  - **Galerkin**: Optimized global polynomial expansion approach.
  - **Kinds**: Direct comparison of 1st-kind ($K\phi = f$) vs 2nd-kind ($\phi - \lambda K\phi = f$) formulations.

## Building and Running

### Prerequisites

- C++17 compiler (e.g., g++)
- SDL2 and SDL2_ttf development libraries

### Build & Run

```bash
g++ fredholm_suite.cpp -o fredholm_suite -lSDL2 -lSDL2_ttf -I .
./fredholm_suite
```

## Interactive Features

- **Data Probing**: Hover over graphs to see precise coordinate values and crosshairs.
- **Divergence Warning**: Real-time alerts when parameters lead to numerical instability ($|\lambda| \rho(K) > 1$).
- **Tabbed Navigation**: 10 distinct modes covering the full spectrum of integral equation theory.

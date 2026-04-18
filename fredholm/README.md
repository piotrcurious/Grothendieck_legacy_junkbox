# Fredholm Education Suite Ultimate

This directory contains a high-fidelity visualization and interactive education suite for Fredholm and Volterra Theory. It covers basic integral equation theory, spectral analysis, and advanced applications in physics and engineering.

## Components

- **FredholmEngine.h**: A robust C++ header library implementing:
  - **Fredholm Solver**: Nystrom method with 16-point Gauss-Legendre quadrature.
  - **Volterra Solver**: Trapezoidal-rule based evolution for causal systems.
  - **Spectral Engine**: QR algorithm for computing kernel eigenvalues and eigenfunctions.
  - **Numerical Utilities**: Linear system solver with partial pivoting and singularity detection.
- **fredholm_suite.cpp**: A multi-mode interactive SDL2 application:
  - **Theory**: Basic Fredholm equations ($\phi = f + \lambda K \phi$) with Gaussian/Lorentzian kernel selection.
  - **Comp**: Signal jitter reduction using Fredholm-based smoothing.
  - **BVP**: Solving Boundary Value Problems (ODEs) using Green's functions.
  - **Deblur**: Signal restoration using Tikhonov Regularization (Fredholm equation of the 2nd kind).
  - **Spectral**: Visualization of the kernel's eigenfunctions and natural modes.
  - **Volterra**: Evolution of causal systems where the integral only depends on past history.
  - **Alt (Fredholm Alternative)**: Demonstration of resonance near kernel eigenvalues.

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

## How it Works

The suite focuses on equations of the form:
$$\phi(x) = f(x) + \lambda \int K(x, y)\phi(y)dy$$

It demonstrates how local interactions reach global balance (Fredholm) or evolve through time (Volterra), and how these mathematical structures appear in everything from beam deflection (BVP) to image processing (Deblur).

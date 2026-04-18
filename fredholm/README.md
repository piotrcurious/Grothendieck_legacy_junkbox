# Fredholm Theory Education Suite

This directory contains a concrete visualization and interactive education suite for Fredholm Theory, demonstrating its versatility across various mathematical and engineering domains.

## Components

- **FredholmEngine.h**: A header-only C++ library implementing:
  - Nystrom method for solving Fredholm integral equations of the second kind.
  - Gauss-Legendre quadrature for numerical integration.
  - Linear system solver with partial pivoting.
  - `AdaptiveCompensator`: A real-time filter for smoothing quantized or noisy signals.
- **fredholm_suite.cpp**: An interactive SDL2-based application featuring four demonstration modes:
  - **Theory Mode**: Basic Fredholm equation visualization ($\phi = f + \lambda K \phi$) with interactive kernel heatmap.
  - **Compensator Mode**: Practical application for signal jitter reduction and quantization recovery.
  - **BVP Mode**: Solving Boundary Value Problems (ODEs) like $-u'' + Vu = f$ by converting them to Fredholm equations using Green's functions.
  - **Deblur Mode**: Signal deconvolution/restoration using Tikhonov Regularization, showing how ill-posed problems are stabilized via Fredholm theory.

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

## Interactive Controls

- **Tabbed Navigation**: Switch between Theory, Compensator, BVP, and Deblur modes.
- **Sliders**: Adjust mathematical parameters (Sigma, Lambda, Frequency, Potential, Regularization Alpha) in real-time.
- **Visual Feedback**: Graphs and heatmaps update instantly to reflect the solution of the integral equations.

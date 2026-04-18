# Fredholm Theory Education Suite

This directory contains a concrete visualization and interactive education suite for Fredholm Theory, specifically focusing on Fredholm integral equations of the second kind and their application in signal processing.

## Components

- **FredholmEngine.h**: A header-only C++ library implementing:
  - Nystrom method for solving Fredholm integral equations.
  - Gauss-Legendre quadrature for numerical integration.
  - Linear system solver with partial pivoting.
  - `AdaptiveCompensator`: A practical application of Fredholm theory for smoothing quantized or noisy signals.
- **fredholm_suite.cpp**: An interactive SDL2-based application that demonstrates:
  - **Theory Mode**: Visualize how a source function $f(x)$ and a kernel $K(x, y)$ produce a solution $\phi(x)$.
  - **Application Mode**: Real-time demonstration of using Fredholm operators to recover smoothness from quantized and noisy angular data.

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

The suite solves the equation:
$$\phi(x) = f(x) + \lambda \int_a^b K(x, y)\phi(y)dy$$

In the interactive demo:
- **Red Graph**: The input source function $f(x)$.
- **Green Graph**: The resulting solution $\phi(x)$.
- **Heatmap**: The interaction kernel $K(x, y)$.
- **Sliders**: Allow real-time adjustment of kernel width ($\sigma$), interaction strength ($\lambda$), and source frequency.

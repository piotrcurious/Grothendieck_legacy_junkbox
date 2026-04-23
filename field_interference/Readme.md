# Field Interference Explorers

A collection of interactive and high-performance tools for exploring the structural density and "interference" patterns of various field systems: Algebraic numbers, Finite fields, and Transcendental extensions.

## Core Features

- **Multi-threaded C++ Engine**: High-performance root-density generation for high-fidelity heatmaps.
- **Unified Mathematical Libraries**: Shared headers for robust Galois field arithmetic (`galois_math.h`) and stabilized Durand-Kerner root solvers (`complex_math.h`).
- **Standardized Projections**: Consistent implementation of Mobius, Log-Polar, Euler Space, and Joukowsky mappings across the entire suite.
- **Field Analytics**: Visualization of element inverses ("Extension Reciprocals"), multiplicative generator orbits, and Grothendieck-style variety restricted functions.

## Suite Overview

### 1. Educational Galois Demos (`demo1/`)
- **`demo01.cpp`**: Visualizes additive lattices and multiplicative orbits in $GF(p^n)$, including reciprocal structure.
- **`demo02.cpp`**: Grothendieck viewpoint on functions restricted to varieties $V(g)$ over finite fields.
- **`demo03.cpp`**: Relationship between companion matrices and their eigenvalues in the complex plane.

### 2. High-Performance Explorers (`interference/`)
- Advanced OpenGL visualizers with hardware-accelerated texture mapping for real-time zooming and panning of algebraic density fields.

### 3. Python Analysis Tools
- **`field_interference_unified.py`**: Versatile tool for generating high-resolution distributions and lattice visualizations.
- **`transcendental_field_explorer.py`**: Visualizes the resonance and density patterns of extensions $Q(\alpha)$ for transcendental $\alpha$ (pi, e, phi, etc.).

## Build and Run

### Requirements
- **Linux**: FLTK 1.3+, OpenGL/GLU development headers.
- **Python**: numpy, matplotlib, scipy.

### Quick Start
To build all 11 C++ targets:
```bash
./field_interference/build_all.sh
```

## Mathematical Implementation Details
- **Galois Fields**: Rugged polynomial arithmetic including Extended Euclidean Algorithm for modular inverses and Ben-Or/Rabin style irreducibility verification.
- **Root Finding**: Stabilized Durand-Kerner method with random initialization perturbations to handle clustered roots and high-degree polynomials.

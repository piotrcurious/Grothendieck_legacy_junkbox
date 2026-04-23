# Field Interference Explorers

A collection of interactive and high-performance tools for exploring the structural density and "interference" patterns of various field systems: Algebraic numbers, Finite fields, and Transcendental extensions.

## Core Features

- **Multi-threaded C++ Engine**: High-performance root-density generation for high-fidelity heatmaps.
- **Unified Mathematical Libraries**: Shared headers for robust Galois field arithmetic (`galois_math.h`) and stabilized Durand-Kerner root solvers (`complex_math.h`).
- **Standardized Projections**: Consistent implementation of Mobius, Log-Polar, Euler Space, and Joukowsky mappings across the entire suite.
- **Field Analytics**: Visualization of element inverses ("Extension Reciprocals"), multiplicative generator orbits, and Grothendieck-style variety restricted functions.

---

## 1. Educational Galois Demos (`demo1/`)

Pedagogical examples designed to illustrate core algebraic concepts.

### Finite Field Interference (`demo01.cpp`)
Visualizes the interplay between the additive vector space structure of $GF(p^n)$ and its multiplicative cyclic group. Now supports "Extension Reciprocals" to visualize the field of fractions.

![Finite Field Demo](docs/images/demo01_screenshot.png)

### Grothendieck Viewpoint (`demo02.cpp`)
Illustrates how functions restricted to polynomial quotient rings $GF(p)[x]/(g)$ are well-defined on the variety $V(g)$.

![Grothendieck Viewpoint](docs/images/demo02_screenshot.png)

### Companion Matrices & Eigenvalues (`demo03.cpp`)
Visualizes the relationship between algebraic field extensions $Q(\alpha)$, companion matrices, and their roots in the complex plane.

![Companion Matrix Demo](docs/images/demo03_screenshot.png)

---

## 2. High-Performance Explorers (`interference/`)

Advanced C++ implementations using FLTK and OpenGL for deep visualization of large-scale algebraic data.

### Root Density Heatmaps (`demo07/`)
Features multi-threaded root-density generation and hardware-accelerated texture mapping. Supports real-time panning, zooming, and "Invert View" mappings.

![Root Density Demo](docs/images/demo07_screenshot.png)

### Professional 3D Explorer (`demo0c/`)
Visualizes field structures in 3D, including Basis Towers, Riemann Sphere projections, and Torus mappings.

![3D Explorer](docs/images/demo0c_screenshot.png)

---

## 3. Python Analysis Tools

### Unified Field Explorer (`field_interference_unified.py`)
Versatile tool for generating high-resolution distributions and visualizing finite field lattice connections and orbits.

![Unified Explorer](docs/images/unified_explorer.png)

### Transcendental Field Explorer (`transcendental_field_explorer.py`)
Visualizes the resonance of $\mathbb{Q}(\alpha)$ for transcendental $\alpha$ (like $\pi$, $e$, or $\phi$), exploring subfield micro-structures.

![Transcendental Explorer](docs/images/transcendental_explorer.png)

---

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

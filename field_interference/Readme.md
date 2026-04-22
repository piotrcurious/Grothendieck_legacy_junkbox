# Field Interference Explorers

A collection of interactive and high-performance tools for exploring the structural density and "interference" patterns of various field systems: Algebraic numbers, Finite fields, and Transcendental extensions.

## Project Overview

This repository provides a suite of visualizers that bridge the gap between abstract field theory and numerical computation. By visualizing the roots of random polynomials, the cyclic orbits of finite field generators, and the dense embeddings of transcendental extensions, we can observe the emergent geometric patterns that characterize different algebraic structures.

---

## 1. Educational Galois Demos (`demo1/`)

Pedagogical examples designed to illustrate core algebraic concepts.

### Finite Field Interference (`demo01.cpp`)
Visualizes the interplay between the **additive vector space** structure of $GF(p^n)$ (the lattice) and its **multiplicative cyclic group** (the generator orbit).

![Finite Field Demo](docs/images/demo01_screenshot.png)

- **Irreducible Search**: Automatically finds valid irreducible polynomials for extension degrees 1-5 using a rigorous GCD-based test.
- **Generator Orbit**: Traces the cyclic multiplicative structure with color gradients using mathematically correct polynomial reduction.

### Grothendieck Viewpoint (`demo02.cpp`)
Illustrates how functions defined on polynomial quotient rings $GF(p)[x]/(g)$ are essentially functions restricted to the variety $V(g)$.

![Grothendieck Viewpoint](docs/images/demo02_screenshot.png)

### Companion Matrices & Field Extensions (`demo03.cpp`)
Demonstrates the relationship between algebraic field extensions $Q(\alpha)$, companion matrices, and their roots (eigenvalues) in the complex plane.

![Companion Matrix Demo](docs/images/demo03_screenshot.png)

---

## 2. High-Performance Explorers (`interference/`)

Advanced C++ implementations using FLTK and OpenGL for deep visualization of large-scale algebraic data.

### Root Density Heatmaps (`demo07/`)
Features high-performance OpenGL texture rendering for root-density heatmaps, allowing smooth real-time panning and zooming into the fractal-like structures of algebraic numbers.

![Root Density Demo](docs/images/demo07_screenshot.png)

- **Rational Resonance**: Highlights roots that are near rational convergents of their real components, exposing constructive interference zones.
- **Hardware Acceleration**: Uses OpenGL textures for fluid interactive exploration.

### Professional 3D Explorer (`demo0c/`)
Visualizes field structures in 3D, including "Basis Towers" for extensions, a Riemann Sphere projection, and Torus mapping.

![3D Explorer](docs/images/demo0c_screenshot.png)
![Torus Mapping](docs/images/demo0c_torus.png)

---

## 3. Python Analysis Tools

### Unified Field Explorer (`field_interference_unified.py`)
A versatile tool for generating high-resolution distributions of algebraic numbers and visualizing finite field lattice connections and multiplicative orbits.

![Unified Explorer](docs/images/unified_explorer.png)

### Transcendental Field Explorer (`transcendental_field_explorer.py`)
Visualizes the resonance of $\mathbb{Q}(\alpha)$ for transcendental $\alpha$ (like $\pi$, $e$, or $\zeta(3)$), exploring how these extensions form dense subfields that "interfere" with the standard complex plane.

![Transcendental Explorer](docs/images/transcendental_explorer.png)

- **Vectorized Engine**: High-performance NumPy implementation supporting up to 2 million samples with chunked processing for memory efficiency.
- **Coordinate Mappings**:
    - **Standard**: Standard complex plane.
    - **Log-Polar**: Exposes radial and angular scaling symmetries. ![Log-Polar](docs/images/transcendental_logpolar.png)
    - **Reciprocal**: Visualizes the field near the origin and at infinity. ![Reciprocal](docs/images/transcendental_reciprocal.png)
    - **Joukowsky**: Aerodynamic transformation highlighting conformal symmetries. ![Joukowsky](docs/images/transcendental_joukowsky.png)
    - **Mobius**: Standard Mobius transformation $(z-1)/(z+1)$. ![Mobius](docs/images/transcendental_mobius.png)
    - **Euler Space**: Mapping $z \to \exp(i \pi z / \text{base})$ to expose periodic symmetries. ![Euler Space](docs/images/transcendental_eulerspace.png)
- **Advanced Metrics**:
    - **Rational Resonance**: Intensity based on proximity to continued fraction convergents (real) or Lattice anchors (Gaussian/Eisenstein integers). ![Resonance](docs/images/transcendental_resonance.png)
    - **Rotation Sensitivity**: Visualizes how field density shifts under infinitesimal base rotations. ![Sensitivity](docs/images/transcendental_sensitivity.png)
    - **Phase Alignment**: Measures coherence of expansion terms. ![Alignment](docs/images/transcendental_alignment.png)
    - **Algebraic Mode**: Now supports 'Binary', 'Littlewood', and 'Standard' coefficient sets. ![Algebraic Littlewood](docs/images/algebraic_littlewood.png)
- **Custom Bases**: Supports arbitrary complex expressions including `gamma` and `zeta` functions with extended `_safe_eval`.
- **Export**: High-resolution PNG images and JSON raw data export for further numerical analysis.

---

## Requirements

### C++ Explorers
- **FLTK 1.3+**, **OpenGL / GLU**
- **Build**: `g++ -std=c++17 -O3 <file>.cpp -o explorer -lfltk -lfltk_gl -lGL -lGLU -lm`

### Python Tools
- `numpy`, `matplotlib`, `scipy`

## Usage Instructions

1. **Navigate** to a demo directory (e.g., `interference/demo07`).
2. **Build** the executable or run the Python tool.
3. **Interact**: Use side panels for parameter tuning. In 3D: Left-click (Pan), Right-click (Rotate), Scroll (Zoom).

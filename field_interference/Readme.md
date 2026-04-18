# Field Interference Explorers

A collection of interactive and high-performance tools for exploring the structural density and "interference" patterns of various field systems: Algebraic numbers, Finite fields, and Transcendental extensions.

## Core Showcases

### 1. Educational Galois Demos (`demo1/`)
- **`demo01.cpp`**: Visualizes the interplay between the additive vector space structure of $GF(p^n)$ and its multiplicative cyclic group.
- **`demo02.cpp`**: Illustrates the Grothendieck viewpoint—showing how functions defined on polynomial quotient rings GF(p)[x]/(g) descend to the variety $V(g)$.
- **`demo03.cpp`**: Demonstrates the relationship between algebraic field extensions, companion matrices, and their roots (eigenvalues) in the complex plane.
- **`gf_explorer.py`**: High-level visualization of finite field embeddings.

### 2. Transcendental Explorer
- **`transcendental_field_explorer.py`**: An interactive tool for visualizing the resonance of $\mathbb{Q}(\alpha)$ for transcendental $\alpha$ (like $\pi$ or $e$), allowing for custom base evaluation.

### 3. Production Field Explorers
- **`unified2/demo05.cpp`**: A consolidated tool for exploring root-density heatmaps of random polynomials (algebraic interference) and finite field embeddings.
- **`interference/demo0c/`**: 3D edition featuring helical flow visualizations and Galois tower constructions.

## Requirements
- **Python**: `numpy`, `matplotlib`, `scipy`, `sympy`
- **C++**: `FLTK 1.3+`, `OpenGL`, `Mesa/GLU`

## Build Instructions
Example:
\`\`\`bash
g++ -std=c++17 -O2 unified2/demo05.cpp -o field_explorer -lfltk -lfltk_gl -lGL -lGLU -lm
./field_explorer
\`\`\`

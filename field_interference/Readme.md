# Field Interference Explorers

This directory contains a suite of tools and visualizations for exploring "Field Interference" — the structural resonance and density patterns of algebraic numbers, finite fields, and transcendental extensions.

## Core Concepts

- **Algebraic Interference**: The density of roots of polynomials with restricted coefficients (e.g., Littlewood polynomials) in the complex plane. This reveals fractal-like structures and "holes" around roots of unity.
- **Finite Field Interference**: Visualization of $GF(p^n)$ extensions as vector spaces over $GF(p)$, showing the additive and multiplicative coupling.
- **Transcendental Interference**: Exploring the structure of the ring $\mathbb{Q}(\alpha)$ where $\alpha$ is a transcendental number like $\pi$ or $e$, by visualizing the values of polynomials evaluated at $\alpha$.
- **Grothendieck Viewpoint**: Demos illustrating the descent of functions to varieties and the relationship between global polynomial rings and local field evaluations.

## Project Structure

### Python Tools (Interactive)
- `field_interference_unified.py`: A unified explorer for Algebraic and Finite field structures using Matplotlib.
- `transcendental_field_explorer.py`: Dedicated tool for exploring transcendental extensions with custom base support.
- `demo1/gf_explorer.py`: Vector space visualization of finite field extensions.
- `demo1/field_interference.py`: High-resolution algebraic number density plotter.

### C++ Demos (High Performance)
- `interference/`: Progressive demos (05 through 0c) using FLTK and OpenGL.
- `unified2/`: Production-ready consolidated demos.
- `demo1/`: Educational demos focusing on the logic of Galois theory and Grothendieck's approach.

## Requirements

### Python
- `numpy`
- `matplotlib`
- `scipy`
- `sympy`

### C++
- `FLTK 1.3+`
- `OpenGL`
- `Mesa/GLU`

## Building and Running C++ Demos
Example for building `unified2/demo05.cpp`:
```bash
g++ -std=c++17 -O2 unified2/demo05.cpp -o unified_explorer -lfltk -lfltk_gl -lGL -lGLU -lm
./unified_explorer
```

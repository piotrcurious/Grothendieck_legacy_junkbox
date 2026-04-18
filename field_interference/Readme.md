# Field Interference Explorers

A collection of interactive and high-performance tools for exploring the structural density and "interference" patterns of various field systems: Algebraic numbers, Finite fields, and Transcendental extensions.

## Project Structure

### 1. Educational Galois Demos (`demo1/`)
Focused, pedagogical examples of core algebraic concepts.
- **`demo01.cpp`**: Visualizes the interplay between the additive vector space structure of $GF(p^n)$ and its multiplicative cyclic group.
- **`demo02.cpp`**: Illustrates the Grothendieck viewpoint—showing how functions defined on polynomial quotient rings GF(p)[x]/(g) descend to the variety $V(g)$.
- **`demo03.cpp`**: Demonstrates the relationship between algebraic field extensions, companion matrices, and their roots (eigenvalues) in the complex plane.
- **`gf_explorer.py`**: Python-based high-level visualization of finite field embeddings.

### 2. High-Performance Explorers (`interference/`)
Advanced C++ implementations using FLTK and OpenGL for deep visualization.
- **`demo07/`**: Enhanced Production Version. Features high-performance OpenGL texture rendering for root-density heatmaps, allowing smooth real-time panning and zooming.
- **`demo08/`**: Synchronized Viewport Edition. Links different mathematical views (Algebraic vs. Finite) under a unified coordinate system.
- **`demo0c/`**: Professional 3D Edition. Visualizes finite field multiplicative flow as 3D helices and maps Galois extensions into "Z-towers" of basis components.

### 3. Python Analysis Tools
- **`field_interference_unified.py`**: Unified tool for generating algebraic number distribution heatmaps and finite field lattice connections.
- **`transcendental_field_explorer.py`**: Visualizes the resonance of $\mathbb{Q}(\alpha)$ for transcendental $\alpha$ (like $\pi$ or $e$), exploring dense subfields of $\mathbb{C}$.

## Requirements
- **Python**: `numpy`, `matplotlib`, `scipy`, `sympy`
- **C++**: `FLTK 1.3+`, `OpenGL`, `Mesa/GLU`

## Build Instructions
Example (for Linux):
```bash
cd interference/demo07
g++ -std=c++17 -O3 interference.cpp -o explorer -lfltk -lfltk_gl -lGL -lGLU -lm
./explorer
```

All C++ tools support interactive panning (Left Mouse), zooming (Mouse Wheel or Right Mouse), and dynamic parameter tuning via the side panel.

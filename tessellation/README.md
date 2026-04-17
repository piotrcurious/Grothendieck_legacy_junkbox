# Algebraic Tessellation & Graphics

Reinventing traditional computer graphics algorithms (like Bresenham's) through the lens of algebraic geometry and Grothendieck's schemes.

## Concepts
- **Parametric Equations**: Representing lines and curves as algebraic manifolds rather than just pixel sequences.
- **Error Minimization**: Using the error term from the algebraic equation to guide the integer lattice traversal.
- **Sheaves & Cyclotomic Polynomials**: Applying higher-level algebraic structures to create complex patterns and tessellations.

## Examples

### 1. Cyclotomic Lattice Projection (`simple_2.tk`)
Visualizes the ring of integers $\mathbb{Z}[\zeta_n]$ of the $n$-th cyclotomic field.
- **Lattice Mode**: Projects the $\phi(n)$-dimensional lattice into the 2D complex plane. Points are colored by their **Minkowski norm**.
- **Spectral Mode**: Simulates a diffraction pattern (Bragg peaks) of the projected lattice, illustrating the connection to **quasicrystallography**.
- **Galois Symmetry**: Visualizes the action of the Galois group $(\mathbb{Z}/n\mathbb{Z})^\times$ on the lattice points.

### 2. Morphisms of Number Schemes (`morphisms_2.tcl`)
Visualizes the "Relative Point of View" through digit-expansion fractals.
- **IFS Generation**: Uses an Iterated Function System with grid-based deduplication for efficient fractal generation.
- **Scheme Morphisms**: Maps one number scheme (fractal) to another via various morphisms:
    - `ScaleFunctor`: Categorical scaling by the base $\beta$.
    - `Composition`: Sequential application of squaring and scaling.
    - `Nonlinear`: Mapping via $z \mapsto z^2$ or $z \mapsto \exp(z)$.

### 3. Geometric Morphism Seed (`seed_2.tcl`)
Demonstrates local deformations and transformations of geometric objects.
- **Seeds**: Choose from House, Circle, Star, or a Monomial curve.
- **Jacobian Traces**: Visualizes the local stretch/rotation (Jacobian) at each point.
- **Phase Space Flow**: Visualizes the trajectories of points as the morphism parameter $T$ evolves.

## Contents
- **Bresenham Reinvented**: `line*.md`, `circle1.md`, `ellipse*.md`.
- **Generative Art**: `spirals*.md` demonstrating spiral effects using these algorithms.
- **Advanced Structures**: `tesselator_cyclotomic.md`, `tesselator_sheaf.md`.

## Technical Details
All scripts support a `--headless` mode for automated asset generation.
Usage:
```bash
wish tessellation/simple_2.tk --headless
wish tessellation/morphisms_2.tcl --headless
wish tessellation/seed_2.tcl --headless
```
This generates `simple.svg`, `morphisms.svg`, and `seed.svg` respectively.

## State
Functional prototypes and educational material. The directory contains both JavaScript examples for web-based visualization and TCL/Tk scripts for desktop experimentation.

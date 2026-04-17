# Tesselation Junkbox

This directory contains experimental Tcl/Tk visualizations of algebraic geometry concepts applied to computer graphics and number theory.

## Examples

### 1. Cyclotomic Lattice Projection (`simple.tk`)
Visualizes the ring of integers $\mathbb{Z}[\zeta_n]$ of the $n$-th cyclotomic field.
- **Lattice Mode**: Projects the $\phi(n)$-dimensional lattice into the 2D complex plane. Points are colored by their **Minkowski norm**.
- **Spectral Mode**: Simulates a diffraction pattern (Bragg peaks) of the projected lattice, illustrating the connection to **quasicrystallography**.
- **Galois Symmetry**: Visualizes the action of the Galois group $(\mathbb{Z}/n\mathbb{Z})^\times$ on the lattice points.

### 2. Morphisms of Number Schemes (`morphisms.tcl`)
Visualizes the "Relative Point of View" through digit-expansion fractals.
- **IFS Generation**: Uses an Iterated Function System with grid-based deduplication for efficient fractal generation.
- **Scheme Morphisms**: Maps one number scheme (fractal) to another via various morphisms:
    - `ScaleFunctor`: Categorical scaling by the base $\beta$.
    - `Composition`: Sequential application of squaring and scaling.
    - `Nonlinear`: Mapping via $z \mapsto z^2$ or $z \mapsto \exp(z)$.

### 3. Geometric Morphism Seed (`seed.tcl`)
Demonstrates local deformations and transformations of geometric objects.
- **Seeds**: Choose from House, Circle, Star, or a Monomial curve.
- **Jacobian Traces**: Visualizes the local stretch/rotation (Jacobian) at each point.
- **Phase Space Flow**: Visualizes the trajectories of points as the morphism parameter $T$ evolves.

## Technical Details
All scripts support a `--headless` mode for automated asset generation.
Usage:
```bash
wish tesselation/simple.tk --headless
wish tesselation/morphisms.tcl --headless
wish tesselation/seed.tcl --headless
```
This generates `simple.svg`, `morphisms.svg`, and `seed.svg` respectively.

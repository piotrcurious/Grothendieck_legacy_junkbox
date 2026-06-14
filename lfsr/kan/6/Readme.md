# Enhanced NTL LFSR Suite: Algebraic Geography

This repository provides an advanced C++ framework for Linear Feedback Shift Registers (LFSRs) and Finite Field constructions, utilizing the **NTL (Number Theory Library)** and **GMP (GNU Multi-Precision Library)**. It treats LFSRs not merely as bit-stream generators, but as geometric objects defined over an algebraic locus.

## Core Features: The Algebraic Geography Model

### 1. Geometric Traverser (Quotient Geometry)
A mathematically guaranteed traversal of any interval $[0, N-1]$ exactly once, producing a true permutation without rejection sampling.
- **Quotient Geometry:** Decomposes $N = 2^k \cdot M$ and uses subgroup actions in suitable field extensions $GF(2^n)$.
- **Scalable Ranking:** Implements an $O(M)$ rank bijection for small ranges and a **Lazy Rank** (hash-based) mode for large $N$ to prevent excessive memory usage.
- **State Management:** Supports `seek(pos)` to jump to any point in the sequence in $O(\log n)$ time.

### 2. Geometric Atlas & Morphisms (Kan Extensions)
Views the global algebraic state through multiple local "Charts," effectively implementing the Kan extension philosophy.
- **Morphisms:** Explicit transition maps between charts. The suite includes an optimized **Trace Reconstruction** morphism that recovers the full $n$-bit field state from $n$ Trace bits using precomputed dual basis matrices.
- **Projections:** Support for Companion, Trace, and Decimation charts.

### 3. Locus Exploration & Certificates
- **Primitive Locus:** Tools to find all primitive polynomials for a given degree, revealing the parameter space of maximal-period recurrences.
- **Kan Certificates:** Determines the minimal bit-length required from multiple charts to uniquely identify the global algebraic object.

## Getting Started

### Dependencies
- **NTL:** [https://shoup.net/ntl/](https://shoup.net/ntl/)
- **GMP:** [https://gmplib.org/](https://gmplib.org/)

### Compilation
Requires C++17 or later. Link against `ntl`, `gmp`, and `pthread`.

```bash
g++ -O3 -std=c++17 lfsr_improved.cpp -o lfsr_improved -lntl -lgmp -pthread
```

### Execution
Ensure `LD_LIBRARY_PATH` includes the location of your libraries.

```bash
LD_LIBRARY_PATH=/path/to/lib ./lfsr_improved
```

## Testing
The `main()` function serves as a verification suite for:
1. **Trace Reconstruction:** Bit-perfect recovery of internal state.
2. **Locus Exploration:** Generation of primitive polynomials.
3. **Scalability:** Demonstration of Lazy Rank mode for large ranges.

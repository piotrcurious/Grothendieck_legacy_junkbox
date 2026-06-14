# Enhanced NTL LFSR Suite: Quotient Geometry & Geometric Atlas

This repository contains a high-level C++ implementation of Linear Feedback Shift Registers (LFSRs) and Finite Field constructions, leveraging the **NTL (Number Theory Library)** and **GMP (GNU Multi-Precision Library)**.

## Core Architecture: Quotient Geometry & Kan Extensions

The suite is designed around the idea of treating LFSRs as geometric objects.

### 1. Geometric Traverser (Quotient Geometry)
The `GeometricTraverser` implements a mathematically guaranteed traversal of any interval $[0, N-1]$ exactly once.
- **True Permutation:** Unlike rejection sampling, this uses subgroups of roots of unity in large field extensions to produce a bijection.
- **State Management:** Supports `jump(steps)` for $O(\log \text{steps})$ fast-forward and `seek(pos)` to jump to any position in the sequence.
- **Scale:** Verified for ranges up to 65,536 and beyond.

### 2. Geometric Atlas (Kan Extensions)
Aligned with the Kan extension philosophy, the same global algebraic state can be viewed through different "Charts":
- **Companion Chart:** Projection to the polynomial basis bit-0.
- **Trace Chart:** Projection through the field trace $\text{Tr}: GF(2^n) \to GF(2)$.
- **Matrix Chart:** Custom linear forms based on the field degree.

### 3. Primitive Locus Exploration
Tools to generate "Full Sequence LFSRs" by finding all primitive polynomials for a given degree $n$. This reveals the parameter space of maximal-period recurrences.

## Getting Started

### Dependencies
- **NTL:** [https://shoup.net/ntl/](https://shoup.net/ntl/)
- **GMP:** [https://gmplib.org/](https://gmplib.org/)

### Compilation
Use C++17 or later and link against `ntl`, `gmp`, and `pthread`.

```bash
g++ -O3 -std=c++17 lfsr_improved.cpp -o lfsr_improved -lntl -lgmp -pthread
```

### Execution
```bash
LD_LIBRARY_PATH=/path/to/lib ./lfsr_improved
```

## Testing
The `main()` function includes:
1. **Primitive Locus Generation:** Finding all primitive polynomials for $n=5$.
2. **Jump & Seek Verification:** Confirming fast-forward and absolute positioning.
3. **Atlas View Demo:** Displaying bit-streams through Companion and Trace projections.
4. **Large Range Stress Test:** Uniqueness verification for $N=65536$.

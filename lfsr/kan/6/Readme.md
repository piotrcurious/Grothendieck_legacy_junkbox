# Enhanced NTL LFSR Suite: Quotient Geometry Edition

This repository contains a high-level C++ implementation of Linear Feedback Shift Registers (LFSRs) and Finite Field constructions, leveraging the **NTL (Number Theory Library)** and **GMP (GNU Multi-Precision Library)**.

## Core Architecture: Quotient Geometry

The centerpiece of this suite is the `GeometricTraverser`, which implements a mathematically guaranteed traversal of any interval $[0, N-1]$ exactly once. Unlike standard LFSR-based "rejection sampling" methods, this implementation uses **Quotient Geometry** to produce a true permutation.

### Mathematical Foundation
For any range size $N$, we decompose it as $N = 2^k \cdot M$, where $M$ is odd.
1. **Odd Part ($M$):** We find the smallest extension degree $n$ such that $M$ divides $2^n - 1$.
2. **Subgroup Action:** In the field $GF(2^n)$, we select a primitive element $\alpha$ and define $\zeta = \alpha^{(2^n-1)/M}$, which generates a cyclic subgroup of order $M$ (the $M$-th roots of unity).
3. **Bijection:** Traversal is performed by multiplication in the subgroup $\langle\zeta\rangle$. We map these algebraic elements to $[0, M-1]$ using their rank in the field's natural bit-ordering.
4. **Product Construction:** The final traversal is the product of the power-of-two counter and the geometric odd-part traverser.

This approach is "geometric" because the interval is obtained as a quotient of a cyclic field action rather than by filtering an orbit.

## Features

- **Algebraic Inference:** Recovers field parameters and primitive elements from observed bit-stream prefixes in $O(n)$ time.
- **Geometric Traversal:** Guaranteed one-time visit of all elements in $[0, N-1]$ without rejection sampling for the odd component.
- **Reproducibility:** Fully deterministic sequences when provided with a seed.
- **Mersenne-Scale Testing:** Verified for ranges up to 65,535 and beyond.

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
./lfsr_improved
```

## Testing
The `main()` function contains a suite of demonstrations:
1. **Field Inference:** Deducing $GF(2^8)$ parameters from a short sequence.
2. **Geometric Traversal:** Completeness and uniqueness checks for various $N$ (100, 65535, 127, 1024).
3. **Reproducibility:** Verification that seeded instances produce bit-identical paths.

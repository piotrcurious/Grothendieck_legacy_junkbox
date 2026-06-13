# Enhanced NTL LFSR Suite

This repository contains a robust C++ implementation of Linear Feedback Shift Registers (LFSRs) leveraging the **NTL (Number Theory Library)** and **GMP (GNU Multi-Precision Library)**. It provides tools for algebraic inference, pseudo-random range traversal, and maximal length sequence generation.

## Features

### 1. Algebraic Inference Layer
- **O(n) Alpha Recovery:** Efficiently recovers the primitive element (alpha) from observed sequences using a direct string-to-polynomial parser. This is significantly faster than the naive $O(2^w)$ enumeration approach.
- **Field Width Inference:** Automatically identifies the underlying field extension degree `w` from a short observed prefix.

### 2. Pseudo-random Range Traversal
- **Complete Coverage:** The `RangeTraverser` class ensures that every integer in a specified range `[0, max_val]` is visited exactly once before the sequence repeats.
- **Pseudo-randomness:** Uses an underlying LFSR over a field extension $GF(2^n)$ (where $2^n-1 \ge max\_val$) to generate a non-repeating, pseudo-random sequence.
- **Efficient Mapping:** Employs a discarding mechanism to stay within the target range while maintaining the cycle properties of the LFSR.
- **Reproducibility:** Supports optional seeding for deterministic, reproducible sequences.

### 3. Maximal Orbit Generation
- **Full Sequence Generation:** The `build_orbit` function generates a complete maximal length sequence ($2^n-1$ states) for any degree $n < 62$.
- **Primitive Polynomials:** Utilizes NTL's `BuildSparseIrred` to automatically select optimal primitive polynomials for any given degree.

## Getting Started

### Dependencies
- **NTL:** [https://shoup.net/ntl/](https://shoup.net/ntl/)
- **GMP:** [https://gmplib.org/](https://gmplib.org/)

### Compilation
To compile the suite, link against NTL, GMP, and Pthreads. Ensure you use C++17 or later.

```bash
g++ -O3 -std=c++17 lfsr_improved.cpp -o lfsr_improved -lntl -lgmp -pthread
```

If the libraries are installed in a custom location (e.g., `/usr/local` or a home directory), specify the paths:

```bash
g++ -O3 -std=c++17 lfsr_improved.cpp -o lfsr_improved \
    -I/path/to/include -L/path/to/lib \
    -lntl -lgmp -pthread
```

### Execution
Run the compiled binary. If using custom library paths, update `LD_LIBRARY_PATH`:

```bash
LD_LIBRARY_PATH=/path/to/lib ./lfsr_improved
```

## Testing
The `main()` function in `lfsr_improved.cpp` includes a comprehensive test suite that verifies:
- **Inference Accuracy:** Correct recovery of field parameters from sequence prefixes.
- **Range Traversal Integrity:** Uniqueness and completeness for ranges up to 65,535 and beyond.
- **Orbit Completeness:** Verification of maximal length sequence properties.
- **Determinism:** Ensuring seeded traversers produce identical sequences.

## Practical Applications
- **Pseudo-random Sampling:** Visit every element in a large dataset exactly once in random order without storing a visited set.
- **Cryptographic Primitives:** Generate keystreams or nonces with guaranteed period properties.
- **Simulation:** Efficiently shuffle or traverse state spaces in Monte Carlo simulations.

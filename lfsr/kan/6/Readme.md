# Enhanced NTL LFSR Suite: Canonical Kan Extensions

This repository contains a high-level C++ implementation of Linear Feedback Shift Registers (LFSRs) and Finite Field constructions, leveraging the **NTL (Number Theory Library)** and **GMP (GNU Multi-Precision Library)**.

## Core Philosophy: Resolving Underdetermined Kan Extensions

Standard Kan extensions for bit-streams can be "underdetermined" due to random primitive choices or insufficient local data. This suite resolves these ambiguities through:

### 1. Canonical Constructions
- **Minimal Primitives:** Replaces random selection with the lexicographically smallest bit-pattern that satisfies primitivity. This makes the global object uniquely determined by the field parameters.
- **Standardized Roots:** Fixes the subgroup action to the primary $N$-th root, ensuring bit-identical traversals across all platforms.

### 2. Chart Morphisms (Geometric Atlas)
Local "Charts" provide different views of the same algebraic state. We implement explicit **Morphisms** to transition between these views:
- **Trace Reconstruction:** A linear-system solver that recovers the full $n$-bit field state from just $n$ bits of a Trace-stream. This proves that local Trace observations are not underdetermined when sufficient context is available.

### 3. Multi-Chart Consensus (Inference)
Resolves field-inference ambiguities by seeking a single global object that simultaneously satisfies multiple local projections (e.g., both Companion and Trace bit-streams).

## Features

- **Geometric Traverser:** Mathematically guaranteed $[0, N-1]$ traversal using Quotient Geometry. Supports `seek(pos)` and arbitrary range sizes.
- **Advanced Inference:** O(n) recovery of field width and primitive elements using Multi-Chart Consensus.
- **Algebraic Morphisms:** Tools to glue local implementation fragments into a coherent global state.

## Getting Started

### Dependencies
- **NTL:** [https://shoup.net/ntl/](https://shoup.net/ntl/)
- **GMP:** [https://gmplib.org/](https://gmplib.org/)

### Compilation
Use C++17 or later.

```bash
g++ -O3 -std=c++17 lfsr_improved.cpp -o lfsr_improved -lntl -lgmp -pthread
```

### Testing
The `main()` function verifies:
1. **Trace -> State Morphism:** Successful bit-perfect reconstruction.
2. **Canonical Determinism:** Seeded instances produce bit-identical sequences.
3. **Consensus Inference:** Resolving field width from multi-chart observations.

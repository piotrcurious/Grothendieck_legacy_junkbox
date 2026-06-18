# GF(p^n) LFSR Suite: Kan Extension Edition

This suite demonstrates the use of Kan extensions to construct and infer Linear Feedback Shift Registers (LFSRs) over arbitrary Galois fields $GF(p^n)$.

## Categorical Architecture

- **Global Object**: A point in the variety of monic irreducible polynomials $f(x)$ over $\mathbb{F}_p$ of degree $n$, where $x \pmod{f(x)}$ is primitive.
- **Left Kan Extension (Lan)**: The realization of the global object into an atlas of local implementation fragments (Charts). Supported charts: `Companion`, `Matrix`, `Trace`, `Decimation`, `Reciprocal`.
- **Right Kan Extension (Ran)**: The inference of the global object from local bit-stream observations.

## Prerequisites

The suite requires the **NTL (Number Theory Library)** and **GMP (GNU Multi-Precision Library)**.

### Installation (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y libntl-dev libgmp-dev
```

## Compilation

Use a C++17 compatible compiler:

```bash
g++ -O2 -std=c++17 gfq_lfsr.cpp -lntl -lgmp -pthread -o gfq_lfsr
```

## Running

```bash
./gfq_lfsr
```

The output will demonstrate:
1.  **Lan Realization**: Generating consistent bit-streams from different algebraic presentations.
2.  **Ran Inference**: Reconstructing the global field parameters (characteristic $p$, degree $n$) and initial state from observed bit-streams.
3.  **Core Utilities**: High-level tools for range traversal and orbit generation.

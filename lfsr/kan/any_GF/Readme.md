# GF(p^n) LFSR Suite: Kan Extension Edition

This suite provides a robust, mathematically-grounded implementation of Linear Feedback Shift Registers (LFSRs) over arbitrary finite fields $GF(p^n)$. It utilizes a categorical framework based on **Kan Extensions** to bridge the gap between abstract Galois field theory and concrete software realizations.

## 1. Categorical Architecture

The system is structured as a mapping between two fundamental categories:
- **$\mathcal{P}_n$ (Presentations)**: The category of local algebraic descriptions (Companion matrices, Trace maps, Decimation orbits).
- **$\mathcal{L}_n$ (Recurrent Objects)**: The category of global linear recurrences over finite fields.

There is a natural inclusion functor $i: \mathcal{P}_n \hookrightarrow \mathcal{L}_n$.

### The Global Object
A point in the **Variety of Recurrences** $V_{p,n}$ defined by:
- $p \in \text{Primes}$, $n \in \mathbb{Z}^+$.
- $f(x) \in \mathbb{F}_p[x]$ (Monic, Irreducible, $\deg(f)=n$).
- $\alpha \in GF(p^n)^*$ (Primitive generator).

### Left Kan Extension (Lan) — Realization
The **Left Kan Extension** $(\text{Lan}_i F) : \mathcal{L}_n \to \mathbf{Impl}$ constructs the global implementation by gluing all compatible local descriptions.
- **Companion Chart**: Polynomial basis representation.
- **Matrix Chart**: Linear map transition $v_{t+1} = M v_t$ over $\mathbb{F}_p$.
- **Trace Chart**: Orthogonal projection to the base field $\mathbb{F}_p$.
- **Decimation Chart**: Orbit sampling at interval $k$, corresponding to a generator morphism $\alpha \mapsto \alpha^k$.
- **Reciprocal Chart**: Duality transformation via $f(x) \mapsto x^n f(1/x)$.

### Right Kan Extension (Ran) — Inference
The **Right Kan Extension** $(\text{Ran}_j G)$ provides the "Universal Inference Machine". It determines the global object from local observations by satisfying the universal property: any other inference machine must factor through Ran.

Reconstruction strategies:
- **Algebraic Recovery**: Generator identification via $\alpha = s_{i+1} / s_i$.
- **Linear-Algebraic Decoding**: Solving the linear system $Ax = b$ for Trace projections.
- **Variety/Locus Search**: Exhaustive exploration of the variety of irreducible polynomials and the primitive locus when structural information is missing.

## 2. Mathematical Detail

### NTL Context Management
Operating over arbitrary $GF(p^n)$ requires a two-level modular context in NTL:
1. `ZZ_pContext`: Manages the prime characteristic $p$.
2. `ZZ_pEContext`: Manages the extension modulus $f(x)$.

The `FieldContext` class encapsulates both, ensuring that `activate()` restores the exact algebraic environment required for arithmetic. This is critical when comparing objects from different fields (e.g., in inference) or different charts.

### Bit-Perfect Consistency
The **Categorical Gluing Test** verifies that the diagram commutes: specifically, that the Companion chart and Matrix chart, despite having different internal data structures and transition logic, produce bit-perfectly identical projections of the same global orbit.

## 3. Complexity & Performance

The categorical approach optimizes for structural clarity, but involves specific trade-offs:

### Realization (Lan)
- **Time Complexity**: $O(n^3)$ over $\mathbb{F}_p$ for the Matrix Chart (initialization). Orbit sampling is $O(\text{poly}(n) \log p)$.
- **Memory Complexity**: $O(n^2)$ to store the transition matrix and field modulus.

### Inference (Ran)
- **State-level**: $O(\text{poly}(n))$ using algebraic inversion.
- **Trace-level**: $O(n^3)$ using Berlekamp-Massey and linear system solving.
- **Variety Search**: $O(V \cdot P \cdot \text{poly}(n))$, where $V$ is the variety size (irreducible polynomials searched) and $P$ is the primitive locus size. Search is capped to prevent memory explosion.

## 4. Usage

### Installation
The suite requires NTL and GMP:
```bash
sudo apt-get update
sudo apt-get install -y libntl-dev libgmp-dev
```

### Compilation
```bash
g++ -O2 -std=c++17 gfq_lfsr.cpp -lntl -lgmp -pthread -o gfq_lfsr
```

### Running
Executing `./gfq_lfsr` runs the integrated test suite, demonstrating:
1. **Lan Realizations**: Multi-chart generation across $p=2, 3, 5, 97$.
2. **Ran Inferences**: Structural reconstruction from Trace, Reciprocal, and Companion observations.
3. **Categorical Commutativity**: Verification of bit-perfect consistency between algebraic presentations.
4. **Algebraic Certificates**: Determining the minimal uniqueness criteria for field parameter recovery.
5. **Traditional Suite**: Comprehensive range traversal, orbit validation, and reproducibility tests.

# Universal LFSR Fitter over Prime Fields GF(p) using Gröbner Bases and Linear Decoding

This module provides a mathematically rigorous framework for fitting Linear Feedback Shift Register (LFSR) recurrences of order $m$ to observed sequence data over any prime field $\mathbb{F}_p$.

## Key Innovations and Architecture

### 1. Explicit Algebraic Contract
- **Prime Field Requirement**: The modulus $p$ must be a prime integer ($p \ge 2$).
- **Normalized Elements**: All sequence values, initial states, and coefficients are normalized into $\{0, 1, \dots, p-1\}$.
- **Dimension & Length Guards**: Requiring `len(initial_state) == len(coeffs) == m` and rejecting input sequences with `len(seq) < m`.

### 2. Linear Gröbner Generator Extractor & Gauss-Jordan Elimination over GF(p)
Because the recurrence equations $a_{n+m} = \sum_{j=0}^{m-1} c_j a_{n+j} \pmod p$ are linear in the unknowns $c_0, c_1, \dots, c_{m-1}$, the reduced Gröbner basis $G$ consists solely of **linear generators**:
$$g_k(c) = \sum_{j=0}^{m-1} a_{kj} c_j + b_k \equiv 0 \pmod p$$

The module asserts `total_degree(g) <= 1` for all $g \in G$. The generators are decoded into the matrix equation $A c \equiv -b \pmod p$ and solved using **Gauss-Jordan elimination over $\mathbb{F}_p$**.

### 3. Solution Taxonomy (`LFSRSolution`)
Instead of guessing or returning a single brute-force vector, fitting returns an explicit `LFSRSolution` object with three mathematically distinct outcomes:
- **`UNIQUE`**: Exactly one coefficient vector fits the sequence (system has full rank $m$).
- **`UNDERDETERMINED`**: Multiple recurrence vectors fit (system is consistent with rank $< m$). The result provides a **parametric affine family** $c = c_{\text{particular}} + \sum t_i v_i \pmod p$ over $\mathbb{F}_p$.
- **`INCONSISTENT`**: No LFSR of order $m$ fits the sequence (e.g. noisy or non-linear data).

### 4. Canonical Predicate & Polynomial Representations
- **`recurrence_residual(seq, coeffs, m, p)`**: Canonical predicate computing element-wise residuals $( \sum c_j a_{n+j} - a_{n+m} ) \pmod p$.
- **Orientation Converters**: Explicit conversion between recurrence coefficients $(c_0, \dots, c_{m-1})$ for $a_{n+m} = \sum c_j a_{n+j}$ and characteristic polynomials $P(x) = x^m - \sum c_j x^j \pmod p$ or connection polynomials $C(x) = 1 - \sum c_{m-j} x^j \pmod p$.

### 5. State Transition Guards and Iterators
- Enforces validity checks on $c \in \mathbb{F}_p^m$.
- Protects against exponential state-space memory explosion ($p^m$) via a configurable `max_states` guard in `lfsr_state_transition()`, complemented by a lazy generator `lfsr_state_transition_iter()`.

---

## Python Code Example (`universal.py`)

```python
from lfsr.fitter.base1.universal import (
    generate_lfsr_sequence,
    fit_lfsr,
    recurrence_residual,
    coeffs_to_characteristic_poly,
    lfsr_state_transition,
)

# Parameters
p = 3  # GF(3)
m = 3  # Order 3
true_coeffs = (1, 2, 0)
initial_state = (1, 0, 2)

# Generate sequence
seq = generate_lfsr_sequence(true_coeffs, initial_state, total_length=10, modulus=p)

# Fit LFSR recurrence using Gröbner basis + GF(p) RREF
sol = fit_lfsr(seq, m, modulus=p)

print("Status:", sol.status)
print("Particular Solution:", sol.particular)
print("Parametric Family:", sol.parametric_str())

# Check canonical residuals
residuals = recurrence_residual(seq, sol.particular, m, p)
assert all(r == 0 for r in residuals)
```

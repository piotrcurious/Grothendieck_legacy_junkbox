# GF(2) LFSR Fitter using Gröbner Bases and Linear Decoding

This module provides a simple, high-level interface for fitting Linear Feedback Shift Register (LFSR) recurrences over GF(2) (binary arithmetic) using Gröbner bases and linear elimination.

## Recurrence Convention

The binary LFSR recurrence convention is:
$$a_{n+m} = c_0 a_n + c_1 a_{n+1} + \dots + c_{m-1} a_{n+m-1} \pmod 2$$

For example, $x^3 + x + 1$ corresponds to $(c_0, c_1, c_2) = (1, 1, 0)$, giving the recurrence:
$$a_{n+3} = a_n + a_{n+1} \pmod 2$$

## How It Works

1. **Sequence Generation**: `generate_lfsr_sequence(coeffs, initial_state, total_length)` generates the bit stream.
2. **Gröbner Basis Computation**: `fit_lfsr_using_groebner(seq, m)` forms linear equations over $\mathbb{F}_2$ and computes a reduced Gröbner basis modulo 2.
3. **Linear Generator Extractor**: `fit_lfsr(seq, m)` parses the basis generators, asserts `total_degree <= 1`, and solves the resulting system $A c = -b \pmod 2$ via Gauss-Jordan elimination.
4. **Solution Taxonomy (`LFSRSolution`)**:
   - `unique`: Exactly one coefficient vector fits.
   - `underdetermined`: Short or degenerate sequences yield an affine family of valid LFSRs.
   - `inconsistent`: No LFSR of order $m$ fits.
5. **State Transitions**: `lfsr_state_transition(coeffs, m)` builds the $2^m$ state transition dictionary with safety guards against large $m$.

---

## Complete Example Code

```python
from lfsr.fitter.base1.gf2_simple import (
    generate_lfsr_sequence,
    fit_lfsr,
    lfsr_state_transition,
    recurrence_residual,
)

# 1. Generate bit sequence
true_coeffs = (1, 1, 0)
initial_state = (1, 0, 0)
seq = generate_lfsr_sequence(true_coeffs, initial_state, total_length=10)

# 2. Fit LFSR coefficients
sol = fit_lfsr(seq, m=3)
print("Solution status:", sol.status)
print("Fitted coefficients:", sol.particular)

# 3. Canonical residual check
residuals = recurrence_residual(seq, sol.particular, m=3, p=2)
assert all(r == 0 for r in residuals)

# 4. State transition table
transitions = lfsr_state_transition(sol.particular, m=3)
for state, next_state in sorted(transitions.items()):
    print(f"State {state} -> Next state {next_state}")
```

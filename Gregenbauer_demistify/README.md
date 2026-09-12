# Gegenbauer Demystified: Unified Algebraic, Geometric, and Numerical Pipeline

This module implements a compressed algebraic-geometric representation theory and operational computational solver for Gegenbauer polynomials $C_n^{(\lambda)}(x)$ and normalized zonal spherical functions $\phi_n(x)$ on the symmetric space $S^{d-1} \cong SO(d)/SO(d-1)$ ($\lambda = \frac{d-2}{2}$).

---

## 1. Overview & Computational Pipeline

By stripping away redundant physical vocabulary, the theory is compressed into two essential layers:
1. **Algebraic Geometry Layer:** Representation spaces $V_n$ are realized as graded components $R(Q)_n$ of the projective quadric coordinate ring $R(Q) = \mathbb{C}[z_1, \dots, z_d]/(q(z))$, with dimensions governed directly by its Hilbert series $H_{R(Q)}(t) = \frac{1-t^2}{(1-t)^d}$.
2. **Computational Asymptotics Layer:** Numerical evaluation is structured around three exact differential equations and a normalized scalar recurrence operator $M_x$ on $R(Q)$, governing four operational regimes on the $(n, \theta)$-plane.

```
  A. Geometry
  SO(d)/SO(d-1), Q^{d-2} ⊂ ℙ^{d-1}
        │
        ▼
  B. Exact Algebra
  R(Q) = ℂ[z]/(q), V_n = R(Q)_n, H_{R(Q)}(t) = (1-t^2)/(1-t)^d
        │
        ▼
  C. Exact Radial Equations
  Compact Radial / Half-Density / Tangent-Limit
        │
        ▼
  D. Exact Normalized Recurrence
  M_x: ϕ_n ↦ x ϕ_n, ϕ_{n+1} = [(2(n+λ))/(n+2λ)] x ϕ_n - [n/(n+2λ)] ϕ_{n-1}
        │
        ▼
  E. Singular Scaling
  N = n + λ, z = N θ
        │
        ▼
  F. Numerical Regimes
  Endpoint / Overlap / Interior / Direct Recurrence
        │
        ▼
  G. Validation
  Quotient-Algebra Remainder ↔ Scalar Recurrence ↔ Bessel Boundary Layer ↔ Exact Anchors
```

---

## 2. Directory Structure & Key Files

```
Gregenbauer_demistify/
├── README.md                              # Main documentation (this file)
├── repair_plan.md                         # 13-point architectural repair plan
├── theoretical_framework.md              # Complete manuscript & mathematical proofs
├── 1.md                                   # Executive summary & core pipeline formulas
├── gegenbauer_proof.pl                   # Formal assertion-based SWI-Prolog knowledgebase
├── algebraic_geometry_combinatorics.py   # Quotient algebra R(Q), normal form, Hilbert series
├── gegenbauer_asymptotics.py             # Scaled recurrence, WKB, Bessel, phase map classifier
├── computational_layer.py                # Pareto optimization solver across bases and precisions
└── test_gegenbauer.py                    # Pytest test suite (14 unit tests & Prolog bridge)
```

---

## 3. Key Mathematical Formulations

### Quadric Quotient Algebra & Hilbert Series
The degree-$n$ representation space $V_n \cong \mathcal{H}_n(\mathbb{R}^d)$ corresponds to degree-$n$ graded pieces of $R(Q) = \mathbb{C}[z_1, \dots, z_d]/(q)$ where $q = \sum_{i=1}^d z_i^2$:
$$R(Q)_n \cong \operatorname{Sym}^n(\mathbb{C}^d) / q \operatorname{Sym}^{n-2}(\mathbb{C}^d) \cong V_n.$$

Dimension growth is extracted directly from the Hilbert series:
$$H_{R(Q)}(t) = \frac{1 - t^2}{(1 - t)^d} \implies \dim V_n = \binom{n+d-1}{d-1} - \binom{n+d-3}{d-1}.$$

Normalization functional identity:
$$C_n^{(\lambda)}(1) = \frac{\lambda}{n + \lambda} \dim V_n.$$

### Three Exact Backbone Equations
1. **Compact Radial Equation:** $\phi'' + 2\lambda \cot\theta \, \phi' + n(n+2\lambda)\phi = 0$.
2. **Half-Density Equation ($u_n = (\sin\theta)^\lambda \phi_n$):** $-u'' + \lambda(\lambda-1)\csc^2\theta \, u = N^2 u$, where $N = n + \lambda$.
3. **Tangent-Limit Equation ($\theta = z/N$):** $\Phi'' + \frac{2\lambda}{z}\Phi' + \Phi = 0 \implies \Phi(z) = \mathcal{J}_{\lambda-1/2}(z)$.

### Normalized Recurrence Operator $M_x$
The degree-shifting multiplication operator $M_x \phi_n = x \phi_n$ satisfies:
$$M_x \phi_n = a_n \phi_{n+1} + b_n \phi_{n-1}, \qquad a_n = \frac{n + 2\lambda}{2(n + \lambda)}, \quad b_n = \frac{n}{2(n + \lambda)}, \quad a_n + b_n = 1.$$

Production Scaled Recurrence:
$$\phi_{n+1}(x) = \frac{2(n + \lambda)}{n + 2\lambda} x \phi_n(x) - \frac{n}{n + 2\lambda} \phi_{n-1}(x).$$

---

## 4. Operational Phase Diagram & Pareto Solver

The computational layer models 6 algebraic geometry expression permutations on the Computational Cost (FLOPs) vs. Relative Numerical Error plane:
1. **Normalized Three-Term Recurrence $\phi_n(x)$**
2. **Quotient Ring Normal Form Polynomial Remainder**
3. **Hypergeometric $_2F_1$ Series Expansion**
4. **Interior WKB / Weyl Semiclassical Wave**
5. **Mehler-Heine Bessel Boundary-Layer Kernel**
6. **Composite Matched Asymptotic Expansion**

Numerical contexts support IEEE Binary Base 2, Decimal Base 10, Fixed-Point Q16.16, Logarithmic Number Systems (LNS), and precisions (`float32`, `float64`, `float128`, `mpmath`).

---

## 5. Running Tests and Formal Proofs

### SWI-Prolog Formal Proof Verification
To verify the assertion-based logical proofs in SWI-Prolog:
```bash
swipl -g "run_all_proofs" -t halt Gregenbauer_demistify/gegenbauer_proof.pl
```

### Python Unit Test Suite
To run the 14 pytest unit tests covering Prolog assertions, quotient ring normal forms, Hilbert series growth, exact test anchors ($S^2, S^3, S^4$), phase diagram map selection, and the Pareto solver:
```bash
PYTHONPATH=. pytest -v Gregenbauer_demistify/test_gegenbauer.py
```

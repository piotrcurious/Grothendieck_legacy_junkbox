# VIII-Layer Unified Computational Framework for Gegenbauer Polynomials and Spherical Harmonics on $SO(d)/SO(d-1)$

This repository implements a mathematically closed, VIII-Layer unified architectural framework and computational solver for Gegenbauer polynomials $C_n^{(\lambda)}(x)$ and normalized zonal spherical functions $\phi_n(x)$ on the symmetric space $S^{d-1} \cong SO(d)/SO(d-1)$ ($\lambda = \frac{d-2}{2}, d \ge 3$).

---

## 1. VIII-Layer Architectural Pipeline

```
  Layer I. Representation Geometry & Fischer Decomposition
  SO(d)/SO(d-1), Sym^n = ℋ_n ⊕ q Sym^{n-2}, Q^{d-2} ⊂ ℙ^{d-1}, R(Q)_n ≅ ℋ_n(ℂ^d)
        │
        ▼
  Layer II. Spherical Fixed Line, Rank-One Projector & Bi-K-Invariance
  v_n ∈ V_n^K, ||v_n|| = 1  ⟹  P_K = P_n = v_n ⊗ v_n^*  ⟹  ϕ_n(g) = Tr(P_n π_n(g)) ∈ C^∞(K\G/K) ≅ C^∞([-1, 1])
        │
        ▼
  Layer III. Exact Operator Equivalence & Schrödinger Eigenvalues
  -Δ_{S^{d-1}} ϕ_n = E_n ϕ_n  |  L_x ↔ L_θ ↔ H_λ u_n = N_n^2 u_n,  N_n^2 = E_n + λ^2
        │
        ▼
  Layer IV. Jacobi Spectral Operator & Unitary Matrix Realization
  (M_x f)(x) = x f(x),  U M_x U^{-1} = J = J^*,  ||J|| = 1,  α_n = 1/2 - λ(λ-1)/(4n^2) + O(n^{-3})
        │
        ▼
  Layer V. Two-Endpoint Boundary Coordinates
  N = n + λ,  z_+ = N θ,  z_- = N (π - θ)
        │
        ▼
  Layer VI. Composite Matched Asymptotic Framework & Candidate Envelopes
  F_comp = F_north + F_south + F_interior - F_{+,overlap} - F_{-,overlap},  B_K^best = min_{M} B_{K,M}
        │
        ▼
  Layer VII. Modular & Multi-Backend Arithmetic Execution Layer
  ├── VII-A: Floating-Point & Fixed-Point (FLOAT32, FLOAT64, LONGDOUBLE, Q16.16, LNS)
  ├── VII-B: Exact Rational Symbolic Algebra (Q[λ, x], RatCert, Fraction Recurrence)
  ├── VII-C: Scalable Residue Number System (RNS / CRT with A-Priori Magnitude Bounds)
  ├── VII-D: Finite-Field & NTT Specializations (F_p, Primitive Roots ω_{L_NTT}, p > max(2, N_max))
  └── VII-E: Golub-Welsch Spectral Matrix Truncation (J_m = tridiag(α_0, ..., α_{m-2}))
        │
        ▼
  Layer VIII. Verification Invariants, Error Taxonomy & Certification Layer
  Taxonomy (R_structural, E_forward, κ, E_backend) ↔ Exact Anchors ϕ_n^{(k)}(±1) ↔ Cross-Backend E_{A,B}^R / C_{A,B}^{(p)}
```

---

## 2. Directory Structure & Key Files

```
Gregenbauer_demistify/
├── README.md                              # Main documentation (this file)
├── repair_plan.md                         # 13-point architectural repair plan
├── 1_repair.md                            # 17-point layer refinement plan
├── 2_repair.md                            # 22-point audit repair plan
├── 3_repair.md                            # Executive repair & layer specification
├── REPAIR_AUDIT.md                        # Full 52-point audit verification report
├── theoretical_framework.md              # Complete manuscript & mathematical proofs
├── 1.md                                   # Executive summary & core pipeline formulas
├── gegenbauer_proof.pl                   # Formal assertion-based SWI-Prolog knowledgebase
├── algebraic_geometry_combinatorics.py   # Quotient algebra R(Q), exact Q[λ,x], RNS/CRT, Golub-Welsch
├── gegenbauer_asymptotics.py             # Scaled recurrence, WKB, Bessel, phase map classifier
├── computational_layer.py                # Pareto optimization solver across bases, capability certs, and precisions
└── test_gegenbauer.py                    # Pytest test suite (30 unit tests & Prolog bridge)
```

---

## 3. Key Mathematical Formulations & Certificates

### Quadric Quotient Algebra & Homogeneous Coordinate Ring
The degree-$n$ representation space $V_n \cong \mathcal{H}_n(\mathbb{R}^d)$ corresponds to degree-$n$ graded pieces of the homogeneous coordinate ring $R(Q) = \mathbb{C}[z_1, \dots, z_d]/(q)$ where $q = \sum_{i=1}^d z_i^2$:
$$R(Q)_n \cong \operatorname{Sym}^n(\mathbb{C}^d) / q \operatorname{Sym}^{n-2}(\mathbb{C}^d) \cong V_n, \qquad H_{R(Q)}(t) = \frac{1 - t^2}{(1 - t)^d} \implies \dim V_n = \binom{n+d-1}{d-1} - \binom{n+d-3}{d-1}.$$

### Exact Rational Algebra $\mathbb{Q}[\lambda, x]$ vs. Jacobi Spectral Path $\overline{\mathbb{Q}}$
- **Exact Polynomial Path:** Evaluates $C_n^{(\lambda)}(x) \in \mathbb{Q}$ using three-term recurrence over reduced fraction inputs $\operatorname{RatCert} = (a, b, c, d, N_{\max})$ where $\lambda = a/b, x = c/d$.
- **Jacobi Spectral Path:** Evaluates $J_m = \operatorname{tridiag}(\alpha_0, \dots, \alpha_{m-2}) \in \mathbb{R}^{m \times m}$ operating in algebraic extensions $\overline{\mathbb{Q}}$ due to $\alpha_n = \frac{1}{2}\sqrt{\frac{(n+1)(n+2\lambda)}{(n+\lambda)(n+\lambda+1)}} = \frac{1}{2} - \frac{\lambda(\lambda-1)}{4n^2} + O(n^{-3})$.

### Modular & RNS/CRT Admissibility Certificates
- **Polynomial Certificate $\mathcal{A}_C(p)$:** Requires prime $p > \max(2, N_{\max})$, $p \nmid b$, and $p \nmid d$.
- **Normalized Spherical Certificate $\mathcal{A}_\phi(p, n)$:** Requires $\mathcal{A}_C(p) \land (p \nmid C_n^{(\lambda)}(1))$.
- **Conditional Exactness Chain:** $\mathcal{A}_{\text{rec}}(m_i) \land \mathcal{A}_{\text{norm}}(m_i) \land \mathcal{A}_{\text{CRT}}(m_i) \implies \varepsilon_A^{\text{arithmetic}} = \varepsilon_A^{\text{forward}} = 0$.

### Layer VIII Verification Taxonomy & Invariants
- **Error Taxonomy:** Distinguishes $R_{\text{structural}}$ (equation residuals), $E_{\text{forward}}$ ($|\hat{\phi}-\phi|$), $\kappa$ (conditioning), and $E_{\text{backend}}$ (discrepancies).
- **Scale-Invariant Residuals:**
  - Recurrence ($n \ge 1$): $\widehat{R}_{\text{rec}}(n, x) = \frac{|x \hat{\phi}_n - a_n \hat{\phi}_{n+1} - b_n \hat{\phi}_{n-1}|}{|x \hat{\phi}_n| + |a_n \hat{\phi}_{n+1}| + |b_n \hat{\phi}_{n-1}| + \tau}$.
  - Interior ODE ($-1 < x < 1$): $\widehat{R}_{\text{ODE}}(x) = \frac{|(1-x^2)\hat{\phi}'' - (2\lambda+1)x\hat{\phi}' + E_n\hat{\phi}|}{|1-x^2||\hat{\phi}''| + |(2\lambda+1)x||\hat{\phi}'| + E_n|\hat{\phi}| + \tau}$.
  - Interior Schrödinger ($0 < \theta < \pi$): $\widehat{R}_{\text{Schr}}(\theta) = \frac{|-\hat{u}'' + \lambda(\lambda-1)\csc^2\theta \, \hat{u} - N_n^2 \hat{u}|}{|\hat{u}''| + |\lambda(\lambda-1)\csc^2\theta \, \hat{u}| + N_n^2 |\hat{u}| + \tau}$.
- **High-Order Endpoint Derivative Formulas:** $\phi_n^{(k)}(1) = \frac{2^k (\lambda)_k C_{n-k}^{(\lambda+k)}(1)}{C_n^{(\lambda)}(1)}$, $\phi_n^{(k)}(-1) = (-1)^{n-k} \phi_n^{(k)}(1)$, and $\phi_n^{(k)} \equiv 0$ for $k > n$.
- **Gauss-Gegenbauer Moment Invariants:** $\sum_{k=1}^m w_k x_k^j = \int_{-1}^1 x^j (1-x^2)^{\lambda-1/2} dx = \begin{cases} 0, & j \text{ is odd}, \\ B(r+1/2, \lambda+1/2), & j = 2r \text{ is even}. \end{cases}$

---

## 4. Running Tests and Formal Proofs

### SWI-Prolog Formal Proof Verification
To verify assertion-based logical proofs in SWI-Prolog:
```bash
swipl -g "run_all_proofs" -t halt Gregenbauer_demistify/gegenbauer_proof.pl
```

### Python Unit Test Suite
To run the 30 pytest unit tests covering Prolog assertions, quotient ring normal forms, Hilbert series growth, exact test anchors ($S^2, S^3, S^4$), phase diagram map selection, high-precision reference convergence ($p_{\text{ref}} \ge 384$ bits), exact rational bit-lengths, RNS/CRT integer recovery, and Pareto optimization solver:
```bash
PYTHONPATH=. pytest -v Gregenbauer_demistify/test_gegenbauer.py
```

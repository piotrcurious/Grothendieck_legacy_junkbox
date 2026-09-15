# VIII-Layer Unified Computational Framework for Gegenbauer Polynomials and Spherical Harmonics on $SO(d)/SO(d-1)$

This repository implements a mathematically closed, VIII-Layer unified architectural framework and computational solver for Gegenbauer polynomials $C_n^{(\lambda)}(x)$, normalized zonal spherical functions $\phi_n(x)$, and orthonormal Jacobi basis functions $e_n(x)$ on the symmetric space $S^{d-1} \cong SO(d)/SO(d-1)$ ($\lambda = \frac{d-2}{2}, d \ge 3$).

---

## 1. VIII-Layer Architectural Pipeline

```
  Layer I. Representation Geometry & Fischer Decomposition
  G = SO(d), K = SO(d-1), Sym^n(ℂ^d) = ℋ_n(ℂ^d) ⊕ q Sym^{n-2}(ℂ^d), Q^{d-2} ⊂ ℙ^{d-1}, R(Q)_n ≅ ℋ_n(ℂ^d), Res_S: ℋ_n(ℂ^d) ──∼──→ 𝒴_n^ℂ(S^{d-1})
        │
        ▼
  Layer II. Spherical Fixed Line, Rank-One Projector & Bi-K-Invariance
  V_n^K = ℂ v_n, ||v_n|| = 1  ⟹  P_{K,n} = v_n ⊗ v_n^*  ⟹  x(g) = ⟨g e_d, e_d⟩ ∈ [-1, 1], ϕ_n(g) = ⟨v_n, π_n(g) v_n⟩, |ϕ_n(x)| ≤ 1
        │
        ▼
  Layer III. Exact Differential Operators, Normalization Types & Dual Recurrences
  Types: C_n^{(λ)}(x) [Poly] | ϕ_n(x) = C_n/C_n(1) [Zonal, ϕ_n(1)=1] | e_n(x) = h_n ϕ_n(x) [Orthonormal, ||e_n||_λ=1]
  Domains & Conjugation: L_x (x ∈ (-1,1)) ──x=cos θ──> L_θ (θ ∈ (0,π)) ──u=(sin θ)^λ ϕ──> H_λ u_n = N_n^2 u_n (Singular SL at 0,π for λ∈(0,1))
  Dual Recurrences: a_n, b_n ⟷ α_n via α_n^2 = a_n b_{n+1} = (n+1)(n+2λ) / [4(n+λ)(n+λ+1)] (Exact Symbolic Identity I_dual)
        │
        ▼
  Layer IV. Jacobi Spectral Operator & Unitary Matrix Realization
  (M_x f)(x) = x f(x),  U M_x U^{-1} = J = J^*,  ||J|| = 1,  α_n = 1/2 + λ(1-λ)/(4 n^2) + O(n^{-3}) as n → ∞
        │
        ▼
  Layer V. Two-Endpoint Boundary Coordinates
  N_n = n + λ,  z_+ = N_n θ,  z_- = N_n (π - θ)
        │
        ▼
  Layer VI. Two-Overlap Composite Uniform Asymptotic Schema & Selector
  Composite Approximation: F_comp = F_north + F_south + F_interior - F_{+O} - F_{-O}
  Evaluation Selector: M^* = argmin_{M ∈ ℳ_valid} B_M  |  MatchCert_K Schema ⇏ RemCert_K Schema
        │
        ▼
  Layer VII. Modular & Multi-Backend Arithmetic Execution Layer
  ├── VII-A: Floating-Point & Fixed-Point (FLOAT32, FLOAT64, LONGDOUBLE, C_fixed, C_LNS)
  ├── VII-B: Exact Rational Symbolic Algebra (Q[λ, x] ──symbolic rec──> C_n ──eval──> Q ──CRT/RNS──> integer residues)
  ├── VII-C: Scalable RNS / CRT (Execution metadata N_max, CRT Certificate C_CRT = A_denom ∧ B_num/den ∧ (2UV < M) ∧ gcd(u,v)=1)
  ├── VII-D1: Finite-Field Arithmetic (Split: C_n cert vs ϕ_n cert requiring p ∤ u_n for bad primes)
  ├── VII-D2: NTT Acceleration Primitive (L_conv = L_1+L_2-1 ≤ L_NTT | (p-1))
  └── VII-E: Golub-Welsch Spectral Matrix Truncation (J_m = tridiag(α_0, ..., α_{m-2}), R_J^abs, R_J_hat)
        │
        ▼
  Layer VIII. Executable Residual Taxonomy, Four Truth Classes & Provenance Optimizer
  Truth Classes: (ALGEBRAIC_EXACT, ARITHMETIC_EXACT, ANALYTIC_CERTIFIED, NUMERICAL_APPROX)
  Exactness Semantics: C_A^exact ⟹ Zero execution error relative to specified exact algorithm.
  Decomposition: E_total ≤ E_analytic + E_arithmetic + E_conditioning + E_implementation
  Residual Schema: Residual(type, domain, scale, absolute, normalized, conditioning, backend, theorem_status, tau_M)
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
The degree-$n$ representation space $V_n \cong \mathcal{H}_n(\mathbb{C}^d)$ corresponds to degree-$n$ graded pieces of the homogeneous coordinate ring $R(Q) = \mathbb{C}[z_1, \dots, z_d]/(q)$ where $q = \sum_{i=1}^d z_i^2$:
$$R(Q)_n \cong \operatorname{Sym}^n(\mathbb{C}^d) / q \operatorname{Sym}^{n-2}(\mathbb{C}^d) \cong \mathcal{H}_n(\mathbb{C}^d), \qquad H_{R(Q)}(t) = \frac{1 - t^2}{(1 - t)^d} \implies \dim \mathcal{H}_n(\mathbb{C}^d) = \binom{n+d-1}{d-1} - \binom{n+d-3}{d-1}.$$

### Distinct Object Types
1. Unnormalized Gegenbauer polynomial $C_n^{(\lambda)}(x)$.
2. Normalized zonal spherical function $\phi_n(x) = C_n^{(\lambda)}(x) / C_n^{(\lambda)}(1)$ ($\phi_n(1)=1$).
3. Orthonormal Jacobi basis function $e_n(x) = h_n \phi_n(x)$ where $h_n = \|\phi_n\|_\lambda^{-1}$ ($\|e_n\|_\lambda = 1$).

### Exact Rational Algebra $\mathbb{Q}[\lambda, x]$ vs. Jacobi Spectral Path $\overline{\mathbb{Q}}$
- **Exact Polynomial Path:** Evaluates $C_n^{(\lambda)}(x) \in \mathbb{Q}$ using three-term recurrence over reduced fraction inputs $\mathcal{A}_{\text{rat}} = \operatorname{RatCert} \land (\gcd(a,b)=\gcd(c,d)=1) \land (b,d>0) \land (C_n^{(\lambda)}(1) \neq 0)$ where $\lambda = a/b, x = c/d$.
- **Jacobi Spectral Path:** Evaluates $J_m = \operatorname{tridiag}(\alpha_0, \dots, \alpha_{m-2}) \in \mathbb{R}^{m \times m}$ operating in algebraic extensions $\overline{\mathbb{Q}}$ due to $\alpha_n = \frac{1}{2}\sqrt{\frac{(n+1)(n+2\lambda)}{(n+\lambda)(n+\lambda+1)}} = \frac{1}{2} + \frac{\lambda(1-\lambda)}{4n^2} + O(n^{-3})$ as $n \to \infty$.

### Modular & RNS/CRT Admissibility Certificates
- **Polynomial Certificate $\mathcal{A}_C(p)$:** Requires prime $p > \max(2, N_{\max})$, $p \nmid b$, and $p \nmid d$. Maximum degree $N_{\max}$ is defined as execution metadata.
- **Normalized Spherical Certificate $\mathcal{A}_\phi(p, n)$:** Requires $\mathcal{A}_C(p) \land (p \nmid v_n) \land (u_n \not\equiv 0 \pmod p)$ where $C_n^{(\lambda)}(1) = u_n/v_n$ ($\gcd(u_n, v_n) = 1$). For bad primes ($p \mid u_n$), normalization fails in $\mathbb{F}_p$.
- **Exactness Semantics:** Algebraic / Arithmetic Exactness $\implies$ Zero execution error relative to specified exact algorithm. Total error $E_{\text{total}} \le E_{\text{analytic}} + E_{\text{arithmetic}} + E_{\text{conditioning}} + E_{\text{implementation}}$.

---

## 4. Running Tests and Formal Proofs

### SWI-Prolog Formal Proof Verification
To verify assertion-based logical proofs in SWI-Prolog:
```bash
swipl -g "run_all_proofs" -t halt Gregenbauer_demistify/gegenbauer_proof.pl
```

### Python Unit Test Suite
To run the pytest unit tests covering Prolog assertions, quotient ring normal forms, Hilbert series growth, exact test anchors ($S^2, S^3, S^4$), phase diagram map selection, high-precision reference convergence ($p_{\text{ref}} \ge 384$ bits), exact rational bit-lengths, RNS/CRT integer recovery, and Pareto optimization solver:
```bash
PYTHONPATH=. pytest -v Gregenbauer_demistify/test_gegenbauer.py
```

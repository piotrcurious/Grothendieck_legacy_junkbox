# VIII-Layer Unified Computational Framework for Gegenbauer Polynomials and Spherical Harmonics on $SO(d)/SO(d-1)$

This repository implements a mathematically closed, VIII-Layer unified architectural framework and computational solver for Gegenbauer polynomials $C_n^{(\lambda)}(x)$ and normalized zonal spherical harmonics on $\mathbb{S}^{d-1}$.

---

## 1. VIII-Layer Architectural Pipeline

```
  Layer I. Representation Geometry & Fischer Decomposition
  G = SO(d), K = SO(d-1), Sym^n = ℋ_n ⊕ q Sym^{n-2}
  Q^{d-2} ⊂ ℙ^{d-1}, R(Q)_n ≅ ℋ_n(ℂ^d), Res_S: ℋ_n(ℂ^d) ──∼──→ 𝒴_n^ℂ(S^{d-1})
        │
        ▼
  Layer II. Spherical Fixed Line, Rank-One Projector & Bi-K-Invariance
  v_n ∈ V_n^K, ‖v_n‖ = 1  ⟹  P_K = P_n = v_n ⊗ v_n^*
  ϕ_n is K-bi-invariant, radial representative ϕ_n(x) ∈ C^∞([-1, 1])
  x = cos θ, |ϕ_n(x)| ≤ 1
        │
        ▼
  Layer III. Exact Operator Equivalence & Schrödinger Eigenvalues
  -Δ_{S^{d-1}} ϕ_n = E_n ϕ_n  |  L_x ↔ L_θ ↔ H_λ u_n = N_n^2 u_n
  N_n = n + λ  |  Dual Recurrences a_n, b_n ⟷ α_n
        │
        ▼
  Layer IV. Jacobi Spectral Operator & Unitary Matrix Realization
  (M_x f)(x) = x f(x),  U M_x U^{-1} = J = J^*,  ‖J‖ = 1
  α_n = 1/2 + O(n^{-2}) as n → ∞
        │
        ▼
  Layer V. Two-Endpoint Boundary Coordinates
  N_n = n + λ,  z_+ = N_n θ,  z_- = N_n (π - θ)
        │
        ▼
  Layer VI. Composite Matched Asymptotic Framework & Candidate Envelopes
  F_comp = F_north + F_south + F_interior - F_{+,overlap} - F_{-,overlap}
  F_{±O}^{(K)} = Match^{(K)}(F_ep, F_int) mod O(N_n^{-K})
  B_K^best = min_{M} B_{K,M}
        │
        ▼
  Layer VII. Modular & Multi-Backend Arithmetic Execution Layer
  ├── VII-A: Floating-Point & Fixed-Point (FLOAT32, FLOAT64, LONGDOUBLE, Q16.16, LNS)
  ├── VII-B: Exact Rational Symbolic Algebra (Q[λ, x], RatCert, Fraction Recurrence)
  ├── VII-C: Scalable Residue Number System (RNS / CRT Certificate C_CRT = A_denom ∧ B_num/den ∧ (2UV < M))
  ├── VII-D1: Finite-Field Polynomial Arithmetic (F_p, A_ϕ(p, n), Rational Reduction ρ_p)
  ├── VII-D2: Number Theoretic Transform Acceleration Primitive (NTT: L_conv = L_1+L_2-1 ≤ L_NTT | (p-1) → fast conv)
  └── VII-E: Golub-Welsch Spectral Matrix Truncation (J_m = tridiag(α_0, ..., α_{m-2}))
        │
        ▼
  Layer VIII. Verification Invariants, Error Taxonomy & Certification Layer
  Taxonomy (R_structural, E_forward, κ, E_backend)
  ↔ Exact Anchors ϕ_n^{(k)}(±1)
  ↔ Spectral Residuals R_J^abs / R_J_hat
  ↔ Cross-Backend E_{A,B}^R / C_{A,B}^{(p)}
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

The degree-$n$ representation space $V_n \cong \mathcal{H}_n(\mathbb{R}^d)$ corresponds to degree-$n$ graded pieces of the homogeneous coordinate ring $R(Q) = \mathbb{C}[z_1, \dots, z_d]/(q)$ where $q$ is a non-degenerate quadratic form.

$$R(Q)_n \cong \operatorname{Sym}^n(\mathbb{C}^d) / q \operatorname{Sym}^{n-2}(\mathbb{C}^d) \cong V_n$$

$$H_{R(Q)}(t) = \frac{1 - t^2}{(1 - t)^d} \implies \dim V_n = \binom{n+d-1}{d-1} - \binom{n+d-3}{d-1}$$

### Exact Rational Algebra $\mathbb{Q}[\lambda, x]$ vs. Jacobi Spectral Path $\overline{\mathbb{Q}}$

- **Exact Polynomial Path:** Evaluates $C_n^{(\lambda)}(x) \in \mathbb{Q}$ using three-term recurrence over reduced fraction inputs. Certification condition:
$$\mathcal{A}_{\text{rat}} = \operatorname{RatCert} \land \left(\gcd(a,b) = 1\right) \land (p \nmid b \text{ for all primes } p)$$

- **Jacobi Spectral Path:** Evaluates $J_m = \operatorname{tridiag}(\alpha_0, \dots, \alpha_{m-2}) \in \mathbb{R}^{m \times m}$ operating in algebraic extensions $\overline{\mathbb{Q}}$ due to $\alpha_n \in \overline{\mathbb{Q}}$.

### Modular & RNS/CRT Admissibility Certificates

- **Polynomial Certificate $\mathcal{A}_C(p)$:** Requires prime $p > \max(2, N_{\max})$, $p \nmid b$, and $p \nmid d$. Maximum degree $N_{\max}$ is defined as the maximum evaluated polynomial degree.

- **Normalized Spherical Certificate $\mathcal{A}_\phi(p, n)$:** Requires
$$\mathcal{A}_C(p) \land (p \nmid v_n) \land (u_n \not\equiv 0 \pmod p)$$
where $C_n^{(\lambda)}(1) = u_n/v_n$ with $\gcd(u_n, v_n) = 1$.

- **Conditional Exactness Chain:**
$$\mathcal{C}_A^{\text{exact}} \implies \varepsilon_A^{\text{certified}} = \varepsilon_A^{\text{forward}} = 0$$

### Layer VIII Verification Taxonomy & Invariants

- **Error Taxonomy:** Distinguishes four sources:
  - $R_{\text{structural}}$: equation residuals
  - $E_{\text{forward}}$: propagated error $|\hat{\phi}-\phi|$
  - $\kappa$: condition number
  - $E_{\text{backend}}$: computational discrepancies

- **Scale-Invariant & Operator Residuals:**

  - **Recurrence** ($n \ge 1$):
$$\widehat{R}_{\text{rec}}(n, x) = \frac{\left|x \hat{\phi}_n - a_n \hat{\phi}_{n+1} - b_n \hat{\phi}_{n-1}\right|}{\left|x \hat{\phi}_n\right| + \left|a_n \hat{\phi}_{n+1}\right| + \left|b_n \hat{\phi}_{n-1}\right| + \epsilon_{\text{floor}}}$$

  - **Interior ODE** ($-1 < x < 1$):
$$\widehat{R}_{\text{ODE}}(x) = \frac{\left|(1-x^2)\hat{\phi}'' - (2\lambda+1)x\hat{\phi}' + E_n\hat{\phi}\right|}{\left|(1-x^2)\right|\left|\hat{\phi}''\right| + \left|(2\lambda+1)x\right|\left|\hat{\phi}'\right| + E_n\left|\hat{\phi}\right| + \epsilon_{\text{floor}}}$$

  - **Interior Schrödinger** ($0 < \theta < \pi$):
$$\widehat{R}_{\text{Schr}}(\theta) = \frac{\left|-\hat{u}'' + \lambda(\lambda-1)\csc^2\theta \, \hat{u} - N_n^2 \hat{u}\right|}{\left|\hat{u}''\right| + \left|\lambda(\lambda-1)\csc^2\theta\right|\left|\hat{u}\right| + N_n^2\left|\hat{u}\right| + \epsilon_{\text{floor}}}$$

  - **Jacobi Spectral Eigenpair Residuals:**
$$R_J^{\text{abs}}(v, x_k) = \|J_m v - x_k v\|$$
$$\widehat{R}_J(v, x_k) = \frac{\|J_m \hat{v} - \hat{x}_k \hat{v}\|}{\|J_m \hat{v}\| + |\hat{x}_k|\|\hat{v}\| + \epsilon_{\text{floor}}}$$

- **High-Order Endpoint Derivative Formulas:**
$$\phi_n^{(k)}(1) = \frac{2^k (\lambda)_k C_{n-k}^{(\lambda+k)}(1)}{C_n^{(\lambda)}(1)}$$
$$\phi_n^{(k)}(-1) = (-1)^{n-k} \phi_n^{(k)}(1)$$
where $(\lambda)_k = \lambda(\lambda+1)\cdots(\lambda+k-1)$ is the Pochhammer symbol.

- **Gauss-Gegenbauer Weight Normalization & Moment Invariants:**
$$\sum_{k=1}^m w_k = \mu_0 = B\left(\tfrac{1}{2}, \lambda+\tfrac{1}{2}\right)$$
$$\sum_{k=1}^m w_k x_k^j = \int_{-1}^1 x^j (1-x^2)^{\lambda-1/2} dx = \begin{cases} 0, & \text{if } j \text{ odd} \\ \text{explicit formula}, & \text{if } j \text{ even} \end{cases}$$

---

## 4. Running Tests and Formal Proofs

### SWI-Prolog Formal Proof Verification

To verify assertion-based logical proofs in SWI-Prolog:

```bash
swipl -g "run_all_proofs" -t halt Gregenbauer_demistify/gegenbauer_proof.pl
```

### Python Unit Test Suite

To run the 30 pytest unit tests covering Prolog assertions, quotient ring normal forms, Hilbert series growth, exact test anchors ($S^2, S^3, S^4$), phase diagram map selection, and high-precision reference values:

```bash
PYTHONPATH=. pytest -v Gregenbauer_demistify/test_gegenbauer.py
```

---

## 5. Implementation Highlights

### Mathematical Rigor & Formal Verification

- **Assertion-based Prolog knowledgebase**: Encodes Fischer decomposition, spherical restriction, and Schrödinger eigenvalue correspondence
- **Exact algebraic closure**: All polynomial evaluations computed over $\mathbb{Q}[\lambda, x]$ with certified denominators
- **Multi-backend error quantification**: Forward error, structural residuals, and cross-backend discrepancies all measured and certified

### Computational Efficiency

- **Modular arithmetic**: RNS/CRT acceleration for large polynomial evaluations
- **Number-theoretic transforms**: NTT-accelerated convolution for FFT-free Fourier analysis
- **Spectral matrix truncation**: Golub-Welsch quadrature and truncated Jacobi matrices for eigenvalue extraction

### Extensibility

The framework supports:
- User-defined dimension $d$ and parameter $\lambda > -1/2$
- Arbitrary precision via multiple backend implementations
- Domain-specific optimizations (e.g., small-dimensional spaces, integer polynomial parameters)

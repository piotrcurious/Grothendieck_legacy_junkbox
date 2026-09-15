# VIII-Layer Unified Computational Framework for Gegenbauer Polynomials and Spherical Harmonics on $SO(d)/SO(d-1)$

## Abstract

This document presents an architecturally closed VIII-Layer framework for Gegenbauer polynomials $C_n^{(\lambda)}(x)$, normalized zonal spherical functions $\phi_n(x)$, and orthonormal Jacobi basis functions $e_n(x)$ on the real sphere $S^{d-1} \cong SO(d)/SO(d-1)$, where physical parameter $\lambda = \frac{d-2}{2} \in \left\{ \frac{1}{2}, 1, \frac{3}{2}, 2, \dots \right\}$ ($d \ge 3$). The framework establishes formal operator morphisms connecting representation geometry, quotient algebras, exact differential operators, self-adjoint Jacobi spectral matrices, two-endpoint singular asymptotic schemas, multi-backend numerical execution (including exact rational, RNS/CRT, and finite-field sub-backends), executable residual taxonomy, typed error bounds, and high-precision verification invariants.

---

## 1. VIII-Layer Architectural Pipeline

```
  Layer I. Representation Geometry & Fischer Decomposition
  G = SO(d), K = SO(d-1), Sym^n(ℂ^d) = ℋ_n(ℂ^d) ⊕ q Sym^{n-2}(ℂ^d), Q^{d-2} ⊂ ℙ^{d-1}, R(Q)_n ≅ ℋ_n(ℂ^d) [via Fischer], Res_S: ℋ_n(ℂ^d) ──∼──→ 𝒴_n^ℂ(S^{d-1})
        │
        ▼
  Layer II. Spherical Fixed Line, Rank-One Projector & Bi-K-Invariance
  V_n^K = ℂ v_n, ||v_n|| = 1  ⟹  P_{K,n} = v_n ⊗ v_n^*  ⟹  Double Coset Map x(g) = ⟨g e_d, e_d⟩ ∈ [-1, 1], ϕ_n(g) = ⟨v_n, π_n(g) v_n⟩, |ϕ_n(x)| ≤ 1
        │
        ▼
  Layer III. Exact Differential Operators, Normalization Types & Dual Recurrences
  Types: C_n^{(λ)}(x) [Poly] | ϕ_n(x) = C_n/C_n(1) [Zonal, ϕ_n(1)=1] | e_n(x) = h_n ϕ_n(x) [Orthonormal, ||e_n||_λ=1]
  Unitary Map: L^2((0,π), (sin θ)^{2λ} dθ) ──u=(sin θ)^λ ϕ──> L^2(0,π)  |  H_λ u_n = N_n^2 u_n
  Physical Sphere Family: d ≥ 3 ⟹ λ ∈ {1/2, 1, 3/2, 2, ...}. Only λ=1/2 (d=3) is subcritical in physical family.
  SL Friedrichs Extension (selecting exponent max(λ, 1-λ) = 1/2 + |λ - 1/2|):
    - Critical λ=1/2 (d=3): u ~ A θ^{1/2} + B θ^{1/2} log θ (Friedrichs B=0 ⟹ u ~ A θ^{1/2})
    - Analytic Continuation 0<λ<1, λ≠1/2: u ~ A θ^λ + B θ^{1-λ} (Friedrichs B=0 for λ>1/2; A=0 for 0<λ<1/2)
  Dual Recurrences: a_n, b_n ⟷ α_n via α_n^2 = a_n b_{n+1} = (n+1)(n+2λ) / [4(n+λ)(n+λ+1)] (Exact Symbolic Identity I_dual)
        │
        ▼
  Layer IV. Jacobi Spectral Operator & Unitary Matrix Realization
  (M_x f)(x) = x f(x),  U M_x U^{-1} = J = J^*  ⟹  ||J|| = 1 and σ(J) = [-1, 1] as consequences
  Self-Contained Spectral Theorem: ||J_m|| < 1 derived via Compactness of S^{m-1} and ⟨J_m v, v⟩ = ∫ x |p_v(x)|^2 w_λ(x) dx < 1
  Subdiagonal Expansion (n → ∞, λ fixed): α_n = 1/2 + λ(1-λ)/(4 n^2) + O(n^{-3})
        │
        ▼
  Layer V. Two-Endpoint Boundary Coordinates
  N_n = n + λ,  z_+ = N_n θ,  z_- = N_n (π - θ)
        │
        ▼
  Layer VI. Two-Overlap Composite Uniform Asymptotic Schema & Quantified Selector
  Composite Approximation Schema: F_comp = F_north + F_south + F_interior - F_{+O} - F_{-O} (Status: MATCHING_SCHEMA)
  Quantified Overlap Contract: |F_endpoint^{(K)} - F_O^{(K)}| ≤ C_{K,λ,Z_0,δ} N_n^{-K} on Z_0 ≤ z_± ≤ δ N_n
  Evaluation Selector: M^*(\theta) = argmin_{M ∈ ℳ_valid} B_M(\theta) requiring ErrorBound.status ∈ {ALGEBRAIC_EXACT, ARITHMETIC_EXACT, ANALYTIC_CERTIFIED, NUMERICAL_CERTIFIED}
        │
        ▼
  Layer VII. Modular & Multi-Backend Arithmetic Execution Layer
  ├── VII-A: Floating-Point & Fixed-Point (FLOAT32, FLOAT64, LONGDOUBLE, C_fixed, C_LNS)
  ├── VII-B: Exact Rational Symbolic Algebra (Q[λ, x] ──symbolic rec──> C_n ──eval──> Q ──CRT/RNS──> integer residues)
  ├── VII-C: Scalable RNS / CRT (N_max := execution metadata, λ=a/b, x=c/r, D_excl = lcm({d_k ∈ D_rec} ∪ {b, r}))
  ├── VII-D1: Finite-Field Arithmetic (PolyCertificate p ∤ D_excl vs ZonalCertificate PolyCertificate ∧ p ∤ u_n v_n)
  ├── VII-D2: NTT Acceleration Primitive (L_conv = L_1+L_2-1 ≤ L_NTT | (p-1))
  └── VII-E: Golub-Welsch Spectral Matrix Truncation (J_m = tridiag(α_0, ..., α_{m-2}), Status: NUMERICAL_CERTIFIED)
        │
        ▼
  Layer VIII. Typed Separation: ExactValue vs ErrorBound vs Residual & Provenance Optimizer
  First-Class Types: ExactValue != ErrorBound != Residual (TheoremStatus Enum: ALGEBRAIC_EXACT, ARITHMETIC_EXACT, ANALYTIC_CERTIFIED, NUMERICAL_CERTIFIED, EMPIRICAL_DIAGNOSTIC)
  Residual != ErrorBound Invariant; Provenance-Aware Decomposition: E_total ≤ E_analytic + E_arithmetic + E_conditioning + E_implementation with E_conditioning ≤ κ · E_input
```

---

## 2. Layer I: Representation Geometry & Fischer Decomposition

Let $G = SO(d)$ act transitively on $S^{d-1} \subset \mathbb{R}^d$ with isotropy subgroup $K = SO(d-1)$, so that $S^{d-1} \cong G/K$. For ambient Euclidean dimension $d \ge 3$, the complexified null quadric $Q^{d-2} \subset \mathbb{P}^{d-1}$ is defined by:
$$Q^{d-2} = \{ [z] \in \mathbb{P}^{d-1} : q(z) = z_1^2 + \dots + z_d^2 = 0 \}.$$

The coordinate ring of the projective quadric $Q^{d-2}$ is $R(Q) = \mathbb{C}[z_1, \dots, z_d] / (q)$. Via Fischer decomposition on polynomial spaces:
$$\operatorname{Sym}^n(\mathbb{C}^d) = \mathcal{H}_n(\mathbb{C}^d) \oplus q \operatorname{Sym}^{n-2}(\mathbb{C}^d), \qquad \mathcal{H}_n(\mathbb{C}^d) = \{p \in \operatorname{Sym}^n(\mathbb{C}^d) : \Delta_{\mathbb{C}^d} p = 0\},$$
yielding the vector space isomorphism (dependent on Fischer decomposition):
$$R(Q)_n \cong \operatorname{Sym}^n(\mathbb{C}^d) / q \operatorname{Sym}^{n-2}(\mathbb{C}^d) \cong \mathcal{H}_n(\mathbb{C}^d).$$

Here, $R(Q)_n$ is the degree-$n$ graded piece of the quotient ring $R(Q)$, which is isomorphic as an $SO(d)$-module to the space of complex harmonic polynomials $\mathcal{H}_n(\mathbb{C}^d)$. The restriction map to $S^{d-1}$:
$$\boxed{\operatorname{Res}_S : \mathcal{H}_n(\mathbb{C}^d) \xrightarrow{\,\,\sim\,\,} \mathscr{Y}_n^\mathbb{C}(S^{d-1}),}$$
is an isomorphism of $SO(d)$-modules mapping complex harmonic polynomials to complex spherical harmonics $\mathscr{Y}_n^\mathbb{C}(S^{d-1})$.

### Hilbert Series & Representation Dimension
$$\dim \mathcal{H}_n(\mathbb{C}^d) = [t^n] \frac{1 - t^2}{(1 - t)^d} = \binom{n + d - 1}{d - 1} - \binom{n + d - 3}{d - 1} = \frac{n + \lambda}{\lambda} C_n^{(\lambda)}(1),$$
where $\lambda = \frac{d-2}{2}$ and $C_n^{(\lambda)}(1) = \frac{(2\lambda)_n}{n!} = \binom{n + 2\lambda - 1}{n}$.

To ensure uniform validity for every $n \ge 0$ (including $n=0$ and $n=1$), binomial coefficients with upper arguments smaller than the lower argument are defined by:
$$\binom{r}{d-1} = 0 \qquad \text{for } r < d - 1, \qquad \text{or equivalently } \operatorname{Sym}^m(\mathbb{C}^d) = 0 \text{ for } m < 0.$$

---

## 3. Layer II: Spherical Fixed Line, Rank-One Projector & Bi-$K$-Invariance

For the symmetric pair $(G, K) = (SO(d), SO(d-1))$, the pair is a rank-one Gelfand pair, meaning the space of $K$-fixed vectors $V_n^K$ in representation $V_n \cong \mathcal{H}_n(\mathbb{C}^d)$ is 1-dimensional:
$$V_n^K = \mathbb{C} v_n \quad (\|v_n\| = 1).$$

Let $P_{K,n} = \int_K \pi_n(k) dk$ project $V_n$ onto $V_n^K$. Inside representation $V_n$, $P_{K,n}$ equals the rank-one orthogonal projector $P_{K,n} = v_n \otimes v_n^* \in \operatorname{End}(V_n)$.

The normalized zonal spherical matrix coefficient is defined by:
$$\phi_n(g) = \langle v_n, \pi_n(g) v_n \rangle, \qquad \phi_n(e) = 1.$$
The double coset space $K \backslash G / K$ is parametrized by the scalar coordinate:
$$x(g) = \langle g e_d, e_d \rangle = \cos\theta \in [-1, 1],$$
which parametrizes double cosets for isotropy $K = SO(d-1)$ fixing $e_d$. The function $\phi_n$ is $K$-bi-invariant ($\phi_n(k_1 g k_2) = \phi_n(g)$), admitting a radial representative $\phi_n(x) \in C^\infty([-1, 1])$:
$$\phi_n(x) = \frac{C_n^{(\lambda)}(x)}{C_n^{(\lambda)}(1)}.$$

---

## 4. Layer III: Domain Operators, Type System & Singular Sturm-Liouville Extensions

### 4.1 Physical Spherical Parameter Family
For the physical geometric family $S^{d-1} \cong SO(d)/SO(d-1)$ with $d \ge 3$, the parameter range is strictly discrete:
$$\boxed{\text{Physical Sphere Family: } d \ge 3 \implies \lambda = \frac{d-2}{2} \in \left\{ \frac{1}{2}, 1, \frac{3}{2}, 2, \ldots \right\}.}$$
Within the physical spherical family, **only $\lambda = 1/2$ ($d=3$, $S^2$) is subcritical/singular in the limit-circle sense**. The range $0 < \lambda < 1$ ($\lambda \ne 1/2$) represents an analytic continuation in $\lambda$, not part of the physical $SO(d)/SO(d-1)$ ($d \ge 3$) parameter family.

### 4.2 Three Operator Representations & Domain Boundaries
The three differential operators are related by changes of variable and unitary transformations between explicit function spaces:
1. **Algebraic Differential Operator $L_x$:**
   $$L_x = (1 - x^2) \frac{d^2}{dx^2} - (2\lambda + 1)x \frac{d}{dx}, \qquad \text{Domain: } x \in (-1, 1) \subset \mathbb{R}.$$
2. **Compact Radial Operator $L_\theta$:** Under coordinate change $x = \cos\theta$:
   $$L_\theta = \frac{d^2}{d\theta^2} + 2\lambda \cot\theta \frac{d}{d\theta}, \qquad \text{Domain: } \theta \in (0, \pi).$$
3. **Sturm-Liouville Hamiltonian $H_\lambda$:** Under unitary transformation $u_n(\theta) = (\sin\theta)^\lambda \phi_n(\cos\theta)$ mapping:
   $$\boxed{L^2\left((0, \pi), (\sin\theta)^{2\lambda} d\theta\right) \xrightarrow{\quad u = (\sin\theta)^\lambda \phi \quad} L^2(0, \pi),}$$
   $$H_\lambda = -\frac{d^2}{d\theta^2} + \lambda(\lambda - 1)\csc^2\theta, \qquad \text{Domain: } \theta \in (0, \pi).$$
   Eigenvalue equation: $H_\lambda u_n = N_n^2 u_n$, where $N_n := n + \lambda$, so $N_n^2 = E_n + \lambda^2$.

### 4.3 Singular Sturm-Liouville Boundary Analysis & Friedrichs Extension Two-Case Rule
Near $\theta \to 0^+$, the potential $\lambda(\lambda-1)\csc^2\theta$ produces singular asymptotic behaviors:

1. **Critical Physical Case $\lambda = 1/2$ (Sphere $S^2$, $d=3$, potential $-\frac{1}{4}\csc^2\theta$):**
   $$\boxed{u(\theta) = A \, \theta^{1/2} + B \, \theta^{1/2} \log\theta + o(\theta^{1/2}) \qquad (\theta \to 0^+).}$$
   Both $\theta^{1/2}$ and $\theta^{1/2}\log\theta$ are square-integrable near $0$. $H_{1/2}$ has deficiency index $(2,2)$ in $L^2(0, \pi)$. The Friedrichs self-adjoint extension selecting smooth spherical harmonics requires the forbidden logarithmic coefficient to vanish:
   $$\boxed{B = 0 \implies u(\theta) \sim A \, \theta^{1/2}.}$$
2. **Subcritical Case $0 < \lambda < 1, \lambda \ne 1/2$ (Analytic Continuation Range):**
   $$\boxed{u(\theta) = A \, \theta^\lambda + B \, \theta^{1-\lambda} + o(\theta^\lambda) \qquad (\theta \to 0^+).}$$
   The Friedrichs extension selects the regular exponent $\max(\lambda, 1-\lambda) = \frac{1}{2} + \left| \lambda - \frac{1}{2} \right|$:
   $$\boxed{\begin{cases} B = 0, & \lambda > 1/2 \implies u(\theta) \sim A \, \theta^\lambda, \\ A = 0, & 0 < \lambda < 1/2 \implies u(\theta) \sim B \, \theta^{1-\lambda}. \end{cases}}$$

---

## 5. Layer IV: Jacobi Spectral Operator & Self-Contained $\|J_m\| < 1$ Proof

On $\mathscr{H}_\lambda = L^2([-1, 1], (1-x^2)^{\lambda-1/2} dx)$, coordinate multiplication $(M_x f)(x) = x f(x)$ is bounded self-adjoint. By unitary equivalence $J = U M_x U^{-1}$ to $M_x$:
$$\boxed{\|J\| = \|M_x\| = 1 \quad \text{and} \quad \sigma(J) = \sigma(M_x) = [-1, 1] \quad \text{(as mathematical consequences).}}$$

### Self-Contained Proof of Strict Inequality $\|J_m\| < 1$
For any finite principal truncation $J_m = P_m M_x P_m |_{\operatorname{span}\{e_0, \dots, e_{m-1}\}} \in \mathbb{R}^{m \times m}$, let $v \in \mathbb{R}^m$ be a unit vector ($\|v\|_2 = 1$), representing polynomial $p_v(x) = \sum_{k=0}^{m-1} v_k e_k(x) \neq 0$.
Quadratic form evaluation yields:
$$\langle J_m v, v \rangle = \int_{-1}^1 x \, [p_v(x)]^2 (1-x^2)^{\lambda-1/2} dx.$$
Because weight $w_\lambda(x) = (1-x^2)^{\lambda-1/2} > 0$ almost everywhere on $(-1, 1)$ and a non-zero degree-$(m-1)$ polynomial $p_v(x)$ cannot be supported exclusively at $x = \pm 1$:
$$|\langle J_m v, v \rangle| < \int_{-1}^1 1 \cdot [p_v(x)]^2 w_\lambda(x) dx = \|p_v\|_\lambda^2 = \|v\|_2^2 = 1.$$
By compactness of the finite-dimensional unit sphere $S^{m-1} \subset \mathbb{R}^m$, the supremum is strictly attained:
$$\boxed{\|J_m\| = \sup_{v \in S^{m-1}} |\langle J_m v, v \rangle| < 1 \quad \text{for all } m < \infty, \quad \sigma(J_m) \subset (-1, 1).}$$

### Correct Subdiagonal Asymptotic Expansion ($n \to \infty$ with fixed $\lambda$)
$$\boxed{\alpha_n = \frac{1}{2} + \frac{\lambda(1-\lambda)}{4n^2} + O(n^{-3}) \qquad \text{as } n \to \infty \quad (\text{fixed } \lambda).}$$

---

## 6. Layer V & VI: Asymptotic Schemas, Quantified Overlap Contract & Selector

### 6.1 Composite Approximation Schema & Quantified Overlap Contract
The composite uniform expression combines endpoint Bessel layers and interior WKB waves:
$$F_{\text{comp}}^{(K)} = F_{\text{north}}^{(K)}(z_+) + F_{\text{south}}^{(K)}(z_-) + F_{\text{interior}}^{(K)}(N_n, \theta) - F_{+O}^{(K)}(z_+) - F_{-O}^{(K)}(z_-).$$
In Layer VI, $F_{\text{comp}}^{(K)}$ is categorized as `MATCHING_SCHEMA` until explicit analytic remainder majorants are derived.

The overlap re-expansion contract is quantified over intermediate overlap domains $Z_0 \le z_\pm \le \delta N_n$:
$$\boxed{\left| F_{\text{endpoint}}^{(K)} - F_O^{(K)} \right| \le C_{K, \lambda, Z_0, \delta} \, N_n^{-K} \qquad \text{for } Z_0 \le z_\pm \le \delta N_n \quad (0 < \delta < \pi/2).}$$

### 6.2 Certified Evaluation Selector ($M^*$)
Pointwise representation selection minimizes local certified forward error bound $B_M(\theta)$:
$$\boxed{M^*(\theta) = \arg\min_{M \in \mathcal{M}_{\text{valid}}} B_M(\theta), \qquad \text{where } \texttt{ErrorBound}_M\text{.status} \in \{\texttt{ALGEBRAIC\_EXACT}, \texttt{ARITHMETIC\_EXACT}, \texttt{ANALYTIC\_CERTIFIED}, \texttt{NUMERICAL\_CERTIFIED}\}.}$$

---

## 7. Layer VII: Modular & Multi-Backend Arithmetic Execution Layer

### VII-A & VII-B. Hardware & Exact Symbolic Sub-Backends
- **Polynomial Path:** $\mathbb{Q}[\lambda, x] \to \mathbb{Q} \to \text{RNS/CRT}$ (truth class `ARITHMETIC_EXACT`).
- **Jacobi Golub-Welsch Path:** Eigendecomposition of symmetric tridiagonal $J_m$. When backed by a backward-stable eigensolver with validated residual bounds, status is `NUMERICAL_CERTIFIED`.

### VII-C. RNS / CRT Sub-Backend with Recurrence-Derived Exclusions
$N_{\max}$ is defined as **execution-plan metadata**. For rational parameter $\lambda = a/b$ and evaluation point $x = c/r$ (where $r$ is the denominator of point $x$), the excluded prime factor product is generated directly from denominators of executed recurrence steps $\mathcal{D}_{\text{rec}} = \operatorname{denominators}(\{a_n, b_n : \text{executed steps}\})$:
$$\boxed{D_{\text{excl}} = \operatorname{lcm}\left( \{d_k : d_k \in \mathcal{D}_{\text{rec}}\} \cup \{b, r\} \right), \qquad \gcd(m_i, D_{\text{excl}}) = 1.}$$

### VII-D. Finite-Field Polynomial Arithmetic & NTT Sub-Backends
Explicit formal dependency for finite-field certificates:
1. **Unnormalized Polynomial Certificate ($\texttt{PolyCertificate}$):**
   $$\boxed{\texttt{PolyCertificate}(p, n) \iff \left( p > N_{\max} \land p \nmid D_{\text{excl}} \right).}$$
2. **Normalized Zonal Spherical Certificate ($\texttt{ZonalCertificate}$):**
   For canonical reduced fraction $C_n^{(\lambda)}(1) = u_n/v_n$ ($\gcd(u_n, v_n)=1$):
   $$\boxed{\texttt{ZonalCertificate}(p, n) \iff \left( \texttt{PolyCertificate}(p, n) \land p \nmid v_n \land p \nmid u_n \right).}$$
   For bad primes ($p \mid u_n$ or $p \mid v_n$), $C_n^{(\lambda)}(1)$ is non-invertible in $\mathbb{F}_p$, failing normalization.

---

## 8. Layer VIII: Typed Separation, Executable Residuals & Provenance Optimizer

### VIII-A. Typed Hierarchy & Theorem Status Enum
Hard type separation:
$$\boxed{\texttt{ExactValue} \neq \texttt{ErrorBound} \neq \texttt{Residual}}$$

`TheoremStatus` Enum:
$$\boxed{\{\texttt{ALGEBRAIC\_EXACT}, \texttt{ARITHMETIC\_EXACT}, \texttt{ANALYTIC\_CERTIFIED}, \texttt{NUMERICAL\_CERTIFIED}, \texttt{EMPIRICAL\_DIAGNOSTIC}\}}$$

Provenance-aware total computational forward error decomposes as:
$$\boxed{E_{\text{total}} \le E_{\text{analytic}} + E_{\text{arithmetic}} + E_{\text{conditioning}} + E_{\text{implementation}}, \qquad E_{\text{conditioning}} \le \kappa \cdot E_{\text{input}}.}$$

### VIII-B. Provenance Invariant
$$\boxed{\texttt{Residual} \not\implies \texttt{ErrorBound}}$$
Unless a theorem explicitly supplies an implication mapping equation residual to forward error bound, `Residual` objects remain diagnostics and do not participate in certified optimization.

# VIII-Layer Unified Computational Framework for Gegenbauer Polynomials and Spherical Harmonics on $SO(d)/SO(d-1)$

## Abstract

This document presents an architecturally closed VIII-Layer framework for Gegenbauer polynomials $C_n^{(\lambda)}(x)$, normalized zonal spherical functions $\phi_n(x)$, and orthonormal Jacobi basis functions $e_n(x)$ on the real sphere $S^{d-1} \cong SO(d)/SO(d-1)$, where parameter $\lambda = \frac{d-2}{2}$ ($d \ge 3$). The framework establishes formal operator morphisms connecting representation geometry, quotient algebras, exact differential operators, self-adjoint Jacobi spectral matrices, two-endpoint singular asymptotic schemas, multi-backend numerical execution (including exact rational, RNS/CRT, and finite-field sub-backends), executable residual taxonomy, typed error bounds, and high-precision verification invariants.

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
  SL Limit-Circle: λ=1/2 (d=3) ⟹ u ~ A θ^{1/2} + B θ^{1/2} log θ (Friedrichs B=0); 0<λ<1, λ≠1/2 ⟹ u ~ A θ^λ + B θ^{1-λ} (Friedrichs B=0)
  Dual Recurrences: a_n, b_n ⟷ α_n via α_n^2 = a_n b_{n+1} = (n+1)(n+2λ) / [4(n+λ)(n+λ+1)] (Exact Symbolic Identity I_dual)
        │
        ▼
  Layer IV. Jacobi Spectral Operator & Unitary Matrix Realization
  (M_x f)(x) = x f(x),  U M_x U^{-1} = J = J^*  ⟹  ||J|| = 1 and σ(J) = [-1, 1] as consequences; ||J_m|| < 1 derived via J_m = P_m M_x P_m
  Subdiagonal Expansion (n → ∞, λ fixed): α_n = 1/2 + λ(1-λ)/(4 n^2) + O(n^{-3})
        │
        ▼
  Layer V. Two-Endpoint Boundary Coordinates
  N_n = n + λ,  z_+ = N_n θ,  z_- = N_n (π - θ)
        │
        ▼
  Layer VI. Two-Overlap Composite Uniform Asymptotic Schema & Quantified Selector
  Composite Approximation: F_comp = F_north + F_south + F_interior - F_{+O} - F_{-O}
  Quantified Overlap Contract: |F_endpoint^{(K)} - F_O^{(K)}| ≤ C_{K,λ,Z_0,δ} N_n^{-K} on Z_0 ≤ z_± ≤ δ N_n
  Evaluation Selector: M^*(\theta) = argmin_{M ∈ ℳ_valid} B_M(\theta) requiring ErrorBound.status ∈ {VERIFIED_EXACT, ANALYTIC_BOUNDED, NUMERICAL_CERTIFIED}
        │
        ▼
  Layer VII. Modular & Multi-Backend Arithmetic Execution Layer
  ├── VII-A: Floating-Point & Fixed-Point (FLOAT32, FLOAT64, LONGDOUBLE, C_fixed, C_LNS)
  ├── VII-B: Exact Rational Symbolic Algebra (Q[λ, x] ──symbolic rec──> C_n ──eval──> Q ──CRT/RNS──> integer residues)
  ├── VII-C: Scalable RNS / CRT (N_max := execution metadata, Excluded primes p ∤ ∏_{d_k ∈ D_rec} d_k · b · d)
  ├── VII-D1: Finite-Field Arithmetic (PolyCertificate p ∤ ∏_{d_k ∈ D_rec} d_k · b · d vs ZonalCertificate PolyCertificate ∧ p ∤ u_n v_n)
  ├── VII-D2: NTT Acceleration Primitive (L_conv = L_1+L_2-1 ≤ L_NTT | (p-1))
  └── VII-E: Golub-Welsch Spectral Matrix Truncation (J_m = tridiag(α_0, ..., α_{m-2}), Status: NUMERICAL_CERTIFIED)
        │
        ▼
  Layer VIII. Typed Separation: ExactValue vs ErrorBound vs Residual & Provenance Optimizer
  First-Class Types: ExactValue != ErrorBound != Residual (TheoremStatus Enum: ALGEBRAIC_EXACT, ARITHMETIC_EXACT, ANALYTIC_CERTIFIED, NUMERICAL_CERTIFIED, EMPIRICAL_DIAGNOSTIC)
  Residual != ErrorBound Invariant; Decomposition: E_total ≤ E_analytic + E_arithmetic + E_conditioning + E_implementation with E_conditioning ≤ κ · E_input
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

### 4.1 Distinct Objects at Type / API Level
The framework explicitly distinguishes three polynomial/spherical objects at the type level:
1. **Unnormalized Gegenbauer Polynomial $C_n^{(\lambda)}(x)$:** Standard orthogonal polynomial with $C_n^{(\lambda)}(1) = \frac{(2\lambda)_n}{n!}$.
2. **Normalized Zonal Spherical Function $\phi_n(x)$:** $K$-bi-invariant representative with $\phi_n(1) = 1$:
   $$\phi_n(x) = \frac{C_n^{(\lambda)}(x)}{C_n^{(\lambda)}(1)}.$$
3. **Orthonormal Jacobi Basis Element $e_n(x)$:** Orthonormal in $\mathscr{H}_\lambda = L^2([-1, 1], (1-x^2)^{\lambda-1/2} dx)$ with $\|e_n\|_\lambda = 1$:
   $$h_n = \|\phi_n\|_\lambda^{-1}, \qquad e_n(x) = h_n \phi_n(x).$$

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

### 4.3 Singular Sturm-Liouville Boundary Analysis & Friedrichs Self-Adjoint Extension
Near $\theta \to 0^+$, the inverse-square potential $\lambda(\lambda-1)\csc^2\theta$ produces singular asymptotic behaviors depending on $\lambda$:

1. **Critical Case $\lambda = 1/2$ (Sphere $S^2$, $d=3$, potential $-\frac{1}{4}\csc^2\theta$):**
   $$\boxed{u(\theta) = A \, \theta^{1/2} + B \, \theta^{1/2} \log\theta + o(\theta^{1/2}) \qquad (\theta \to 0^+).}$$
   Both $\theta^{1/2}$ and $\theta^{1/2}\log\theta$ are square-integrable near $0$. $H_{1/2}$ has deficiency index $(2,2)$ in $L^2(0, \pi)$. The Friedrichs self-adjoint extension selecting smooth spherical harmonics requires the forbidden logarithmic coefficient to vanish:
   $$\boxed{B = 0 \implies u(\theta) \sim A \, \theta^{1/2}.}$$
2. **Subcritical Case $0 < \lambda < 1, \lambda \ne 1/2$ (Continuous Parameter Range):**
   $$\boxed{u(\theta) = A \, \theta^\lambda + B \, \theta^{1-\lambda} + o(\theta^\lambda) \qquad (\theta \to 0^+).}$$
   Both solutions are in $L^2(0, \pi)$, giving deficiency index $(2,2)$. The Friedrichs extension enforces $B = 0$:
   $$\boxed{B = 0 \implies u(\theta) \sim A \, \theta^\lambda.}$$

---

## 5. Layer IV: Jacobi Spectral Operator & Asymptotic Expansion

On $\mathscr{H}_\lambda$, coordinate multiplication $(M_x f)(x) = x f(x)$ is bounded self-adjoint. By unitary equivalence $J = U M_x U^{-1}$ to $M_x$:
$$\boxed{\|J\| = \|M_x\| = 1 \quad \text{and} \quad \sigma(J) = \sigma(M_x) = [-1, 1] \quad \text{(as mathematical consequences).}}$$

For finite principal truncation $J_m = P_m M_x P_m |_{\operatorname{span}\{e_0, \dots, e_{m-1}\}} \in \mathbb{R}^{m \times m}$, since $M_x$ has no eigenvalues and spectrum $\sigma(M_x) = [-1, 1]$ is continuous, strict inequality $\|J_m\| < 1$ follows as a theorem consequence:
$$\boxed{\|J_m\| < \|M_x\| = 1 \quad \text{for all finite } m < \infty, \quad \sigma(J_m) \subset (-1, 1).}$$

### Correct Subdiagonal Asymptotic Expansion ($n \to \infty$ with fixed $\lambda$)
$$\boxed{\alpha_n = \frac{1}{2} + \frac{\lambda(1-\lambda)}{4n^2} + O(n^{-3}) \qquad \text{as } n \to \infty \quad (\text{fixed } \lambda).}$$

---

## 6. Layer V & VI: Asymptotic Schemas, Quantified Overlap Contract & Selector

### 6.1 Composite Approximation Schema & Quantified Overlap Contract
The composite uniform expression combines endpoint Bessel layers and interior WKB waves:
$$F_{\text{comp}}^{(K)} = F_{\text{north}}^{(K)}(z_+) + F_{\text{south}}^{(K)}(z_-) + F_{\text{interior}}^{(K)}(N_n, \theta) - F_{+O}^{(K)}(z_+) - F_{-O}^{(K)}(z_-).$$

The overlap re-expansion contract is quantified over intermediate overlap domains $Z_0 \le z_\pm \le \delta N_n$:
$$\boxed{\left| F_{\text{endpoint}}^{(K)} - F_O^{(K)} \right| \le C_{K, \lambda, Z_0, \delta} \, N_n^{-K} \qquad \text{for } Z_0 \le z_\pm \le \delta N_n \quad (0 < \delta < \pi/2).}$$

### 6.2 Certified Evaluation Selector ($M^*$)
Pointwise representation selection minimizes local certified forward error bound $B_M(\theta)$:
$$\boxed{M^*(\theta) = \arg\min_{M \in \mathcal{M}_{\text{valid}}} B_M(\theta), \qquad \text{where } \texttt{ErrorBound}_M\text{.status} \in \{\texttt{VERIFIED\_EXACT}, \texttt{ANALYTIC\_BOUNDED}, \texttt{NUMERICAL\_CERTIFIED}\}.}$$

---

## 7. Layer VII: Modular & Multi-Backend Arithmetic Execution Layer

### VII-A & VII-B. Hardware & Exact Symbolic Sub-Backends
- **Polynomial Path:** $\mathbb{Q}[\lambda, x] \to \mathbb{Q} \to \text{RNS/CRT}$ (truth class `ARITHMETIC_EXACT`).
- **Jacobi Golub-Welsch Path:** Eigendecomposition of symmetric tridiagonal $J_m$. When backed by a backward-stable eigensolver with validated residual bounds, status is `NUMERICAL_CERTIFIED`.

### VII-C. RNS / CRT Sub-Backend with Recurrence-Derived Exclusions
$N_{\max}$ is defined as **execution-plan metadata**. The excluded prime factor product is generated directly from denominators of executed recurrence steps $\mathcal{D}_{\text{rec}} = \operatorname{denominators}(\{a_n, b_n : \text{executed steps}\})$:
$$\boxed{D = \prod_{d_k \in \mathcal{D}_{\text{rec}}} d_k \cdot b \cdot d, \qquad \gcd(m_i, D) = 1.}$$

### VII-D. Finite-Field Polynomial Arithmetic & NTT Sub-Backends
Explicit logical dependency for finite-field certificates:
1. **Unnormalized Polynomial Certificate ($\texttt{PolyCertificate}$):**
   $$\boxed{\texttt{PolyCertificate}(p, n) \iff \left( p > N_{\max} \land p \nmid D \right).}$$
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

Total computational forward error decomposes as:
$$\boxed{E_{\text{total}} \le E_{\text{analytic}} + E_{\text{arithmetic}} + E_{\text{conditioning}} + E_{\text{implementation}}, \qquad E_{\text{conditioning}} \le \kappa \cdot E_{\text{input}}.}$$

### VIII-B. Provenance Invariant
$$\boxed{\texttt{Residual} \not\implies \texttt{ErrorBound}}$$
Unless a theorem explicitly supplies an implication mapping equation residual to forward error bound, `Residual` objects remain diagnostics and do not participate in certified optimization.

# VIII-Layer Unified Computational Framework for Gegenbauer Polynomials and Spherical Harmonics on $SO(d)/SO(d-1)$

## Abstract

This document presents an architecturally closed VIII-Layer framework for Gegenbauer polynomials $C_n^{(\lambda)}(x)$, normalized zonal spherical functions $\phi_n(x)$, and orthonormal Jacobi basis functions $e_n(x)$ on the real sphere $S^{d-1} \cong SO(d)/SO(d-1)$, where physical parameter $\lambda = \frac{d-2}{2} \in \left\{ \frac{1}{2}, 1, \frac{3}{2}, 2, \dots \right\}$ ($d \ge 3$). The framework establishes formal operator morphisms connecting representation geometry, quotient algebras, exact differential operators, self-adjoint Jacobi spectral matrices, two-endpoint singular asymptotic schemas, multi-backend numerical execution (including exact rational, RNS/CRT, and finite-field sub-backends), executable residual taxonomy, typed error bounds, and high-precision verification invariants.

---

## 1. VIII-Layer Architectural Pipeline

```
  Layer I. Representation Geometry & Fischer Decomposition
  G = SO(d), K = SO(d-1), Sym^n(ℂ^d) = ℋ_n(ℂ^d) ⊕ q Sym^{n-2}(ℂ^d), Q^{d-2} ⊂ ℙ^{d-1}, R(Q)_n ≅ ℋ_n(ℂ^d) [Fischer Equivariant Map], Res_S: ℋ_n(ℂ^d) ──∼──→ 𝒴_n^ℂ(S^{d-1})
        │
        ▼
  Layer II. Spherical Fixed Line, Rank-One Projector & Bi-K-Invariance
  V_n^K = ℂ v_n, ||v_n|| = 1  ⟹  P_{K,n} = v_n ⊗ v_n^*  ⟹  Double Coset Map x(g) = ⟨g e_d, e_d⟩ ∈ [-1, 1], ϕ_n(g) = ⟨v_n, π_n(g) v_n⟩, |ϕ_n(x)| ≤ 1
        │
        ▼
  Layer III. Exact Differential Operators, Normalization Types & Dual Recurrences
  Initial Data Anchors: ϕ_0(x) = 1, ϕ_1(x) = x, e_n = h_n ϕ_n
  Types: C_n^{(λ)}(x) [Poly] | ϕ_n(x) = C_n/C_n(1) [Zonal, ϕ_n(1)=1] | e_n(x) = h_n ϕ_n(x) [Orthonormal, ||e_n||_λ=1]
  Operator Morphism Chain: L^2([-1,1], (1-x^2)^{λ-1/2} dx) ──(S_λ f)(θ) = f(cos θ)──> L^2((0,π), (sin θ)^{2λ} dθ) ──(T_λ g)(θ) = (sin θ)^λ g(θ)──> L^2(0,π)
  Resulting Morphism: u_n = T_λ S_λ ϕ_n = (sin θ)^λ ϕ_n(cos θ)  |  H_λ u_n = N_n^2 u_n  |  Isometry: ||T_λ S_λ f||_{L^2(0,π)} = ||f||_{ℋ_λ} for arbitrary f
  Precedence Classifier: Classify(λ) = PhysicalClassifier(λ) if λ ∈ PhysicalSphereDomain else AnalyticClassifier(λ)
  Physical Sphere Family: d ≥ 3 ⟹ λ ∈ {1/2, 1, 3/2, 2, ...}. Invariants: ϕ_n(-x) = (-1)^n ϕ_n(x), |ϕ_n(x)| ≤ 1
  SL Friedrichs Extension (selecting exponent max(λ, 1-λ) = 1/2 + |λ - 1/2|):
    - Critical λ=1/2 (d=3): u ~ A θ^{1/2} + B θ^{1/2} log θ (Friedrichs B=0 ⟹ u ~ A θ^{1/2})
    - d=4 (λ=1): Regular Endpoint, H_1 = -∂_θ^2, u ~ A θ + B (Friedrichs B=0)
    - d≥5 (λ≥3/2): Limit-Point (LP): coeff_{+, singular} unused, coeff_{-, singular} unused
  Analytic Continuation 0<λ<1, λ≠1/2: u ~ A θ^λ + B θ^{1-λ} (Friedrichs B=0 for λ>1/2; A=0 for 0<λ<1/2)
  Dual Recurrences: a_n, b_n ⟷ α_n via α_n^2 = a_n b_{n+1} = (n+1)(n+2λ) / [4(n+λ)(n+λ+1)] (Exact Symbolic Identity I_dual)
        │
        ▼
  Layer IV. Jacobi Spectral Operator & Unitary Matrix Realization
  (M_x f)(x) = x f(x),  U M_x U^{-1} = J = J^*  ⟹  ||J|| = 1 and σ(J) = [-1, 1] as consequences; ||J_m|| < 1 derived via compactness
  Orthonormal Jacobi Recurrence Split: n=0: x e_0 - α_0 e_1 = 0; n≥1: x e_n - α_n e_{n+1} - α_{n-1} e_{n-1} = 0
  Subdiagonal Expansion (n → ∞, λ fixed): α_n = 1/2 + λ(1-λ)/(4 n^2) + O(n^{-3})
        │
        ▼
  Layer V. Two-Endpoint Boundary Coordinates
  N_n = n + λ,  z_+ = N_n θ,  z_- = N_n (π - θ)
        │
        ▼
  Layer VI. Two-Overlap Composite Uniform Asymptotic Schema & Quantified Selector
  Target Semantics: F_Q(θ) = TargetValue(Q, θ), F(θ) ≡ F_Q(θ) for fixed Q
  Global Coverage: 𝒟_north ∪ 𝒟_interior ∪ 𝒟_south = 𝒟_global
  Finite Region Table Reconstruct_r: (exact identity, I_r, s_r, 𝒟_r) for NORTH, INTERIOR, SOUTH, NORTH_INTERIOR_OVERLAP, INTERIOR_SOUTH_OVERLAP, GLOBAL_OVERLAP
  Status Lattice: Status(B_comp) = min_⪯ {Status(R_i) : i ∈ I_r}
  Selector Candidate Interface: Candidate { domain 𝒟_M, target Q, status, ErrorBound.valid, B_M(θ) } requiring target Q matching
        │
        ▼
  Layer VII. Modular & Multi-Backend Arithmetic Execution Layer
  ├── VII-A: Floating-Point & Fixed-Point (FLOAT32, FLOAT64, LONGDOUBLE, C_fixed, C_LNS)
  ├── VII-B: Exact Rational Symbolic Algebra (Q[λ, x] ──symbolic rec──> C_n ──eval──> Q ──CRT/RNS──> integer residues)
  ├── VII-C: Scalable RNS / CRT (N_max := metadata, Trace P_rec_inv, D_rec(P) = lcm_{q ∈ P_rec_inv} den_red(q), D_den = lcm(D_rec, D_norm, D_eval))
  ├── VII-D1: Finite-Field Arithmetic (D_norm^{used}(P, ZONAL) = 1; Canonical u_n/v_n, c/r; ZonalAdmissible(p, P, n, x); Bad_zonal := ¬ZonalAdmissible)
  ├── VII-D2: NTT Acceleration Primitive (L_conv = L_1+L_2-1 ≤ L_NTT | (p-1))
  └── VII-E: Golub-Welsch Spectral Truncation (J_m = tridiag(α_0, ..., α_{m-2}), Typed Implication: CertSensitivity_Q(A, κ_Q) ∧ R_Q ≤ B_back ∧ E_{Q,conv} ≤ B_{Q,conv} ⟹ E_Q ≤ κ_Q B_back + B_{Q,conv})
        │
        ▼
  Layer VIII. Typed Separation: ExactValue vs ErrorBound vs Residual & Provenance Optimizer
  First-Class Types: ExactValue != ErrorBound != Residual (TheoremStatus Enum: ALGEBRAIC_EXACT, ARITHMETIC_EXACT, ANALYTIC_CERTIFIED, NUMERICAL_CERTIFIED, EMPIRICAL_DIAGNOSTIC)
  Extended Certificate Tuple Invariant: Every certificate carries (target Q, domain 𝒟, backend, status, validity_conditions)
  Residual != ErrorBound Invariant; Staged Perturbation Chain for Target Q: F_0(Q) ──E_0──> F_1(Q) ──E_1──> ... ──E_{k-1}──> F_k(Q) ⟹ |F_0(Q) - F_k(Q)| ≤ ∑_{i=0}^{k-1} E_i
  Canonical Test Matrix: d ∈ {3, 4, 5}, n ∈ {0, 1, 2, 3} validating initial data anchors, recurrences, and cheap invariants (norm isometry & parity/boundedness)
```

---

## 2. Layer I: Representation Geometry & Fischer Decomposition

Let $G = SO(d)$ act transitively on $S^{d-1} \subset \mathbb{R}^d$ with isotropy subgroup $K = SO(d-1)$, so that $S^{d-1} \cong G/K$. For ambient Euclidean dimension $d \ge 3$, the complexified null quadric $Q^{d-2} \subset \mathbb{P}^{d-1}$ is defined by:
$$Q^{d-2} = \{ [z] \in \mathbb{P}^{d-1} : q(z) = z_1^2 + \dots + z_d^2 = 0 \}.$$

The coordinate ring of the projective quadric $Q^{d-2}$ is $R(Q) = \mathbb{C}[z_1, \dots, z_d] / (q)$. In graded degree $n$:
$$R(Q)_n = \operatorname{Sym}^n(\mathbb{C}^d) / q \operatorname{Sym}^{n-2}(\mathbb{C}^d).$$
Via the Fischer decomposition on polynomial spaces $\operatorname{Sym}^n(\mathbb{C}^d) = \mathcal{H}_n(\mathbb{C}^d) \oplus q \operatorname{Sym}^{n-2}(\mathbb{C}^d)$ associated with Euclidean quadratic form $q(z) = \sum z_i^2$, Fischer decomposition supplies a canonical $SO(d)$-equivariant harmonic representative isomorphism:
$$\boxed{R(Q)_n \xrightarrow{\,\,\sim\,\,} \mathcal{H}_n(\mathbb{C}^d).}$$

The restriction map to $S^{d-1}$:
$$\boxed{\operatorname{Res}_S : \mathcal{H}_n(\mathbb{C}^d) \xrightarrow{\,\,\sim\,\,} \mathscr{Y}_n^\mathbb{C}(S^{d-1}),}$$
is an isomorphism of $SO(d)$-modules mapping complex harmonic polynomials to complex spherical harmonics $\mathscr{Y}_n^\mathbb{C}(S^{d-1})$.

### Hilbert Series & Representation Dimension Conventions
$$\dim \mathcal{H}_n(\mathbb{C}^d) = [t^n] \frac{1 - t^2}{(1 - t)^d} = \binom{n + d - 1}{d - 1} - \binom{n + d - 3}{d - 1} = \frac{n + \lambda}{\lambda} C_n^{(\lambda)}(1),$$
where $\lambda = \frac{d-2}{2}$ and $C_n^{(\lambda)}(1) = \frac{(2\lambda)_n}{n!} = \binom{n + 2\lambda - 1}{n}$.

To ensure uniform validity for every $n \ge 0$ (including $n=0$ and $n=1$), the following separate conventions are specified:
$$\boxed{\binom{r}{d-1} = 0 \quad (r < d - 1), \qquad \operatorname{Sym}^m(\mathbb{C}^d) = 0 \quad (m < 0).}$$

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

**Parity and Boundedness Invariants:**
$$\boxed{\phi_n(-x) = (-1)^n \phi_n(x), \qquad |\phi_n(x)| \le 1 \quad \forall x \in [-1, 1].}$$

---

## 4. Layer III: Domain Operators, Type System & Singular Sturm-Liouville Extensions

### 4.1 Parameter Domains & Explicit Classifier Precedence ($\operatorname{Classify}(\lambda)$)
The framework distinguishes two named parameter domains with unambiguous explicit classification precedence:
$$\boxed{
\operatorname{Classify}(\lambda) =
\begin{cases}
\operatorname{PhysicalClassifier}(\lambda), & \lambda \in \texttt{PhysicalSphereDomain}, \\
\operatorname{AnalyticClassifier}(\lambda), & \text{otherwise}.
\end{cases}
}$$
1. **$\texttt{PhysicalSphereDomain}$:** For $S^{d-1} \cong SO(d)/SO(d-1)$ ($d \ge 3$), parameter $\lambda = \frac{d-2}{2} \in \left\{ \frac{1}{2}, 1, \frac{3}{2}, 2, \dots \right\}$.
2. **$\texttt{AnalyticContinuationDomain}$:** For continuous parameter range $\lambda > 0$.

### 4.2 Two-Sided Endpoint Classification Function & Machine LP Condition ($\operatorname{EndpointClass}(\lambda)$)
The Sturm-Liouville operator $H_\lambda = -\partial_\theta^2 + \lambda(\lambda-1)\csc^2\theta$ on $(0, \pi)$ admits two-sided endpoint boundary conditions at both left endpoint $\theta \to 0^+$ and right endpoint $t = \pi - \theta \to 0^+$, represented as an explicit two-sided coefficient tuple $((A_+, B_+), (A_-, B_-))$:

$$\boxed{
\begin{array}{c|c|c|l}
\text{Domain} & \text{Parameter } \lambda & \operatorname{EndpointClass}(\lambda) & \text{Two-Sided Asymptotic Coefficient Condition (Friedrichs)} \\ \hline
\texttt{PhysicalDomain} & d=3 \ (\lambda=1/2) & \texttt{CRITICAL\_LC} & r_+ = r_- = 1/2, \ u \sim A_+ \theta^{1/2} + B_+ \theta^{1/2} \log\theta \ (\theta \to 0^+) \implies B_+ = 0; \\
& & & u \sim A_- t^{1/2} + B_- t^{1/2} \log t \ (t = \pi - \theta \to 0^+) \implies B_- = 0 \\[1.5mm]
\texttt{PhysicalDomain} & d=4 \ (\lambda=1) & \texttt{REGULAR} & H_1 = -\partial_\theta^2, \ u(\theta) \sim A_+ \theta + B_+ \implies B_+ = 0; \ u(t) \sim A_- t + B_- \implies B_- = 0 \\[1.5mm]
\texttt{PhysicalDomain} & d \ge 5 \ (\lambda \ge 3/2) & \texttt{LIMIT\_POINT} & \texttt{LP}: \operatorname{coeff}_{+, \text{singular}} \text{ unused}, \ \operatorname{coeff}_{-, \text{singular}} \text{ unused} \\[1.5mm]
\hline
\texttt{AnalyticDomain} & 1/2 < \lambda < 3/2, \lambda \ne 1 & \texttt{LIMIT\_CIRCLE} & u \sim A_+ \theta^\lambda + B_+ \theta^{1-\lambda} \implies B_+ = 0; \ u \sim A_- t^\lambda + B_- t^{1-\lambda} \implies B_- = 0 \\[1.5mm]
\texttt{AnalyticDomain} & 0 < \lambda < 1/2 & \texttt{LIMIT\_CIRCLE} & u \sim A_+ \theta^\lambda + B_+ \theta^{1-\lambda} \implies A_+ = 0; \ u \sim A_- t^\lambda + B_- t^{1-\lambda} \implies A_- = 0
\end{array}
}$$

### 4.3 Three Operator Representations & Domain Operator Morphism Isometry Chain
The three differential operators are related by changes of variable and unitary transformations between explicit function spaces via operator morphism chain $T_\lambda S_\lambda$:
1. **Algebraic Differential Operator $L_x$:**
   $$L_x = (1 - x^2) \frac{d^2}{dx^2} - (2\lambda + 1)x \frac{d}{dx}, \qquad \text{Domain: } x \in (-1, 1) \subset \mathbb{R}.$$
2. **Compact Radial Coordinate Transformation $S_\lambda$:** Under coordinate change $x = \cos\theta$:
   $$\boxed{S_\lambda : L^2\left([-1, 1], (1-x^2)^{\lambda-1/2} dx\right) \longrightarrow L^2\left((0, \pi), (\sin\theta)^{2\lambda} d\theta\right), \qquad (S_\lambda f)(\theta) = f(\cos\theta).}$$
3. **Sturm-Liouville Unitary Transformation $T_\lambda$:**
   $$\boxed{T_\lambda : L^2\left((0, \pi), (\sin\theta)^{2\lambda} d\theta\right) \longrightarrow L^2(0, \pi), \qquad (T_\lambda g)(\theta) = (\sin\theta)^\lambda g(\theta).}$$
   Combining $T_\lambda$ and $S_\lambda$ yields the exact Sturm-Liouville eigenfunction $u_n$:
   $$\boxed{u_n = T_\lambda S_\lambda \phi_n = (\sin\theta)^\lambda \phi_n(\cos\theta), \qquad H_\lambda u_n = N_n^2 u_n \quad (N_n = n + \lambda).}$$

   **Norm Isometry Invariant:** The morphism chain satisfies the exact norm isometry for any arbitrary test function $f \in \mathcal{H}_\lambda$:
   $$\boxed{\|T_\lambda S_\lambda f\|_{L^2(0, \pi)} = \|f\|_{\mathcal{H}_\lambda}, \qquad \text{and for } f = \phi_n: \quad \|u_n\|_{L^2(0, \pi)}^2 = \|\phi_n\|_\lambda^2.}$$

### 4.4 Exact Zonal Norm Formula & Dual Recurrence Invariant
The $L^2$ norm of normalized zonal functions $\phi_n(x)$ on $\mathcal{H}_\lambda = L^2([-1, 1], (1-x^2)^{\lambda-1/2} dx)$ has the exact closed-form expression:
$$\boxed{\|\phi_n\|_\lambda^2 = \int_{-1}^1 \phi_n(x)^2 (1-x^2)^{\lambda - 1/2} dx = \frac{\pi 2^{1-2\lambda} \Gamma(n+2\lambda)}{n!(n+\lambda) \Gamma(\lambda)^2 \left[ C_n^{(\lambda)}(1) \right]^2}.}$$

The exact norm weight $h_n = \|\phi_n\|_\lambda^{-1}$ provides the direct cross-layer algebraic bridge converting polynomial-normalized recurrence $a_n$ to orthonormal Jacobi recurrence $\alpha_n$:
$$\boxed{h_n = \|\phi_n\|_\lambda^{-1} \implies \alpha_n = a_n \frac{h_n}{h_{n+1}} = b_{n+1} \frac{h_{n+1}}{h_n}, \qquad \alpha_n^2 = a_n b_{n+1} = \frac{(n+1)(n+2\lambda)}{4(n+\lambda)(n+\lambda+1)}.}$$

---

## 5. Layer IV: Jacobi Spectral Operator & Rayleigh-Ritz $\|J_m\| < 1$ Proof

On $\mathcal{H}_\lambda = L^2([-1, 1], (1-x^2)^{\lambda-1/2} dx)$, coordinate multiplication $(M_x f)(x) = x f(x)$ is bounded self-adjoint. By unitary equivalence $J = U M_x U^{-1}$ to $M_x$:
$$\boxed{\|J\| = \|M_x\| = 1 \quad \text{and} \quad \sigma(J) = \sigma(M_x) = [-1, 1] \quad \text{(as mathematical consequences).}}$$

### Orthonormal Jacobi Basis Recurrence Split
The orthonormal Jacobi basis recurrence $x e_n - \alpha_n e_{n+1} - \alpha_{n-1} e_{n-1} = 0$ is split explicitly for $n=0$ vs $n \ge 1$:
$$\boxed{x e_0(x) - \alpha_0 e_1(x) = 0, \qquad n=0}$$
$$\boxed{x e_n(x) - \alpha_n e_{n+1}(x) - \alpha_{n-1} e_{n-1}(x) = 0, \qquad n \ge 1.}$$

### Rigorous Finite-Compression Spectral Bound Proof ($\|J_m\| < 1$)
For finite principal truncation $J_m = P_m M_x P_m |_{\operatorname{span}\{e_0, \dots, e_{m-1}\}} \in \mathbb{R}^{m \times m}$, since $J_m$ is real symmetric / self-adjoint, its operator norm equals the supremum of its Rayleigh quotient:
$$\|J_m\| = \max_{v \in S^{m-1}} |\langle J_m v, v \rangle|, \qquad S^{m-1} = \{v \in \mathbb{R}^m : \|v\|_2 = 1\}.$$

Every unit vector $v \in S^{m-1}$ represents a non-zero degree-$(m-1)$ polynomial $p_v(x) = \sum_{k=0}^{m-1} v_k e_k(x) \neq 0$:
$$\langle J_m v, v \rangle = \int_{-1}^1 x \, [p_v(x)]^2 (1-x^2)^{\lambda-1/2} dx.$$
Define continuous function $f(v) = |\langle J_m v, v \rangle|$ on $S^{m-1}$.
Because weight $w_\lambda(x) = (1-x^2)^{\lambda-1/2} > 0$ almost everywhere on $(-1, 1)$ and a non-zero polynomial $p_v(x)$ cannot be supported exclusively at $x = \pm 1$:
$$f(v) = |\langle J_m v, v \rangle| < \int_{-1}^1 1 \cdot [p_v(x)]^2 w_\lambda(x) dx = \|p_v\|_\lambda^2 = \|v\|_2^2 = 1 \qquad \forall v \in S^{m-1}.$$
Since $f(v)$ is continuous and $S^{m-1}$ is compact in $\mathbb{R}^m$, the maximum is strictly attained at some $v^* \in S^{m-1}$:
$$\boxed{\|J_m\| = \max_{v \in S^{m-1}} f(v) = f(v^*) < 1 \quad \text{for all } m < \infty, \quad \sigma(J_m) \subset (-1, 1).}$$

---

## 6. Layer V & VI: Asymptotic Schemas, Quantified Overlap Contract & Selector

### 6.1 Target Semantics & Piecewise Composite Approximation Schema
For a fixed target $Q$, the Layer VI exact target value is normalized:
$$\boxed{F_Q(\theta) = \operatorname{TargetValue}(Q, \theta), \qquad F(\theta) \equiv F_Q(\theta).}$$

The piecewise composite uniform expression combines endpoint Bessel layers and interior WKB waves over global domain coverage:
$$\boxed{\mathcal{D}_{\text{north}} \cup \mathcal{D}_{\text{interior}} \cup \mathcal{D}_{\text{south}} = \mathcal{D}_{\text{global}}.}$$

### 6.2 Finite Region Table & Remainder Reconstruction Identities
Define certified remainder objects $R_i$:
$$\begin{aligned}
R_{\text{north}} &= F - F_{\text{north}} \quad (\|R_{\text{north}}\| \le B_{\text{north}} \text{ on } \mathcal{D}_{\text{north}}), \\
R_{\text{interior}} &= F - F_{\text{interior}} \quad (\|R_{\text{interior}}\| \le B_{\text{interior}} \text{ on } \mathcal{D}_{\text{interior}}), \\
R_{\text{south}} &= F - F_{\text{south}} \quad (\|R_{\text{south}}\| \le B_{\text{south}} \text{ on } \mathcal{D}_{\text{south}}), \\
R_{+O} &= F_{\text{north}} - F_{+O} \quad (\|R_{+O}\| \le B_{+O} \text{ on } \mathcal{D}_{\text{north}} \cap \mathcal{D}_{\text{interior}}), \\
R_{-O} &= F_{\text{south}} - F_{-O} \quad (\|R_{-O}\| \le B_{-O} \text{ on } \mathcal{D}_{\text{south}} \cap \mathcal{D}_{\text{interior}}).
\end{aligned}$$

The region-dependent reconstruction rules are defined explicitly by a finite region table:
$$\boxed{\operatorname{Reconstruct}_r = (\text{exact algebraic identity}, I_r, s_r, \mathcal{D}_r)}$$

$$\boxed{
\begin{array}{l|l|l|c}
\text{Region } r & \text{Domain } \mathcal{D}_r & \text{Exact Remainder Reconstruction Identity } F_Q - F_{\text{comp},r} & \text{Active Terms } I_r \\ \hline
\texttt{NORTH} & \mathcal{D}_{\text{north}} \setminus \mathcal{D}_{\text{interior}} & F_Q - F_{\text{north}} = R_{\text{north}} & \{\text{north}\} \\[1.5mm]
\texttt{INTERIOR} & \mathcal{D}_{\text{interior}} \setminus (\mathcal{D}_{\text{north}} \cup \mathcal{D}_{\text{south}}) & F_Q - F_{\text{interior}} = R_{\text{interior}} & \{\text{interior}\} \\[1.5mm]
\texttt{SOUTH} & \mathcal{D}_{\text{south}} \setminus \mathcal{D}_{\text{interior}} & F_Q - F_{\text{south}} = R_{\text{south}} & \{\text{south}\} \\[1.5mm]
\texttt{NORTH\_INTERIOR\_OVERLAP} & \mathcal{D}_{\text{north}} \cap \mathcal{D}_{\text{interior}} \setminus \mathcal{D}_{\text{south}} & F_Q - (F_{\text{north}} + F_{\text{interior}} - F_{+O}) = R_{\text{interior}} - R_{+O} & \{\text{interior}, +O\} \\[1.5mm]
\texttt{INTERIOR\_SOUTH\_OVERLAP} & \mathcal{D}_{\text{interior}} \cap \mathcal{D}_{\text{south}} \setminus \mathcal{D}_{\text{north}} & F_Q - (F_{\text{interior}} + F_{\text{south}} - F_{-O}) = R_{\text{interior}} - R_{-O} & \{\text{interior}, -O\} \\[1.5mm]
\texttt{GLOBAL\_OVERLAP} & \mathcal{D}_{\text{north}} \cap \mathcal{D}_{\text{interior}} \cap \mathcal{D}_{\text{south}} & F_Q - F_{\text{comp}} = R_{\text{interior}} - R_{+O} - R_{-O} & \{\text{interior}, +O, -O\}
\end{array}
}$$

Applying the triangle inequality to $\operatorname{Reconstruct}_r(Q, \theta)$ yields the region-dependent composite error bound:
$$\boxed{|F_Q(\theta) - F_{\text{comp},r}^{(K)}(\theta)| \le \sum_{i \in I_r(\theta)} B_i(\theta) =: B_{\text{comp},r}(\theta).}$$

**Status Lattice Minimum Rule:**
The certification status of $B_{\text{comp},r}$ is determined by the lattice minimum across all active remainder terms in region $r$:
$$\boxed{\operatorname{Status}(B_{\text{comp},r}) = \min_{\preceq} \{\operatorname{Status}(R_i) : i \in I_r\},}$$
where status order is $\texttt{ALGEBRAIC\_EXACT} \succ \texttt{ARITHMETIC\_EXACT} \succ \texttt{ANALYTIC\_CERTIFIED} \succ \texttt{NUMERICAL\_CERTIFIED} \succ \texttt{EMPIRICAL\_DIAGNOSTIC}$.

Under status $\texttt{MATCHING\_SCHEMA}$, the asymptotic growth exponent is declared as $\gamma_K := \text{unspecified}$. It is promoted to $\texttt{ANALYTIC\_CERTIFIED}$ only after proving the uniform majorant theorem:
$$\boxed{G^\pm_{K, \lambda, Z_0, \delta}(z; N_n) \le C^\pm_{K, \lambda, Z_0, \delta} (1 + z)^{\gamma_K} \quad \text{uniformly for } N_n \ge N_0 \text{ and } Z_0 \le z \le \delta N_n.}$$

### 6.3 Certified Domain-Compatible Evaluation Selector ($M^*$)
Candidate evaluation representations are defined by explicit Candidate interfaces:
$$\boxed{\text{Candidate } \{ \text{domain } \mathcal{D}_M, \text{ target } Q, \text{ status}, \texttt{ErrorBound.valid}, B_M(\theta) \}.}$$
Selection requires target $Q$ matching ($\texttt{valid} \implies B_M(\theta)$ bounds the exact target quantity $F_Q(\theta)$):
$$\boxed{M^*(\theta) = \arg\min_{\substack{M \\ \theta \in \mathcal{D}_M \\ \text{Candidate}_M.\text{target} = Q \\ \texttt{ErrorBound}_M\text{.valid}}} B_M(\theta), \quad \text{where } \texttt{ErrorBound}_M\text{.status} \in \{\texttt{ALGEBRAIC\_EXACT}, \dots, \texttt{NUMERICAL\_CERTIFIED}\}.}$$

---

## 7. Layer VII: Modular & Multi-Backend Arithmetic Execution Layer

### VII-A & VII-B. Hardware & Exact Symbolic Sub-Backends
- **Polynomial Path:** $\mathbb{Q}[\lambda, x] \to \mathbb{Q} \to \text{RNS/CRT}$ (truth class `ARITHMETIC_EXACT`).
- **Jacobi Golub-Welsch Path:** Eigendecomposition of symmetric tridiagonal $J_m$. When backed by a backward-stable eigensolver with validated residual bounds, status is `NUMERICAL_CERTIFIED` via a formal target-specific `NumericalCertificate` using explicit typed implication with certified sensitivity hypothesis:
  $$\boxed{\operatorname{CertifiedSensitivity}_Q(A, \kappa_Q) \land R_Q \le B_{\text{back}} \land E_{Q, \text{conv}} \le B_{Q, \text{conv}} \implies E_Q \le \kappa_Q B_{\text{back}} + B_{Q, \text{conv}} := B_{Q, \text{forward}},}$$
  where target $Q \in \{\texttt{NODE}, \texttt{WEIGHT}, \texttt{EIGENVECTOR}, \texttt{QUADRATURE}\}$ and $\kappa_Q$ is explicitly certified (including eigenvalue-gap conditioning for eigenvectors/weights).

### VII-C. Execution-Trace Recurrence Inversions & Backend Normalization Denominators
Define execution-trace primitive inversions: $P_{\text{rec\_inv}}(P) = \{ q : \texttt{INVERT}(q) \text{ occurs in recurrence trace of } P \}$. Recurrence denominators follow the actual executed arithmetic graph:
$$\boxed{D_{\text{rec}}(P) = \operatorname{lcm}_{q \in P_{\text{rec\_inv}}(P)} \operatorname{den}_{\text{red}}(q).}$$

Evaluation point denominator for canonical fraction $x = c/r$ ($\gcd(c,r)=1, r>0$):
$$\boxed{D_{\text{eval}} = r.}$$

Backend normalization denominators are defined per plan $P$ and target $Q$:
$$\boxed{D_{\text{norm}}(P, Q) = \operatorname{lcm}\{\text{rational denominators actually inverted by } P \text{ for target } Q\}.}$$

The zonal finite-field backend excludes normalization denominators via typed predicate:
$$\boxed{D_{\text{norm}}^{\text{used}}(P, \texttt{ZONAL}) = 1 \quad \iff \quad \operatorname{UsesNormalizationDenominators}(P, \texttt{ZONAL}) = \text{false}.}$$

Aggregate excluded-denominator modulus:
$$\boxed{D_{\text{den}} = \operatorname{lcm}(D_{\text{rec}}(P), D_{\text{norm}}(P, Q), D_{\text{eval}}), \qquad \text{with empty LCM convention } \operatorname{lcm}(\varnothing) = 1.}$$

### VII-D. Explicit Canonical Preconditions & Semantically Exact Bad Zonal Prime Predicate
Canonical reduced fraction representation preconditions:
$$\boxed{C_n^{(\lambda)}(1) = \frac{u_n}{v_n}, \quad \gcd(u_n, v_n) = 1, \ v_n > 0; \qquad x = \frac{c}{r}, \ \gcd(c, r) = 1, \ r > 0.}$$

Finite-field certificates and bad prime predicate:
1. **Unnormalized Polynomial Certificate ($\texttt{PolyCertificate}$):**
   $$\boxed{\texttt{PolyCertificate}(p, P, n) \iff \operatorname{Prime}(p) \land \gcd(p, D_{\text{rec}}(P)) = 1.}$$
2. **Point Evaluation Certificate ($\texttt{PointCertificate}$):**
   $$\boxed{\texttt{PointCertificate}(p, x) \iff \gcd(p, D_{\text{eval}}) = 1 \quad (\text{i.e. } p \nmid r).}$$
3. **Polynomial Evaluation Certificate ($\texttt{PolyEvaluationCertificate}$):**
   $$\boxed{\texttt{PolyEvaluationCertificate}(p, P, n, x) = \texttt{PolyCertificate}(p, P, n) \land \texttt{PointCertificate}(p, x).}$$
4. **Normalized Zonal Admissibility ($\operatorname{ZonalAdmissible}$):**
   $$\boxed{\operatorname{ZonalAdmissible}(p, P, n, x) \iff \operatorname{Prime}(p) \land \gcd(p, D_{\text{rec}}(P)) = 1 \land p \nmid r \land p \nmid u_n v_n.}$$
5. **Semantically Exact Bad Zonal Prime Predicate ($\operatorname{Bad}_{\text{zonal}}$):**
   $$\boxed{\operatorname{Bad}_{\text{zonal}}(p, P, n, x) \iff \neg \operatorname{ZonalAdmissible}(p, P, n, x) \iff p \mid r \lor p \mid v_n \lor p \mid u_n \lor p \mid D_{\text{rec}}(P).}$$

   Note explicit separation: $\text{Zonal finite-field certificate} \not\Rightarrow \text{orthonormal-basis certificate}$.

---

## 8. Layer VIII: Typed Separation, Executable Residuals & Provenance Optimizer

### VIII-A. Typed Hierarchy & Extended Metadata Tuple Invariant
Hard type separation: $\texttt{ExactValue} \neq \texttt{ErrorBound} \neq \texttt{Residual}$.

`TheoremStatus` Enum:
$$\boxed{\{\texttt{ALGEBRAIC\_EXACT}, \texttt{ARITHMETIC\_EXACT}, \texttt{ANALYTIC\_CERTIFIED}, \texttt{NUMERICAL\_CERTIFIED}, \texttt{EMPIRICAL\_DIAGNOSTIC}\}}$$

**Extended Certificate Metadata Tuple Invariant:**
$$\boxed{\text{Every certificate carries } (\text{target } Q, \text{domain } \mathcal{D}, \text{backend}, \text{status}, \text{validity\_conditions}).}$$

Provenance-aware total computational forward error decomposes across a target-specific staged perturbation chain $F_0(Q) \xrightarrow{E_0} F_1(Q) \xrightarrow{E_1} \dots \xrightarrow{E_{k-1}} F_k(Q)$:
$$\boxed{|F_0(Q) - F_k(Q)| \le \sum_{i=0}^{k-1} |F_i(Q) - F_{i+1}(Q)| \le \sum_{i=0}^{k-1} E_i \quad \text{provided } \operatorname{Cert}(E_i) \text{ holds for target } Q.}$$

### VIII-B. Mandatory Cross-Layer Consistency Invariants
The Layer VIII canonical test matrix ($d \in \{3, 4, 5\}, n \in \{0, 1, 2, 3\}$) explicitly validates:
1. **Initial Data Anchors:** $\phi_0(x) = 1, \phi_1(x) = x$.
2. **Recurrence Invariants:**
   - $n=0$: $x \phi_0 - \phi_1 = 0$.
   - $n \ge 1$: $R_{\text{rec}}(n, x) = x \phi_n - a_n \phi_{n+1} - b_n \phi_{n-1} = 0$.
3. **Orthonormal Jacobi Basis Recurrence:**
   - $n=0$: $x e_0 - \alpha_0 e_1 = 0$.
   - $n \ge 1$: $x e_n - \alpha_n e_{n+1} - \alpha_{n-1} e_{n-1} = 0$.
4. **Dual Conversion Square Invariant:** $I_{\text{dual}}(n) = \alpha_n^2 - a_n b_{n+1} = 0$.
5. **Cheap Invariants:**
   - Parity and Boundedness: $\phi_n(-x) = (-1)^n \phi_n(x)$ and $|\phi_n(x)| \le 1$.
   - Arbitrary Test Function Norm Isometry: $\|T_\lambda S_\lambda f\|_{L^2(0, \pi)} = \|f\|_{\mathcal{H}_\lambda}$.

### VIII-C. Provenance Invariant
$$\boxed{\texttt{Residual} \not\implies \texttt{ErrorBound}}$$
Unless a theorem explicitly supplies an implication mapping equation residual to forward error bound, `Residual` objects remain diagnostics and do not participate in certified optimization.

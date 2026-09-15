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
  Operator Morphism T_λ: L^2((0,π), (sin θ)^{2λ} dθ) ──(T_λ ϕ)(θ) = (sin θ)^λ ϕ(cos θ)──> L^2(0,π)  |  H_λ u_n = N_n^2 u_n
  Two-Sided EndpointBoundaryCondition ((A_+, B_+) at θ → 0+, (A_-, B_-) at t = π-θ → 0+):
    - d=3 (λ=1/2): Critical Limit-Circle (LC), r_+ = r_- = 1/2, u ~ A_± θ^{1/2} + B_± θ^{1/2} log θ (forbidden_log_coeff B_± = 0)
    - d=4 (λ=1): Regular Endpoint, H_1 = -∂_θ^2, u ~ A_± θ + B_± (singular_branch_coeff B_± = 0)
    - d≥5 (λ≥3/2): Limit-Point (LP), singular branch θ^{1-λ} ∉ L^2(0,π), no boundary parameter required
  AnalyticContinuationDomain (λ > 0, λ ∉ PhysicalDomain): 1/2 < λ < 3/2 ⟹ LC (B_± = 0); 0 < λ < 1/2 ⟹ LC (A_± = 0)
  Exact Norm Invariant: ||ϕ_n||_λ^2 = (π 2^{1-2λ} Γ(n+2λ)) / (n! (n+λ) Γ(λ)^2 [C_n^{(λ)}(1)]^2) ⟹ h_n = ||ϕ_n||_λ^{-1} ⟹ α_n = a_n (h_n/h_{n+1})
        │
        ▼
  Layer IV. Jacobi Spectral Operator & Unitary Matrix Realization
  (M_x f)(x) = x f(x),  U M_x U^{-1} = J = J^*  ⟹  ||J|| = 1 and σ(J) = [-1, 1] as consequences
  Rayleigh-Ritz Spectral Theorem: ||J_m|| = max_{||v||=1} |⟨J_m v, v⟩| < 1 derived via compactness of unit sphere S^{m-1} and real symmetry of J_m
  Subdiagonal Expansion (n → ∞, λ fixed): α_n = 1/2 + λ(1-λ)/(4 n^2) + O(n^{-3})
        │
        ▼
  Layer V. Two-Endpoint Boundary Coordinates
  N_n = n + λ,  z_+ = N_n θ,  z_- = N_n (π - θ)
        │
        ▼
  Layer VI. Two-Overlap Composite Uniform Asymptotic Schema & Quantified Selector
  Composite Approximation Schema: F_comp = F_north + F_south + F_interior - F_{+O} - F_{-O} (Status: MATCHING_SCHEMA)
  Quantified Overlap Contract: |F_endpoint^{(K)} - F_O^{(K)}| ≤ N_n^{-K} G_{K,λ,Z_0,δ}(z_±) on Z_0 ≤ z_± ≤ δ N_n
  Majorant Function Contract: G_{K,λ,Z_0,δ} : [Z_0, δ N_n] → ℝ_{≥0} (MATCHING_SCHEMA unproved vs ANALYTIC_CERTIFIED proved)
  Evaluation Selector: M^*(\theta) = argmin_{M, \theta ∈ 𝒟_M, B_M \text{ cert}} B_M(\theta)
        │
        ▼
  Layer VII. Modular & Multi-Backend Arithmetic Execution Layer
  ├── VII-A: Floating-Point & Fixed-Point (FLOAT32, FLOAT64, LONGDOUBLE, C_fixed, C_LNS)
  ├── VII-B: Exact Rational Symbolic Algebra (Q[λ, x] ──symbolic rec──> C_n ──eval──> Q ──CRT/RNS──> integer residues)
  ├── VII-C: Scalable RNS / CRT (N_max := execution metadata, D_rec(P) = lcm({den_red(c) : c ∈ P_ops}), D_eval = lcm(D_rec, r))
  ├── VII-D1: Finite-Field Arithmetic (PolyCert gcd(p, D_rec)=1 ∧ PointCert gcd(p, r)=1 vs ZonalCert PolyCert ∧ PointCert ∧ p ∤ u_n v_n)
  │          Failure Taxonomy: PointLocalizationFailure (p|r), NormalizationRepresentationFailure (p|v_n), NormalizationSingularityFailure (p|u_n)
  ├── VII-D2: NTT Acceleration Primitive (L_conv = L_1+L_2-1 ≤ L_NTT | (p-1))
  └── VII-E: Golub-Welsch Spectral Matrix Truncation (J_m = tridiag(α_0, ..., α_{m-2}), Status: NUMERICAL_CERTIFIED via NumericalCertificate)
        │
        ▼
  Layer VIII. Typed Separation: ExactValue vs ErrorBound vs Residual & Provenance Optimizer
  First-Class Types: ExactValue != ErrorBound != Residual (TheoremStatus Enum: ALGEBRAIC_EXACT, ARITHMETIC_EXACT, ANALYTIC_CERTIFIED, NUMERICAL_CERTIFIED, EMPIRICAL_DIAGNOSTIC)
  Residual != ErrorBound Invariant; Staged Perturbation Chain: F_0 ──E_0──> F_1 ──E_1──> ... ──E_{k-1}──> F_k ⟹ |F_0 - F_k| ≤ ∑_{i=0}^{k-1} E_i
  Selector Candidate Conversion Mapping:
    - ALGEBRAIC_EXACT / ARITHMETIC_EXACT ⟹ 0 (relative to certified exact target)
    - ANALYTIC_CERTIFIED ⟹ B_theorem
    - NUMERICAL_CERTIFIED ⟹ B_forward (via Composable Invariant B_forward ≥ κ B_back + B_conv)
  Canonical Test Matrix: d ∈ {3, 4, 5}, n ∈ {0, 1, 2, 3} validating:
    - Initial Data Anchors: ϕ_0 = 1, ϕ_1 = x
    - Recurrence Invariants: n=0: x ϕ_0 - ϕ_1 = 0; n≥1: x ϕ_n - a_n ϕ_{n+1} - b_n ϕ_{n-1} = 0
    - Dual Conversion Square Invariant: I_dual(n) = α_n^2 - a_n b_{n+1} = 0
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

### 4.1 Parameter Domains & Endpoint Boundary Conditions
The framework distinguishes two named parameter domains:
1. **$\texttt{PhysicalSphereDomain}$:** For $S^{d-1} \cong SO(d)/SO(d-1)$ ($d \ge 3$), parameter $\lambda = \frac{d-2}{2} \in \left\{ \frac{1}{2}, 1, \frac{3}{2}, 2, \dots \right\}$.
2. **$\texttt{AnalyticContinuationDomain}$:** For continuous parameter range $\lambda > 0$.

### 4.2 Two-Sided Endpoint Classification Function ($\operatorname{EndpointClass}(\lambda)$)
The Sturm-Liouville operator $H_\lambda = -\partial_\theta^2 + \lambda(\lambda-1)\csc^2\theta$ on $(0, \pi)$ admits identical two-sided endpoint boundary conditions at both endpoints ($\theta \to 0^+$ and $t = \pi - \theta \to 0^+$) represented as an explicit tuple $(A_\pm, B_\pm)$:

$$\boxed{
\begin{array}{c|c|c|l}
\text{Domain} & \text{Parameter } \lambda & \operatorname{EndpointClass}(\lambda) & \text{Two-Sided Asymptotic Coefficient Condition (Friedrichs)} \\ \hline
\texttt{PhysicalDomain} & d=3 \ (\lambda=1/2) & \texttt{CRITICAL\_LC} & r_+ = r_- = 1/2, \ u \sim A_\pm \theta^{1/2} + B_\pm \theta^{1/2} \log\theta \implies B_\pm = 0 \\[1.5mm]
\texttt{PhysicalDomain} & d=4 \ (\lambda=1) & \texttt{REGULAR} & H_1 = -\partial_\theta^2, \ u(\theta) \sim A_\pm \theta + B_\pm \implies B_\pm = 0 \\[1.5mm]
\texttt{PhysicalDomain} & d \ge 5 \ (\lambda \ge 3/2) & \texttt{LIMIT\_POINT} & \text{Singular branch } \theta^{1-\lambda} \notin L^2(0,\pi); \text{ no boundary parameter required} \\[1.5mm]
\hline
\texttt{AnalyticDomain} & 1/2 < \lambda < 3/2, \lambda \ne 1 & \texttt{LIMIT\_CIRCLE} & u(\theta) \sim A_\pm \theta^\lambda + B_\pm \theta^{1-\lambda} \implies B_\pm = 0 \\[1.5mm]
\texttt{AnalyticDomain} & 0 < \lambda < 1/2 & \texttt{LIMIT\_CIRCLE} & u(\theta) \sim A_\pm \theta^\lambda + B_\pm \theta^{1-\lambda} \implies A_\pm = 0
\end{array}
}$$

### 4.3 Three Operator Representations & Domain Operator Morphisms
The three differential operators are related by changes of variable and unitary transformations between explicit function spaces via operator morphism $T_\lambda$:
1. **Algebraic Differential Operator $L_x$:**
   $$L_x = (1 - x^2) \frac{d^2}{dx^2} - (2\lambda + 1)x \frac{d}{dx}, \qquad \text{Domain: } x \in (-1, 1) \subset \mathbb{R}.$$
2. **Compact Radial Operator $L_\theta$:** Under coordinate change $x = \cos\theta$:
   $$L_\theta = \frac{d^2}{d\theta^2} + 2\lambda \cot\theta \frac{d}{d\theta}, \qquad \text{Domain: } \theta \in (0, \pi).$$
3. **Sturm-Liouville Hamiltonian $H_\lambda$:** Under operator morphism $T_\lambda$:
   $$\boxed{T_\lambda : L^2\left((0, \pi), (\sin\theta)^{2\lambda} d\theta\right) \longrightarrow L^2(0, \pi), \qquad (T_\lambda \phi)(\theta) = (\sin\theta)^\lambda \phi(\cos\theta),}$$
   $$H_\lambda = -\frac{d^2}{d\theta^2} + \lambda(\lambda - 1)\csc^2\theta, \qquad \text{Domain: } \theta \in (0, \pi).$$
   Eigenvalue equation: $H_\lambda u_n = N_n^2 u_n$, where $N_n := n + \lambda$, so $N_n^2 = E_n + \lambda^2$.

### 4.4 Exact Zonal Norm Formula & Dual Recurrence Invariant
The $L^2$ norm of normalized zonal functions $\phi_n(x)$ on $\mathscr{H}_\lambda = L^2([-1, 1], (1-x^2)^{\lambda-1/2} dx)$ has the exact closed-form expression:
$$\boxed{\|\phi_n\|_\lambda^2 = \int_{-1}^1 \phi_n(x)^2 (1-x^2)^{\lambda - 1/2} dx = \frac{\pi 2^{1-2\lambda} \Gamma(n+2\lambda)}{n!(n+\lambda) \Gamma(\lambda)^2 \left[ C_n^{(\lambda)}(1) \right]^2}.}$$

The exact norm weight $h_n = \|\phi_n\|_\lambda^{-1}$ provides the direct cross-layer algebraic bridge converting polynomial-normalized recurrence $a_n$ to orthonormal Jacobi recurrence $\alpha_n$:
$$\boxed{h_n = \|\phi_n\|_\lambda^{-1} \implies \alpha_n = a_n \frac{h_n}{h_{n+1}} = b_{n+1} \frac{h_{n+1}}{h_n}, \qquad \alpha_n^2 = a_n b_{n+1} = \frac{(n+1)(n+2\lambda)}{4(n+\lambda)(n+\lambda+1)}.}$$

---

## 5. Layer IV: Jacobi Spectral Operator & Rayleigh-Ritz $\|J_m\| < 1$ Proof

On $\mathscr{H}_\lambda = L^2([-1, 1], (1-x^2)^{\lambda-1/2} dx)$, coordinate multiplication $(M_x f)(x) = x f(x)$ is bounded self-adjoint. By unitary equivalence $J = U M_x U^{-1}$ to $M_x$:
$$\boxed{\|J\| = \|M_x\| = 1 \quad \text{and} \quad \sigma(J) = \sigma(M_x) = [-1, 1] \quad \text{(as mathematical consequences).}}$$

### Rigorous Self-Contained Proof of $\|J_m\| < 1$
For finite principal truncation $J_m = P_m M_x P_m |_{\operatorname{span}\{e_0, \dots, e_{m-1}\}} \in \mathbb{R}^{m \times m}$, since $J_m$ is real symmetric / self-adjoint, its operator norm equals the supremum of its Rayleigh quotient:
$$\|J_m\| = \max_{v \in S^{m-1}} |\langle J_m v, v \rangle|, \qquad S^{m-1} = \{v \in \mathbb{R}^m : \|v\|_2 = 1\}.$$

Every unit vector $v \in S^{m-1}$ represents a non-zero degree-$(m-1)$ polynomial $p_v(x) = \sum_{k=0}^{m-1} v_k e_k(x) \neq 0$:
$$\langle J_m v, v \rangle = \int_{-1}^1 x \, [p_v(x)]^2 (1-x^2)^{\lambda-1/2} dx.$$
Define continuous function $f(v) = |\langle J_m v, v \rangle|$ on $S^{m-1}$.
Because weight $w_\lambda(x) = (1-x^2)^{\lambda-1/2} > 0$ almost everywhere on $(-1, 1)$ and a non-zero polynomial $p_v(x)$ cannot be supported exclusively at $x = \pm 1$:
$$f(v) = |\langle J_m v, v \rangle| < \int_{-1}^1 1 \cdot [p_v(x)]^2 w_\lambda(x) dx = \|p_v\|_\lambda^2 = \|v\|_2^2 = 1 \qquad \forall v \in S^{m-1}.$$
Since $f(v)$ is continuous and $S^{m-1}$ is compact in $\mathbb{R}^m$, the maximum is strictly attained at some $v^* \in S^{m-1}$:
$$\boxed{\|J_m\| = \max_{v \in S^{m-1}} f(v) = f(v^*) < 1 \quad \text{for all } m < \infty, \quad \sigma(J_m) \subset (-1, 1).}$$

### Subdiagonal Asymptotic Expansion ($n \to \infty$ with fixed $\lambda$)
$$\boxed{\alpha_n = \frac{1}{2} + \frac{\lambda(1-\lambda)}{4n^2} + O(n^{-3}) \qquad \text{as } n \to \infty \quad (\text{fixed } \lambda).}$$

---

## 6. Layer V & VI: Asymptotic Schemas, Quantified Overlap Contract & Selector

### 6.1 Composite Approximation Schema & Quantified Overlap Contract
The composite uniform expression combines endpoint Bessel layers and interior WKB waves:
$$F_{\text{comp}}^{(K)} = F_{\text{north}}^{(K)}(z_+) + F_{\text{south}}^{(K)}(z_-) + F_{\text{interior}}^{(K)}(N_n, \theta) - F_{+O}^{(K)}(z_+) - F_{-O}^{(K)}(z_-).$$

The overlap re-expansion contract is quantified over intermediate overlap domains $Z_0 \le z_\pm \le \delta N_n$:
$$\boxed{\left| F_{\text{endpoint}}^{(K)} - F_O^{(K)} \right| \le N_n^{-K} \, G_{K, \lambda, Z_0, \delta}(z_\pm) \qquad \text{for } Z_0 \le z_\pm \le \delta N_n \quad (0 < \delta < \pi/2),}$$
where $G_{K, \lambda, Z_0, \delta}: [Z_0, \delta N_n] \to \mathbb{R}_{\ge 0}$ is a $z$-dependent majorant function.
- **$\texttt{MATCHING\_SCHEMA}$:** Function $G$ is symbolically specified / unproved.
- **$\texttt{ANALYTIC\_CERTIFIED}$:** Function $G$ is accompanied by a proved majorant theorem.

### 6.2 Certified Domain-Compatible Evaluation Selector ($M^*$)
Pointwise representation selection minimizes local certified forward error bound $B_M(\theta)$ over domain-compatible candidates:
$$\boxed{M^*(\theta) = \arg\min_{\substack{M \\ \theta \in \mathcal{D}_M \\ \texttt{ErrorBound}_M\text{.valid}}} B_M(\theta), \quad \text{where } \texttt{ErrorBound}_M\text{.status} \in \{\texttt{ALGEBRAIC\_EXACT}, \texttt{ARITHMETIC\_EXACT}, \texttt{ANALYTIC\_CERTIFIED}, \texttt{NUMERICAL\_CERTIFIED}\}.}$$

---

## 7. Layer VII: Modular & Multi-Backend Arithmetic Execution Layer

### VII-A & VII-B. Hardware & Exact Symbolic Sub-Backends
- **Polynomial Path:** $\mathbb{Q}[\lambda, x] \to \mathbb{Q} \to \text{RNS/CRT}$ (truth class `ARITHMETIC_EXACT`).
- **Jacobi Golub-Welsch Path:** Eigendecomposition of symmetric tridiagonal $J_m$. When backed by a backward-stable eigensolver with validated residual bounds, status is `NUMERICAL_CERTIFIED` via a formal `NumericalCertificate`:
  $$\boxed{\texttt{NumericalCertificate} = (A, B_{\text{back}}, R, \kappa, B_{\text{conv}}, B_{\text{forward}}) \implies B_{\text{forward}} \ge \kappa \cdot B_{\text{back}} + B_{\text{conv}}.}$$

### VII-C. RNS / CRT Sub-Backend with Execution-Plan Denominators
$N_{\max}$ is defined as **execution-plan metadata**. For an execution plan $P$, rational parameter $\lambda = a/b$, and evaluation point $x = c/r$:
$$\boxed{D_{\text{recurrence}}(P) = \operatorname{lcm}\left( \{ \operatorname{den}_{\text{red}}(c) : c \in P_{\text{rational\_operations}} \} \cup \{b\} \right), \qquad D_{\text{eval}} = \operatorname{lcm}(D_{\text{recurrence}}(P), r).}$$

### VII-D. Finite-Field Polynomial Arithmetic & NTT Sub-Backends
Explicit formal dependency for finite-field certificates:
1. **Unnormalized Polynomial Certificate ($\texttt{PolyCertificate}$):**
   $$\boxed{\texttt{PolyCertificate}(p, P) \iff \gcd(p, D_{\text{recurrence}}(P)) = 1.}$$
2. **Point Evaluation Certificate ($\texttt{PointCertificate}$):**
   $$\boxed{\texttt{PointCertificate}(p, r) \iff \gcd(p, r) = 1.}$$
3. **Normalized Zonal Spherical Certificate ($\texttt{ZonalCertificate}$):**
   For canonical reduced fraction $C_n^{(\lambda)}(1) = u_n/v_n$ ($\gcd(u_n, v_n)=1$):
   $$\boxed{\texttt{ZonalCertificate}(p, P, n) \iff \left( \texttt{PolyCertificate}(p, P) \land \texttt{PointCertificate}(p, r) \land p \nmid v_n \land p \nmid u_n \right).}$$

   *Failure Mode Taxonomy:*
   - `PointLocalizationFailure` ($p \mid r$): Evaluation point $x=c/r$ non-local in $\mathbb{F}_p$.
   - `NormalizationRepresentationFailure` ($p \mid v_n$): Rational representation $u_n/v_n$ of $C_n(1)$ non-local in $\mathbb{F}_p$.
   - `NormalizationSingularityFailure` ($p \mid u_n$): $C_n(1) \equiv 0 \pmod p$, normalization division by zero in $\mathbb{F}_p$.

---

## 8. Layer VIII: Typed Separation, Executable Residuals & Provenance Optimizer

### VIII-A. Typed Hierarchy & Theorem Status Enum
Hard type separation:
$$\boxed{\texttt{ExactValue} \neq \texttt{ErrorBound} \neq \texttt{Residual}}$$

`TheoremStatus` Enum:
$$\boxed{\{\texttt{ALGEBRAIC\_EXACT}, \texttt{ARITHMETIC\_EXACT}, \texttt{ANALYTIC\_CERTIFIED}, \texttt{NUMERICAL\_CERTIFIED}, \texttt{EMPIRICAL\_DIAGNOSTIC}\}}$$

Selector Candidate Conversion Mapping:
$$\boxed{
\begin{array}{c|c}
\text{Status} & \text{Certified Forward Error Bound} \\ \hline
\texttt{ALGEBRAIC\_EXACT} & 0 \quad (\text{relative to exact target}) \\
\texttt{ARITHMETIC\_EXACT} & 0 \quad (\text{relative to exact algorithm}) \\
\texttt{ANALYTIC\_CERTIFIED} & B_{\text{theorem}} \\
\texttt{NUMERICAL\_CERTIFIED} & B_{\text{forward}} \quad (\text{via } B_{\text{forward}} \ge \kappa B_{\text{back}} + B_{\text{conv}})
\end{array}
}$$

Provenance-aware total computational forward error decomposes across a staged perturbation chain $F_0 \xrightarrow{E_0} F_1 \xrightarrow{E_1} \dots \xrightarrow{E_{k-1}} F_k$:
$$\boxed{E_{\text{total}} \le \sum_{i=0}^{k-1} E_i \quad \text{provided } \operatorname{Cert}(E_i) \text{ holds for } |F_i - F_{i+1}| \le E_i, \qquad E_{\text{conditioning}} \le \kappa \cdot E_{\text{input}}.}$$

### VIII-B. Mandatory Cross-Layer Consistency Invariants
The Layer VIII canonical test matrix ($d \in \{3, 4, 5\}, n \in \{0, 1, 2, 3\}$) explicitly validates:
1. **Initial Data Anchors:** $\phi_0(x) = 1, \phi_1(x) = x$.
2. **Recurrence Invariants:**
   - $n=0$: $x \phi_0 - \phi_1 = 0$.
   - $n \ge 1$: $R_{\text{rec}}(n, x) = x \phi_n - a_n \phi_{n+1} - b_n \phi_{n-1} = 0$.
3. **Dual Conversion Square Invariant:** $I_{\text{dual}}(n) = \alpha_n^2 - a_n b_{n+1} = 0$.

### VIII-C. Provenance Invariant
$$\boxed{\texttt{Residual} \not\implies \texttt{ErrorBound}}$$
Unless a theorem explicitly supplies an implication mapping equation residual to forward error bound, `Residual` objects remain diagnostics and do not participate in certified optimization.

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
  Resulting Morphism: u_n = T_λ S_λ ϕ_n = (sin θ)^λ ϕ_n(cos θ)  |  H_λ u_n = N_n^2 u_n  |  ||T_λ S_λ f||_{L^2(0,π)} = ||f||_{ℋ_λ} (||u_n||_{L^2(0,π)}^2 = ||ϕ_n||_λ^2)
  Precedence Classifier: Classify(λ) = PhysicalClassifier(λ) if λ ∈ PhysicalSphereDomain else AnalyticClassifier(λ)
  Two-Sided EndpointBoundaryCondition ((A_+, B_+), (A_-, B_-)) with LP execution symmetry:
    - d=3 (λ=1/2): Critical Limit-Circle (LC), r_+ = r_- = 1/2, u ~ A_+ θ^{1/2} + B_+ θ^{1/2} log θ (left B_+ = 0) vs u ~ A_- t^{1/2} + B_- t^{1/2} log t (right B_- = 0)
    - d=4 (λ=1): Regular Endpoint, H_1 = -∂_θ^2, u ~ A_+ θ + B_+ (left B_+ = 0) vs u ~ A_- t + B_- (right B_- = 0)
    - d≥5 (λ≥3/2): Limit-Point (LP), LP: coeff_{+, singular} unused, coeff_{-, singular} unused
  AnalyticContinuationDomain (λ > 0, λ ∉ PhysicalDomain): 1/2 < λ < 3/2 ⟹ LC (B_± = 0); 0 < λ < 1/2 ⟹ LC (A_± = 0)
  Exact Norm Invariant: ||ϕ_n||_λ^2 = (π 2^{1-2λ} Γ(n+2λ)) / (n! (n+λ) Γ(λ)^2 [C_n^{(λ)}(1)]^2) ⟹ h_n = ||ϕ_n||_λ^{-1} ⟹ α_n = a_n (h_n/h_{n+1})
        │
        ▼
  Layer IV. Jacobi Spectral Operator & Unitary Matrix Realization
  (M_x f)(x) = x f(x),  U M_x U^{-1} = J = J^*  ⟹  ||J|| = 1 and σ(J) = [-1, 1] as consequences
  Rayleigh-Ritz Finite-Compression Spectral Bound: ||J_m|| = max_{||v||=1} |⟨J_m v, v⟩| < 1 derived via compactness of unit sphere S^{m-1} and real symmetry of J_m
  Subdiagonal Expansion (n → ∞, λ fixed): α_n = 1/2 + λ(1-λ)/(4 n^2) + O(n^{-3})
        │
        ▼
  Layer V. Two-Endpoint Boundary Coordinates
  N_n = n + λ,  z_+ = N_n θ,  z_- = N_n t = N_n (π - θ)
        │
        ▼
  Layer VI. Two-Overlap Composite Uniform Asymptotic Schema & Quantified Selector
  Composite Approximation Schema: F_comp = F_north + F_south + F_interior - F_{+O} - F_{-O} (Status: MATCHING_SCHEMA)
  Quantified Overlap Majorants: |R_{+,O}^{(K)}| ≤ N_n^{-K} G^+_{K,λ,Z_0,delta}(z_+; N_n) and |R_{-,O}^{(K)}| ≤ N_n^{-K} G^-_{K,λ,Z_0,delta}(z_-; N_n) on Z_0 ≤ z_± ≤ delta N_n
  Executable Uniform Majorant Growth Bound: G^\pm_{K,λ,Z_0,delta}(z; N_n) ≤ C^\pm_{K,λ,Z_0,delta} (1+z)^{\gamma_K} uniformly for N_n ≥ N_0 and Z_0 ≤ z ≤ delta N_n
  Evaluation Selector: M^*(\theta) = argmin_{M, \theta ∈ 𝒟_M, B_M \text{ cert}} B_M(\theta)
        │
        ▼
  Layer VII. Modular & Multi-Backend Arithmetic Execution Layer
  ├── VII-A: Floating-Point & Fixed-Point (FLOAT32, FLOAT64, LONGDOUBLE, C_fixed, C_LNS)
  ├── VII-B: Exact Rational Symbolic Algebra (Q[λ, x] ──symbolic rec──> C_n ──eval──> Q ──CRT/RNS──> integer residues)
  ├── VII-C: Scalable RNS / CRT (N_max := execution metadata, D_rec, D_norm, D_eval; D_excl = lcm(D_rec, D_norm, D_eval))
  ├── VII-D1: Finite-Field Arithmetic (PolyCert Prime(p) ∧ gcd(p, D_rec)=1; PointCert gcd(p, r)=1; PolyEvalCert PolyCert ∧ PointCert; ZonalCert(p, plan, n, x=c/r) PolyEvalCert ∧ p ∤ v_n ∧ C_n(1) ≠ 0 mod p)
  │          Failure Taxonomy: PointLocalizationFailure (p|r), NormalizationRepresentationFailure (p|v_n), NormalizationSingularityFailure (p|u_n)
  ├── VII-D2: NTT Acceleration Primitive (L_conv = L_1+L_2-1 ≤ L_NTT | (p-1))
  └── VII-E: Golub-Welsch Spectral Matrix Truncation (J_m = tridiag(α_0, ..., α_{m-2}), Target-Specific NumericalCertificate_Q)
        │
        ▼
  Layer VIII. Typed Separation: ExactValue vs ErrorBound vs Residual & Provenance Optimizer
  First-Class Types: ExactValue != ErrorBound != Residual (TheoremStatus Enum: ALGEBRAIC_EXACT, ARITHMETIC_EXACT, ANALYTIC_CERTIFIED, NUMERICAL_CERTIFIED, EMPIRICAL_DIAGNOSTIC)
  Residual != ErrorBound Invariant; Staged Perturbation Chain for Target Q: F_0(Q) ──E_0──> F_1(Q) ──E_1──> ... ──E_{k-1}──> F_k(Q) ⟹ |F_0(Q) - F_k(Q)| ≤ ∑_{i=0}^{k-1} E_i
  Two-Stage NumericalCertificate_Q Derivation Theorem: R_Q ≤ B_back ∧ Sensitivity_Q(κ_Q) ∧ E_{Q,conv} ≤ B_{Q,conv} ⟹ E_{Q,forward} ≤ κ_Q B_back + B_{Q,conv} := B_{Q,forward}
  Canonical Test Matrix: d ∈ {3, 4, 5}, n ∈ {0, 1, 2, 3} validating:
    - Initial Data Anchors: ϕ_0 = 1, ϕ_1 = x
    - Recurrence Invariants: n=0: x ϕ_0 - ϕ_1 = 0; n≥1: x ϕ_n - a_n ϕ_{n+1} - b_n ϕ_{n-1} = 0
    - Orthonormal Jacobi Recurrence: n=0: x e_0 - α_0 e_1 = 0; n≥1: x e_n - α_n e_{n+1} - α_{n-1} e_{n-1} = 0
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

   **Norm Isometry Invariant:** The morphism chain satisfies the exact norm isometry:
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

### Subdiagonal Asymptotic Expansion ($n \to \infty$ with fixed $\lambda$)
$$\boxed{\alpha_n = \frac{1}{2} + \frac{\lambda(1-\lambda)}{4n^2} + O(n^{-3}) \qquad \text{as } n \to \infty \quad (\text{fixed } \lambda).}$$

---

## 6. Layer V & VI: Asymptotic Schemas, Quantified Overlap Contract & Selector

### 6.1 Composite Approximation Schema & Quantified Two-Sided Overlap Majorants
The composite uniform expression combines endpoint Bessel layers and interior WKB waves:
$$F_{\text{comp}}^{(K)} = F_{\text{north}}^{(K)}(z_+) + F_{\text{south}}^{(K)}(z_-) + F_{\text{interior}}^{(K)}(N_n, \theta) - F_{+O}^{(K)}(z_+) - F_{-O}^{(K)}(z_-).$$

The two-sided overlap re-expansion contract is quantified over intermediate overlap domains $Z_0 \le z_\pm \le \delta N_n$:
$$\boxed{\left| F_{\text{north}}^{(K)} - F_{+O}^{(K)} \right| \le N_n^{-K} \, G_{K, \lambda, Z_0, \delta}^+(z_+; N_n), \qquad \left| F_{\text{south}}^{(K)} - F_{-O}^{(K)} \right| \le N_n^{-K} \, G_{K, \lambda, Z_0, \delta}^-(z_-; N_n),}$$
with explicit executable uniform growth bound:
$$\boxed{G^\pm_{K, \lambda, Z_0, \delta}(z; N_n) \le C^\pm_{K, \lambda, Z_0, \delta} (1 + z)^{\gamma_K} \quad \text{uniformly for } N_n \ge N_0 \text{ and } Z_0 \le z \le \delta N_n,}$$
where exponent $\gamma_K = -K + 1/2 - \lambda$.

### 6.2 Certified Domain-Compatible Evaluation Selector ($M^*$)
Pointwise representation selection minimizes local certified forward error bound $B_M(\theta)$ over domain-compatible candidates:
$$\boxed{M^*(\theta) = \arg\min_{\substack{M \\ \theta \in \mathcal{D}_M \\ \texttt{ErrorBound}_M\text{.valid}}} B_M(\theta), \quad \text{where } \texttt{ErrorBound}_M\text{.status} \in \{\texttt{ALGEBRAIC\_EXACT}, \texttt{ARITHMETIC\_EXACT}, \texttt{ANALYTIC\_CERTIFIED}, \texttt{NUMERICAL\_CERTIFIED}\}.}$$

---

## 7. Layer VII: Modular & Multi-Backend Arithmetic Execution Layer

### VII-A & VII-B. Hardware & Exact Symbolic Sub-Backends
- **Polynomial Path:** $\mathbb{Q}[\lambda, x] \to \mathbb{Q} \to \text{RNS/CRT}$ (truth class `ARITHMETIC_EXACT`).
- **Jacobi Golub-Welsch Path:** Eigendecomposition of symmetric tridiagonal $J_m$. When backed by a backward-stable eigensolver with validated residual bounds, status is `NUMERICAL_CERTIFIED` via a formal target-specific `NumericalCertificate`:
  $$\boxed{\texttt{NumericalCertificate}_Q = (Q, A, B_{\text{back}}, R, \kappa_Q, B_{Q,\text{conv}}, B_{Q,\text{forward}}) \implies E_{Q,\text{forward}} \le \kappa_Q \cdot B_{\text{back}} + B_{Q,\text{conv}} := B_{Q,\text{forward}},}$$
  where target $Q \in \{\texttt{NODE}, \texttt{WEIGHT}, \texttt{EIGENVECTOR}, \texttt{QUADRATURE}\}$.

### VII-C. Split Denominators & RNS / CRT Execution Plan
The execution plan distinguishes three distinct denominator structures for rational parameters $\lambda = a/b$ and evaluation point $x = c/r$:
1. **Recurrence Denominators ($D_{\text{rec}}$):**
   $$\boxed{D_{\text{rec}} = \operatorname{lcm}\left(\{ \operatorname{den}_{\text{red}}(a_k), \operatorname{den}_{\text{red}}(b_k) : k = 1 \dots n \} \cup \{b\}\right)}$$
2. **Normalization Denominators ($D_{\text{norm}}$):**
   $$\boxed{D_{\text{norm}} = v_n \cdot \operatorname{den}(h_n^2), \quad \text{where } C_n^{(\lambda)}(1) = u_n/v_n}$$
3. **Evaluation Point Denominators ($D_{\text{eval}}$):**
   $$\boxed{D_{\text{eval}} = r, \quad \text{for } x = c/r.}$$

Excluded primes product: $D_{\text{excl}} = \operatorname{lcm}(D_{\text{rec}}, D_{\text{norm}}, D_{\text{eval}})$.

### VII-D. Explicit Certificate Hierarchy & Bad Primes
1. **Unnormalized Polynomial Certificate ($\texttt{PolyCertificate}$):**
   $$\boxed{\texttt{PolyCertificate}(p, n) \iff \operatorname{Prime}(p) \land \gcd(p, D_{\text{rec}}) = 1.}$$
2. **Point Evaluation Certificate ($\texttt{PointCertificate}$):**
   $$\boxed{\texttt{PointCertificate}(p, x) \iff \gcd(p, D_{\text{eval}}) = 1 \quad (\text{i.e. } p \nmid r).}$$
3. **Polynomial Evaluation Certificate ($\texttt{PolyEvaluationCertificate}$):**
   $$\boxed{\texttt{PolyEvaluationCertificate}(p, n, x) = \texttt{PolyCertificate}(p, n) \land \texttt{PointCertificate}(p, x).}$$
4. **Normalized Zonal Spherical Certificate ($\texttt{ZonalCertificate}$):**
   $$\boxed{\texttt{ZonalCertificate}(p, \text{plan}, \text{degree}=n, \text{point}=x=c/r) \iff \texttt{PolyEvaluationCertificate}(p, n, x) \land (p \nmid v_n) \land (p \nmid u_n)}$$

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
\texttt{ALGEBRAIC\_EXACT} & 0 \quad (\text{relative to certified exact target}) \\
\texttt{ARITHMETIC\_EXACT} & 0 \quad (\text{relative to exact algorithm}) \\
\texttt{ANALYTIC\_CERTIFIED} & B_{\text{theorem}} \\
\texttt{NUMERICAL\_CERTIFIED} & B_{Q,\text{forward}} := \kappa_Q B_{\text{back}} + B_{Q,\text{conv}}
\end{array}
}$$

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
5. **Norm Isometry Invariant:** $\|u_n\|_{L^2(0, \pi)}^2 = \|\phi_n\|_\lambda^2$.

### VIII-C. Provenance Invariant
$$\boxed{\texttt{Residual} \not\implies \texttt{ErrorBound}}$$
Unless a theorem explicitly supplies an implication mapping equation residual to forward error bound, `Residual` objects remain diagnostics and do not participate in certified optimization.

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
  V_n^K = ℂ v_n, ||v_n|| = 1  ⟹  P_{K,n} = v_n ⊗ v_n^*  ⟹  Double Coset K\G/K Parametrization x(g) = ⟨g e_d, e_d⟩ ∈ [-1, 1], ϕ_n(g) = ⟨v_n, π_n(g) v_n⟩, |ϕ_n(x)| ≤ 1
        │
        ▼
  Layer III. Exact Differential Operators, Normalization Types & Dual Recurrences
  Types: C_n^{(λ)}(x) [Poly] | ϕ_n(x) = C_n/C_n(1) [Zonal, ϕ_n(1)=1] | e_n(x) = h_n ϕ_n(x) [Orthonormal, ||e_n||_λ=1]
  Unitary Map: L^2((0,π), (sin θ)^{2λ} dθ) ──u=(sin θ)^λ ϕ──> L^2(0,π)  |  H_λ u_n = N_n^2 u_n
  SL Limit-Circle: λ ∈ (0,1) ⟹ Deficiency index (2,2), asymptotics u ~ θ^λ (regular) vs u ~ θ^{1-λ} (singular), Friedrichs Extension u(0)=u(π)=0
  Dual Recurrences: a_n, b_n ⟷ α_n via α_n^2 = a_n b_{n+1} = (n+1)(n+2λ) / [4(n+λ)(n+λ+1)] (Exact Symbolic Identity I_dual)
        │
        ▼
  Layer IV. Jacobi Spectral Operator & Unitary Matrix Realization
  (M_x f)(x) = x f(x),  U M_x U^{-1} = J = J^*  ⟹  ||J|| = 1 and σ(J) = [-1, 1] as consequences,  ||J_m|| < 1 for m < ∞,  α_n = 1/2 + λ(1-λ)/(4 n^2) + O(n^{-3}) as n → ∞
        │
        ▼
  Layer V. Two-Endpoint Boundary Coordinates
  N_n = n + λ,  z_+ = N_n θ,  z_- = N_n (π - θ)
        │
        ▼
  Layer VI. Two-Overlap Composite Uniform Asymptotic Schema & Quantified Selector
  Composite Approximation: F_comp = F_north + F_south + F_interior - F_{+O} - F_{-O}
  Quantified Overlap Contract: F_endpoint^{(K)} - F_O^{(K)} = O(N_n^{-K}) on Z_0 ≤ z_± ≤ δ N_n
  Evaluation Selector: M^*(\theta) = argmin_{M ∈ ℳ_valid} B_M(\theta) using certified ErrorBound objects B_M
        │
        ▼
  Layer VII. Modular & Multi-Backend Arithmetic Execution Layer
  ├── VII-A: Floating-Point & Fixed-Point (FLOAT32, FLOAT64, LONGDOUBLE, C_fixed, C_LNS)
  ├── VII-B: Exact Rational Symbolic Algebra (Q[λ, x] ──symbolic rec──> C_n ──eval──> Q ──CRT/RNS──> integer residues)
  ├── VII-C: Scalable RNS / CRT (N_max := execution metadata, Excluded primes p ∤ ∏_{j ∈ D_rec} j · b · d, CRT Certificate C_CRT)
  ├── VII-D1: Finite-Field Arithmetic (Split: C_n cert p ∤ ∏_{j ∈ D_rec} j · b · d vs ϕ_n bad-prime cert p ∤ v_n ∧ p ∤ u_n)
  ├── VII-D2: NTT Acceleration Primitive (L_conv = L_1+L_2-1 ≤ L_NTT | (p-1))
  └── VII-E: Golub-Welsch Spectral Matrix Truncation (J_m = tridiag(α_0, ..., α_{m-2}), R_J^abs, R_J_hat)
        │
        ▼
  Layer VIII. Typed Separation: ExactValue vs ErrorBound vs Residual & Provenance Optimizer
  First-Class Types: ExactValue != ErrorBound != Residual (TheoremStatus Enum: VERIFIED_EXACT, ANALYTIC_BOUNDED, EMPIRICAL_DIAGNOSTIC)
  Decomposition: E_total ≤ E_analytic + E_arithmetic + E_conditioning + E_implementation with E_conditioning ≤ κ · E_input
```

---

## 2. Layer I: Representation Geometry & Fischer Decomposition

Let $G = SO(d)$ act transitively on $S^{d-1} \subset \mathbb{R}^d$ with isotropy subgroup $K = SO(d-1)$, so that $S^{d-1} \cong G/K$. For ambient Euclidean dimension $d \ge 3$, the complexified null quadric $Q^{d-2} \subset \mathbb{P}^{d-1}$ is defined by:
$$Q^{d-2} = \{ [z] \in \mathbb{P}^{d-1} : q(z) = z_1^2 + \dots + z_d^2 = 0 \}.$$

The coordinate ring of the projective quadric $Q^{d-2}$ is $R(Q) = \mathbb{C}[z_1, \dots, z_d] / (q)$. Via Fischer decomposition on polynomial spaces:
$$\operatorname{Sym}^n(\mathbb{C}^d) = \mathcal{H}_n(\mathbb{C}^d) \oplus q \operatorname{Sym}^{n-2}(\mathbb{C}^d), \qquad \mathcal{H}_n(\mathbb{C}^d) = \{p \in \operatorname{Sym}^n(\mathbb{C}^d) : \Delta_{\mathbb{C}^d} p = 0\},$$
yielding the vector space isomorphism (dependent on the choice of inner product / Fischer decomposition):
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
Double coset space $K \backslash G / K \cong [-1, 1]$ is parametrized via the double-coset map:
$$x(g) = \langle g e_d, e_d \rangle = \cos\theta \in [-1, 1].$$
The function $\phi_n$ is $K$-bi-invariant ($\phi_n(k_1 g k_2) = \phi_n(g)$), admitting a radial representative $\phi_n(x) \in C^\infty([-1, 1])$:
$$\phi_n(x) = \frac{C_n^{(\lambda)}(x)}{C_n^{(\lambda)}(1)}.$$

### Global Parity & Boundedness Invariants
By unitary representation invariance and symmetry:
$$\boxed{\phi_n(-x) = (-1)^n \phi_n(x), \qquad |\phi_n(x)| \le 1 \quad \forall x \in [-1, 1].}$$

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
   Eigenvalue equation: $L_x \phi_n = -E_n \phi_n$, where $E_n = n(n + 2\lambda)$.
2. **Compact Radial Operator $L_\theta$:** Under coordinate change $x = \cos\theta$:
   $$L_\theta = \frac{d^2}{d\theta^2} + 2\lambda \cot\theta \frac{d}{d\theta}, \qquad \text{Domain: } \theta \in (0, \pi).$$
   Eigenvalue equation: $L_\theta \phi_n(\cos\theta) = -E_n \phi_n(\cos\theta)$.
3. **Sturm-Liouville Hamiltonian $H_\lambda$:** Under unitary transformation $u_n(\theta) = (\sin\theta)^\lambda \phi_n(\cos\theta)$ mapping:
   $$\boxed{L^2\left((0, \pi), (\sin\theta)^{2\lambda} d\theta\right) \xrightarrow{\quad u = (\sin\theta)^\lambda \phi \quad} L^2(0, \pi),}$$
   $$H_\lambda = -\frac{d^2}{d\theta^2} + \lambda(\lambda - 1)\csc^2\theta, \qquad \text{Domain: } \theta \in (0, \pi).$$
   Eigenvalue equation: $H_\lambda u_n = N_n^2 u_n$, where $N_n := n + \lambda$, so $N_n^2 = E_n + \lambda^2$.

### 4.3 Endpoint Singularities & Limit-Circle Boundary Analysis for $H_\lambda$
For $0 < \lambda < 1$ (e.g. $d=3 \implies \lambda = 1/2$), the inverse-square potential $\lambda(\lambda-1)\csc^2\theta$ creates a genuine singular Sturm-Liouville problem at endpoints $\theta = 0, \pi$.
Near $\theta \to 0$, local asymptotic solutions behave as:
$$u(\theta) \sim A \, \theta^\lambda + B \, \theta^{1-\lambda} \qquad (\theta \to 0^+).$$
Since $0 < \lambda < 1$, both solutions $\theta^\lambda$ and $\theta^{1-\lambda}$ are square-integrable in $L^2(0, \pi)$ near $\theta = 0$ (since $2\lambda > -1$ and $2(1-\lambda) > -1$). Thus, $H_\lambda$ is in the **limit-circle case** at both endpoints $\theta = 0$ and $\theta = \pi$, giving a deficiency index of $(2, 2)$.

The self-adjoint extension selecting the spherical harmonics $u_n(\theta) = (\sin\theta)^\lambda \phi_n(\cos\theta)$ is the **Friedrichs extension**, which enforces the regular boundary condition $B = 0$:
$$\boxed{u(\theta) \sim \theta^\lambda \quad (\theta \to 0^+), \qquad u(0) = u(\pi) = 0 \quad (\text{for } n+\lambda > 0).}$$

### 4.4 Norm Formula & Exact Symbolic Dual Recurrence Invariant
The $L^2$ norm of normalized zonal functions $\phi_n(x)$ on $\mathscr{H}_\lambda$ has the exact closed-form expression:
$$\boxed{\|\phi_n\|_\lambda^2 = \int_{-1}^1 \phi_n(x)^2 (1-x^2)^{\lambda - 1/2} dx = \frac{\pi 2^{1-2\lambda} \Gamma(n+2\lambda)}{n!(n+\lambda) \Gamma(\lambda)^2 \left[ C_n^{(\lambda)}(1) \right]^2}.}$$

The framework links two recurrence formulations ($n \ge 1$, with $\phi_0 = 1, \phi_1 = x$):
1. **Polynomial-Normalized Recurrence ($\phi_n(1) = 1$):**
   $$x \phi_n = a_n \phi_{n+1} + b_n \phi_{n-1}, \qquad a_n = \frac{n + 2\lambda}{2(n + \lambda)}, \qquad b_n = \frac{n}{2(n + \lambda)} \quad (a_n + b_n = 1).$$
2. **Orthonormal Jacobi Recurrence ($\|e_n\|_\lambda = 1$):**
   $$x e_n = \alpha_n e_{n+1} + \alpha_{n-1} e_{n-1}, \qquad e_n = h_n \phi_n, \qquad \alpha_n = a_n \frac{h_n}{h_{n+1}} = b_{n+1} \frac{h_{n+1}}{h_n}.$$
3. **Exact Closed Form & Symbolic Identity Invariant ($I_{\text{dual}}(n) = 0$):**
   $$\boxed{\alpha_n^2 = a_n b_{n+1} = \frac{(n+1)(n+2\lambda)}{4(n+\lambda)(n+\lambda+1)}, \qquad I_{\text{dual}}(n) = \alpha_n^2 - a_n b_{n+1} \equiv 0.}$$

### 4.5 Dimension Invariant ($I_{\text{dim}}(n) = 0$)
$$\boxed{I_{\text{dim}}(n) = \dim \mathcal{H}_n(\mathbb{C}^d) - \frac{n+\lambda}{\lambda} C_n^{(\lambda)}(1) = 0.}$$

### 4.6 Mandatory Canonical Verification Anchors
- **$\lambda = 1/2$ (Sphere $S^2$):** $\phi_n(x) = P_n(x)$ (Legendre polynomials).
- **$\lambda = 1$ (Sphere $S^3$):** $H_1 = -\partial_\theta^2 \implies \phi_n(\cos\theta) = \frac{\sin((n+1)\theta)}{(n+1)\sin\theta}$ (Chebyshev second kind $U_n(x)/(n+1)$).

---

## 5. Layer IV: Jacobi Spectral Operator & Asymptotic Expansion

On $\mathscr{H}_\lambda$, coordinate multiplication $(M_x f)(x) = x f(x)$ is a bounded self-adjoint operator. By unitary equivalence $J = U M_x U^{-1}$ to $M_x$:
$$\boxed{\|J\| = \|M_x\| = 1 \quad \text{and} \quad \sigma(J) = \sigma(M_x) = [-1, 1] \quad \text{(as mathematical consequences).}}$$

The tridiagonal symmetric matrix representation on orthonormal basis $e_n$ is:
$$J = \begin{pmatrix} 0 & \alpha_0 & 0 & \cdots \\ \alpha_0 & 0 & \alpha_1 & \cdots \\ 0 & \alpha_1 & 0 & \ddots \\ \vdots & \vdots & \ddots & \ddots \end{pmatrix}, \qquad \alpha_n = \frac{1}{2} \sqrt{\frac{(n+1)(n+2\lambda)}{(n+\lambda)(n+\lambda+1)}}.$$

### Correct Asymptotic Expansion ($n \to \infty$ with fixed $\lambda$)
$$\boxed{\alpha_n = \frac{1}{2} + \frac{\lambda(1-\lambda)}{4n^2} + O(n^{-3}) \qquad \text{as } n \to \infty \quad (\text{fixed } \lambda).}$$

For any finite principal truncation $J_m \in \mathbb{R}^{m \times m}$ ($m < \infty$), spectrum $\sigma(J_m) \subset (-1, 1)$ lies strictly inside the open interval, so $\|J_m\| < 1$. Sanity check for $\lambda=0.5$: $\alpha_0 = 1/\sqrt{3}$.

---

## 6. Layer V & VI: Asymptotic Schemas, Quantified Overlap Contract & Evaluation Selector

Define phase space $(n, \theta, \lambda)$ boundary coordinates using unified scale $N_n := n + \lambda$:
$$z_+ = N_n \theta, \qquad z_- = N_n (\pi - \theta), \qquad N_n = n + \lambda.$$

### 6.1 Composite Approximation Schema & Quantified Overlap Contract
The composite uniform expression combines endpoint Bessel layers and interior WKB waves:
$$\boxed{F_{\text{comp}}^{(K)} = F_{\text{north}}^{(K)}(z_+) + F_{\text{south}}^{(K)}(z_-) + F_{\text{interior}}^{(K)}(N_n, \theta) - F_{+O}^{(K)}(z_+) - F_{-O}^{(K)}(z_-).}$$

The overlap re-expansion matching contract is quantified over explicit intermediate overlap domains $Z_0 \le z_\pm \le \delta N_n$:
$$\boxed{F_{\text{endpoint}}^{(K)} - F_O^{(K)} = O(N_n^{-K}) \qquad \text{for } Z_0 \le z_\pm \le \delta N_n \quad (0 < \delta < \pi/2).}$$

### 6.2 Evaluation Selector ($M^*$)
Pointwise representation selection minimizes local certified forward error bound $B_M(\theta)$:
$$\boxed{M^*(\theta) = \arg\min_{M \in \mathcal{M}_{\text{valid}}} B_M(\theta),}$$
where $B_M(\theta)$ is a typed `ErrorBound` object certifying forward error on domain $\mathcal{D}_M$.

---

## 7. Layer VII: Modular & Multi-Backend Arithmetic Execution Layer

### VII-A. Hardware Floating/Fixed Point
Supports `FLOAT32`, `FLOAT64`, `LONGDOUBLE`, `Q16.16` fixed-point, and `LNS`.

### VII-B. Exact Rational Symbolic Algebra Sub-Backend ($\mathbb{Q}[\lambda, x]$)
1. **Exact Polynomial vs Spectral Paths:**
   - **Polynomial Exact Path:** $\mathbb{Q}[\lambda, x] \to \mathbb{Q} \to \text{RNS/CRT}$ (operates strictly in $\mathbb{Q}$, truth class `ARITHMETIC_EXACT`).
   - **Jacobi Spectral Path:** $\overline{\mathbb{Q}} \to \text{Golub-Welsch}$ (operates in algebraic extension $\overline{\mathbb{Q}}$ due to $\alpha_n$ square roots; labeled `NUMERICAL_APPROX` or `ANALYTIC_CERTIFIED` unless a true algebraic-number backend is provided).
2. **Exact Quadrature Moment Formula:**
   For Gauss-Gegenbauer quadrature on $L^2([-1,1], (1-x^2)^{\lambda-1/2}dx)$, even moments satisfy:
   $$\boxed{\int_{-1}^1 x^{2r}(1-x^2)^{\lambda - 1/2} dx = B\left(r + \frac{1}{2}, \lambda + \frac{1}{2}\right) = \frac{\Gamma(r + 1/2)\Gamma(\lambda + 1/2)}{\Gamma(r + \lambda + 1)}.}$$
   Total mass ($r=0$): $\mu_0 = B\left(\frac{1}{2}, \lambda + \frac{1}{2}\right) = \frac{\sqrt{\pi}\,\Gamma(\lambda+1/2)}{\Gamma(\lambda+1)}$.

### VII-C. Scalable Residue Number System (RNS / CRT) Sub-Backend
1. **Algorithm-Derived Exclusion & Execution Metadata $N_{\max}$:**
   $N_{\max}$ is defined as **execution-plan metadata** representing the maximum evaluated degree $n$.
   - **Algorithm Denominators:** Derived directly from selected recurrence steps $D_{\text{recurrence}} = \{1, 2, \dots, N_{\max}\}$. For rational parameter $\lambda = a/b$ and point $x = c/d$:
     $$\text{Excluded Prime Factor Product: } D = \prod_{j \in D_{\text{recurrence}}} j \cdot b \cdot d.$$
   - **Modulus Exclusion:** $\gcd(m_i, D) = 1$.
2. **Reconstruction Bounds:**
   - **Integer Reconstruction:** Exact for $|X| < M/2$ where $M = \prod m_i$.
   - **Rational Reconstruction:** Recovers $u/v$ ($\gcd(u,v)=1$) when $2UV < M$ with $|u| < U, 0 < v < V$.

### VII-D. Finite-Field Polynomial Arithmetic & NTT Sub-Backends
1. **Split Finite-Field Certificates ($C_n$ vs $\phi_n$):**
   - **$C_n^{(\lambda)}(x)$ Certificate:** Requires prime $p > N_{\max}$ and $p \nmid D$ (where $D = \prod_{j \in D_{\text{recurrence}}} j \cdot b \cdot d$).
   - **$\phi_n(x) = C_n^{(\lambda)}(x)/C_n^{(\lambda)}(1)$ Certificate:** For canonical reduced fraction $C_n^{(\lambda)}(1) = u_n/v_n$ ($\gcd(u_n, v_n)=1$), requires in addition:
     $$\boxed{\mathcal{A}_\phi(p, n) \iff \left( p > N_{\max} \land p \nmid D \land p \nmid v_n \land p \nmid u_n \right).}$$
     For bad primes where $p \mid u_n$, $C_n^{(\lambda)}(1) \equiv 0 \pmod p$, so normalization fails in $\mathbb{F}_p$.
2. **NTT Convolution Primitive:** Linear convolution length $L_{\text{conv}} = L_1 + L_2 - 1 \le L_{\text{NTT}} \mid (p-1)$.

---

## 8. Layer VIII: Typed Separation, Executable Residuals & Provenance Optimizer

### VIII-A. Typed Separation & Theorem Status Enum
The framework enforces a hard type distinction between exact values, error bounds, and residuals:
$$\boxed{\texttt{ExactValue} \neq \texttt{ErrorBound} \neq \texttt{Residual}}$$

`TheoremStatus` is represented as an explicit Enum:
```python
class TheoremStatus(Enum):
    VERIFIED_EXACT = "VERIFIED_EXACT"
    ANALYTIC_BOUNDED = "ANALYTIC_BOUNDED"
    EMPIRICAL_DIAGNOSTIC = "EMPIRICAL_DIAGNOSTIC"
```

 computational forward error decomposes as:
$$\boxed{E_{\text{total}} \le E_{\text{analytic}} + E_{\text{arithmetic}} + E_{\text{conditioning}} + E_{\text{implementation}}, \qquad E_{\text{conditioning}} \le \kappa \cdot E_{\text{input}}.}$$

### VIII-B. Executable Machine-Readable Schema (`Residual`)
Every structural equation residual is instantiated as a typed diagnostic record:
```python
@dataclass
class Residual:
    type: str             # 'recurrence', 'ode', 'schrodinger', 'jacobi_eigenpair', 'moment'
    domain: str           # e.g., 'x in (-1, 1)', 'theta in (0, pi)', 'spectrum'
    scale: float          # characteristic magnitude scale factor S_M
    absolute: float       # absolute residual value R_abs
    normalized: float     # normalized residual R_norm
    conditioning: float   # local condition number kappa
    backend: str          # backend identifier
    status: TheoremStatus # TheoremStatus enum
    tau_M: float          # residual-specific regularization floor max(tau_abs, tau_rel * scale)
```

### VIII-C. Feasibility-First Provenance Optimizer
The solver filters candidate representations using `ErrorBound` objects only, while `Residual` objects remain diagnostic:
$$\boxed{\mathcal{M}_{\text{admissible}} = \left\{ M \in \mathcal{M} : \mathcal{C}_M \text{ valid} \land \text{ErrorBound}_M \le \epsilon_{\text{target}} \right\}, \quad M^* = \arg\min_{M \in \mathcal{M}_{\text{admissible}}} \operatorname{Cost}(M).}$$

---

## 9. Mandatory Canonical Verification Test Suite

The framework must be validated against the following canonical anchors:
1. **$\lambda = 1/2$ Anchor:** $\phi_n(x) = P_n(x)$ (Legendre polynomials).
2. **$\lambda = 1$ Anchor:** $\phi_n(\cos\theta) = \frac{\sin((n+1)\theta)}{(n+1)\sin\theta}$ (Chebyshev second kind $U_n(x)/(n+1)$).
3. **Degree Grid:** $n = 0, 1, 2, 3$.
4. **Coordinate Grid:** $x \in \{-1, -1/2, 0, 1/2, 1\}$.
5. **Endpoint Derivatives:** $k = 0, \dots, n$ at $x = \pm 1$.
6. **Recurrence Identity:** $x \phi_n = a_n \phi_{n+1} + b_n \phi_{n-1}$.
7. **Operator Equivalence:** $L_x \leftrightarrow L_\theta \leftrightarrow H_\lambda$ on respective interior domains.
8. **Invariants:** $I_{\text{dim}}(n) = 0$ and $I_{\text{dual}}(n) = \alpha_n^2 - a_n b_{n+1} = 0$.
9. **Jacobi Eigenvalue Bounds:** Eigenvalues $\sigma(J_m) \subset (-1, 1)$, $\|J_m\| < 1$.
10. **Quadrature Moments:** $\sum w_k x_k^{2r} = B(r+1/2, \lambda+1/2)$.
11. **RNS/CRT Reconstruction:** Rational $\to$ RNS $\to$ exact rational recovery.
12. **Finite-Field Modular Congruence:** Rational $\to$ finite-field $\to$ congruence modulo $p$ (with bad prime check $p \nmid u_n v_n$).

# VIII-Layer Unified Computational Framework for Gegenbauer Polynomials and Spherical Harmonics on $SO(d)/SO(d-1)$

## Abstract

This document presents an architecturally closed VIII-Layer framework for Gegenbauer polynomials $C_n^{(\lambda)}(x)$ and normalized zonal spherical functions $\phi_n(x)$ on the real sphere $S^{d-1} \cong SO(d)/SO(d-1)$, where parameter $\lambda = \frac{d-2}{2}$ ($d \ge 3$). The framework establishes formal operator morphisms connecting representation geometry, quotient algebras, exact differential operators, dual self-adjoint Jacobi spectral and polynomial recurrences, candidate two-endpoint singular asymptotic envelopes, multi-backend numerical execution (including exact rational, RNS/CRT, and finite-field sub-backends), and high-precision verification invariants.

---

## 1. VIII-Layer Architectural Pipeline

```
  Layer I. Representation Geometry & Fischer Decomposition
  G = SO(d), K = SO(d-1), Sym^n = ℋ_n ⊕ q Sym^{n-2}, Q^{d-2} ⊂ ℙ^{d-1}, R(Q)_n ≅ ℋ_n(ℂ^d), ℋ_n(ℝ^d) ──→ 𝒴_n(S^{d-1})
        │
        ▼
  Layer II. Spherical Fixed Line, Rank-One Projector & Bi-K-Invariance
  v_n ∈ V_n^K, ||v_n|| = 1  ⟹  P_{K,n} = v_n ⊗ v_n^*  ⟹  ϕ_n is K-bi-invariant with radial representative ϕ_n(x) ∈ C^∞([-1, 1]), x = cos θ, |ϕ_n(x)| ≤ 1
        │
        ▼
  Layer III. Exact Differential Operators & Dual Recurrence Duality
  -Δ_{S^{d-1}} ϕ_n = E_n ϕ_n  |  L_x ↔ L_θ ↔ H_λ u_n = N_n^2 u_n  |  Dual Recurrence Conversion: a_n, b_n ⟷ α_n via h_n
        │
        ▼
  Layer IV. Jacobi Spectral Operator & Unitary Matrix Realization
  (M_x f)(x) = x f(x),  U M_x U^{-1} = J = J^*,  ||J|| = 1,  α_n = 1/2 + O(n^{-2}) as n → ∞
        │
        ▼
  Layer V. Two-Endpoint Boundary Coordinates
  N_n = n + λ,  z_+ = N_n θ,  z_- = N_n (π - θ)
        │
        ▼
  Layer VI. Two-Overlap Composite Uniform Asymptotic Expansion & Theorem Layer
  F_comp = F_north + F_south + F_interior - F_{+,overlap} - F_{-,overlap},  Match^{(K)} = coeff-wise re-expansion mod O(N_n^{-K})
  Assembly Protocol: Match^{(K)} ⟹ Overlap Consistency; Theorem Backend ⟹ |R_{K,M}| ≤ B_{K,M}
        │
        ▼
  Layer VII. Modular & Multi-Backend Arithmetic Execution Layer
  ├── VII-A: Floating-Point & Fixed-Point (FLOAT32, FLOAT64, LONGDOUBLE, Q16.16, LNS)
  ├── VII-B: Exact Rational Symbolic Algebra (Symbolic Q[λ, x], Rational Eval λ, x ∈ Q, Exact Recurrence)
  ├── VII-C: Scalable Residue Number System (RNS / CRT with Certificate C_CRT = A_denom ∧ B_num/den ∧ (2UV < M))
  ├── VII-D1: Finite-Field Polynomial Arithmetic (F_p, A_ϕ(p, n): p > N_max ∧ p ∤ v_n ∧ u_n ≢ 0 mod p)
  ├── VII-D2: Number Theoretic Transform Acceleration (NTT Primitive: L_conv = a+b-1 ≤ L_NTT | (p-1))
  └── VII-E: Golub-Welsch Spectral Matrix Truncation (J_m = tridiag(α_0, ..., α_{m-2}), R_J^abs, R_J_hat)
        │
        ▼
  Layer VIII. Verification Invariants, Error Taxonomy & Feasibility-First Optimizer
  Four Truth Classes (ALGEBRAIC_EXACT, ARITHMETIC_EXACT, ANALYTIC_CERTIFIED, NUMERICAL_APPROX)
  Taxonomy (R_structural, E_forward, κ, E_backend) ↔ Invariants (Parity, Boundedness, Moments, Dual Conversion)
  Exactness C_A^exact = A_A ∧ I_A ∧ T_A ⟹ ε_A^cert = 0 ↔ Feasibility-First Optimizer over M_admissible
```

---

## 2. Layer I: Representation Geometry & Fischer Decomposition

Let $G = SO(d)$ act transitively on $S^{d-1} \subset \mathbb{R}^d$ with isotropy subgroup $K = SO(d-1)$, so that $S^{d-1} \cong G/K$. For ambient Euclidean dimension $d \ge 3$, the complexified null quadric $Q^{d-2} \subset \mathbb{P}^{d-1}$ is defined by:
$$Q^{d-2} = \{ [z] \in \mathbb{P}^{d-1} : q(z) = z_1^2 + \dots + z_d^2 = 0 \}.$$

The coordinate ring of the projective quadric $Q^{d-2}$ is $R(Q) = \mathbb{C}[z_1, \dots, z_d] / (q)$. Via Fischer decomposition:
$$\operatorname{Sym}^n(\mathbb{C}^d) = \mathcal{H}_n(\mathbb{C}^d) \oplus q \operatorname{Sym}^{n-2}(\mathbb{C}^d), \qquad \mathcal{H}_n(\mathbb{C}^d) = \{p \in \operatorname{Sym}^n(\mathbb{C}^d) : \Delta_{\mathbb{C}^d} p = 0\},$$
yielding the canonical representation-theoretic isomorphism:
$$V_n \cong R(Q)_n \cong \operatorname{Sym}^n(\mathbb{C}^d) / q \operatorname{Sym}^{n-2}(\mathbb{C}^d) \cong \mathcal{H}_n(\mathbb{C}^d).$$

The space $V_n \cong \mathcal{H}_n(\mathbb{C}^d)$ represents the space of degree-$n$ spherical harmonics. The restriction map to $S^{d-1}$:
$$\mathcal{H}_n(\mathbb{C}^d) \longrightarrow \mathscr{Y}_n(S^{d-1}) = \{ p|_{S^{d-1}} : p \in \mathcal{H}_n(\mathbb{R}^d) \},$$
maps harmonic polynomials to spherical harmonics $\mathscr{Y}_n(S^{d-1})$.

### Hilbert Series & Representation Dimension
$$\dim V_n = [t^n] \frac{1 - t^2}{(1 - t)^d} = \binom{n + d - 1}{d - 1} - \binom{n + d - 3}{d - 1} = \frac{n + \lambda}{\lambda} C_n^{(\lambda)}(1),$$
where $C_n^{(\lambda)}(1) = \frac{(2\lambda)_n}{n!} = \binom{n + 2\lambda - 1}{n}$.

To ensure uniform validity for every $n \ge 0$ (including $n=0$ and $n=1$), binomial coefficients with upper arguments smaller than the lower argument are defined by:
$$\binom{r}{d-1} = 0 \qquad \text{for } r < d - 1, \qquad \text{or equivalently } \operatorname{Sym}^m(\mathbb{C}^d) = 0 \text{ for } m < 0.$$

---

## 3. Layer II: Spherical Fixed Line, Rank-One Projector & Bi-$K$-Invariance

Let $P_{K,n} = \int_K \pi_n(k) dk$ project $V_n$ onto the 1-dimensional $K$-fixed subspace $V_n^K = \mathbb{C} v_n$ ($\|v_n\| = 1$). Inside representation $V_n$, $P_{K,n}$ equals the rank-one orthogonal projector $P_{K,n} = v_n \otimes v_n^* \in \operatorname{End}(V_n)$.

Trace pairing defines the zonal spherical function $\phi_n(g) = \operatorname{Tr}(P_{K,n} \pi_n(g)) = \langle v_n, \pi_n(g) v_n \rangle$. The function $\phi_n$ is $K$-bi-invariant ($\phi_n(k_1 g k_2) = \phi_n(g)$ and $\phi_n(e) = 1$), admitting a radial representative $\phi_n(x) \in C^\infty([-1, 1])$ via $x = \cos\theta$:
$$\phi_n(x) = \frac{C_n^{(\lambda)}(x)}{C_n^{(\lambda)}(1)}.$$

### Global Parity & Boundedness Invariants
By unitary representation invariance and symmetry:
$$\boxed{\phi_n(-x) = (-1)^n \phi_n(x), \qquad |\phi_n(x)| \le 1 \quad \forall x \in [-1, 1].}$$

---

## 4. Layer III: Exact Operator Equivalence & Dual Recurrence Duality

On the sphere $S^{d-1}$, $\phi_n$ is the eigenfunction of the Laplace-Beltrami operator $-\Delta_{S^{d-1}}$:
$$-\Delta_{S^{d-1}} \phi_n = E_n \phi_n, \qquad E_n = n(n + 2\lambda).$$

Equivalence across three operator representations:
1. **Algebraic Differential Operator $L_x$:** $(1 - x^2) \phi'' - (2\lambda + 1)x \phi' + E_n \phi = 0$.
2. **Compact Radial Operator $L_\theta$:** $\phi'' + 2\lambda \cot\theta \, \phi' + E_n \phi = 0$.
3. **Sturm-Liouville Hamiltonian $H_\lambda$:** $-u_n'' + \lambda(\lambda - 1)\csc^2\theta \, u_n = N_n^2 u_n$, where $u_n = (\sin\theta)^\lambda \phi_n$ and $N_n := n + \lambda$, so $N_n^2 = E_n + \lambda^2$.

### Dual Recurrence Architectures & Exact Conversion Square
The framework distinguishes two distinct three-term recurrences and provides their exact conversion square:
1. **Polynomial-Normalized Recurrence ($\phi_n(1) = 1$):**
   $$x \phi_n = a_n \phi_{n+1} + b_n \phi_{n-1}, \qquad a_n = \frac{n + 2\lambda}{2(n + \lambda)}, \qquad b_n = \frac{n}{2(n + \lambda)}, \qquad a_n + b_n = 1.$$
2. **Orthonormal Jacobi Recurrence ($\|e_n\| = 1$):**
   $$x e_n = \alpha_n e_{n+1} + \alpha_{n-1} e_{n-1}, \qquad e_n = h_n \phi_n, \qquad h_n = \|\phi_n\|^{-1}.$$
3. **Exact Commutative Conversion Square:**
   $$\boxed{\alpha_n = a_n \frac{h_n}{h_{n+1}}, \qquad \alpha_{n-1} = b_n \frac{h_n}{h_{n-1}}.}$$

### Mandatory Canonical Verification Families
- **$\lambda = 1/2$ (Sphere $S^2$):** $\phi_n(x) = P_n(x)$ (Legendre polynomials).
- **$\lambda = 1$ (Sphere $S^3$):** $H_1 = -\partial_\theta^2 \implies \phi_n(\theta) = \frac{\sin((n+1)\theta)}{(n+1)\sin\theta}$.

---

## 5. Layer IV: Jacobi Spectral Operator & Unitary Matrix Realization

On the Hilbert space $\mathscr{H}_\lambda = L^2([-1, 1], (1-x^2)^{\lambda-1/2} dx)$, coordinate multiplication $(M_x f)(x) = x f(x)$ is a bounded self-adjoint multiplication operator with $\|M_x\| = 1$ and spectrum $\sigma(M_x) = [-1, 1]$.

Let $U : \mathscr{H}_\lambda \to \ell^2(\mathbb{N}_0)$ map orthonormal basis $e_n = h_n \phi_n$ to canonical basis $\mathbf{e}_n$. The unitary matrix realization $J = U M_x U^{-1} = J^*$ is a tridiagonal symmetric matrix:
$$J = \begin{pmatrix} 0 & \alpha_0 & 0 & \cdots \\ \alpha_0 & 0 & \alpha_1 & \cdots \\ 0 & \alpha_1 & 0 & \ddots \\ \vdots & \vdots & \ddots & \ddots \end{pmatrix}, \qquad \alpha_n = \frac{1}{2} \sqrt{\frac{(n+1)(n+2\lambda)}{(n+\lambda)(n+\lambda+1)}} = \frac{1}{2} - \frac{\lambda(\lambda-1)}{4n^2} + O(n^{-3}) \quad \text{as } n \to \infty.$$
Matrix invariants: $J = J^*, \|J\| = 1, \sigma(J) = [-1, 1]$. Sanity check for $\lambda=0.5$: $\alpha_0 = 1/\sqrt{3}$. Exact $\alpha_n$ remains canonical for small $n$.

---

## 6. Layer V & VI: Singular Scaling & Two-Overlap Uniform Asymptotics

Define 3D phase space $(n, \theta, \lambda)$ boundary coordinates using unified singular scale $N_n := n + \lambda$:
$$z_+ = N_n \theta, \qquad z_- = N_n (\pi - \theta), \qquad N_n = n + \lambda.$$

### Composite Matched Asymptotic Assembly Protocol & Theorem Layer
Layer VI establishes a formal asymptotic assembly protocol. For truncation order $K$ ($F^{(K)} = \sum_{j=0}^{K-1} N_n^{-j} F_j$), overlap approximants $F_{+O}^{(K)}$ and $F_{-O}^{(K)}$ are defined by coefficient-wise common asymptotic re-expansion modulo $O(N_n^{-K})$ in the corresponding overlap scaling (e.g. $\theta = z_+ / N_n, z_+ \to \infty, \theta \to 0$ for North overlap):
$$\boxed{F_{\pm O}^{(K)} = \operatorname{Match}^{(K)}\left( F_{\text{endpoint}}^{(K)}, F_{\text{interior}}^{(K)} \right) = \text{coefficient-wise common re-expansion modulo } O(N_n^{-K}),}$$
preserving $F_{\text{endpoint}}^{(K)} - F_{\pm O}^{(K)} = O(N_n^{-K})$ and $F_{\text{interior}}^{(K)} - F_{\pm O}^{(K)} = O(N_n^{-K})$ in the overlap scaling.

$\operatorname{Match}^{(K)}$ establishes formal **overlap consistency**, whereas proving $|R_{K, M}| \le B_{K, M}$ requires an independent asymptotic remainder theorem.

For each branch $M \in \{N, S, I, +O, -O\}$, branch remainder is defined by $R_M = \phi - F_M^{(K)}$. The composite uniform expansion is:
$$F_{\text{comp}}^{(K)} = F_{\text{north}}^{(K)}(z_+) + F_{\text{south}}^{(K)}(z_-) + F_{\text{interior}}^{(K)}(N_n, \theta) - F_{+,\text{overlap}}^{(K)}(z_+) - F_{-,\text{overlap}}^{(K)}(z_-),$$
and composite remainder $R_{\text{comp}} = \phi - F_{\text{comp}}^{(K)} = R_N + R_S + R_I - R_{+O} - R_{-O}$ satisfies the exact identity:
$$\boxed{R_{\text{comp}} = R_N + R_S + R_I - R_{+O} - R_{-O} \implies |R_{\text{comp}}| \le B_{K, \text{comp}} \le B_{K, N} + B_{K, S} + B_{K, I} + B_{K, +O} + B_{K, -O},}$$
where $|R_{K, M}| \le B_{K, M}(N_n, \theta, \lambda)$ provides local branch certificates.

1. **Interior Domain Envelope ($\mathcal{D}_{\text{int}}(\delta) = \{\theta : \delta \le \theta \le \pi - \delta\}$):**
   $$|R_{K, \text{int}}| \le B_{K, \text{int}}(N_n, \theta, \lambda) \le N_n^{-K} A_{\lambda, K}(\theta), \qquad A_{\lambda, K}(\theta) \lesssim (\sin\theta)^{-\lambda - \eta_K},$$
   where $A_{\lambda, K}(\theta) > 0$ is derived by the asymptotic theorem backend.
2. **North / South Endpoint Envelopes ($z_+ = N_n\theta \le Z_+, z_- = N_n(\pi-\theta) \le Z_-$):**
   Nonnegative majorant envelopes bounding the full omitted asymptotic series:
   $$|R_{K, \pm}| \le B_{K, \pm}, \qquad B_{K, +}(N_n, z_+, \lambda) = N_n^{-K} C_{\lambda, K}^+ A_{K, \lambda}^+(z_+; N_n), \qquad B_{K, -}(N_n, z_-, \lambda) = N_n^{-K} C_{\lambda, K}^- A_{K, \lambda}^-(z_-; N_n),$$
   where $A_{K, \lambda}^\pm(z; N_n) > 0$ is a strictly positive majorant (avoiding zeroes at Bessel oscillating nodes).
3. **Coverage Partition vs. Multiple Representation Overlaps:**
   Domain coverage requirement: $\mathcal{D}_+ \cup \mathcal{D}_- \cup \mathcal{D}_I = [0, \pi]$ with overlapping regions $\mathcal{D}_+ \cap \mathcal{D}_I \neq \emptyset$. Branch-ordering ambiguity in overlaps is resolved via envelope minimization over valid representations $\mathcal{M}_{\text{valid}}(N_n, \theta, \lambda) = \{M : \mathcal{C}_M \text{ is valid at } (N_n, \theta, \lambda)\}$:
   $$B_K^{\text{best}}(N_n, \theta, \lambda) = \min_{M \in \mathcal{M}_{\text{valid}}} B_{K, M}(N_n, \theta, \lambda).$$

4. **Backend Capability Certificates $\mathcal{C}_M$:**
   Each execution backend $M \in \mathcal{M}$ provides a capability certificate tuple $\mathcal{C}_M = (\mathcal{D}_M, \mathcal{P}_M, \mathcal{E}_M, \mathcal{R}_M)$ specifying valid domain $\mathcal{D}_M$, parameter admissibility $\mathcal{P}_M$, error model $\mathcal{E}_M$, and residual checkers $\mathcal{R}_M$.

For solver optimization across representations $\mathcal{M} = \{\text{rec}, \text{Bessel}_+, \text{Bessel}_-, \text{WKB}, \text{comp}\}$, adaptive execution selects $M^* = \arg\min_{M \in \mathcal{M}_{\text{admissible}}} \operatorname{Cost}(M)$, where total estimated selection error decomposes as:
$$\widehat{E}_M = \widehat{E}_M^{\text{trunc}} + \widehat{E}_M^{\text{arith}} + \widehat{E}_M^{\text{cond}} + \widehat{E}_M^{\text{model}}.$$

---

## 7. Layer VII: Modular & Multi-Backend Arithmetic Execution Layer

Layer VII implements diverse arithmetic realizations of Gegenbauer polynomials and zonal functions $\phi_n(x)$ across distinct numerical backends:

### VII-A. Floating-Point & Fixed-Point Hardware Execution
Supports `FLOAT32`, `FLOAT64`, `LONGDOUBLE`, `Q16.16` fixed-point, and Logarithmic Number Systems (`LNS`).

### VII-B. Exact Rational Symbolic Algebra Sub-Backend ($\mathbb{Q}[\lambda, x]$)
1. **Symbolic Polynomial Identity vs. Evaluation Theorem:**
   - **Symbolic Polynomial Identity:** $C_n^{(\lambda)}(x) \in \mathbb{Q}[\lambda, x]$.
   - **Evaluation Theorem:** $\lambda, x \in \mathbb{Q} \implies C_n^{(\lambda)}(x) \in \mathbb{Q}$ (with reduced fraction certificate $\mathcal{A}_{\text{rat}} = \operatorname{RatCert}(\lambda, x, N_{\max}) \land (\gcd(a,b)=\gcd(c,d)=1) \land (b,d>0) \land (C_n^{(\lambda)}(1) \neq 0)$ where $\lambda = a/b, x = c/d$). Exact rational arithmetic eliminates floating-point rounding errors subject to exact rational arithmetic semantics.
2. **Polynomial Path vs. Jacobi Spectral Path:**
   - **Polynomial Exact Path:** $\mathbb{Q}[\lambda, x] \to \mathbb{Q} \to \text{RNS/CRT}$ (operates strictly in $\mathbb{Q}$).
   - **Jacobi Spectral Path:** $\overline{\mathbb{Q}} \to \text{Golub-Welsch}$ (operates in algebraic extension $\overline{\mathbb{Q}}$ due to $\alpha_n$ square roots).
3. **Three-Term Polynomial Recurrence:**
   $$C_0^{(\lambda)}(x) = 1, \qquad C_1^{(\lambda)}(x) = 2\lambda x, \qquad n C_n^{(\lambda)}(x) = 2(n+\lambda-1)x C_{n-1}^{(\lambda)}(x) - (n+2\lambda-2) C_{n-2}^{(\lambda)}(x).$$
4. **Fraction Bit-Length Tracking:** Denominator and numerator growth is tracked via separate metrics $B_{\text{bits}}^{(C)}(n)$ and $B_{\text{bits}}^{(\phi)}(n)$:
   $$B_{\text{bits}}^{(C)}(n) = \max\{ \operatorname{bitlen}(\operatorname{num}(C_n)), \operatorname{bitlen}(\operatorname{den}(C_n)) \}, \qquad B_{\text{bits}}^{(\phi)}(n) = \max\{ \operatorname{bitlen}(\operatorname{num}(\phi_n)), \operatorname{bitlen}(\operatorname{den}(\phi_n)) \}.$$
5. **Differential Derivative Generation (Interior $-1 < x < 1$):**
   $$\frac{d}{dx} C_n^{(\lambda)}(x) = 2\lambda C_{n-1}^{(\lambda+1)}(x), \qquad y'' = \frac{(2\lambda+1)x y' - n(n+2\lambda)y}{1-x^2} \qquad (-1 < x < 1).$$
6. **Golub-Welsch Spectral Quadrature:**
   Golub-Welsch spectral decomposition isolates the algebraic spectral data $(x_k, v_{k,1}^2)$ (where $x_k \in \sigma(J_m) = \{x_1, \dots, x_m\}$ are the eigenvalues of symmetric $m \times m$ Jacobi truncation $J_m$) from the global transcendental scalar normalization:
   $$\mu_0 = \int_{-1}^1 (1-x^2)^{\lambda-1/2} dx = \frac{\sqrt{\pi}\,\Gamma(\lambda+1/2)}{\Gamma(\lambda+1)} = B\left(\frac{1}{2}, \lambda + \frac{1}{2}\right).$$
   Gauss-Gegenbauer $m$-point quadrature $\int_{-1}^1 f(x)(1-x^2)^{\lambda-1/2} dx \approx \sum_{k=1}^m w_k f(x_k)$ (with $w_k = \mu_0 v_{k,1}^2$) avoids direct endpoint evaluation at $x = \pm 1$, reducing endpoint singularity exposure. Reserving $n$ for Gegenbauer degree and $m$ for quadrature order.

### VII-C. Scalable Residue Number System (RNS / CRT) Sub-Backend
Single binary unsigned hardware wrap channels ($\texttt{uint}_b \simeq \mathbb{Z}/2^b \mathbb{Z}$) are distinguished from multi-modulus prime/coprime RNS systems ($\prod \mathbb{Z}/m_i \mathbb{Z}$).
1. **Admissibility & Modular Recurrence:** Evaluated over pairwise coprime moduli $m_1, \dots, m_k$. For rational parameter $\lambda = a/b$ and rational point $x = c/d$, recurrence step admissibility requires:
   $$\gcd\left( m_i, b \cdot d \cdot \operatorname{lcm}(1, \dots, N_{\max}) \right) = 1.$$
   For prime moduli $p_i$, this simplifies to $p_i > \max(2, N_{\max})$, $p_i \nmid b$, and $p_i \nmid d$.
2. **Bounded Rational / Integer CRT Reconstruction Certificate ($\mathcal{C}_{\text{CRT}}$):**
   $$\boxed{\mathcal{C}_{\text{CRT}} = \mathcal{A}_{\text{denom}} \land \mathcal{B}_{\text{num/den}} \land (2UV < M),}$$
   where $M = \prod_{i=1}^k m_i$. Integer reconstruction $X \equiv \sum r_i M_i (M_i^{-1} \bmod m_i) \pmod M$ is exact for $|X| < M/2$. Rational reconstruction recovers canonical reduced $u/v$ ($\gcd(u,v)=1$) subject to $|u| < U, 0 < v < V$ with $2UV < M$.

### VII-D. Finite-Field Polynomial Arithmetic & NTT Acceleration Sub-Backends
Conceptually separates finite-field polynomial arithmetic from NTT transform acceleration:
1. **Finite-Field Polynomial Arithmetic Sub-Backend ($\mathbb{F}_p$):**
   For canonical reduced fraction $C_n^{(\lambda)}(1) = \frac{u_n}{v_n}$ with $\gcd(u_n, v_n) = 1$, the normalized finite-field certificate is:
   $$\boxed{\mathcal{A}_\phi(p, n) \iff \left( p > N_{\max} \land p \nmid b \cdot d \land p \nmid v_n \land u_n \not\equiv 0 \pmod p \right).}$$
   Define the rational localization reduction map $\rho_p : \mathbb{Z}_{(p)} \to \mathbb{F}_p$. Then $\rho_p(C_n^{(\lambda)}(1)) = u_n v_n^{-1} \in \mathbb{F}_p^\times$, field parameters $\lambda_p = a b^{-1} \in \mathbb{F}_p$, point $x_p = c d^{-1} \in \mathbb{F}_p$, and normalized zonal evaluation:
   $$\boxed{\phi_{n,p}(x_p) = \rho_p(C_n^{(\lambda)}(x)) \left( u_n v_n^{-1} \right)^{-1} \in \mathbb{F}_p.}$$
2. **Number Theoretic Transform (NTT) Fast Convolution Acceleration Primitive:**
   Over finite prime fields $\mathbb{F}_p$ where desired convolution length $L_{\text{conv}} = a + b - 1$ satisfies $L_{\text{conv}} \le L_{\text{NTT}} \mid (p-1)$ (using $L_{\text{NTT}}$ to avoid collision with $N_n = n+\lambda$), primitive $L_{\text{NTT}}$-th roots of unity $\omega_{L_{\text{NTT}}} \in \mathbb{F}_p$ provide fast polynomial coefficient-domain multiplication:
   $$\boxed{\text{NTT} : L_{\text{conv}} = a + b - 1 \le L_{\text{NTT}} \mid (p-1) \implies \text{coefficient-domain fast convolution.}}$$

### VII-E. Golub-Welsch Spectral Matrix Truncation
For Gauss-Gegenbauer quadrature calculations, the symmetric tridiagonal Jacobi matrix operator $J$ is truncated via orthogonal projection $P_m$ to its leading $m \times m$ principal truncation $J_m = \operatorname{tridiag}(\alpha_0, \dots, \alpha_{m-2}) = P_m J P_m |_{\operatorname{span}\{e_0, \dots, e_{m-1}\}} \in \mathbb{R}^{m\times m}$. Its eigenvalues $\sigma(J_m) = \{x_1, \dots, x_m\}$ yield quadrature nodes $x_k$, and weights are $w_k = \mu_0 |(v_k)_1|^2$ using normalized eigenvector $v_k$ of $J_m$.

---

## 8. Layer VIII: Verification Invariants & Cross-Backend Certification Layer

Layer VIII provides formal verification and cross-backend error certification comparing results across independent arithmetic realizations:

### VIII-A. Formal Error Taxonomy & Four Truth Classes
Layer VIII explicitly distinguishes four truth/exactness classes:
$$\boxed{\texttt{ALGEBRAIC\_EXACT} \quad \text{(symbolic identity)}}$$
$$\boxed{\texttt{ARITHMETIC\_EXACT} \quad \text{(exact rational / RNS reconstruction)}}$$
$$\boxed{\texttt{ANALYTIC\_CERTIFIED} \quad \text{(proved asymptotic bounds)}}}$$
$$\boxed{\texttt{NUMERICAL\_APPROX} \quad \text{(floating-point residual diagnostics)}}$$

Layer VIII explicitly distinguishes four error concepts:
1. **Structural Residual ($R_{\text{structural}}$):** Normalized equation residual (e.g. $\widehat{R}_{\text{rec}}, \widehat{R}_{\text{ODE}}, \widehat{R}_{\text{Schr}}, R_J^{\text{abs}}, \widehat{R}_J$).
2. **Forward Error ($E_{\text{forward}}$):** Discrepancy $|\hat{\phi} - \phi|$ from exact ground truth. Note: $R_{\text{structural}} = 0 \centernot\implies E_{\text{forward}} = 0$ and $E_{\text{backend}} \approx 0 \centernot\implies E_{\text{forward}} = 0$.
3. **Conditioning ($\kappa$):** Problem sensitivity under perturbation.
4. **Backend Discrepancy ($E_{\text{backend}}$):** Cross-implementation error $E_{A,B} = |\hat{\phi}^{(A)} - \hat{\phi}^{(B)}|$.

Zero arithmetic forward error $\varepsilon_A^{\text{certified}} = 0$ is guaranteed by full implementation correctness certificates incorporating algorithmic correctness theorem $\mathcal{T}_A$:
$$\boxed{\mathcal{C}_A^{\text{exact}} \implies \varepsilon_A^{\text{certified}} = 0, \qquad \mathcal{C}_A^{\text{exact}} = \mathcal{A}_A \land \mathcal{I}_{\text{impl}} \land \mathcal{T}_A,}$$
where backend-specific exactness certificates are defined by:
$$\boxed{\mathcal{C}_{\text{rat}}^{\text{exact}} = \mathcal{A}_{\text{rat}} \land \mathcal{I}_{\text{impl}} \land \mathcal{T}_{\text{rat}}, \qquad \mathcal{C}_{\text{RNS}}^{\text{exact}} = \mathcal{A}_{\text{rec}} \land \mathcal{A}_{\text{norm}} \land \mathcal{A}_{\text{CRT}} \land \mathcal{I}_{\text{impl}} \land \mathcal{T}_{\text{RNS}}.}$$

### VIII-B. Scale-Invariant Structural Residual Invariants & Residual-Specific Floors
To ensure comparable residual evaluations across independent backends and distinct residual types, the regularization floor is explicitly backend-dependent and residual-scale aware:
$$\boxed{\tau_M \ge 0, \qquad \tau_M = \max(\tau_{\text{abs}}, \tau_{\text{rel}} S_M),}$$
where $S_M \in \{S_M^{\text{rec}}, S_M^{\text{ODE}}, S_M^{\text{Schr}}, S_J\}$ is the characteristic magnitude scale factor for representation backend $M$ and specific equation type (with $\tau_J = \max(\tau_{\text{abs}}, \tau_{\text{rel}} S_J)$).

- **Normalized Recurrence Residual:** $\widehat{R}_{\text{rec}}(n, x) = \frac{|x \hat{\phi}_n - a_n \hat{\phi}_{n+1} - b_n \hat{\phi}_{n-1}|}{|x \hat{\phi}_n| + |a_n \hat{\phi}_{n+1}| + |b_n \hat{\phi}_{n-1}| + \tau_M}$ defined for $n \ge 1$ (with initial conditions $\phi_0 = 1, \phi_1 = x$).
- **Normalized ODE Residual:** Defined for interior $x \in (-1, 1)$ to avoid endpoint $1-x^2=0$ cancellation:
  $$\widehat{R}_{\text{ODE}}(x) = \frac{|(1-x^2)\hat{\phi}'' - (2\lambda+1)x\hat{\phi}' + E_n\hat{\phi}|}{|1-x^2||\hat{\phi}''| + |(2\lambda+1)x||\hat{\phi}'| + E_n|\hat{\phi}| + \tau_M} \approx 0.$$
- **Normalized Schrödinger Residual:** Defined for interior $\theta \in (0, \pi)$ to avoid $\csc^2\theta$ endpoint singularity:
  $$\widehat{R}_{\text{Schr}}(\theta) = \frac{|-\hat{u}'' + \lambda(\lambda-1)\csc^2\theta \, \hat{u} - N_n^2 \hat{u}|}{|\hat{u}''| + |\lambda(\lambda-1)\csc^2\theta \, \hat{u}| + N_n^2 |\hat{u}| + \tau_M}.$$
- **Jacobi Matrix Spectral Eigenpair Residuals:** Absolute and normalized verification residuals for Golub-Welsch spectral backend:
  $$\boxed{R_J^{\text{abs}}(v, x) = \|J_m v - x v\|, \qquad \widehat{R}_J(v, x) = \frac{\|J_m \hat{v} - \hat{x} \hat{v}\|}{\|J_m \hat{v}\| + |\hat{x}| \|\hat{v}\| + \tau_J}.}$$
- **Endpoint Anchors & Normalization Residual:** $R_+ = |\hat{\phi}_n(1) - 1|$, $R_- = |\hat{\phi}_n(-1) - (-1)^n|$, $R_{\text{norm}} = |C_n^{(\lambda)}(1)\hat{\phi}_n(x) - \hat{C}_n^{(\lambda)}(x)|$.
- **Exact High-Order Endpoint Derivative Formulas:** Total domain $k \in \mathbb{N}_0$ with $\phi_n^{(k)} \equiv 0$ for $k > n$:
  $$\phi_n^{(k)}(1) = \frac{2^k (\lambda)_k C_{n-k}^{(\lambda+k)}(1)}{C_n^{(\lambda)}(1)} \quad (0 \le k \le n), \qquad \phi_n^{(k)}(-1) = (-1)^{n-k} \phi_n^{(k)}(1), \qquad R_{\pm, k} = |\hat{\phi}_n^{(k)}(\pm 1) - \phi_n^{(k)}(\pm 1)|.$$
- **Gauss-Gegenbauer Quadrature Weight Normalization & Moment Invariants:** $m$-point Gauss quadrature exactness certified against closed-form moments ($0 \le j \le 2m-1$):
  $$\boxed{\sum_{k=1}^m w_k = \mu_0 = B\left(\frac{1}{2}, \lambda+\frac{1}{2}\right) \quad (j=0),} \qquad \sum_{k=1}^m w_k x_k^j = \begin{cases} 0, & j \text{ is odd}, \\ B\left(r+\frac{1}{2}, \lambda+\frac{1}{2}\right), & j = 2r \text{ is even}. \end{cases}$$

### VIII-C. Feasibility-First Optimizer over Admissible Representations
Rather than minimizing an arbitrary additive heuristic score, the solver employs a feasibility-first candidate selection architecture over admissible representation backends $\mathcal{M}_{\text{admissible}}$:
$$\boxed{\mathcal{M}_{\text{admissible}} = \left\{ M \in \mathcal{M} : \mathcal{C}_M \land B_M \le \epsilon_{\text{target}} \land E_M^{\text{arith}} \le \epsilon_{\text{target}} \right\}, \qquad M^* = \arg\min_{M \in \mathcal{M}_{\text{admissible}}} \operatorname{Cost}(M).}$$

### VIII-D. Cross-Backend Error Certification ($E_{A,B}$) & Commutative Reduction
Layer VIII-D is structured into two distinct verification parts:

1. **Real-Valued Numerical Discrepancy Bounds ($E_{A,B}^{\mathbb{R}}$):**
   $$E_{A,B}^{\mathbb{R}}(x) = |\hat{\phi}_n^{(A)}(x) - \hat{\phi}_n^{(B)}(x)| \le \varepsilon_A^{\text{certified}} + \varepsilon_B,$$
   where $\varepsilon_A^{\text{certified}} = 0$ if $\mathcal{C}_A^{\text{exact}}$ holds. For independently converged high-precision references (e.g. mpmath at nominal working precision $p_{\text{ref}} \ge p_{\text{target}} + p_{\text{guard}}$, e.g., $p_{\text{ref}} \ge 384$ bits for nominal $10^{-100}$ scale resolution), precision is empirically certified via convergence diagnostic $E_{\text{conv}} = |\phi_{p_2} - \phi_{p_1}| < \varepsilon_{\text{ref}}$ ($p_2 > p_1$). Rigorous forward error certification requires interval/ball arithmetic or analytic enclosures. Total error satisfies:
   $$\boxed{|F - \widehat{\widetilde{F}}| \le B_{\text{analytic}} + E_{\text{arithmetic}}.}$$
2. **Finite-Field Modular Congruence Certificate ($C_{A,B}^{(p)}$):**
   $$\boxed{C_{A,B}^{(p)}(x_p) = \rho_p(\phi_n^{\mathbb{Q}}(x)) - \phi_{n,p}^{\mathbb{F}_p}(x_p) \equiv 0 \pmod p,}$$
   provided the finite-field normalized certificate $\mathcal{A}_\phi(p, n)$ is satisfied ($p > N_{\max}, p \nmid b d, p \nmid v_n, u_n \not\equiv 0 \pmod p$).

Key certified verification pairs:
1. $E_{\text{rational}, \text{float64}}^{\mathbb{R}}$: Exact rational vs standard double-precision recurrence.
2. $E_{\text{RNS}, \text{mpmath}}^{\mathbb{R}}$: Bounded CRT reconstructed integer/rational vs independently converged 384+ bit mpmath reference.
3. $C_{\text{finitefield}, \text{symbolic}}^{(p)}$: Commutative modular reduction congruence test.

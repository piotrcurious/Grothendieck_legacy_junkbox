# Semiclassical Representation Theory and Matched Asymptotics of Gegenbauer Polynomials

## Executive Summary

The asymptotic behavior of Gegenbauer polynomials $C_n^{(\lambda)}(x)$ as degree $n \to \infty$ represents the semiclassical limit ($\hbar \sim 1/n \to 0$) of spherical representations on compact rank-one Riemannian symmetric spaces $SO(d)/SO(d-1)$ (or compact Gelfand pairs $(SO(d), SO(d-1))$).

This document provides a mathematically precise 3-regime asymptotic framework, culminating in the explicit demystification of the **singular scaling limit** and the **representation geometry of the projectivized null quadric**:
1. **Interior Semiclassical Schrödinger Analysis (Regime I)**: Half-density conjugation converts the radial differential realization of the Casimir/Laplacian eigenvalue equation $-n(n+2\rho) = -(n+\rho)^2 + \rho^2$ into the exact 1D Schrödinger Hamiltonian $H_\lambda = -\partial_\theta^2 + \lambda(\lambda-1)\csc^2\theta$ with spectral parameter $E_N = N^2 = (n+\rho)^2$.
2. **Microscopic Endpoint Blow-Up & Contraction (Regime II)**: Tangent blow-up $z = N\theta$ and rescaled half-density $U_N(z) = N^\lambda u(z/N)$ convert the Schrödinger operator into the microscopic inverse-square Hamiltonian $-U_{zz} + \frac{\lambda(\lambda-1)}{z^2} U = U$, whose flat radial form $U = z^\lambda \phi$ is the flat Euclidean radial Helmholtz equation on $\mathbb{R}^{d-1}$ yielding the normalized Bessel kernel $\mathcal{J}_{\lambda-1/2}(z)$.
3. **Matched Asymptotic Overlap Bridge (Regime III)**: Common asymptotic overlap in $1 \ll z \ll N$ unifying interior WKB waves with singular orbit boundary layers.
4. **Representation Geometry & Commutative Gelfand Algebra**: Harmonic quotient decomposition $\operatorname{Sym}^n(\mathbb{C}^d)/(q) \cong V_n$, projectivized null quadric $Q^{d-2} \subset \mathbb{P}^{d-1}$, representation-growth stripping via normalization identity $C_n^{(\lambda)}(1) = \frac{\lambda}{n+\lambda}\dim V_n$, and the exact normalized Jacobi recurrence in the commutative spherical function algebra.

---

1. Representation-Theoretic Setup on $SO(d)/SO(d-1)$
------------------------------------------------------

Let $S^{d-1} \cong G/H = SO(d)/SO(d-1)$ be the real compact rank-one Riemannian symmetric space of dimension $d-1$, where $d \ge 3$. The dimension parameter $\lambda$ is related to $d$ via:
$$\lambda = \frac{d-2}{2}$$

The irreducible spherical representation $V_n$ of $SO(d)$ corresponds to highest weight $n\omega_1$ (space of degree-$n$ homogeneous harmonic polynomials on $\mathbb{R}^d$).

The radial orbit space $H \backslash G / H \cong [0, \pi]$ is parameterized by polar angle $\theta \in [0, \pi]$, where $x = \cos\theta \in [-1, 1]$. The normalized zonal spherical function of $V_n$ is:
$$\phi_n(\theta) = \frac{C_n^{(\lambda)}(\cos\theta)}{C_n^{(\lambda)}(1)}$$
satisfying $\phi_n(0) = 1$.

### Orbit Stratification
Under the cohomogeneity-one $H$-action on $S^{d-1}$:
1. **Principal $H$-Orbits ($\theta \in (0, \pi)$)**: Smooth $H$-orbits isomorphic to $S^{d-2}$ with non-vanishing radial volume measure $J(\theta) = (\sin\theta)^{2\lambda}$ (since $2\lambda = d-2$).
2. **Singular Orbits ($\theta = 0, \pi$)**: Two collapsed orbits at the north and south poles where $J(\theta) = 0$.

---

2. Half-Density Reduction & The Exact Schrödinger Operator
------------------------------------------------------------

The radial Laplacian acting on $H$-invariant functions on $S^{d-1}$ is:
$$\Delta_{\text{rad}} = \frac{1}{(\sin\theta)^{2\lambda}} \frac{d}{d\theta} \left( (\sin\theta)^{2\lambda} \frac{d}{d\theta} \right) = \frac{d^2}{d\theta^2} + 2\lambda \cot\theta \frac{d}{d\theta}$$

The zonal function satisfies $\Delta_{\text{rad}} \phi_n = -n(n + 2\lambda) \phi_n$.

### Exact Half-Density Transformation & Spectral Parameter
Conjugating $\Delta_{\text{rad}}$ with the radial half-density $J(\theta)^{1/2} = (\sin\theta)^\lambda$ via $u(\theta) = (\sin\theta)^\lambda \phi_n(\theta)$ eliminates the first derivative term, producing the exact 1D stationary Schrödinger equation:
$$\boxed{ -u''(\theta) + \frac{\lambda(\lambda - 1)}{\sin^2\theta} u(\theta) = (n + \lambda)^2 u(\theta) }$$

The radial Casimir eigenvalue identity:
$$-n(n + 2\rho) = -(n + \rho)^2 + \rho^2 \quad \left(\text{where } \rho = \lambda = \frac{d-2}{2}\right)$$
shows that half-density conjugation packages the radial differential realization of the Casimir eigenvalue equation into a Schrödinger operator:
$$\boxed{ H_\lambda = -\frac{d^2}{d\theta^2} + \lambda(\lambda - 1)\csc^2\theta }$$
whose natural semiclassical spectral parameter is $E_N = N^2 = (n+\rho)^2$. Its appearance is mathematically analogous to a Langer correction.

### Universal Potential Classification & Indicial Behavior
Near $\theta \to 0$, $V_{\text{eff}}(\theta) \sim \frac{\lambda(\lambda-1)}{\theta^2}$. The indicial equation $r(r-1) = \lambda(\lambda-1)$ has roots $r = \lambda$ and $r = 1-\lambda$.
- **For $\lambda > 1$ ($d > 4$)**: $\lambda(\lambda-1) > 0$, forming a repulsive inverse-square potential.
- **For $\lambda = 1$ ($d = 4$, $S^3$)**: $\lambda(\lambda-1) = 0$, giving $V_{\text{eff}} = 0$. Here $C_n^{(1)}(\cos\theta) = U_n(\cos\theta) = \frac{\sin((n+1)\theta)}{\sin\theta}$ is the zonal spherical function on $S^3$.
- **For $\lambda = 1/2$ ($d = 3$, $S^2$ Legendre $P_n$)**: $\lambda(\lambda-1) = -1/4$, yielding the critically attractive inverse-square potential $V_{\text{eff}} = -\frac{1}{4}\csc^2\theta$. The indicial roots coalesce at $r = 1/2$, and the Legendre spherical solution selects the non-logarithmic regular / Friedrichs-type branch $u \sim \theta^{1/2}$.

---

3. The Three-Regime Asymptotic Theory
-------------------------------------

```
                             [ FULL DOMAIN θ ∈ [0, π] ]
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
  [ REGIME I: Interior ]        [ REGIME II: Micro Blow-Up ]     [ REGIME III: Overlap ]
  - θ ∈ [ε, π-ε]                - z = Nθ = O(1)                  - 1 << z << N
  - Semiclassical WKB           - Rescaled Half-Density          - Asymptotic Agreement
  - Momentum p(θ) = N + O(1/N)  - Flat Euclidean Helmholtz      - Bessel Hankel ~ WKB
  - Cosine standing wave        - Normalized Kernel J_{λ-1/2}    - Smooth Matching Bridge
```

### Regime I: Fixed Interior Angle ($\theta \in [\epsilon, \pi - \epsilon]$)
In the interior, the exact local WKB momentum is:
$$p(\theta) = \sqrt{N^2 - \lambda(\lambda-1)\csc^2\theta} = N - \frac{\lambda(\lambda-1)}{2N}\csc^2\theta + O(N^{-3})$$
Uniformly on compact interior subsets, $p(\theta) = N + O(N^{-1})$, so the phase integral $\int^\theta p(t) dt = N\theta + O(N^{-1})$.

The rank-one Weyl group $W \cong \mathbb{Z}_2$ identifies the two radial WKB branches $\pm p$. The spherical regularity condition at $z=0$ fixes their linear combination and connection phase $-\frac{\lambda\pi}{2}$:
$$\boxed{ C_n^{(\lambda)}(\cos\theta) = \frac{2^{1-\lambda}}{\Gamma(\lambda)} n^{\lambda-1} (\sin\theta)^{-\lambda} \cos\left( (n + \lambda)\theta - \frac{\lambda \pi}{2} \right) + O(n^{\lambda-2}) }$$

---

### Regime II: Microscopic Endpoint Blow-Up and Euclidean Contraction ($\theta \sim N^{-1}$)
Near $\theta = 0$, set $z = N\theta = (n+\lambda)\theta$.
With $\theta = z/N$, $u(z/N) = \left(\sin\frac{z}{N}\right)^\lambda \phi_n(z/N) \sim N^{-\lambda} z^\lambda \phi(z)$. Define the properly rescaled half-density:
$$\boxed{ U_N(z) := N^\lambda u(z/N) }$$

Using the Taylor expansion $\csc^2(z/N) = \frac{N^2}{z^2} + \frac{1}{3} + O\left(\frac{z^2}{N^2}\right)$, the rescaled Schrödinger equation becomes:
$$-U_{zz} + \frac{\lambda(\lambda-1)}{z^2} U = \left[ 1 - \frac{\lambda(\lambda-1)}{3N^2} + O\left(\frac{z^2}{N^4}\right) \right] U$$
For bounded $z = O(1)$, $U_N(z)$ converges to $U(z) = z^\lambda \mathcal{J}_{\lambda-1/2}(z)$, satisfying the microscopic inverse-square Schrödinger equation:
$$-U_{zz} + \frac{\lambda(\lambda-1)}{z^2} U = U$$

Undoing the half-density $U(z) = z^\lambda \phi(z)$ yields the flat Euclidean radial Helmholtz equation on $\mathbb{R}^{d-1}$ (dimension $d-1 = 2\lambda+1$) at momentum magnitude $1$:
$$\boxed{ \phi'' + \frac{2\lambda}{z} \phi' + \phi = 0 }$$

The unique regular solution normalized to $\phi(0) = 1$ is the normalized Bessel kernel:
$$\boxed{ \mathcal{J}_{\lambda-1/2}(z) = 2^{\lambda - 1/2} \Gamma(\lambda + 1/2) \frac{J_{\lambda - 1/2}(z)}{z^{\lambda - 1/2}} }$$

---

### Regime III: Matched Asymptotic Overlap Zone ($1 \ll z \ll N$)
In the intermediate overlap zone ($1 \ll z \ll N \iff 1/N \ll \theta \ll 1$), both leading forms possess a common asymptotic overlap:

1. **Large-$z$ Limit of Endpoint Bessel Kernel**:
   Using $J_{\nu}(z) \sim \sqrt{\frac{2}{\pi z}} \cos\left( z - \frac{\nu \pi}{2} - \frac{\pi}{4} \right)$ with $\nu = \lambda - 1/2$:
   $$\mathcal{J}_{\lambda - 1/2}(z) \sim \frac{2^\lambda \Gamma(\lambda+1/2)}{\sqrt{\pi}} z^{-\lambda} \cos\left( z - \frac{\lambda \pi}{2} \right)$$

2. **Small-$\theta$ Limit of Interior WKB**:
   Using $C_n^{(\lambda)}(1) \sim \frac{n^{2\lambda-1}}{\Gamma(2\lambda)}$ and Legendre duplication $\Gamma(2\lambda) = \frac{2^{2\lambda-1}}{\sqrt{\pi}} \Gamma(\lambda)\Gamma(\lambda+1/2)$:
   $$\phi_n(\theta) = \frac{C_n^{(\lambda)}(\cos\theta)}{C_n^{(\lambda)}(1)} \sim \frac{2^\lambda \Gamma(\lambda+1/2)}{\sqrt{\pi}} (n\sin\theta)^{-\lambda} \cos\left( N\theta - \frac{\lambda \pi}{2} \right)$$

Since $n\sin\theta = z\left(1 + O(N^{-1}) + O(z^2/N^2)\right)$, the two leading asymptotic expansions agree in the overlap region.

---

4. Demystifying the "Singular Scaling Limit"
--------------------------------------------

The central theoretical insight is that the endpoint Bessel kernel behavior is not an ad-hoc special-function identity; it is the unique manifestation of a single **unified singular scaling limit** viewed simultaneously through four complementary perspectives:

$$\boxed{
\begin{array}{ccc}
\textbf{Compact Gelfand Pair} & \xrightarrow[\theta = z/N,\; N \to \infty]{\text{High-Weight Tangent Contraction}} & \textbf{Euclidean Gelfand Pair} \\
(SO(d), SO(d-1)) && (E(d-1), SO(d-1)) \\[3mm]
\downarrow && \downarrow \\[1mm]
\textbf{Spherical Radial Laplacian} & \xrightarrow[\text{Blow-Up}]{\Delta_{\text{rad}}} & \textbf{Euclidean Radial Laplacian} \\
\partial_\theta^2 + 2\lambda\cot\theta\,\partial_\theta && \partial_z^2 + \frac{2\lambda}{z}\partial_z \\[3mm]
\downarrow && \downarrow \\[1mm]
\textbf{Compact Zonal Function} & \xrightarrow[\text{Mehler-Heine}]{N \to \infty} & \textbf{Euclidean Spherical Kernel} \\
\phi_n(\theta) = \frac{C_n^{(\lambda)}(\cos\theta)}{C_n^{(\lambda)}(1)} && \mathcal{J}_{\lambda-1/2}(z) = \frac{1}{|S^{d-2}|} \int_{S^{d-2}} e^{i z \omega_1} d\omega
\end{array}
}$$

### The 4 Unified Perspectives of the Singular Scaling Limit:
1. **(i) Geometric Tangent-Space Limit**: The $N^{-1}$ microscopic blow-up flattens the compact sphere $S^{d-1}$ into its tangent space $T_p S^{d-1} \cong \mathbb{R}^{d-1}$.
2. **(ii) High-Weight Tangent Contraction of Gelfand Pairs**: Decomposing $\mathfrak{so}(d) = \mathfrak{so}(d-1) \oplus \mathfrak{p}$ and rescaling transvection generators $P_i^{(n)} = \frac{1}{n+\rho} X_i$ ($[P_i^{(n)}, P_j^{(n)}] \to 0$) contracts the compact Gelfand pair $(SO(d), SO(d-1))$ into the Euclidean Gelfand pair $(E(d-1), SO(d-1))$.
3. **(iii) Singular Schrödinger Operator Blow-Up**: Rescaling the compact Schrödinger operator $H_\lambda$ by $N^{-2}$ and half-density scaling $U_N(z) = N^\lambda u(z/N)$ yields the flat inverse-square Bessel Hamiltonian $-U_{zz} + \frac{\lambda(\lambda-1)}{z^2} U = U$.
4. **(iv) Mehler–Heine Matrix Coefficient Limit**: The zonal spherical functions $\phi_n(z/N)$ converge uniformly on compact $z$-sets to the Euclidean radial spherical function $\mathcal{J}_{\lambda-1/2}(z) = \frac{1}{|S^{d-2}|} \int_{S^{d-2}} e^{i z \omega_1} d\omega$.

---

5. Antipodal Parity & The South Pole Layer
------------------------------------------

At the south pole $\theta = \pi$, setting $\zeta = N(\pi - \theta)$, the boundary layer is governed by the antipodal parity identity:
$$C_n^{(\lambda)}(-x) = (-1)^n C_n^{(\lambda)}(x) \implies \phi_n(\theta) \sim (-1)^n \mathcal{J}_{\lambda-1/2}(\zeta)$$

The two singular boundary layers at $\theta=0$ and $\theta=\pi$ are mapped into each other by antipodal reflection, completing the uniform asymptotic description across $[0, \pi]$.

---

6. Representation Geometry of Projective Quadrics & Commutative Gelfand Algebra
-------------------------------------------------------------------------------

To connect Gegenbauer polynomials to algebraic geometry, we clearly distinguish the two quadric varieties and formulate the recurrence in the commutative algebra of spherical functions.

### 6.1 Representation Null Quadric & Harmonic Quotient
1. **Sphere Geometry vs Representation Geometry**: While the complexified sphere compactifies to $Q^{d-1} = \{z_1^2 + \dots + z_d^2 - z_0^2 = 0\} \subset \mathbb{P}^d$ ($SO(d+1)$ conformal action), the irreducible $SO(d)$ representation $V_n = \mathcal{H}_n(\mathbb{C}^d)$ is encoded by the projectivized null quadric $Q^{d-2} \subset \mathbb{P}^{d-1}$ given by $q(z) = z_1^2 + \dots + z_d^2 = 0$.
2. **Harmonic Polynomial Quotient**: Homogeneous degree-$n$ polynomials / symmetric tensors $\operatorname{Sym}^n(\mathbb{C}^d)$ decompose into trace spaces via $q$:
   $$\operatorname{Sym}^n(\mathbb{C}^d) = \bigoplus_{j=0}^{\lfloor n/2 \rfloor} q^j \mathcal{H}_{n-2j}(\mathbb{C}^d)$$
   Quotienting by the trace ideal $(q)$ directly isolates the harmonic representation:
   $$\boxed{ H^0(Q^{d-2}, \mathcal{O}_{Q^{d-2}}(n)) \cong \frac{\text{Sym}^n(\mathbb{C}^d)}{(q)} \cong \mathcal{H}_n(\mathbb{C}^d) \cong V_n }$$

### 6.2 Representation-Growth Stripping via Normalization
Taking the dimension of $H^0(Q^{d-2}, \mathcal{O}(n))$ gives:
$$\dim V_n = \binom{n+d-1}{d-1} - \binom{n+d-3}{d-1} = \frac{2n+d-2}{d-2} \binom{n+d-3}{n} = \frac{n+\lambda}{\lambda} \binom{n+2\lambda-1}{n}$$

Since $C_n^{(\lambda)}(1) = \binom{n+2\lambda-1}{n}$, we obtain the exact normalization dimension identity:
$$\boxed{ C_n^{(\lambda)}(1) = \frac{\lambda}{n+\lambda} \dim V_n = \frac{\rho}{n+\rho} \dim V_n }$$

As $n \to \infty$, $\dim V_n \sim \frac{2}{(d-2)!} n^{d-2}$. Normalizing by $C_n^{(\lambda)}(1) = \frac{\rho}{n+\rho} \dim V_n$ strips one power of representation growth, preparing the zonal function $\phi_n = C_n^{(\lambda)} / C_n^{(\lambda)}(1)$ for the $O(1)$ semiclassical limit.

### 6.3 Commutative Gelfand Algebra & Exact Normalized Jacobi Recurrence
Because $(SO(d), SO(d-1))$ is a compact Gelfand pair, every irreducible spherical representation $V_n$ has a 1-dimensional $H$-fixed subspace $V_n^H$, and the normalized zonal spherical functions $\phi_n = C_n^{(\lambda)} / C_n^{(\lambda)}(1)$ form the spherical basis.

Pointwise multiplication by the degree-1 fundamental spherical function $\phi_1(x) = x$ defines an exact Jacobi operator in the spherical function algebra:
$$\boxed{ x \cdot \phi_n(x) = \frac{n+2\lambda}{2(n+\lambda)} \phi_{n+1}(x) + \frac{n}{2(n+\lambda)} \phi_{n-1}(x) }$$

As $n \to \infty$, $\frac{n+2\lambda}{2(n+\lambda)} \to \frac{1}{2}$ and $\frac{n}{2(n+\lambda)} \to \frac{1}{2}$, so the Jacobi operator limits to $x \phi_n \sim \frac{1}{2}(\phi_{n+1} + \phi_{n-1})$, which is the discrete free-wave / discrete semiclassical momentum operator!

Undoing the normalization $\phi_n = C_n^{(\lambda)} / C_n^{(\lambda)}(1)$ recovers the standard Gegenbauer three-term recurrence:
$$x \cdot C_n^{(\lambda)}(x) = \frac{n+1}{2(n+\lambda)} C_{n+1}^{(\lambda)}(x) + \frac{n+2\lambda-1}{2(n+\lambda)} C_{n-1}^{(\lambda)}(x)$$

#### Case Study: $SO(3)$ ($d=3, \lambda=1/2$) Parity Demonstration
For $SO(3)$, $V_1 \otimes V_n \cong V_{n+1} \oplus V_n \oplus V_{n-1}$. However, spherical functions satisfy $\phi_1(-x) \phi_n(-x) = (-1)^{n+1} \phi_1(x) \phi_n(x)$, whereas $\phi_n(x)$ has parity $(-1)^n$. Thus, the $V_n^H$ channel is strictly forbidden by parity! This proves why spherical multiplication isolates strictly the $n \pm 1$ channels.

### 6.4 Hypergeometric Representation
The terminating hypergeometric polynomial representation:
$$C_n^{(\lambda)}(x) = \binom{n+2\lambda-1}{n} {}_2F_1\left(-n, n+2\lambda; \lambda+1/2; \frac{1-x}{2}\right) = \sum_{k=0}^n \frac{(-1)^k \binom{n}{k} (n+2\lambda)_k}{k! (\lambda+1/2)_k} \left(\frac{1-x}{2}\right)^k$$
packages the degree $n$ and dimension parameter $\lambda = (d-2)/2$ into explicit coefficient parameters $(-n, n+2\lambda, \lambda+1/2)$ governing the spherical representation.

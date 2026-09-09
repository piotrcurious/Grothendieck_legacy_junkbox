# Semiclassical Representation Theory and Matched Asymptotics of Gegenbauer Polynomials

## Executive Summary

The asymptotic behavior of Gegenbauer polynomials $C_n^{(\lambda)}(x)$ as degree $n \to \infty$ represents the semiclassical limit ($\hbar \sim 1/n \to 0$) of spherical representations on compact rank-one Riemannian symmetric spaces $SO(d)/SO(d-1)$ (or compact Gelfand pairs $(SO(d), SO(d-1))$).

This document provides a mathematically precise, 3-regime asymptotic theory unifying:
1. **Interior Semiclassical Schrödinger Analysis (Regime I)**: Half-density conjugation of the radial Laplacian to the exact Schrödinger Hamiltonian $H_\lambda = -\partial_\theta^2 + \lambda(\lambda-1)\csc^2\theta$ with Harish-Chandra / Weyl spectral shift $E_N = N^2 = (n+\rho)^2$.
2. **Microscopic Endpoint Blow-Up & Contraction (Regime II)**: Tangent blow-up $z = N\theta$ converting the compact Schrödinger problem into the flat inverse-square Euclidean radial Helmholtz equation on $\mathbb{R}^{d-1}$, yielding the normalized Bessel kernel $\mathcal{J}_{\lambda-1/2}(z)$.
3. **Matched Asymptotic Overlap Bridge (Regime III)**: Uniform asymptotic agreement in the intermediate overlap zone $1 \ll z \ll N$ connecting interior WKB waves to singular orbit boundary layers.

---

1. Representation-Theoretic Setup on $SO(d)/SO(d-1)$
------------------------------------------------------

Let $S^{d-1} \cong G/H = SO(d)/SO(d-1)$ be the real compact rank-one Riemannian symmetric space of dimension $d-1$, where $d \ge 3$. The dimension parameter $\lambda$ is related to $d$ via:
$$\lambda = \frac{d-2}{2}$$

The irreducible spherical representation $V_n$ of $SO(d)$ corresponds to highest weight $n\omega_1$ (degree-$n$ homogeneous harmonic polynomials on $\mathbb{R}^d$).

The radial orbit space $H \backslash G / H \cong [0, \pi]$ is parameterized by polar angle $\theta \in [0, \pi]$, where $x = \cos\theta \in [-1, 1]$. The normalized zonal spherical function of $V_n$ is:
$$\phi_n(\theta) = \frac{C_n^{(\lambda)}(\cos\theta)}{C_n^{(\lambda)}(1)}$$
satisfying $\phi_n(0) = 1$.

### Orbit Stratification
Under the cohomogeneity-one $H$-action on $S^{d-1}$:
1. **Principal $H$-Orbits ($\theta \in (0, \pi)$)**: Smooth $H$-orbits isomorphic to $S^{d-2}$ with non-vanishing radial volume measure $J(\theta) = (\sin\theta)^{2\lambda}$ (since $2\lambda = d-2$).
2. **Singular Orbits ($\theta = 0, \pi$)**: Two collapsed orbits at the north and south poles where $J(\theta) = 0$.

---

2. Half-Density Reduction & The Schrödinger Operator
-----------------------------------------------------

The radial Laplacian acting on $H$-invariant functions on $S^{d-1}$ is:
$$\Delta_{\text{rad}} = \frac{1}{(\sin\theta)^{2\lambda}} \frac{d}{d\theta} \left( (\sin\theta)^{2\lambda} \frac{d}{d\theta} \right) = \frac{d^2}{d\theta^2} + 2\lambda \cot\theta \frac{d}{d\theta}$$

The zonal function satisfies $\Delta_{\text{rad}} \phi_n = -n(n + 2\lambda) \phi_n$.

### Exact Half-Density Transformation
Conjugating $\Delta_{\text{rad}}$ with the radial half-density $J(\theta)^{1/2} = (\sin\theta)^\lambda$ via $u(\theta) = (\sin\theta)^\lambda \phi_n(\theta)$ eliminates the first derivative term, producing the exact 1D stationary Schrödinger equation:
$$\boxed{ -u''(\theta) + \frac{\lambda(\lambda - 1)}{\sin^2\theta} u(\theta) = (n + \lambda)^2 u(\theta) }$$

The clean effective Hamiltonian is:
$$\boxed{ H_\lambda = -\frac{d^2}{d\theta^2} + \lambda(\lambda - 1)\csc^2\theta }$$
with exact eigenvalue:
$$\boxed{ E_N = N^2 = (n + \lambda)^2 = (n + \rho)^2 }$$

### Interpretation of the $\rho$-Shift & Singularities
1. **Harish-Chandra / Weyl Spectral Shift**: The identity $n(n+2\rho) = (n+\rho)^2 - \rho^2$ shows that $N = n+\rho$ (where $\rho = \lambda = \frac{d-2}{2}$) is the rank-one $\rho$-shifted Casimir eigenvalue. Half-density reduction packages the radial problem into a Schrödinger operator whose natural semiclassical energy is $E = N^2$. Its appearance is mathematically analogous to a Langer correction.
2. **Universal Potential Classification**:
   - **For $\lambda > 1$ ($d > 4$)**: $\lambda(\lambda-1) > 0$, forming a repulsive inverse-square potential.
   - **For $\lambda = 1$ ($d = 4$, $S^3$)**: $\lambda(\lambda-1) = 0$, giving $V_{\text{eff}} = 0$.
   - **For $\lambda = 1/2$ ($d = 3$, $S^2$ Legendre)**: $\lambda(\lambda-1) = -1/4$, yielding the critically attractive inverse-square potential $V_{\text{eff}} = -\frac{1}{4}\csc^2\theta$. The local indicial roots coalesce at $r = 1/2$, and the Legendre spherical solution selects the non-logarithmic regular branch $u \sim \theta^{1/2}$.

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
  - Semiclassical WKB           - Transvection Contraction      - Asymptotic Agreement
  - Momentum p(θ) = N + O(1/N)  - Flat Euclidean Helmholtz      - Bessel Hankel ~ WKB
  - Cosine standing wave        - Normalized Kernel J_{λ-1/2}    - Smooth Matching Bridge
```

### Regime I: Fixed Interior Angle ($\theta \in [\epsilon, \pi - \epsilon]$)
In the interior, the local WKB momentum is:
$$p(\theta) = \sqrt{N^2 - \lambda(\lambda-1)\csc^2\theta} = N - \frac{\lambda(\lambda-1)}{2N}\csc^2\theta + O(N^{-3})$$
Uniformly on compact interior subsets, $p(\theta) = N + O(N^{-1})$, so the semiclassical phase integral $\int^\theta p(t) dt = N\theta + O(N^{-1})$.

The rank-one Weyl group $W \cong \mathbb{Z}_2$ exchanges the two oscillatory WKB branches $\pm p$. The spherical boundary condition selects the Weyl-symmetric combination, whose connection phase $-\frac{\lambda\pi}{2}$ is fixed by the singular orbit problem:
$$\boxed{ C_n^{(\lambda)}(\cos\theta) = \frac{2^{1-\lambda}}{\Gamma(\lambda)} n^{\lambda-1} (\sin\theta)^{-\lambda} \cos\left( (n + \lambda)\theta - \frac{\lambda \pi}{2} \right) + O(n^{\lambda-2}) }$$

#### Sanity Checks:
- **$\lambda = 1$ ($d = 4$, $S^3$ zonal harmonics / Chebyshev $U_n$)**: Gives $C_n^{(1)}(\cos\theta) = U_n(\cos\theta) = \frac{\sin((n+1)\theta)}{\sin\theta}$, matching exactly on $S^3$.
- **$\lambda = 1/2$ ($d = 3$, $S^2$ Legendre $P_n$)**: Gives $P_n(\cos\theta) \sim \sqrt{\frac{2}{\pi n \sin\theta}} \cos\left( (n+\frac{1}{2})\theta - \frac{\pi}{4} \right)$, matching the classical Legendre formula.

---

### Regime II: Microscopic Endpoint Scaling ($\theta \sim N^{-1}$)
Near $\theta = 0$, set $z = N\theta = (n+\lambda)\theta$.
Under this blow-up, $\csc^2\theta = \frac{N^2}{z^2} + O(1)$. Dividing the Schrödinger equation by $N^2$ yields the microscopic tangent equation:
$$-u_{zz} + \frac{\lambda(\lambda-1)}{z^2} u = u + O(N^{-2}z^2) u$$

Undoing the half-density factor $u(z) = z^\lambda \phi(z)$ transforms this into the flat Euclidean radial Helmholtz equation on $\mathbb{R}^{d-1}$ (dimension $d-1 = 2\lambda+1$) at momentum magnitude $1$:
$$\phi'' + \frac{2\lambda}{z} \phi' + \phi = 0$$

The unique regular solution normalized to $\phi(0) = 1$ is the normalized Bessel kernel:
$$\boxed{ \mathcal{J}_{\lambda-1/2}(z) = 2^{\lambda - 1/2} \Gamma(\lambda + 1/2) \frac{J_{\lambda - 1/2}(z)}{z^{\lambda - 1/2}} }$$

#### Representation-Theoretic Contraction
The family of spherical representations $V_n$, under rescaled transvection generators $P_i^{(n)} = \frac{1}{n+\rho} X_i$ ($[P_i^{(n)}, P_j^{(n)}] \to 0$), contracts such that its zonal matrix coefficients limit to those of the $E(d-1)$ Euclidean motion group:
$$\lim_{n \to \infty} \frac{C_n^{(\lambda)}\left(\cos(z/(n+\lambda))\right)}{C_n^{(\lambda)}(1)} = \mathcal{J}_{\lambda - 1/2}(z)$$

---

### Regime III: Matched Asymptotic Overlap Zone ($1 \ll z \ll N$)
In the intermediate overlap zone ($1 \ll z \ll N \iff 1/N \ll \theta \ll 1$), both expansions are simultaneously valid:

1. **Large-$z$ Limit of Endpoint Bessel Kernel**:
   Using $J_{\nu}(z) \sim \sqrt{\frac{2}{\pi z}} \cos\left( z - \frac{\nu \pi}{2} - \frac{\pi}{4} \right)$ with $\nu = \lambda - 1/2$:
   $$\mathcal{J}_{\lambda - 1/2}(z) \sim \frac{2^\lambda \Gamma(\lambda+1/2)}{\sqrt{\pi}} z^{-\lambda} \cos\left( z - \frac{\lambda \pi}{2} \right)$$

2. **Small-$\theta$ Limit of Interior WKB**:
   Using $C_n^{(\lambda)}(1) \sim \frac{n^{2\lambda-1}}{\Gamma(2\lambda)}$ and Legendre duplication $\Gamma(2\lambda) = \frac{2^{2\lambda-1}}{\sqrt{\pi}} \Gamma(\lambda)\Gamma(\lambda+1/2)$:
   $$\phi_n(\theta) = \frac{C_n^{(\lambda)}(\cos\theta)}{C_n^{(\lambda)}(1)} \sim \frac{2^\lambda \Gamma(\lambda+1/2)}{\sqrt{\pi}} (n\sin\theta)^{-\lambda} \cos\left( (n + \lambda)\theta - \frac{\lambda \pi}{2} \right)$$

Since $n\sin\theta = z + O(z/N)$ for $\theta \ll 1$, the two leading asymptotic expansions agree in the overlap region.

---

4. Antipodal Parity & The South Pole Layer
------------------------------------------

At the south pole $\theta = \pi$, setting $\zeta = (n+\lambda)(\pi - \theta)$, the layer structure is governed by the antipodal parity identity:
$$C_n^{(\lambda)}(-x) = (-1)^n C_n^{(\lambda)}(x) \implies \phi_n(\theta) \sim (-1)^n \mathcal{J}_{\lambda-1/2}(\zeta)$$

The two singular boundary layers at $\theta=0$ and $\theta=\pi$ are mapped into each other by antipodal reflection, providing full coverage across the entire domain $[0, \pi]$.

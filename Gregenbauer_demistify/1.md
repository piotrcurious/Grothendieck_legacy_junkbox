# Semiclassical Representation Theory and Matched Asymptotics of Gegenbauer Polynomials

## Executive Summary

The asymptotic behavior of Gegenbauer polynomials $C_n^{(\lambda)}(x)$ as degree $n \to \infty$ represents the semiclassical limit ($\hbar \sim 1/n \to 0$) of spherical representations on compact rank-one Riemannian symmetric spaces $SO(d)/SO(d-1)$ (or compact Gelfand pairs $(SO(d), SO(d-1))$).

This document establishes a rigorous framework unifying three distinct structural layers:
1. **Interior Semiclassical Schrödinger Analysis**: Transformation of the radial Laplacian via half-density conjugation into a 1D Schrödinger operator $H_\lambda = -\partial_\theta^2 + \lambda(\lambda-1)\csc^2\theta$ with Harish-Chandra/Weyl spectral shift $E_n = (n+\rho)^2$.
2. **Explicit Three-Level Contraction**: Concurrent contraction of the manifold ($S^{d-1} \to \mathbb{R}^{d-1}$), Lie algebra ($\mathfrak{so}(d) \to \mathfrak{se}(d-1)$), and spherical function to the radial Euclidean plane-wave kernel $\mathcal{J}_{\lambda-1/2}(z)$.
3. **Matched Asymptotic Bridge**: Asymptotic unification of the interior WKB standing waves with the two singular-orbit boundary layers ($\theta=0$ and $\theta=\pi$), related by Weyl reflection.

---

1. Representation-Theoretic Setup on $SO(d)/SO(d-1)$
------------------------------------------------------

Let $S^{d-1} \cong G/H = SO(d)/SO(d-1)$ be the real compact rank-one Riemannian symmetric space of dimension $d-1$, where $d \ge 3$. The dimension parameter $\lambda$ is related to $d$ via:
$$\lambda = \frac{d-2}{2}$$

The irreducible spherical representation $V_n$ of $SO(d)$ corresponds to highest weight $n\omega_1$ (space of degree-$n$ homogeneous harmonic polynomials on $\mathbb{R}^d$).

The double coset space $H \backslash G / H$ is parameterized by polar angle $\theta \in [0, \pi]$, where $x = \cos\theta \in [-1, 1]$. The normalized zonal spherical function of $V_n$ is:
$$\phi_n(\theta) = \frac{C_n^{(\lambda)}(\cos\theta)}{C_n^{(\lambda)}(1)}$$
satisfying $\phi_n(0) = 1$.

### Orbit Stratification
Under the cohomogeneity-one $H$-action, the double coset space stratifies into:
1. **Principal $H$-Orbits ($\theta \in (0, \pi)$)**: Smooth $H$-orbits isomorphic to $S^{d-2}$ with non-vanishing radial volume measure $J(\theta) = (\sin\theta)^{2\lambda}$ (since $2\lambda = d-2$).
2. **Singular Orbits ($\theta = 0, \pi$)**: Two collapsed orbits at the north and south poles where $J(\theta) = 0$.

---

2. Interior Regime: Semiclassical Schrödinger Operator & Weyl Asymptotics
-------------------------------------------------------------------------

The radial Laplacian acting on $H$-invariant functions on $S^{d-1}$ is:
$$\Delta_{\text{rad}} = \frac{1}{(\sin\theta)^{2\lambda}} \frac{d}{d\theta} \left( (\sin\theta)^{2\lambda} \frac{d}{d\theta} \right) = \frac{d^2}{d\theta^2} + 2\lambda \cot\theta \frac{d}{d\theta}$$

The zonal spherical function $\phi_n(\theta)$ satisfies $\Delta_{\text{rad}} \phi_n = -n(n + 2\lambda) \phi_n$.

### Half-Density Conjugation & The Correct Effective Hamiltonian
To eliminate the first-derivative term, we conjugate $\Delta_{\text{rad}}$ with the radial half-density $J(\theta)^{1/2} = (\sin\theta)^\lambda$:
$$u(\theta) = J(\theta)^{1/2} \phi_n(\theta) = (\sin\theta)^\lambda \phi_n(\theta)$$

Substituting $u(\theta)$ into the differential equation yields the 1D stationary Schrödinger equation:
$$\boxed{ -u''(\theta) + \frac{\lambda(\lambda - 1)}{\sin^2\theta} u(\theta) = (n + \lambda)^2 u(\theta) }$$

The clean effective Hamiltonian is:
$$\boxed{ H_\lambda = -\frac{d^2}{d\theta^2} + \lambda(\lambda - 1)\csc^2\theta }$$
with eigenvalue:
$$\boxed{ E_n = (n + \lambda)^2 = (n + \rho)^2 }$$

### Harish-Chandra / Langer Spectral Interpretation
1. **Harish-Chandra / Weyl Spectral Shift**: The shift $n \mapsto n + \rho$ (where $\rho = \lambda = \frac{d-2}{2}$) is the rank-one Harish-Chandra/Weyl spectral shift, and simultaneously the Langer-type shift appearing after radial half-density reduction.
2. **Langer Indicial Behavior**: Near $\theta \to 0$, $V_{\text{eff}}(\theta) \sim \frac{\lambda(\lambda-1)}{\theta^2}$. The indicial equation $r(r-1) = \lambda(\lambda-1)$ gives roots $r = \lambda$ and $r = 1-\lambda$. The regular spherical solution chooses $u \sim \theta^\lambda$, so $\phi_n = u / \theta^\lambda \sim 1$.
   *(Note: $\lambda = 1/2$ for $S^2$ gives the critical attractive inverse-square potential $-\frac{1}{4}\theta^{-2}$.)*

### Semiclassical WKB Branch Pairing & Phase Separation
- **Weyl Pairing**: The restricted root system is rank-one with Weyl group $W \cong \mathbb{Z}_2 = \{1, -1\}$. Semiclassical WKB momentum solutions $\pm p = \pm (n+\rho)$ yield phases $e^{\pm i (n+\rho)\theta}$. Weyl invariance forces the pairing of $\pm p$ into a cosine standing wave.
- **Connection Phase Shift**: The phase shift $-\frac{\lambda \pi}{2}$ is determined by the singular-orbit connection problem at $z=0$ (Bessel boundary condition).

In the interior regime $\theta \in (\epsilon, \pi - \epsilon)$, the asymptotic formula for $C_n^{(\lambda)}(\cos\theta)$ is:
$$\boxed{ C_n^{(\lambda)}(\cos\theta) = \frac{2^{1-\lambda}}{\Gamma(\lambda)} n^{\lambda-1} (\sin\theta)^{-\lambda} \cos\left( (n + \lambda)\theta - \frac{\lambda \pi}{2} \right) + O(n^{\lambda-2}) }$$

### Sanity Checks:
- **$\lambda = 1$ ($S^1$ / Chebyshev $U_n$)**: Gives $C_n^{(1)}(\cos\theta) = \frac{\sin((n+1)\theta)}{\sin\theta}$, matching exactly.
- **$\lambda = 1/2$ ($S^2$ / Legendre $P_n$)**: Gives $P_n(\cos\theta) \sim \sqrt{\frac{2}{\pi n \sin\theta}} \cos\left( (n+\frac{1}{2})\theta - \frac{\pi}{4} \right)$, recovering the classic Legendre WKB formula.

---

3. Endpoint Singular Orbits & Explicit Inönü–Wigner Contraction
----------------------------------------------------------------

At the poles $\theta \to 0$ and $\theta \to \pi$, the WKB approximation fails due to the centrifugal barrier $\frac{\lambda(\lambda-1)}{\theta^2}$.

To resolve the boundary layer near $\theta = 0$, we set $z = (n + \lambda)\theta$.
*(Note: The classical Mehler–Heine formula uses $n\theta = O(1)$; replacing $n$ by $n+\rho$ gives an asymptotically equivalent but spectrally more natural boundary coordinate since $\frac{n+\lambda}{n} = 1 + O(1/n)$).*

### The Three-Level Contraction Hierarchy
1. **Geometric Manifold Contraction**: $S^{d-1} \xrightarrow{n \to \infty} T_p S^{d-1} \cong \mathbb{R}^{d-1}$.
2. **Explicit Lie Algebra Contraction**: Decompose $\mathfrak{so}(d) = \mathfrak{h} \oplus \mathfrak{p}$, where $\mathfrak{h} = \mathfrak{so}(d-1)$ and $\mathfrak{p} \cong \mathbb{R}^{d-1}$. Rescale transvection generators by large momentum: $P_i^{(n)} = \frac{1}{n+\rho} X_i$ for $X_i \in \mathfrak{p}$.
   $$[P_i^{(n)}, P_j^{(n)}] = \frac{1}{(n+\rho)^2} [X_i, X_j] \xrightarrow{n \to \infty} 0$$
   while $[H, P_i^{(n)}]$ retains the vector representation of $SO(d-1)$. Thus $\mathfrak{so}(d) \to \mathfrak{se}(d-1) = \mathfrak{so}(d-1) \ltimes \mathbb{R}^{d-1}$.
3. **Spherical Function Contraction**: The spherical representation contracts to a Euclidean plane-wave representation whose radial matrix coefficient is the Euclidean spherical function.

### The Euclidean Radial Helmholtz Kernel
In flat $\mathbb{R}^{d-1}$ (where $d-1 = 2\lambda+1$), the Euclidean radial Helmholtz equation at momentum magnitude $1$ is:
$$\phi'' + \frac{d-2}{z} \phi' + \phi = 0 \quad \iff \quad \boxed{ \phi'' + \frac{2\lambda}{z} \phi' + \phi = 0 }$$

The regular solution with $\phi(0) = 1$ is the normalized Bessel kernel:
$$\phi_\infty(z) = 2^{\lambda - 1/2} \Gamma(\lambda + 1/2) \frac{J_{\lambda - 1/2}(z)}{z^{\lambda - 1/2}} = \mathcal{J}_{\lambda - 1/2}(z)$$

This provides the exact Mehler–Heine limit:
$$\lim_{n \to \infty} \frac{C_n^{(\lambda)}\left(\cos(z/(n+\lambda))\right)}{C_n^{(\lambda)}(1)} = \mathcal{J}_{\lambda - 1/2}(z)$$

---

4. Matched Asymptotic Overlap & Boundary Layers
----------------------------------------------

The overlap region is defined by $1/n \ll \theta \ll 1 \iff 1 \ll z \ll n$.

### Large-$z$ Expansion of the Endpoint Bessel Kernel
For $z = (n+\lambda)\theta \gg 1$, using $J_{\nu}(z) \sim \sqrt{\frac{2}{\pi z}} \cos\left( z - \frac{\nu \pi}{2} - \frac{\pi}{4} \right)$ with $\nu = \lambda - 1/2$:
$$\mathcal{J}_{\lambda - 1/2}(z) \sim \frac{2^{\lambda - 1/2} \Gamma(\lambda + 1/2)}{\sqrt{\pi/2}} z^{-\lambda} \cos\left( z - \frac{\lambda \pi}{2} \right)$$

### Small-$\theta$ Expansion of Interior WKB
Using $C_n^{(\lambda)}(1) = \frac{\Gamma(n+2\lambda)}{\Gamma(2\lambda)\Gamma(n+1)} \sim \frac{n^{2\lambda-1}}{\Gamma(2\lambda)}$ and Legendre's duplication formula $\Gamma(2\lambda) = \frac{2^{2\lambda-1}}{\sqrt{\pi}} \Gamma(\lambda)\Gamma(\lambda+1/2)$:
$$\phi_n(\theta) = \frac{C_n^{(\lambda)}(\cos\theta)}{C_n^{(\lambda)}(1)} \sim \frac{2^\lambda \Gamma(\lambda+1/2)}{\sqrt{\pi}} (n\sin\theta)^{-\lambda} \cos\left( (n + \lambda)\theta - \frac{\lambda \pi}{2} \right)$$

Since $n\sin\theta \sim n\theta \sim z$ for $\theta \ll 1$, the two expansions match identically!

### Two Endpoint Layers & Weyl Reflection
Near the south pole $\theta = \pi$, setting $\zeta = (n+\lambda)(\pi - \theta)$:
$$C_n^{(\lambda)}(-x) = (-1)^n C_n^{(\lambda)}(x) \implies \phi_n(\theta) \sim (-1)^n \mathcal{J}_{\lambda-1/2}(\zeta)$$
The two singular boundary layers at $\theta=0$ and $\theta=\pi$ are mapped into each other by the action of the Weyl reflection $w \in W \cong \mathbb{Z}_2$.

---

5. Summary Architecture
------------------------

```
   [ SO(d)/SO(d-1) Radial Laplacian ]
                  │
                  ▼ (Half-Density Conjugation)
   [ H_λ = -d²/dθ² + λ(λ-1)csc²θ ]
                  │
                  ├──────────────────────────────────────────┐
                  ▼                                          ▼
   (Interior: 0 < θ < π)                     (Singular Orbits: θ = 0, π)
   - Semiclassical Momentum p = n + ρ        - Zoom: z = (n+λ)θ, ζ = (n+λ)(π-θ)
   - Weyl Pairing: ±p -> cos(...)            - Lie Algebra: so(d) -> se(d-1)
   - Inverse Half-Density: (sin θ)^(-λ)      - Euclidean Radial Kernel J_{λ-1/2}
                  │                                          │
                  └───────────────────┬──────────────────────┘
                                      ▼
             [ Matched Asymptotics Overlap: 1/n << θ << 1 ]
```

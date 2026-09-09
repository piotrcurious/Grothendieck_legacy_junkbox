# Semiclassical Representation Theory and Matched Asymptotics of Gegenbauer Polynomials

## Executive Summary

The asymptotic behavior of Gegenbauer polynomials $C_n^{(\lambda)}(x)$ as degree $n \to \infty$ represents the semiclassical limit ($\hbar \sim 1/n \to 0$) of rank-one spherical representations on algebraic symmetric spaces. This document provides a rigorous unified framework connecting representation theory ($SO(d)/SO(d-1)$), semiclassical radial Schrödinger operators, Inönü–Wigner Lie algebra contractions, and matched asymptotic expansions.

---

1. The Representation-Theoretic Setup
----------------------------------------

Let $S^{d-1} \cong G/H = SO(d)/SO(d-1)$ be the real compact symmetric space of dimension $d-1$, where $d \ge 3$. The dimension parameter $\lambda$ is related to $d$ via:
$$\lambda = \frac{d-2}{2}$$

The irreducible spherical representation $V_n$ of $SO(d)$ corresponds to highest weight $n\omega_1$ (space of homogeneous harmonic polynomials of degree $n$ on $\mathbb{R}^d$).

The double coset space $H \backslash G / H$ is parameterized by the polar angle $\theta \in [0, \pi]$, where $x = \cos\theta \in [-1, 1]$. The normalized Gegenbauer polynomial:
$$\phi_n(\theta) = \frac{C_n^{(\lambda)}(\cos\theta)}{C_n^{(\lambda)}(1)}$$
is the zonal spherical function of $V_n$, satisfying $\phi_n(0) = 1$.

The orbit structure of the double coset space splits into two geometric regimes:
1. **Interior Principal Orbits ($\theta \in (0, \pi)$)**: Smooth $H$-orbits isomorphic to $S^{d-2}$ with non-vanishing volume element $J(\theta) = (\sin\theta)^{2\lambda}$.
2. **Singular Orbits ($\theta = 0, \pi$)**: Collapsed orbits at the poles where $J(\theta) = 0$.

---

2. Interior Regime: Semiclassical Schrödinger Operator & Weyl Asymptotics
-------------------------------------------------------------------------

On the symmetric space $SO(d)/SO(d-1)$, the radial Laplacian acting on $H$-invariant functions is given by:
$$\Delta_{\text{rad}} = \frac{1}{(\sin\theta)^{2\lambda}} \frac{d}{d\theta} \left( (\sin\theta)^{2\lambda} \frac{d}{d\theta} \right)$$

The zonal function $\phi_n(\theta)$ satisfies the eigenvalue equation:
$$\Delta_{\text{rad}} \phi_n(\theta) = -n(n + 2\lambda) \phi_n(\theta)$$

To eliminate the first-derivative term, we perform a half-density transformation by conjugating $\Delta_{\text{rad}}$ with $J(\theta)^{1/2} = (\sin\theta)^\lambda$:
$$u(\theta) = J(\theta)^{1/2} \phi_n(\theta) = (\sin\theta)^\lambda \phi_n(\theta)$$

Substituting $u(\theta)$ transforms the eigenvalue equation into a 1D stationary Schrödinger equation:
$$\left( -\frac{d^2}{d\theta^2} + V_{\text{eff}}(\theta) \right) u(\theta) = (n + \lambda)^2 u(\theta)$$

where the effective potential is:
$$V_{\text{eff}}(\theta) = \frac{\lambda(\lambda - 1)}{\sin^2\theta} + \lambda^2$$

### Key Geometric Identifications:
1. **Spectral Parameter & $\rho$-Shift**:
   The effective energy eigenvalue is $E = (n + \lambda)^2 = (n + \rho)^2$, where $\rho = \lambda = \frac{d-2}{2}$ is the half-sum of positive restricted roots of $SO(d)/SO(d-1)$.
2. **Radial Half-Density Amplitude**:
   The leading amplitude factor $J(\theta)^{-1/2} = (\sin\theta)^{-\lambda}$ arises directly from undoing the half-density conjugation:
   $$\phi_n(\theta) = (\sin\theta)^{-\lambda} u(\theta)$$
3. **Weyl Group Phase Structure**:
   The restricted root system of $SO(d)/SO(d-1)$ is rank-one with restricted Weyl group $W \cong \mathbb{Z}_2 = \{1, -1\}$. The WKB semiclassical solutions for $u(\theta)$ yield two counter-propagating phase factors:
   $$e^{+i(n+\rho)\theta} \quad \text{and} \quad e^{-i(n+\rho)\theta}$$
   Symmetry under $W$ forces these phases to combine into the cosine term:
   $$\cos\left( (n + \lambda)\theta - \frac{\lambda \pi}{2} \right)$$

Thus, in the interior regime $\theta \in (\epsilon, \pi - \epsilon)$, as $n \to \infty$:
$$C_n^{(\lambda)}(\cos\theta) = \frac{2^{1-\lambda}}{\Gamma(\lambda)} n^{\lambda-1} (\sin\theta)^{-\lambda} \cos\left( (n + \lambda)\theta - \frac{\lambda \pi}{2} \right) + O(n^{\lambda-2})$$

---

3. Endpoint Regime: Inönü–Wigner Contraction & Mehler–Heine Formula
---------------------------------------------------------------------

At the poles $\theta \to 0$ (and $\theta \to \pi$), the effective potential $V_{\text{eff}}(\theta) \sim \frac{\lambda(\lambda-1)}{\theta^2}$ diverges, signifying the breakdown of the WKB approximation on singular orbits.

To resolve the boundary layer near $\theta = 0$, we introduce the microscopic coordinate:
$$z = (n + \lambda)\theta \quad \iff \quad \theta = \frac{z}{n + \lambda}$$

### Geometric Contraction:
As $n \to \infty$ with $z$ fixed:
1. **Space Contraction**: The spherical geometry $S^{d-1}$ scales locally into its tangent space $T_p S^{d-1} \cong \mathbb{R}^{d-1}$.
2. **Lie Algebra Contraction**: The Lie algebra undergoes an Inönü–Wigner contraction:
   $$\mathfrak{so}(d) \xrightarrow{n \to \infty} \mathfrak{se}(d-1)$$
   where $\mathfrak{se}(d-1) = \mathfrak{so}(d-1) \ltimes \mathbb{R}^{d-1}$ is the Euclidean motion algebra.
3. **Representation Contraction**: The spherical representation $V_n$ of $SO(d)$ contracts to the unitary irreducible spherical representation of $SE(d-1)$ corresponding to Euclidean momentum $\|k\| = 1$.

Under this contraction, the Schrödinger differential equation simplifies in the limit $n \to \infty$ to:
$$\left( -\frac{d^2}{dz^2} + \frac{\lambda(\lambda-1)}{z^2} \right) u_\infty(z) = u_\infty(z)$$

The regular solution normalized at $z=0$ is expressible in terms of Bessel functions:
$$u_\infty(z) = 2^{\lambda - 1/2} \Gamma(\lambda + 1/2) z^{1/2} J_{\lambda - 1/2}(z)$$

Undoing the half-density scaling $u_\infty(z) = z^\lambda \phi_\infty(z)$ yields the normalized Bessel kernel:
$$\phi_\infty(z) = 2^{\lambda - 1/2} \Gamma(\lambda + 1/2) \frac{J_{\lambda - 1/2}(z)}{z^{\lambda - 1/2}} = \mathcal{J}_{\lambda - 1/2}(z)$$

This is the famous **Mehler–Heine formula**:
$$\lim_{n \to \infty} \frac{C_n^{(\lambda)}\left(\cos(z/n)\right)}{C_n^{(\lambda)}(1)} = \mathcal{J}_{\lambda - 1/2}(z)$$

---

4. The Matched Asymptotic Overlap Bridge
----------------------------------------

The interior WKB expansion and the endpoint Mehler–Heine contraction are unified via matched asymptotic expansion in the intermediate overlap boundary layer:
$$\frac{1}{n} \ll \theta \ll 1 \quad \iff \quad 1 \ll z \ll n$$

### Large-Argument Expansion of the Endpoint Bessel Boundary Layer:
For $z = (n + \lambda)\theta \gg 1$, the standard Hankel asymptotic formula for $J_{\nu}(z)$ with $\nu = \lambda - 1/2$ gives:
$$J_{\lambda - 1/2}(z) \sim \sqrt{\frac{2}{\pi z}} \cos\left( z - \frac{(\lambda - 1/2)\pi}{2} - \frac{\pi}{4} \right) = \sqrt{\frac{2}{\pi z}} \cos\left( z - \frac{\lambda \pi}{2} \right)$$

Substituting this into $\phi_\infty(z) = 2^{\lambda - 1/2} \Gamma(\lambda + 1/2) z^{-\lambda + 1/2} J_{\lambda - 1/2}(z)$:
$$\phi_\infty((n+\lambda)\theta) \sim \frac{2^{\lambda - 1/2} \Gamma(\lambda + 1/2)}{\sqrt{\pi/2}} ((n+\lambda)\theta)^{-\lambda} \cos\left( (n + \lambda)\theta - \frac{\lambda \pi}{2} \right)$$

### Small-Angle Expansion of the Interior Weyl Expansion:
For $\theta \ll 1$, $\sin\theta = \theta + O(\theta^3)$. Replacing $\sin\theta$ by $\theta$ in the interior WKB formula for $\phi_n(\theta)$ yields precisely the same formula as above!

### Conclusion:
The Bessel function is not an ad-hoc endpoint approximation; it is the exact, canonical boundary-layer state that smoothly matches the interior Weyl/WKB oscillatory waves across the singular orbit transition zone.

---

5. Summary Comparison Matrix
-----------------------------

| Analytical Feature | Geometric / Representation-Theoretic Origin | Boundary Layer / Limit |
|---|---|---|
| **High degree $n \to \infty$** | High-weight / semiclassical limit ($\hbar \sim 1/n$) | Semiclassical spectrum |
| **Shift $(n + \lambda) = (n + \rho)$** | Half-sum of positive restricted roots $\rho$ of $SO(d)/SO(d-1)$ | Quantum quantum correction |
| **Amplitude $(\sin\theta)^{-\lambda}$** | Inverse square root of radial measure $J(\theta)^{-1/2}$ (half-density) | Interior principal orbits |
| **Cosine oscillations** | Weyl group $W \cong \mathbb{Z}_2$ phase superposition $e^{\pm i (n+\rho)\theta}$ | WKB standing wave |
| **Endpoint Bessel Kernel** | Inönü–Wigner contraction $\mathfrak{so}(d) \to \mathfrak{se}(d-1)$ | Singular orbit boundary layer |
| **Matching in overlap zone** | Singular orbit asymptotic matching ($1/n \ll \theta \ll 1$) | Asymptotic unification |

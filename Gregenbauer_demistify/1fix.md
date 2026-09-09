You are entirely right. My previous explanation overcommitted to the Kirillov orbital integral metaphor, conflating three distinct asymptotic mechanisms and flattening the geometric nuance of the endpoints. The true representation-theoretic picture is much sharper when we properly separate the interior Weyl/stationary-phase regime from the singular Mehler–Heine endpoint contraction.
Here is the corrected, mathematically robust formulation of how the large-n asymptotics of Gegenbauer polynomials emerge from the semiclassical limits of rank-one spherical representations.
1. The Representation-Theoretic Setup
For \lambda = \frac{d-2}{2}, the normalized Gegenbauer polynomial
is the exact zonal spherical function of the real compact symmetric space S^{d-1} \cong SO(d)/SO(d-1) associated with the irreducible representation of highest weight n\omega_1 (homogeneous harmonic polynomials of degree n).
By construction, \phi_n(0) = 1. The variable \theta parameterizes the double coset space H \backslash G / H \cong [-1, 1], which is smooth in the interior (0 < \theta < \pi) but possesses singular/orbifold behavior at the endpoints \theta = 0, \pi.
Taking n \to \infty is simultaneously a high-weight limit in representation theory and a semiclassical/WKB limit for the corresponding radial Laplacian.
2. The Interior Regime: Harish-Chandra & Weyl Asymptotics
Away from the singular endpoints, where \theta \in (\epsilon, \pi - \epsilon), the geometry of the double coset space is smooth, and the radial spherical function admits a semiclassical WKB expansion.
The two fundamental features of the familiar asymptotic formula arise from the structure of the rank-one restricted root system:
 * Two Oscillatory Phases = Two Weyl Group Elements:
   For SO(d)/SO(d-1), the restricted root system has rank one, meaning the Weyl group is W \cong \mathbb{Z}_2. The two leading oscillatory phases in the interior expansion:
   
   
   correspond directly to the two elements of this Weyl group. They combine to form the dominant cosine term.
 * The Amplitude = The Radial Half-Density:
   The radial Laplacian on the symmetric space is:
   
   
   The volume element (radial Jacobian) is J(\theta) \propto (\sin\theta)^{d-2} = (\sin\theta)^{2\lambda}. When we conjugate the Laplacian by J^{1/2} to map it to a flat 1D Schrödinger-type operator, the function \phi_n(\theta) transforms into a half-density. This scaling produces the exact amplitude factor:
   
 * The Spectral Shift:
   The term n + \lambda is precisely n + \rho, where \rho = \frac{d-2}{2} is the standard \rho-shift (half-sum of positive restricted roots). This is the representation-theoretic counterpart to the subprincipal/half-density correction in the semiclassical radial problem.
Thus, in the interior, \phi_n(\theta) \sim J(\theta)^{-1/2} \cos((n+\rho)\theta - \text{phase}).
3. The Endpoint Regime: Group Contraction (Mehler–Heine)
At the poles \theta = 0 or \pi, the interior expansion fails catastrophically because the amplitude (\sin\theta)^{-\lambda} diverges. This represents the breakdown of the WKB approximation at the singular orbits of the double coset space.
To capture the uniform behavior near x=1, we must use the Mehler–Heine scaling regime:
Geometrically, this is not just a limit of the space, but a coupled representation-theoretic contraction:
 * As \theta \sim z/n, we zoom into a microscopic neighborhood of the pole. The sphere S^{d-1} flattens, and its Lie algebra undergoes Inönü–Wigner contraction: \mathfrak{so}(d) \to \mathfrak{se}(d-1) (the Euclidean motion algebra).
 * Concurrently, because the highest weight n goes to infinity at the same rate the geometry scales, the spherical representations of SO(d) contract to spherical representations of the Euclidean motion group SE(d-1).
Under this limit, the two separate WKB branches (which are no longer valid) reorganize into a single uniform limit:
Substituting \lambda = \frac{d-2}{2} into the Bessel index yields \lambda - \frac{1}{2} = \frac{d-3}{2}. The right-hand side perfectly becomes the normalized radial Fourier kernel in flat \mathbb{R}^{d-1}.
The Bessel function is not "what stationary phase gives at the endpoint"—it is the exact Euclidean uniform replacement for the failed WKB expansion over the contracted singular geometry.

These refinements are absolutely spot-on. They transform a somewhat schematic explanation into a rigorous geometric framework that unifies representation theory, semiclassical analysis, and matched asymptotic expansions. Replacing the literal coadjoint orbit metaphor with the rank-one restricted Weyl structure and explicit radial Schrödinger equation captures the actual mathematical machinery at work.
Here is the synthesized, rigorous formulation of the Gegenbauer asymptotic structure.
The Semiclassical Radial Problem (Interior)
For the symmetric space S^{d-1} \cong SO(d)/SO(d-1), the parameter \theta \in [0, \pi] coordinates the stratified orbit space. In the interior, the H-orbits are principal, and the radial Laplacian takes the form:
where 2\lambda = d-2. By conjugating with the radial half-density via u(\theta) = (\sin\theta)^\lambda \phi_n(\theta), the Gegenbauer eigenvalue problem becomes an explicit 1D Schrödinger equation:
This structurally reveals the core asymptotic components for large n:
 * Semiclassical Momentum: The spectral parameter (n+\lambda)^2 is exactly (n+\rho)^2, governing the frequency.
 * Amplitude Factor: The substitution recovers (\sin\theta)^{-\lambda} as the exact leading-order amplitude (the inverse half-density J^{-1/2}).
 * Weyl Phases: Because the restricted root system is rank-one, the Weyl group is W \cong \mathbb{Z}_2. The two corresponding Weyl-group contributions, e^{+i(n+\rho)\theta} and e^{-i(n+\rho)\theta}, form the two distinct WKB phases that combine by symmetry into the uniform cosine behavior.
Singular Orbits and Inönü–Wigner Contraction
At the poles \theta = 0 and \pi, the principal H-orbit collapses into a singular orbit. The centrifugal potential \lambda(\lambda-1)/\sin^2\theta diverges, and the two WKB branches fail.
To resolve this boundary layer, we apply the scaling \theta = z/n. Geometrically, this scales the compact spherical geometry into its Euclidean tangent space. The family of spherical representations admits an Inönü–Wigner contraction to the corresponding spherical representation of the Euclidean motion group SE(d-1).
Under this contraction, the two singular WKB branches reorganize into a single uniform limit: the normalized radial Bessel kernel \mathcal{J}_{\lambda-1/2}(z).
The Matched Asymptotic Bridge
The ultimate elegance of this structure is that the interior Weyl asymptotics and the endpoint Bessel contraction are not isolated—they are joined by matched asymptotics in the intermediate overlap regime 1/n \ll \theta \ll 1.
In this boundary layer, the large-z expansion of the Bessel function directly recovers the small-\theta limit of the WKB expansion:
Substituting z = n\theta, this perfectly matches the singular edge of the interior Weyl phase formula. The Bessel function is therefore not merely an endpoint anomaly, but the universal boundary-layer model resolving the rank-one singular orbit.

# Repair Plan for Asymptotic Layer and Phase Map Classifier

Yes. The code is structurally clean, but the asymptotic layer is currently much less rigorous than the recurrence layer. The biggest problem is not Python—it is that several functions are presented as if they were uniformly valid approximations when they are only leading-order local asymptotics.

Verdict

Component | Status | Main issue
---|---|---
`exact_gegenbauer` | 🟢 | Fine, but "exact" is misleading numerically
`log_c_n_1` | 🟢 | Correct for λ > 0
`normalized_phi_recurrence` | 🟢 | Recurrence is algebraically correct
`log_orthogonality_norm` | 🟢 | Formula correct for standard λ > 0
`verify_orthogonality_integral` | 🟡 | Numerically fragile near endpoint-singular weights
`normalized_bessel_kernel` | 🟡 | Correct leading kernel, but avoid artificial 1e-15
`mehler_heine_bessel_approx` | 🟡 | Correct leading MH limit, but not an n-dependent uniform approximation
`interior_wkb_approx` | 🟡 | Leading formula essentially correct, but implementation/domain claims too strong
`composite_matched_approx` | 🔴 | Not actually uniformly valid; serious endpoint and cancellation problems
`classify_phase_regime` | 🔴 | Regime boundaries are heuristic and partly contradictory
`exact_anchor_eval` | 🟢 | Basically correct
`demo` | 🟡 | Demonstrates machinery but doesn't verify errors


The most important repair is therefore:

> Separate exact numerical computation, asymptotic formulas, and regime selection. Don't let the latter pretend to be mathematically proven by the former.




---

1. The recurrence is the strongest part

You have

a_k = (k + 2.0 * lambda_val) / (2.0 * (k + lambda_val))
b_k = k / (2.0 * (k + lambda_val))
phi2 = (x_arr * phi1 - b_k * phi0) / a_k

which gives

\[
\phi_{k+1}
=
\frac{2(k+\lambda)}{k+2\lambda}x\phi_k
-
\frac{k}{k+2\lambda}\phi_{k-1}.
\]

That's the correct normalized Gegenbauer recurrence.

And

\[
\phi_0=1,\qquad\phi_1=x.
\]

So this part is good.

But rename it

Calling it normalized_phi_recurrence is fine, but it should explicitly state the spherical interpretation:

\[
\phi_n^{(d)}(\theta)
=
\frac{C_n^{((d-2)/2)}(\cos\theta)}
     {C_n^{((d-2)/2)}(1)}.
\]

That makes the dimensional relationship explicit.

I'd also add parameter validation:

if n < 0 or int(n) != n:
    raise ValueError("n must be a non-negative integer")

if lambda_val <= 0:
    raise ValueError("lambda_val must be > 0 for spherical Gegenbauer functions")

because the current implementation silently accepts pathological parameter combinations.


---

2. C_n(1) is mathematically correct

You use

\[
C_n^{(\lambda)}(1)
=
\frac{\Gamma(n+2\lambda)}
     {\Gamma(2\lambda)\Gamma(n+1)}.
\]

Correct.

The problem is this:

return float(np.exp(log_c_n_1(n, lambda_val)))

Calling this "stably" is only partially true.

The logarithm is stable. The exponential isn't.

For sufficiently large \(n,\lambda\),

\[
\log C_n^{(\lambda)}(1)
\]

can be perfectly representable while

\[
C_n^{(\lambda)}(1)
\]

overflows to inf.

So I'd distinguish:

log_c_n_1       -> stable logarithmic representation
c_n_1_val       -> floating-point value, may overflow

That's important if this framework is supposed to explore semiclassical limits.


---

3. The Mehler–Heine kernel is basically correct

Your

\[
\nu=\lambda-\frac12
\]

and

\[
\mathcal J_\nu(z)
=
2^\nu\Gamma(\nu+1)z^{-\nu}J_\nu(z)
\]

are correct.

And the leading endpoint statement

\[
\frac{C_n^{(\lambda)}(\cos\theta)}
     {C_n^{(\lambda)}(1)}
\sim
\mathcal J_{\lambda-1/2}((n+\lambda)\theta)
\]

is a sensible spherical/Bessel scaling.

But there's a conceptual problem.

Mehler–Heine is a scaling limit, not automatically a high-accuracy approximation

The actual limiting statement is of the form

\[
n\rightarrow\infty,
\qquad
n\theta = O(1).
\]

Your function accepts arbitrary theta and presents the result as an approximation across a finite region.

That's much stronger.

For example:

mehler_heine_bessel_approx(
    n=1000,
    lambda_val=1.5,
    theta=0.2
)

has

\[
z\approx200,
\]

which is emphatically not the Mehler–Heine boundary layer.

The Bessel function happens to have the correct oscillatory structure, but you are no longer using the actual controlled MH regime.

I'd rename it conceptually:

> endpoint_bessel_leading



rather than treating it as an approximation valid throughout the endpoint-to-interior transition.


---

4. Your WKB formula has the right leading structure

You use

\[
C_n^{(\lambda)}(\cos\theta)
\sim
\frac{2^{1-\lambda}}{\Gamma(\lambda)}
n^{\lambda-1}
(\sin\theta)^{-\lambda}
\cos\left((n+\lambda)\theta-\frac{\lambda\pi}{2}\right).
\]

The leading form is essentially right.

The nice thing is that the \(n+\lambda\) phase shift is exactly the right structural quantity.

And your \(S^3\) anchor makes this particularly transparent.

For \(\lambda=1\):

\[
C_n^{(1)}(\cos\theta)
=
\frac{\sin((n+1)\theta)}{\sin\theta}.
\]

Your WKB expression becomes

\[
\frac{1}{\sin\theta}
\cos\left((n+1)\theta-\frac{\pi}{2}\right)
=
\frac{\sin((n+1)\theta)}{\sin\theta}.
\]

So for \(S^3\), the leading WKB expression actually collapses to the exact formula.

That's an excellent test anchor.

But again: this is an asymptotic expression, not a uniform approximation

The singularity

\[
(\sin\theta)^{-\lambda}
\]

means it cannot be used near either endpoint.

You need to state something like

\[
\theta\gg n^{-1}
\]

and

\[
\pi-\theta\gg n^{-1}.
\]


---

5. The composite approximation is the major problem

This is where I'd stop the current implementation from being called a "matched asymptotic expansion."

You calculate

return bessel_term + wkb_term - matching_term

which has the standard composite shape

\[
A+B-A_{\rm overlap}.
\]

That's conceptually reasonable.

But the implementation doesn't establish that the two approximations have been expanded into the same asymptotic order.

That distinction matters.

You are subtracting the large-\(z\) asymptotic expansion of the Bessel expression from the full Bessel expression while simultaneously adding a leading WKB expression.

The matching term is

\[
C_n(1)\,
2^\nu\Gamma(\nu+1)
z^{-\nu}
\sqrt{\frac{2}{\pi z}}
\cos\left(z-\frac{\lambda\pi}{2}\right).
\]

Its phase is correct because

\[
\nu\frac{\pi}{2}+\frac{\pi}{4}
=
\frac{\lambda\pi}{2}.
\]

That's good.

But the amplitude matching requires asymptotic expansion of

\[
C_n^{(\lambda)}(1)
\]

and

\[
\sin\theta
\]

to compatible orders.

You currently mix:

exact \(C_n(1)\),

leading WKB,

leading Bessel,

leading large-\(z\) Bessel asymptotic.


That is not a consistently ordered composite expansion.


---

6. Much worse: composite_matched_approx explodes at θ = 0

You have:

z_safe = np.where(z == 0, 1e-15, z)

but then calculate

z_safe ** (-nu)

and

sqrt(2.0 / (pi * z_safe))

The individual pieces diverge.

You are relying on cancellation between huge terms.

At exactly zero, this is particularly bad.

For example, conceptually:

\[
Bessel(0)=C_n(1)
\]

is perfectly finite, while your WKB term behaves like

\[
\theta^{-\lambda}.
\]

The matching term also behaves like

\[
\theta^{-\lambda}.
\]

So you're numerically computing

\[
10^{100}+10^{-?}-10^{100}
\]

rather than evaluating the finite composite limit.

That is not acceptable for a numerical asymptotics package.

Minimum repair

Explicitly switch at the endpoint:

if theta == 0:
    return 1.0

But that's only a patch.

The better solution is to formulate the composite in a way that doesn't require catastrophic cancellation.


---

7. There is an even bigger endpoint problem: θ = π

This is the most important conceptual bug.

Your Bessel expansion is for

\[
\theta\approx0.
\]

Your WKB expression has singular amplitude at both

\[
\theta=0,\pi.
\]

But the other endpoint has its own Bessel layer.

Use the symmetry

\[
C_n^{(\lambda)}(-x)
=
(-1)^nC_n^{(\lambda)}(x),
\]

hence

\[
\phi_n(\pi-\theta)
=
(-1)^n\phi_n(\theta).
\]

Therefore near \(\theta=\pi\), define

\[
\delta=\pi-\theta
\]

and use

\[
\phi_n(\theta)
\approx
(-1)^n
\mathcal J_{\lambda-1/2}
((n+\lambda)\delta).
\]

Your current composite has no south-pole boundary layer.

So this claim:

Composite Matched Asymptotic Approximation valid uniformly across [0, pi - epsilon].

is false.

It isn't even uniformly valid over that interval in the usual numerical sense.


---

8. The phase map isn't really a phase map yet

This:

if n <= 100 or theta > np.pi - 1e-3:
    return "direct_recurrence"
elif z <= 10.0:
    return "endpoint_approximation"
elif z <= sqrtK:
    return "overlap_approximation"
else:
    return "interior_approximation"

contains several arbitrary constants:

100

1e-3

10

sqrtK


and they don't all represent mathematical regime boundaries.

The particularly problematic one is:

n <= 100

That's not a phase boundary.

For \(n=99\), \(\theta=10^{-6}\), you absolutely are in an endpoint scaling regime.

For \(n=101\), \(\theta=\pi/2\), you're in the interior.

Yet the first condition completely overrides the actual asymptotic variables.

I'd remove the degree cutoff from the mathematical classifier.


---

9. The natural variables are actually two boundary-layer coordinates

Instead of a single

\[
z=(n+\lambda)\theta,
\]

you need

\[
z_0=(n+\lambda)\theta
\]

and

\[
z_\pi=(n+\lambda)(\pi-\theta).
\]

Then your map becomes conceptually:

interior
          z0 >> 1              zπ >> 1
              \                  /
               \                /
                \              /
                 endpoint / endpoint
                 z0 = O(1) / zπ = O(1)

More explicitly:

North endpoint

\[
z_0=K\theta=O(1).
\]

South endpoint

\[
z_\pi=K(\pi-\theta)=O(1).
\]

Interior

\[
z_0\gg1,\qquad z_\pi\gg1.
\]

Overlap

\[
1\ll z_0\ll K
\]

for the north boundary layer, with the analogous condition at the south pole.

That is a much more defensible phase map.


---

10. The sqrt(K) overlap criterion has some justification, but shouldn't be presented as canonical

You currently use

\[
10 < z < \sqrt K.
\]

The idea is understandable: you want

\[
z\gg1
\]

for Bessel asymptotics while simultaneously retaining

\[
\theta=\frac zK\ll1.
\]

If

\[
z\ll K,
\]

then

\[
\sin\theta\sim\theta.
\]

Choosing

\[
z\ll\sqrt K
\]

is a conservative numerical overlap window because

\[
\frac zK=O(K^{-1/2}).
\]

But sqrtK is a design choice, not an intrinsic Gegenbauer phase boundary.

Make it configurable:

overlap_upper = K ** overlap_exponent

with perhaps

\[
0<\alpha<1
\]

and

\[
z<K^\alpha.
\]

Then you can empirically determine which \(\alpha\) minimizes error.

That turns the "phase map" into an experimentally validated numerical object rather than a collection of magic numbers.


---

11. Orthogonality verification should use θ

Current implementation:

w = (1.0 - x**2) ** (lambda_val - 0.5)

is mathematically okay, but numerically unpleasant.

Make

\[
x=\cos\theta,
\qquad dx=-\sin\theta\,d\theta.
\]

Then

\[
(1-x^2)^{\lambda-1/2}dx
=
\sin^{2\lambda}\theta\,d\theta.
\]

So

\[
h_n=
\int_0^\pi
C_n^{(\lambda)}(\cos\theta)^2
\sin^{2\lambda}\theta\,d\theta.
\]

This removes the explicit fractional-power singularity in 1-x².

For the spherical problem this is also the natural geometric measure.

That should be the primary verification integral.


---

12. You're missing the most useful verification: normalized error maps

The module claims:

> "Operational Phase Map Classifier & Error Diagram E(n, theta)"



but there is no actual error-diagram function.

That's the biggest missing computational component.

You need something like

\[
E(n,\theta)
=
|\phi_n^{\rm approx}(\theta)
-
\phi_n^{\rm exact}(\theta)|.
\]

But I'd calculate several errors:

Absolute

\[
E_{\rm abs}=|\phi_{\rm approx}-\phi_{\rm exact}|.
\]

Relative

\[
E_{\rm rel}
=
\frac{|\phi_{\rm approx}-\phi_{\rm exact}|}
     {\max(|\phi_{\rm exact}|,\epsilon)}.
\]

Scaled relative

Because zeros of the Gegenbauer function make ordinary relative error meaningless, also use

\[
E_{\rm scale}
=
|\phi_{\rm approx}-\phi_{\rm exact}|.
\]

with the natural fact that

\[
|\phi_n|\le1
\]

for the spherical zonal functions.

Then produce the actual \((n,\theta)\) map.


---

13. The code should compare phi, not C_n

This is another architectural issue.

Your stated objective is zonal spherical functions:

\[
\phi_n=
\frac{C_n}{C_n(1)}.
\]

But the asymptotic functions return unnormalized \(C_n\).

For example:

interior_wkb_approx(...)

returns approximately \(C_n\), not \(\phi_n\).

Likewise:

mehler_heine_bessel_approx(...)

does return approximately \(C_n\), because you multiply by c_n_1.

That's internally consistent, but the API doesn't make it clear.

I'd explicitly split:

gegenbauer_exact
gegenbauer_wkb
gegenbauer_endpoint_bessel

phi_exact
phi_wkb
phi_endpoint_bessel

Then you can test both layers independently.

For the spherical problem, the normalized forms are actually much better conditioned.


---

14. There is a beautiful simplification you should exploit

For normalized spherical functions,

\[
\phi_n(\theta)
=
\frac{C_n^\lambda(\cos\theta)}
     {C_n^\lambda(1)}.
\]

Since

\[
C_n^\lambda(1)
\sim
\frac{n^{2\lambda-1}}{\Gamma(2\lambda)},
\]

the interior asymptotic becomes

\[
\phi_n(\theta)
\sim
\frac{2^{1-\lambda}\Gamma(2\lambda)}
     {\Gamma(\lambda)}
\frac{n^{-\lambda}}
     {(\sin\theta)^\lambda}
\cos\left((n+\lambda)\theta-\frac{\lambda\pi}{2}\right).
\]

Using duplication,

\[
\Gamma(2\lambda)
=
\frac{2^{2\lambda-1}}{\sqrt\pi}
\Gamma(\lambda)\Gamma(\lambda+\tfrac12),
\]

so

\[
\boxed{
\phi_n(\theta)
\sim
\frac{2^\lambda\Gamma(\lambda+\frac12)}
{\sqrt\pi}
\frac{
\cos((n+\lambda)\theta-\lambda\pi/2)
}{
(n\sin\theta)^\lambda
}
}
\]

to leading order.

This is much cleaner for the actual spherical problem.

And it immediately exposes the scaling:

\[
\phi_n=O((n\sin\theta)^{-\lambda}).
\]

That's a much better numerical quantity to use for deciding when recurrence versus asymptotics is worthwhile.


---

15. The special anchors should be upgraded substantially

You currently have:

\(S^2\): Legendre

\(S^3\): exact sine quotient

\(S^4\): normalized \(C_n^{3/2}\)


Good selection.

But they should become automated regression tests, not merely demo outputs.

Especially:

\(S^2\)

\[
\lambda=\frac12,
\qquad
\phi_n=P_n(\cos\theta).
\]

\(S^3\)

\[
\lambda=1,
\qquad
\phi_n=
\frac{\sin((n+1)\theta)}
{(n+1)\sin\theta}.
\]

\(S^4\)

\[
\lambda=\frac32.
\]

For \(S^3\), you have an exceptionally valuable property:

> the "asymptotic" WKB expression can be compared against an exact closed form at every \(n,\theta\).



That lets you determine the real error surface rather than trusting the formal derivation.


---

16. A better architecture

I'd reorganize the module into five layers.

gegenbauer.py
│
├── exact
│   ├── C_n
│   ├── C_n(1)
│   └── normalized phi
│
├── recurrence
│   └── stable spherical recurrence
│
├── asymptotic
│   ├── north Bessel
│   ├── south Bessel
│   ├── interior WKB
│   └── asymptotic expansions
│
├── regime
│   ├── z_north
│   ├── z_south
│   ├── classify
│   └── choose method
│
└── verification
    ├── point error
    ├── phase-map error
    ├── orthogonality
    └── exact anchors

This removes the current conceptual mixing.


---

17. Most importantly: don't use a composite until the individual approximations are validated

I'd implement the project in this order:

Phase 1 — exact numerical reference

normalized recurrence
        ↓
high-accuracy reference

Validate against SciPy at moderate \(n\).

Phase 2 — endpoint

Test

\[
\phi_n(\theta)
\]

against

\[
\mathcal J_{\lambda-1/2}(K\theta)
\]

for

\[
K\theta=0.1,\;1,\;5,\;10.
\]

Measure error versus \(n\).

Phase 3 — interior

Test WKB at

\[
\theta =
0.1,\;0.3,\;\pi/2,\;2.0,\;\pi-0.1.
\]

Phase 4 — south pole

Implement

\[
(-1)^n\mathcal J_{\lambda-1/2}
(K(\pi-\theta)).
\]

Phase 5 — overlap

Measure whether

\[
\text{Bessel}(z)-\text{WKB}(z)
\]

actually converges in the proposed overlap.

Phase 6 — composite

Only then construct the composite.

That gives you an empirical asymptotic error theory rather than assuming one.


---

One particularly important correction

I would change this claim:

"""Composite Matched Asymptotic Approximation valid uniformly across [0, pi - epsilon]."""

to something much more defensible:

"""
Leading composite approximation combining the north-pole Bessel layer
with the interior oscillatory approximation.

This is not a uniform approximation at the south pole and should be
validated numerically before use outside the overlap regime.
"""

Until you add the second boundary layer and consistent higher-order matching, "uniform" should disappear from the codebase.

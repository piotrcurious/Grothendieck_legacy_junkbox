# 3_repair.md: VIII-Layer Closed Architecture, Normalization, and Jacobi Operator Repair Plan

Verdict

The framework is substantially coherent, but I would not call it mathematically closed yet. The main problems are not cosmetic: two statements change the meaning of the architecture, and one normalization formula is simply wrong.

1. Critical errors

A. Layer II contains an incorrect normalization identity

You define

\[
\phi_n(x)=\frac{C_n^{(\lambda)}(x)}{C_n^{(\lambda)}(1)}.
\]

That is correct.

But then you write

\[
\phi_n(x)
=
\frac{\lambda}{n+\lambda}
\frac{C_n^{(\lambda)}(x)}
{\dim V_n/C_n^{(\lambda)}(1)}.
\]

Since

\[
\frac{\dim V_n}{C_n^{(\lambda)}(1)}
=
\frac{n+\lambda}{\lambda},
\]

your second expression becomes

\[
\frac{\lambda^2}{(n+\lambda)^2}C_n^{(\lambda)}(x),
\]

which is not \(\phi_n\).

The correct chain is simply

\[
\boxed{
C_n^{(\lambda)}(1)
=
\frac{(2\lambda)_n}{n!}
}
\]

and

\[
\boxed{
\phi_n(x)
=
\frac{C_n^{(\lambda)}(x)}{C_n^{(\lambda)}(1)}
}
\]

together with

\[
\boxed{
\dim V_n
=
\frac{n+\lambda}{\lambda}C_n^{(\lambda)}(1)
}
\]

or equivalently

\[
\boxed{
\dim V_n
=
\frac{n+\lambda}{\lambda}
\frac{(2\lambda)_n}{n!}.
}
\]

I would remove the erroneous intermediate expression completely.


---

B. The tensor-product statement in Layer IV is false as written

This is the most important structural problem:

> \(V_1\otimes V_n\cong V_{n+1}\oplus V_{n-1}\)



is not true for \(SO(d)\) in general.

For example, already for \(SO(3)\),

\[
V_1\otimes V_n
\cong
V_{n+1}\oplus V_n\oplus V_{n-1}.
\]

For higher \(d\), there is likewise an additional mixed-symmetry/hook representation.

The good news is that you do not need this representation-theoretic claim at all.

Your desired three-term recurrence follows directly from multiplication by the coordinate \(x\) in the radial/K-invariant sector:

\[
x\,\phi_n(x)
=
a_n\phi_{n+1}(x)+b_n\phi_{n-1}(x),
\]

with

\[
a_n=\frac{n+2\lambda}{2(n+\lambda)},
\qquad
b_n=\frac{n}{2(n+\lambda)}.
\]

Thus Layer IV should say something like:

\[
\boxed{
M_x:\mathcal H^K\to\mathcal H^K,
\qquad
(M_xf)(x)=xf(x)
}
\]

and

\[
\boxed{
M_x\phi_n
=
a_n\phi_{n+1}+b_n\phi_{n-1}.
}
\]

That is exact and sufficient.

You can then describe this as the Jacobi operator associated with the orthogonal polynomial system.

This is actually cleaner than the current representation-theoretic construction.


---

2. Layer III is correct, but the operator nomenclature should be tightened

The radial equation

\[
\phi''+2\lambda\cot\theta\,\phi'
+n(n+2\lambda)\phi=0
\]

is correct because

\[
2\lambda=d-2.
\]

The half-density substitution

\[
u=(\sin\theta)^\lambda\phi
\]

also gives

\[
-u''+
\lambda(\lambda-1)\csc^2\theta\,u
=
(n+\lambda)^2u.
\]

So this part is actually good.

But I would explicitly distinguish:

\[
E_n=n(n+2\lambda)
\]

from

\[
N^2=(n+\lambda)^2=E_n+\lambda^2.
\]

That distinction becomes important once you start constructing numerical solvers.

The exact transformed operator is

\[
\boxed{
H_\lambda
=
-\frac{d^2}{d\theta^2}
+
\lambda(\lambda-1)\csc^2\theta
}
\]

with eigenvalues

\[
\boxed{N_n^2=(n+\lambda)^2.}
\]

This is a beautiful structural layer and should probably be one of the central invariants of the framework.


---

3. Layer I is mathematically sound

The quotient

\[
R(Q)=\mathbb C[z_1,\ldots,z_d]/(q)
\]

with

\[
q=\sum z_i^2
\]

is the right homogeneous-coordinate-ring construction.

The identification

\[
R(Q)_n
\simeq
\operatorname{Sym}^n(\mathbb C^d)/
q\operatorname{Sym}^{n-2}(\mathbb C^d)
\]

and its identification with harmonic tensors is correct.

Likewise,

\[
H_{R(Q)}(t)
=
\frac{1-t^2}{(1-t)^d}
\]

is correct.

The dimension formula

\[
\dim V_n
=
\binom{n+d-1}{d-1}
-
\binom{n+d-3}{d-1}
\]

is correct.

I would, however, change one conceptual phrase:

> "\(R(Q)_n\) is isomorphic to the space of spherical harmonics"



strictly speaking, \(R(Q)_n\) is naturally the irreducible harmonic representation, while actual spherical harmonics are its realization as functions on \(S^{d-1}\).

So distinguish

\[
V_n
\simeq
\mathcal H_n(\mathbb C^d)
\]

from

\[
\mathscr Y_n(S^{d-1})
\]

when discussing functions.

That will make later operator maps much cleaner.


---

4. Layer II should explicitly separate three objects

At present, the framework mixes:

1. the abstract representation \(V_n\),


2. the \(K\)-fixed line \(V_n^K\),


3. the radial function \(\phi_n(\cos\theta)\).



They are related but not identical.

A cleaner architecture is

\[
V_n
\supset
V_n^K
=
\mathbb C v_n
\]

and then

\[
v_n
\longmapsto
\phi_n(gK)
=
\langle v_n,\pi_n(g)v_n\rangle.
\]

Finally radialization gives

\[
gK\longmapsto x=\cos\theta
\]

and hence

\[
\phi_n(gK)=\phi_n(x).
\]

That makes the morphism chain explicit:

\[
\boxed{
V_n
\to V_n^K
\to C^\infty(K\backslash G/K)
\to C^\infty([-1,1]).
}
\]

That is more rigorous than calling all of them \(\phi_n\).


---

5. Layer IV should really be called the Jacobi spectral layer

The recurrence itself is excellent:

\[
\phi_{n+1}
=
\frac{2(n+\lambda)}{n+2\lambda}
x\phi_n
-
\frac{n}{n+2\lambda}\phi_{n-1}.
\]

And the normalized coefficients satisfy

\[
a_n+b_n=1.
\]

But the deeper structure is that \(x\) is multiplication by the coordinate and the \(\phi_n\) form a Jacobi polynomial basis.

The natural Hilbert-space operator is

\[
M_x f=x f
\]

with respect to the spherical radial measure

\[
d\mu_\lambda(x)
\propto
(1-x^2)^{\lambda-\frac12}dx.
\]

Thus the operator is a bounded self-adjoint Jacobi operator with spectrum

\[
\sigma(M_x)=[-1,1].
\]

This is useful computationally because it gives you a spectral invariant that is independent of the implementation backend.


---

6. The endpoint asymptotics are basically right

The definition

\[
\mathcal J_\nu(z)
=
2^\nu\Gamma(\nu+1)z^{-\nu}J_\nu(z)
\]

is a good normalization.

The north-pole limit

\[
\phi_n(\theta)
\sim
\mathcal J_{\lambda-\frac12}(N\theta)
\]

and south-pole version

\[
\phi_n(\theta)
\sim
(-1)^n
\mathcal J_{\lambda-\frac12}(N(\pi-\theta))
\]

have the correct structure.

Your interior asymptotic

\[
\phi_n(\theta)
\sim
\frac{2^\lambda\Gamma(\lambda+1/2)}
{\sqrt\pi}
\frac{
\cos(N\theta-\lambda\pi/2)
}{
(n\sin\theta)^\lambda
}
\]

also passes the important sanity checks.

For example:

\(\lambda=\frac12\): Legendre

\[
\phi_n=P_n(\cos\theta)
\]

gives

\[
P_n(\cos\theta)
\sim
\sqrt{\frac{2}{\pi n\sin\theta}}
\cos\left((n+\tfrac12)\theta-\frac{\pi}{4}\right),
\]

exactly matching your formula.

\(\lambda=1\): \(S^3\)

\[
\phi_n
=
\frac{\sin((n+1)\theta)}
{(n+1)\sin\theta},
\]

and your asymptotic becomes

\[
\frac{\sin((n+1)\theta)}
{n\sin\theta},
\]

which is asymptotically correct.

So the constants are not the problem.

The problem is that Layer VI currently gives approximations without an error model.

For an actual computational framework you need something like

\[
\phi_n(\theta)
=
A_0(n,\theta)
+
A_1(n,\theta)n^{-1}
+\cdots+
R_K(n,\theta)
\]

with

\[
|R_K(n,\theta)|
\le
B_K(n,\theta).
\]

Otherwise you have asymptotic formulas, but not a numerical solver-selection mechanism.


---

7. The phase-space partition needs one more dimension

Your current partition is essentially

\[
\text{endpoint} \quad / \quad \text{interior}.
\]

For numerical work I would make it explicitly three-dimensional:

\[
\boxed{
(n,\theta,\lambda)
}
\]

with regimes such as

\[
N\theta\lesssim 1
\]

north endpoint,

\[
N(\pi-\theta)\lesssim1
\]

south endpoint,

and

\[
N\sin\theta\gg1
\]

interior oscillatory.

But there is an important transition region where neither crude Bessel nor crude WKB should be trusted without a uniform formula.

So I would introduce

\[
\mathcal R_{\rm north},
\quad
\mathcal R_{\rm south},
\quad
\mathcal R_{\rm interior},
\quad
\mathcal R_{\rm transition}.
\]

That will make your Layer VII solver dispatch mathematically meaningful rather than heuristic.


---

8. The numerical backend abstraction needs refinement

FLOAT32 / FLOAT64

Fine.

LONGDOUBLE

Not a portable precision class.

On one platform this may mean IEEE 80-bit extended precision; on another it can simply be binary64.

So represent it as a backend property:

precision_bits
exponent_bits
mantissa_bits
rounding_model

rather than assuming long double means a particular precision.

Q16.16

This is fine as an implementation target, but the definition

\[
x_{\rm fp}
=
\lfloor65536x\rfloor/65536
\]

does not correctly describe signed rounding behavior of a real Q16.16 implementation.

You should define the quantizer explicitly, e.g.

\[
Q_{16.16}(x)
=
\operatorname{round}(2^{16}x)\,2^{-16}
\]

and separately specify saturation.

LNS

"LNS" is not enough to define a numerical algorithm.

Because \(\phi_n(x)\) crosses zero, you need signed logarithms:

\[
x=s_xe^{\ell_x},
\qquad
s_x\in\{-1,0,+1\}.
\]

Addition becomes

\[
s_1e^{\ell_1}+s_2e^{\ell_2}
\]

which requires a stable signed log-add operation.

Near zero, the representation can become pathological.

So LNS should be treated as a specialized backend, not as a drop-in numerical type.


---

9. The error metric is useful, but it isn't really an "error surface" yet

You currently have

\[
E_{\rm mixed}
=
\frac{|\phi^{\rm approx}-\phi^{\rm ref}|}
{\mathrm{atol}+\mathrm{rtol}|\phi^{\rm ref}|}.
\]

Good as a pass/fail metric.

But for your framework I would define at least three independent errors:

Forward error

\[
E_f
=
|\hat\phi-\phi|.
\]

Relative error

\[
E_r
=
\frac{|\hat\phi-\phi|}
{\max(|\phi|,\tau)}.
\]

Recurrence residual

This one is particularly important:

\[
R_n
=
\hat\phi_{n+1}
-
\frac{2(n+\lambda)}{n+2\lambda}x\hat\phi_n
+
\frac{n}{n+2\lambda}\hat\phi_{n-1}.
\]

A solver can have a surprisingly small pointwise error while accumulating a structurally bad recurrence residual, or vice versa.

I would therefore define

\[
\boxed{
E_{\rm total}
=
F(E_f,E_r,E_{\rm recurrence},E_{\rm asymptotic})
}
\]

rather than collapsing everything into one scalar.


---

10. The 100-bit reference does not automatically validate the algorithm

This is a subtle but important point.

If you use mpmath to generate the reference, the meaningful validation protocol is more like

high-precision independent reference
        ↓
exact identities
        ↓
recurrence residual
        ↓
asymptotic residual
        ↓
backend forward error

rather than simply

mpmath(100 bits) → compare → result

For example, special test anchors should include:

\[
\phi_n(1)=1,
\]

\[
\phi_n(-1)=(-1)^n,
\]

\[
\phi_0=1,
\]

\[
\phi_1=x,
\]

\[
\phi_n'(1)
=
\frac{n(n+2\lambda)}
{2\lambda+1}
\]

for the appropriate normalized Gegenbauer derivative relation, and exact special families:

\[
\lambda=\frac12:\quad P_n(x),
\]

\[
\lambda=1:\quad
\frac{U_n(x)}{n+1}.
\]

These are much stronger tests than a single arbitrary high-precision reference.


---

Recommended corrected architecture

I would modify your eight layers to:

I. Representation Geometry
   SO(d)/SO(d-1)
   harmonic representation V_n
   quadric quotient R(Q)_n

II. Spherical Fixed Line
   V_n^K = C v_n
   matrix coefficient φ_n(g)
   radial realization φ_n(cos θ)

III. Exact Differential Operators
   Gegenbauer ODE
   radial Laplacian
   Schrödinger operator H_λ

IV. Jacobi Operator
   M_x f = x f
   three-term recurrence
   spectral measure
   self-adjoint Jacobi matrix

V. Endpoint Scaling
   Nθ
   N(π-θ)
   endpoint normal forms

VI. Uniform Asymptotics
   Bessel / Hilb
   interior WKB
   transition formula
   explicit remainder bounds

VII. Numerical Execution
   recurrence
   asymptotic evaluators
   regime dispatcher
   float/fixed/LNS/mp backends

VIII. Verification
   exact identities
   recurrence residuals
   endpoint residuals
   reference error
   conditioning/error surfaces

That version has a much cleaner mathematical flow:

\[
\boxed{
\text{representation}
\rightarrow
\text{radial function}
\rightarrow
\text{ODE}
\rightarrow
\text{Jacobi operator}
\rightarrow
\text{scaling}
\rightarrow
\text{asymptotics}
\rightarrow
\text{algorithm}
\rightarrow
\text{verification}.
}
\]

Overall assessment

I'd rate the current framework roughly 8/10 mathematically, but only 6/10 as a production computational specification.

The strong parts are the \(SO(d)/SO(d-1)\) geometry, quadric/Hilbert-series layer, exact radial–Schrödinger transformation, normalized recurrence, and endpoint/interior asymptotic structure.

The things that need fixing before treating it as a closed framework are:

\[
\boxed{
\text{incorrect Layer-II formula}
}
\]

\[
\boxed{
\text{incorrect }V_1\otimes V_n\text{ decomposition}
}
\]

\[
\boxed{
\text{missing transition asymptotic regime}
}
\]

\[
\boxed{
\text{missing remainder/error bounds}
}
\]

\[
\boxed{
\text{LNS/Q16.16 not specified as actual numerical models}
}
\]

\[
\boxed{
\text{verification needs structural residuals, not only high-precision comparison}
}
\]
